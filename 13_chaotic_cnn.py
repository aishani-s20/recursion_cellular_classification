"""
DenseNet121 - Chaotic CNN (Procedural)
=======================================
DenseNet121 enhanced with chaos-based feature transformation
as described in "Chaotic CNN for Limited Data Image Classification"
(Anusree M, Akhila Henry, Pramod P Nair, arXiv:2604.14645).

The chaotic transformation is inserted between the DenseNet feature
extractor and the classification head. Three maps are supported:
  - 'logistic' : x_{n+1} = r * x * (1 - x),  r=4 (max chaos)
  - 'skew_tent': piecewise-linear with p=0.499 (max entropy)
  - 'sine'     : x_{n+1} = sin(pi * x)

The transformation is applied element-wise to min-max normalised
feature vectors (so values stay in [0, 1]) and adds zero extra
trainable parameters.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# ─── Device ──────────────────────────────────────────────────────────────────
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ─── Config ───────────────────────────────────────────────────────────────────
DATA_DIR   = '/kaggle/input/competitions/recursion-cellular-image-classification'
TRAIN_CSV  = f'{DATA_DIR}/train.csv'
TEST_CSV   = '/kaggle/input/datasets/himanshusardana2/corrected-test-csv-recurrence-cellular/test.csv'

MODEL_NAME  = 'densenet121'
IMG_SIZE    = 320
BATCH_SIZE  = 32
EPOCHS      = 20
LR          = 3e-4
NUM_WORKERS = 2
SEED        = 42
CELL_TYPES  = ['HUVEC']

# ── Chaotic map selection ──────────────────────────────────────────────────────
# Choose one of: 'logistic', 'skew_tent', 'sine'
# Based on the paper's results:
#   • 'skew_tent' tends to give the strongest gain on MNIST and CIFAR-10
#   • 'logistic'  is most consistent on Fashion-MNIST style tasks
#   • 'sine'      is smooth and stable across all settings
CHAOTIC_MAP = 'skew_tent'

# ─── Seed ─────────────────────────────────────────────────────────────────────
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# ════════════════════════════════════════════════════════════════════════════
# Chaotic Transformation Layer
# ════════════════════════════════════════════════════════════════════════════

class ChaoticTransform(nn.Module):
    """
    Applies a single iteration of a chaotic map to a min-max normalised
    feature vector element-wise.  No trainable parameters are added.

    Maps (following the paper exactly):
      logistic  : f* = r * f̃ * (1 - f̃),  r = 4
      skew_tent : f* = f̃/p          if f̃ < p
                       (1-f̃)/(1-p)  if f̃ >= p,   p = 0.499
      sine      : f* = sin(π * f̃)
    """

    VALID_MAPS = ('logistic', 'skew_tent', 'sine')

    def __init__(self, map_type: str = 'skew_tent'):
        super().__init__()
        assert map_type in self.VALID_MAPS, \
            f"map_type must be one of {self.VALID_MAPS}, got '{map_type}'"
        self.map_type = map_type
        # Fixed hyper-parameters from the paper
        self.r = 4.0          # logistic: maximum-chaos region
        self.p = 0.499        # skew tent: maximum-entropy parameter

    def _normalize(self, f: torch.Tensor) -> torch.Tensor:
        """Min-max normalise each sample's feature vector to [0, 1]."""
        f_min = f.min(dim=1, keepdim=True).values
        f_max = f.max(dim=1, keepdim=True).values
        # Avoid division by zero for constant feature vectors
        denom = (f_max - f_min).clamp(min=1e-8)
        return (f - f_min) / denom

    def _logistic(self, f: torch.Tensor) -> torch.Tensor:
        return self.r * f * (1.0 - f)

    def _skew_tent(self, f: torch.Tensor) -> torch.Tensor:
        p = self.p
        return torch.where(f < p, f / p, (1.0 - f) / (1.0 - p))

    def _sine(self, f: torch.Tensor) -> torch.Tensor:
        return torch.sin(torch.pi * f)

    def forward(self, f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f: feature tensor of shape (B, D) – the raw pooled features
               from the DenseNet backbone.
        Returns:
            f*: transformed features of the same shape.
        """
        f_norm = self._normalize(f)          # → [0, 1]

        if self.map_type == 'logistic':
            f_star = self._logistic(f_norm)
        elif self.map_type == 'skew_tent':
            f_star = self._skew_tent(f_norm)
        else:  # 'sine'
            f_star = self._sine(f_norm)

        return f_star


# ════════════════════════════════════════════════════════════════════════════
# Dataset
# ════════════════════════════════════════════════════════════════════════════

class SimpleDataset(Dataset):
    def __init__(self, df, data_dir, mode='train'):
        self.df       = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode     = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exp, plate, well = row['experiment'], row['plate'], row['well']

        prefix = 'test' if self.mode == 'test' else 'train'
        base   = f'{self.data_dir}/{prefix}/{exp}/Plate{plate}/{well}_s1_w'

        channels = []
        for i in range(1, 7):
            img = cv2.imread(f'{base}{i}.png', cv2.IMREAD_GRAYSCALE)
            channels.append(img if img is not None else np.zeros((512, 512), dtype=np.uint8))

        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)

        if self.mode == 'train':
            return img, row['label']
        return img, row['id_code']


# ════════════════════════════════════════════════════════════════════════════
# Model  –  DenseNet121 + Chaotic Feature Transformation
# ════════════════════════════════════════════════════════════════════════════

class ChaoticDenseNet(nn.Module):
    """
    DenseNet121 backbone  →  Global Average Pool  →  ChaoticTransform
    →  Dropout  →  Linear classifier.

    The chaotic transformation sits between the pooled feature vector and
    the linear head, exactly as described in the paper (Section 3):

        f̃  = Normalize(f)         # element-wise to [0, 1]
        f* = chaotic_map(f̃)       # element-wise, no learnable params
        ŷ  = softmax(W f* + b)
    """

    def __init__(self, num_classes: int, map_type: str = 'skew_tent'):
        super().__init__()

        # ── Backbone ──────────────────────────────────────────────────────
        backbone = timm.create_model(MODEL_NAME, pretrained=True, in_chans=3)

        # Adapt the first conv to accept 6-channel microscopy images
        old_conv = backbone.features.conv0
        out_ch   = int(old_conv.out_channels)
        kernel   = int(old_conv.kernel_size[0]) if hasattr(old_conv.kernel_size, '__len__') else int(old_conv.kernel_size)
        stride   = int(old_conv.stride[0])      if hasattr(old_conv.stride,      '__len__') else int(old_conv.stride)
        padding  = int(old_conv.padding[0])     if hasattr(old_conv.padding,     '__len__') else int(old_conv.padding)

        backbone.features.conv0 = nn.Conv2d(6, out_ch, kernel, stride, padding, bias=False)
        with torch.no_grad():
            w = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 6, 1, 1) / 6
            backbone.features.conv0.weight.copy_(w)

        # Remove the original classifier; keep only the feature extractor
        n_features = backbone.get_classifier().in_features
        backbone.reset_classifier(0)   # returns bare feature vector after GAP
        self.backbone = backbone

        # ── Chaotic transformation (no new parameters) ─────────────────────
        self.chaotic = ChaoticTransform(map_type=map_type)

        # ── Classification head ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f      = self.backbone(x)          # (B, n_features) – GAP applied by timm
        f_star = self.chaotic(f)           # (B, n_features) – chaos-transformed
        return self.classifier(f_star)     # (B, num_classes)


# ════════════════════════════════════════════════════════════════════════════
# Training helpers
# ════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out  = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        correct  += (out.argmax(1) == labels).sum().item()
        total    += labels.size(0)

        pbar.set_postfix({
            'loss': f'{loss_sum / len(loader):.4f}',
            'acc':  f'{100. * correct / total:.2f}%',
        })

    return loss_sum / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val'):
            imgs, labels = imgs.to(device), labels.to(device)
            out  = model(imgs)
            loss = criterion(out, labels)

            loss_sum += loss.item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)

    return loss_sum / len(loader), 100 * correct / total


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print(f"DenseNet121 + Chaotic CNN  [{CHAOTIC_MAP} map]")
    print("=" * 65)

    # ── Load data ─────────────────────────────────────────────────────────
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df  = pd.read_csv(TEST_CSV)

    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    train_df = train_df[train_df['cell_type'].isin(CELL_TYPES)].reset_index(drop=True)
    print(f"Training samples: {len(train_df)}")

    # Labels
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    sirnas          = sorted(train_df['sirna_id'].unique())
    sirna_to_label  = {s: i for i, s in enumerate(sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    num_classes     = len(sirnas)
    print(f"Classes: {num_classes}")

    # Split
    train_data, val_data = train_test_split(
        train_df, test_size=0.15,
        stratify=train_df['label'], random_state=SEED,
    )

    # Dataloaders
    train_loader = DataLoader(
        SimpleDataset(train_data, DATA_DIR),
        batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True,
    )
    val_loader = DataLoader(
        SimpleDataset(val_data, DATA_DIR),
        batch_size=BATCH_SIZE, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    print(f"\nCreating ChaoticDenseNet121 [{CHAOTIC_MAP}]...")
    model = ChaoticDenseNet(num_classes=num_classes, map_type=CHAOTIC_MAP).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ── Training loop ─────────────────────────────────────────────────────
    best_acc, patience, counter = 0, 3, 0
    save_path = f'best_densenet121_chaotic_{CHAOTIC_MAP}.pth'

    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss,   val_acc   = validate(model, val_loader, criterion)
        scheduler.step()

        print(
            f"Train: loss={train_loss:.4f}  acc={train_acc:.2f}%  |  "
            f"Val: loss={val_loss:.4f}  acc={val_acc:.2f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            counter  = 0
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model: {best_acc:.2f}%")
        else:
            counter += 1
            print(f"  → No improvement ({counter}/{patience})")
            if counter >= patience:
                print("Early stopping!")
                break

    # ── Inference ─────────────────────────────────────────────────────────
    print("\nRunning inference...")
    model.load_state_dict(torch.load(save_path))
    model.eval()

    test_loader = DataLoader(
        SimpleDataset(test_df, DATA_DIR, mode='test'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    )

    preds, ids = [], []
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Test'):
            out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(img_ids)

    # ── Submission ────────────────────────────────────────────────────────
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    submission     = pd.DataFrame({
        'id_code': ids,
        'sirna':   [label_to_sirna[p] for p in preds],
    })
    out_csv = f'submission_densenet121_chaotic_{CHAOTIC_MAP}.csv'
    submission.to_csv(out_csv, index=False)

    print(f"\n✓ Done!  Best Val Acc: {best_acc:.2f}%")
    print(f"Saved {out_csv}")


if __name__ == '__main__':
    main()
