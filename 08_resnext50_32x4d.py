"""
Recursion Cellular Image Classification
========================================
ResNeXt50-32x4d with 6-channel input, experiment-aware split,
per-channel z-score normalization, mixed precision training,
backbone freeze warmup, cosine LR decay, and dual-site TTA.
"""

import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from tqdm import tqdm
import cv2
import warnings

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    DATA_DIR    = '/kaggle/input/competitions/recursion-cellular-image-classification'
    TRAIN_CSV   = f'{DATA_DIR}/train.csv'
    TEST_CSV    = f'{DATA_DIR}/test.csv'
    PIXEL_STATS = f'{DATA_DIR}/pixel_stats.csv'

    MODEL_NAME   = 'resnext50_32x4d'
    IMG_SIZE     = 384
    BATCH_SIZE   = 16           # Reduced from 24 — safer with mixed precision at 384px
    EPOCHS       = 30
    WARMUP_EP    = 1            # Epochs with backbone frozen
    LR           = 3e-4
    MIN_LR       = 1e-6
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTH = 0.1
    NUM_WORKERS  = 4
    SEED         = 42
    NUM_CLASSES  = 1108         # Updated dynamically in main()
    VAL_EXP_FRAC = 0.15        # Fraction of experiments held out for val


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(Config.SEED)


# =============================================================================
# NORMALIZATION STATS
# =============================================================================

def build_stats_lookup(pixel_stats_path):
    """
    Returns a dict keyed by (experiment, plate, well, site, channel)
    with values (mean, std). Falls back to (127.5, 64.0) if key missing.
    """
    stats  = pd.read_csv(pixel_stats_path)
    lookup = {}
    for _, row in stats.iterrows():
        key = (row['experiment'], int(row['plate']),
               row['well'],       int(row['site']),
               int(row['channel']))
        lookup[key] = (float(row['mean']), float(row['std']))
    return lookup


# =============================================================================
# DATASET
# =============================================================================

class CellularDataset(Dataset):
    def __init__(self, df, data_dir, stats_lookup, mode='train', site=None):
        self.df           = df.reset_index(drop=True)
        self.data_dir     = data_dir
        self.stats_lookup = stats_lookup
        self.mode         = mode
        self.site         = site

    def __len__(self):
        return len(self.df)

    def _load_channels(self, row, site):
        """Load 6 grayscale channels, z-score each independently."""
        exp   = row['experiment']
        plate = int(row['plate'])
        well  = row['well']
        split = 'test' if self.mode == 'test' else 'train'
        base  = f'{self.data_dir}/{split}/{exp}/Plate{plate}/{well}_s{site}_w'

        channels = []
        for ch in range(1, 7):
            path = f'{base}{ch}.png'
            img  = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) \
                   else np.zeros((512, 512), dtype=np.uint8)
            img  = img.astype(np.float32)

            key        = (exp, plate, well, int(site), ch)
            mean, std  = self.stats_lookup.get(key, (127.5, 64.0))
            img        = (img - mean) / (std + 1e-6)
            channels.append(img)

        stacked = np.stack(channels, axis=-1)                   # H x W x 6
        stacked = cv2.resize(stacked, (Config.IMG_SIZE, Config.IMG_SIZE))
        return stacked

    def _augment(self, img):
        """Spatial augmentations — train only."""
        if self.mode != 'train':
            return img
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        if np.random.rand() > 0.5:
            img = np.rot90(img, k=np.random.randint(1, 4)).copy()
        return img

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        site = np.random.choice(['1', '2']) if self.mode == 'train' \
               else (self.site or '1')

        img = self._load_channels(row, site)
        img = self._augment(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float()   # 6 x H x W

        if self.mode in ('train', 'val'):
            return img, int(row['label'])
        else:
            return img, row['id_code']


# =============================================================================
# MODEL
# =============================================================================

class CellularModel(nn.Module):
    """
    ResNeXt50-32x4d with the first conv modified to accept 6 channels.
    Pretrained RGB weights are replicated to channels 4-6.
    """
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)

        # ResNeXt uses 'conv1' as the first layer
        if hasattr(self.backbone, 'conv1'):
            old = self.backbone.conv1
            new = nn.Conv2d(
                in_channels, old.out_channels,
                kernel_size=old.kernel_size,
                stride=old.stride,
                padding=old.padding,
                bias=old.bias is not None          # correctly passes bool
            )
            with torch.no_grad():
                new.weight[:, :3, ...] = old.weight        # copy RGB weights
                new.weight[:, 3:, ...] = old.weight        # replicate to ch 4-6
                if old.bias is not None:
                    # clone into a new Parameter — do NOT share the tensor
                    new.bias = nn.Parameter(old.bias.clone())
            self.backbone.conv1 = new

        n_feat = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)   # remove original head (timm-native)

        self.head = nn.Sequential(
            nn.BatchNorm1d(n_feat),
            nn.Dropout(0.4),
            nn.Linear(n_feat, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# =============================================================================
# LR SCHEDULE  (linear warmup → cosine decay to MIN_LR)
# =============================================================================

def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine   = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(Config.MIN_LR / Config.LR, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = correct = total = 0

    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        with autocast():
            out  = model(imgs)
            loss = criterion(out, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # gradient clipping
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()                    # step per batch, not per epoch

        total_loss += loss.item()
        _, pred     = out.max(1)
        correct    += pred.eq(labels).sum().item()
        total      += labels.size(0)
        pbar.set_postfix(loss=f'{total_loss/len(loader):.4f}',
                         acc=f'{100.*correct/total:.2f}%')

    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val  ', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                out  = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item()
            _, pred     = out.max(1)
            correct    += pred.eq(labels).sum().item()
            total      += labels.size(0)

    return total_loss / len(loader), 100. * correct / total


# =============================================================================
# TEST-TIME AUGMENTATION
# =============================================================================

def predict_tta(model, test_df, data_dir, stats_lookup, device):
    """
    6 TTA passes: site1/site2 × {original, hflip, vflip}
    Returns (predictions_array, id_codes_list).
    """
    model.eval()
    tta_configs = [
        ('1', None), ('1', 'h'), ('1', 'v'),
        ('2', None), ('2', 'h'), ('2', 'v'),
    ]
    all_probs = []
    ids       = None

    for site, flip in tta_configs:
        ds     = CellularDataset(test_df, data_dir, stats_lookup,
                                 mode='test', site=site)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE,
                            shuffle=False, num_workers=Config.NUM_WORKERS,
                            pin_memory=True)            # fixed: pin_memory added

        probs_list = []
        id_list    = []

        with torch.no_grad():
            for imgs, img_ids in tqdm(loader,
                                      desc=f'TTA site{site} {flip or "orig"}',
                                      leave=False):
                imgs = imgs.to(device)
                if flip == 'h':
                    imgs = torch.flip(imgs, dims=[3])
                elif flip == 'v':
                    imgs = torch.flip(imgs, dims=[2])
                with autocast():
                    out = model(imgs)
                probs_list.append(out.softmax(dim=1).cpu())
                id_list.extend(img_ids)

        all_probs.append(torch.cat(probs_list, dim=0))
        if ids is None:
            ids = id_list

    avg_probs = torch.stack(all_probs).mean(dim=0)
    preds     = avg_probs.argmax(dim=1).numpy()
    return preds, ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Loading CSVs...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df  = pd.read_csv(Config.TEST_CSV)

    # Guard against both integer and string sirna formats
    if train_df['sirna'].dtype == object:
        train_df['sirna_id'] = train_df['sirna'].str.extract(r'(\d+)').astype(int)
    else:
        train_df['sirna_id'] = train_df['sirna'].astype(int)

    unique_sirnas      = sorted(train_df['sirna_id'].unique())
    sirna_to_label     = {s: i for i, s in enumerate(unique_sirnas)}
    label_to_sirna     = {i: s for s, i in sirna_to_label.items()}
    train_df['label']  = train_df['sirna_id'].map(sirna_to_label)
    Config.NUM_CLASSES = len(unique_sirnas)

    print(f"Classes      : {Config.NUM_CLASSES}")
    print(f"Train wells  : {len(train_df)}")
    print(f"Test wells   : {len(test_df)}")

    # Experiment-aware split — no experiment leaks into val
    rng      = np.random.default_rng(Config.SEED)
    exps     = train_df['experiment'].unique()
    n_val    = max(1, int(len(exps) * Config.VAL_EXP_FRAC))
    val_exps = set(rng.choice(exps, size=n_val, replace=False))
    val_df   = train_df[train_df['experiment'].isin(val_exps)].reset_index(drop=True)
    tr_df    = train_df[~train_df['experiment'].isin(val_exps)].reset_index(drop=True)
    print(f"Train exps   : {len(exps)-n_val}  ({len(tr_df)} wells)")
    print(f"Val exps     : {n_val}  ({len(val_df)} wells)")

    print("Building pixel-stats lookup...")
    stats_lookup = build_stats_lookup(Config.PIXEL_STATS)

    train_loader = DataLoader(
        CellularDataset(tr_df,  Config.DATA_DIR, stats_lookup, mode='train'),
        batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        CellularDataset(val_df, Config.DATA_DIR, stats_lookup,
                        mode='val', site='1'),
        batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )

    print(f"Building {Config.MODEL_NAME}...")
    model     = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTH)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    scaler    = GradScaler()

    total_steps  = Config.EPOCHS * len(train_loader)
    warmup_steps = Config.WARMUP_EP * len(train_loader)
    scheduler    = build_scheduler(optimizer, total_steps, warmup_steps)

    best_acc  = 0.0
    best_path = 'best_resnext50_32x4d.pth'

    # Phase 1: freeze backbone so the new head stabilises first
    print(f"\nPhase 1 — backbone frozen for {Config.WARMUP_EP} epoch(s)")
    model.freeze_backbone()

    for epoch in range(Config.EPOCHS):
        if epoch == Config.WARMUP_EP:
            model.unfreeze_backbone()
            print(f"\nPhase 2 — full fine-tune from epoch {epoch+1}")

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")

        tr_loss, tr_acc = train_epoch(model, train_loader, criterion,
                                      optimizer, scheduler, scaler, device)
        vl_loss, vl_acc = validate(model, val_loader, criterion, device)

        print(f"  Train  loss={tr_loss:.4f}  acc={tr_acc:.2f}%")
        print(f"  Val    loss={vl_loss:.4f}  acc={vl_acc:.2f}%")

        if vl_acc > best_acc:
            best_acc = vl_acc
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Saved best model (val acc={best_acc:.2f}%)")

    print(f"\nLoading best weights (val acc={best_acc:.2f}%)...")
    model.load_state_dict(torch.load(best_path, map_location=device))  # fixed: map_location

    print("Running TTA inference...")
    preds, ids = predict_tta(model, test_df, Config.DATA_DIR, stats_lookup, device)

    sirna_preds = [label_to_sirna[int(p)] for p in preds]
    submission  = pd.DataFrame({'id_code': ids, 'sirna': sirna_preds})
    submission.to_csv('submission_resnext50_32x4d.csv', index=False)
    print(f"\nDone. submission_resnext50_32x4d.csv saved  |  Best val acc: {best_acc:.2f}%")
    print(submission.head())


if __name__ == '__main__':
    main()
