"""
Vision Transformer Base - Optimized Version
============================================
ViT-B/16 with proper normalization, augmentations, and mixed precision.
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Config
DATA_DIR = '/kaggle/input/competitions/recursion-cellular-image-classification'
TRAIN_CSV = f'{DATA_DIR}/train.csv'
TEST_CSV = '/kaggle/input/datasets/himanshusardana2/corrected-test-csv-recurrence-cellular/test.csv'

MODEL_NAME = 'vit_base_patch16_224'
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4  # Lower LR for ViT fine-tuning
NUM_WORKERS = 2
SEED = 42
CELL_TYPES = ['HUVEC']
GRAD_CLIP = 1.0  # Gradient clipping

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# ImageNet normalization for 6 channels (RGB replicated)
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406, 0.485, 0.456, 0.406]).view(6, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225, 0.229, 0.224, 0.225]).view(6, 1, 1)


# ============== Dataset with Augmentation ==============
class SimpleDataset(Dataset):
    def __init__(self, df, data_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def augment(self, img):
        """Basic augmentations for training."""
        if self.mode != 'train':
            return img
        
        # Horizontal flip
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        
        # Vertical flip
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        
        # 90-degree rotation
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k=k).copy()
        
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exp, plate, well = row['experiment'], row['plate'], row['well']
        
        prefix = 'test' if self.mode == 'test' else 'train'
        base = f'{self.data_dir}/{prefix}/{exp}/Plate{plate}/{well}_s1_w'
        
        # Load 6 channels
        channels = []
        for i in range(1, 7):
            img = cv2.imread(f'{base}{i}.png', cv2.IMREAD_GRAYSCALE)
            channels.append(img if img is not None else np.zeros((512, 512), dtype=np.uint8))
        
        # Stack and resize
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        
        # Augment
        img = self.augment(img)
        
        # Convert to tensor and normalize [0, 1]
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        
        # Apply ImageNet normalization
        img = (img - IMAGENET_MEAN) / IMAGENET_STD
        
        if self.mode == 'train':
            return img, row['label']
        return img, row['id_code']


# ============== Model ==============
class ViTWrapper(nn.Module):
    """Wrapper to properly handle ViT forward pass with custom classifier."""
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        n_features = backbone.num_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Dropout(0.2),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def create_model(num_classes):
    # Create backbone with no classifier
    backbone = timm.create_model(MODEL_NAME, pretrained=True, in_chans=3, num_classes=0)
    
    # Modify patch embedding for 6 channels
    old_conv = backbone.patch_embed.proj
    
    # Extract values
    out_ch = int(old_conv.out_channels)
    kernel = int(old_conv.kernel_size[0]) if hasattr(old_conv.kernel_size, '__len__') else int(old_conv.kernel_size)
    stride = int(old_conv.stride[0]) if hasattr(old_conv.stride, '__len__') else int(old_conv.stride)
    padding = int(old_conv.padding[0]) if hasattr(old_conv.padding, '__len__') else int(old_conv.padding)
    
    backbone.patch_embed.proj = nn.Conv2d(6, out_ch, kernel, stride, padding)
    
    # Fix: Don't divide by 6 again (already averaged)
    with torch.no_grad():
        w = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 6, 1, 1)
        backbone.patch_embed.proj.weight.copy_(w)
        if old_conv.bias is not None:
            backbone.patch_embed.proj.bias.copy_(old_conv.bias)
    
    # Wrap with custom classifier
    model = ViTWrapper(backbone, num_classes)
    return model


# ============== Training with Mixed Precision ==============
def train_epoch(model, loader, criterion, optimizer, scaler, num_classes):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision forward
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        
        # Scaled backward
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        
        scaler.step(optimizer)
        scaler.update()
        
        loss_sum += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        pbar.set_postfix({
            'loss': f'{loss_sum/len(loader):.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return loss_sum / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return loss_sum / len(loader), 100 * correct / total


# ============== Main ==============
def main():
    print("=" * 60)
    print("ViT-Base/16 - Optimized (With Augmentation + Mixed Precision)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    train_df = train_df[train_df['cell_type'].isin(CELL_TYPES)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    
    # Labels
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {s: i for i, s in enumerate(sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    num_classes = len(sirnas)
    
    print(f"Classes: {num_classes}")
    
    # Split
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=SEED
    )
    
    # Dataloaders
    train_loader = DataLoader(
        SimpleDataset(train_data, DATA_DIR, mode='train'),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        SimpleDataset(val_data, DATA_DIR, mode='train'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True
    )
    
    # Model
    print(f"\nCreating {MODEL_NAME}...")
    model = create_model(num_classes).to(device)
    
    # Label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = GradScaler()  # Mixed precision scaler
    
    # Training
    best_acc, patience, counter = 0, 3, 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, scaler, num_classes)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'best_vit_base.pth')
            print(f"  → Saved best: {best_acc:.2f}%")
        else:
            counter += 1
            print(f"  → No improvement ({counter}/{patience})")
            if counter >= patience:
                print("Early stopping!")
                break
    
    # Inference
    print("\nInference...")
    model.load_state_dict(torch.load('best_vit_base.pth'))
    model.eval()
    
    test_loader = DataLoader(
        SimpleDataset(test_df, DATA_DIR, mode='test'),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS
    )
    
    preds, ids = [], []
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Test'):
            imgs = imgs.to(device)
            with autocast():
                out = model(imgs)
            preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(img_ids)
    
    # Submission
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    submission = pd.DataFrame({
        'id_code': ids,
        'sirna': [label_to_sirna[p] for p in preds]
    })
    submission.to_csv('submission_vit_base.csv', index=False)
    
    print(f"\n✓ Done! Best Val Acc: {best_acc:.2f}%")
    print("Saved submission_vit_base.csv")


if __name__ == '__main__':
    main()
