"""
Google Inception V3 - Inception Module Architecture
=====================================================
Inception V3 for cellular image classification.
Uses parallel multi-scale convolutions (Inception modules).
"""
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Config:
    DATA_DIR = '/kaggle/input/competitions/recursion-cellular-image-classification'
    TRAIN_CSV = f'{DATA_DIR}/train.csv'
    TEST_CSV = '/kaggle/input/datasets/himanshusardana2/corrected-test-csv-recurrence-cellular/test.csv'
    
    MODEL_NAME = 'inception_v3'  # Google Inception V3
    IMG_SIZE = 299  # Inception V3 native size
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 3e-4
    
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108
    
    CELL_TYPES = ['HUVEC']


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)


class CellularDataset(Dataset):
    """Dataset for 6-channel cellular microscopy images."""
    
    def __init__(self, df, data_dir, mode='train', site='1'):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
        self.site = site
    
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image from disk."""
        exp = row['experiment']
        plate = row['plate']
        well = row['well']
        
        if self.mode == 'test':
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s{self.site}_w'
        else:
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s{self.site}_w'
        
        channels = []
        for i in range(1, 7):
            img_path = f'{path_template}{i}.png'
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            else:
                img = np.zeros((512, 512), dtype=np.uint8)
            channels.append(img)
        
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        return img.astype(np.float32) / 255.0
    
    def augment(self, img):
        """Light data augmentation."""
        if self.mode != 'train':
            return img
        
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row)
        img = self.augment(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode == 'train':
            return img, row['label']
        else:
            return img, row['id_code']


class InceptionV3Model(nn.Module):
    """
    Google Inception V3 with modified first conv layer for 6-channel input.
    
    Inception V3 uses parallel branches with different kernel sizes:
    - 1x1 convolutions for dimension reduction
    - 3x3 convolutions for local features
    - 5x5 convolutions for global features
    - Max pooling for spatial reduction
    
    Also includes:
    - Batch Normalization after each conv
    - RMSProp optimizer (we'll use AdamW)
    - Label smoothing
    """
    
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        
        # Load pretrained Inception V3 model
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        
        # Find and modify the first conv layer (Conv2d_1a)
        # Inception V3 uses 'Conv2d_1a' or 'conv_1a' depending on version
        if hasattr(self.backbone, 'Conv2d_1a_1x1'):
            old_conv = self.backbone.Conv2d_1a_1x1
        elif hasattr(self.backbone, 'conv_1a'):
            old_conv = self.backbone.conv_1a
        elif hasattr(self.backbone, 'conv_stem'):
            old_conv = self.backbone.conv_stem
        elif hasattr(self.backbone, 'stem'):
            # Inception V3 has a sequential stem
            old_conv = self.backbone.stem[0]
        else:
            # Fallback: iterate through modules
            old_conv = None
            for name, module in self.backbone.named_modules():
                if isinstance(module, nn.Conv2d) and module.in_channels == 3:
                    old_conv = module
                    break
        
        if old_conv is not None:
            new_conv = nn.Conv2d(
                in_channels,
                old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            with torch.no_grad():
                # Replicate pretrained weights for 6 channels
                if old_conv.kernel_size == (3, 3) or old_conv.kernel_size == (1, 1):
                    # For 1x1 conv: replicate weights
                    new_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_weight = new_weight.repeat(1, in_channels, 1, 1) / in_channels
                else:
                    # For other kernels: average across channels
                    new_weight = old_conv.weight.mean(dim=1, keepdim=True)
                    new_weight = new_weight.repeat(1, in_channels, 1, 1) / in_channels
                
                new_conv.weight.copy_(new_weight)
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            
            # Replace the conv layer
            if hasattr(self.backbone, 'Conv2d_1a_1x1'):
                self.backbone.Conv2d_1a_1x1 = new_conv
            elif hasattr(self.backbone, 'conv_1a'):
                self.backbone.conv_1a = new_conv
            elif hasattr(self.backbone, 'conv_stem'):
                self.backbone.conv_stem = new_conv
            elif hasattr(self.backbone, 'stem'):
                self.backbone.stem[0] = new_conv
        
        # Get feature dimension (Inception V3 has 2048 features before classifier)
        if hasattr(self.backbone, 'fc'):
            n_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()
        elif hasattr(self.backbone, 'classifier'):
            n_features = self.backbone.classifier.in_features
            self.backbone.classifier = nn.Identity()
        elif hasattr(self.backbone, 'head'):
            n_features = self.backbone.head.in_features
            self.backbone.head = nn.Identity()
        else:
            n_features = 2048  # Inception V3 default
        
        # Custom classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        pbar.set_postfix({'loss': f'{running_loss/len(loader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
    
    return running_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    return running_loss / len(loader), 100. * correct / total


def main():
    print("=" * 60)
    print("Google Inception V3 - Multi-Scale Convolution Architecture")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    test_df['cell_type'] = test_df['experiment'].str.split('-').str[0]
    
    # Filter HUVEC only
    train_df = train_df[train_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Cell types: {train_df['cell_type'].unique()}")
    
    # Convert sirna to numeric labels
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {s: i for i, s in enumerate(unique_sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    Config.NUM_CLASSES = len(unique_sirnas)
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    # Split data
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=Config.SEED, stratify=train_df['label']
    )
    print(f"Train: {len(train_data)}, Val: {len(val_data)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        CellularDataset(train_data, Config.DATA_DIR, mode='train'),
        batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        CellularDataset(val_data, Config.DATA_DIR, mode='train'),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating model: {Config.MODEL_NAME}...")
    model = InceptionV3Model(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss, optimizer, scheduler
    # Inception V3 typically uses label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_inceptionv3.pth')
            print(f"Saved best model: {best_acc:.2f}%")
    
    # Inference
    print("\nInference...")
    model.load_state_dict(torch.load('best_inceptionv3.pth'))
    
    test_loader = DataLoader(
        CellularDataset(test_df, Config.DATA_DIR, mode='test'),
        batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS
    )
    
    model.eval()
    predictions, ids = [], []
    
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Inference'):
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, preds = outputs.max(1)
            predictions.extend(preds.cpu().numpy())
            ids.extend(img_ids)
    
    # Create submission
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    predictions = [label_to_sirna[p] for p in predictions]
    
    submission = pd.DataFrame({'id_code': ids, 'sirna': predictions})
    submission.to_csv('submission_inceptionv3.csv', index=False)
    
    print(f"\n✓ Complete! Best Val Acc: {best_acc:.2f}%")
    print(f"Saved submission_inceptionv3.csv")


if __name__ == '__main__':
    main()
