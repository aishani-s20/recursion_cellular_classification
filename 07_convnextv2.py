"""
ConvNeXt V2 - Modern ConvNet Architecture
=========================================
ConvNeXt V2 for cellular image classification.
Uses FCGF (Fully Convolution Groupwise Fusion) block for improved performance.
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
    
    MODEL_NAME = 'convnextv2_tiny'  # Smaller version - convnextv2_nano, convnextv2_tiny, convnextv2_small, convnextv2_base
    IMG_SIZE = 256  # Reduced from 320 to save memory
    BATCH_SIZE = 16  # Reduced from 32 to save memory
    EPOCHS = 20
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
    """Dataset for 6-channel cellular microscopy images - NO augmentation."""
    
    def __init__(self, df, data_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image from disk - single site (site 1), NO augmentation."""
        exp = row['experiment']
        plate = row['plate']
        well = row['well']
        
        if self.mode == 'test':
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s1_w'
        else:
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s1_w'
        
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
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode == 'train':
            return img, row['label']
        else:
            return img, row['id_code']


class ConvNeXtV2Model(nn.Module):
    """ConvNeXt V2 with modified first conv layer for 6-channel input."""
    
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        
        # Load pretrained ConvNeXt V2 model
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3, num_classes=0)
        
        # Find and modify the first conv layer (stem)
        if hasattr(self.backbone, 'stem'):
            old_conv = self.backbone.stem[0]  # First conv in stem (Sequential)
            
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
                new_weight = old_conv.weight.mean(dim=1, keepdim=True)
                new_weight = new_weight.repeat(1, in_channels, 1, 1) / in_channels
                new_conv.weight.copy_(new_weight)
                if old_conv.bias is not None:
                    new_conv.bias = old_conv.bias
            
            # Replace the first conv in stem Sequential
            self.backbone.stem[0] = new_conv
        
        # Get number of features by probing the model
        with torch.no_grad():
            dummy = torch.randn(1, in_channels, Config.IMG_SIZE, Config.IMG_SIZE)
            out = self.backbone(dummy)
            if len(out.shape) == 4:  # (B, C, H, W)
                n_features = out.shape[1]
            else:  # (B, features)
                n_features = out.shape[1]
            del dummy, out
            torch.cuda.empty_cache()
        
        print(f"Detected feature dimension: {n_features}")
        
        # Add global average pooling and classifier
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        # Get backbone features
        features = self.backbone(x)
        
        # If output is (B, C, H, W), apply pooling
        if len(features.shape) == 4:
            features = self.pool(features)
        
        # Classify
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
    # Clear GPU memory
    torch.cuda.empty_cache()
    
    print("=" * 60)
    print("ConvNeXt V2 - Modern ConvNet (No Augmentation)")
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
    model = ConvNeXtV2Model(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    best_acc = 0
    patience = 3
    patience_counter = 0
    
    for epoch in range(Config.EPOCHS):
        torch.cuda.empty_cache()  # Clear memory at start of each epoch
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_convnextv2.pth')
            print(f"Saved best model: {best_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{patience}")
            
            if patience_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
    
    # Inference
    print("\nInference...")
    model.load_state_dict(torch.load('best_convnextv2.pth'))
    
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
    submission.to_csv('submission_convnextv2.csv', index=False)
    
    print(f"\n✓ Complete! Best Val Acc: {best_acc:.2f}%")
    print("Saved submission_convnextv2.csv")


if __name__ == '__main__':
    main()
