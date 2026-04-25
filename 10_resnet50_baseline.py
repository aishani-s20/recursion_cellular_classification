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
    
    MODEL_NAME = 'resnet50'
    IMG_SIZE = 320
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
    def __init__(self, df, data_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image from disk - single site (site 1)."""
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
        img = torch.from_numpy(img).permute(2, 0, 1)  # (H, W, C) -> (C, H, W)
        
        if self.mode == 'train':
            return img, row['label']
        else:
            return img, row['id_code']


class CellularModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        
        if hasattr(self.backbone, 'conv1'):
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            
            # Initialize with pretrained weights (replicate)
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old_conv.weight
                self.backbone.conv1.weight[:, 3:] = old_conv.weight
                if old_conv.bias is not None:
                    self.backbone.conv1.bias = old_conv.bias
        
        # Replace classifier
        n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
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
    print("ResNet50 Baseline - No Augmentation")
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
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
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
            torch.save(model.state_dict(), 'best_resnet50_baseline.pth')
            print(f"Saved best model: {best_acc:.2f}%")
    
    # Inference
    print("\nInference...")
    model.load_state_dict(torch.load('best_resnet50_baseline.pth'))
    
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
    submission.to_csv('submission_resnet50_baseline.csv', index=False)
    
    print(f"\n✓ Complete! Best Val Acc: {best_acc:.2f}%")
    print("Saved submission_resnet50_baseline.csv")


if __name__ == '__main__':
    main()
