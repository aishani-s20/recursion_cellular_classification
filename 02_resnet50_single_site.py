import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import cv2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class Config:
    DATA_DIR = '/kaggle/input/competitions/recursion-cellular-image-classification'
    TRAIN_CSV = f'{DATA_DIR}/train.csv'
    TEST_CSV = None  # No test set; using stratified train/val split
    
    MODEL_NAME = 'resnet50'  
    IMG_SIZE = 320  
    BATCH_SIZE = 32
    EPOCHS = 10  
    LR = 3e-4
    
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108
    
    CELL_TYPES = ['HUVEC']  
    
    CONVERT_SIRNA = True

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

set_seed(Config.SEED)

class CellularDataset(Dataset):
    def __init__(self, df, data_dir, mode='train', transform=None):
        self.df = df
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row):
        """Load 6-channel image"""
        exp = row['experiment']
        plate = row['plate']
        well = row['well']
        
        if self.mode == 'train':
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s1_w'
        else:
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s1_w'
        
        channels = []
        for i in range(1, 7):
            img_path = f'{path_template}{i}.png'
            if os.path.exists(img_path):
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                channels.append(img)
            else:
                # Fallback if file doesn't exist
                channels.append(np.zeros((512, 512), dtype=np.uint8))
        
        # Stack channels and resize
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = self.load_image(row)
        
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        
        if self.transform:
            # Convert to PIL for transforms (handle 6 channels)
            img = torch.from_numpy(img).permute(2, 0, 1)  # C, H, W
        else:
            img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode == 'train':
            label = row['label']
            return img, label
        else:
            return img, row['id_code']

# Model
class CellularModel(nn.Module):
    def __init__(self, model_name, num_classes, in_channels=6):
        super().__init__()
        # Load pretrained model
        self.backbone = timm.create_model(model_name, pretrained=True, in_chans=3)
        
        # Modify first conv layer to accept 6 channels
        # ResNet models in timm use 'conv1' instead of 'conv_stem'
        if hasattr(self.backbone, 'conv1'):
            old_conv = self.backbone.conv1
            self.backbone.conv1 = nn.Conv2d(
                in_channels, old_conv.out_channels,
                kernel_size=old_conv.kernel_size,
                stride=old_conv.stride,
                padding=old_conv.padding,
                bias=old_conv.bias is not None
            )
            # Initialize with average of pretrained weights
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old_conv.weight
                self.backbone.conv1.weight[:, 3:] = old_conv.weight
                if old_conv.bias is not None:
                    self.backbone.conv1.bias = old_conv.bias
        
        # Get number of features
        n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)  # Remove classifier
        
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(n_features, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# Training function
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
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

def compute_metrics(all_labels, all_preds, all_probs):
    """Compute multi-class classification metrics."""
    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs, multi_class='ovr', average='macro')
    f1 = f1_score(all_labels, all_preds, average='macro')
    prec = precision_score(all_labels, all_preds, average='macro')
    rec = recall_score(all_labels, all_preds, average='macro')
    return acc, auc, f1, prec, rec


# Validation function
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_labels, all_preds, all_probs = [], [], []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Validation'):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = outputs.softmax(dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    val_loss = total_loss / len(loader)
    
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)
    
    _, auc, f1, prec, rec = compute_metrics(all_labels, all_preds, all_probs)
    val_acc = 100.0 * (all_preds == all_labels).mean()
    
    return val_loss, val_acc, auc, f1, prec, rec

# Main training pipeline
def main():
    # Load data
    print("Loading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    
    # Extract cell type from experiment column
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    
    # Filter by cell type for speed
    train_df = train_df[train_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    
    print(f"Training samples: {len(train_df)}")
    print(f"Cell types in train: {train_df['cell_type'].unique()}")
    
    # Convert sirna labels to numeric (sirna_1 -> 1, sirna_10 -> 10, etc.)
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    
    # Create label mapping (need to map to 0-indexed consecutive integers)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {sirna: idx for idx, sirna in enumerate(unique_sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    print(f"Number of unique sirnas: {len(unique_sirnas)}")
    print(f"Label range: 0 to {train_df['label'].max()}")
    
    # Update NUM_CLASSES based on actual data
    Config.NUM_CLASSES = len(unique_sirnas)
    
    # Split train/val
    train_data, val_data = train_test_split(train_df, test_size=0.15, random_state=Config.SEED, stratify=train_df['label'])
    
    # Create datasets
    train_dataset = CellularDataset(train_data, Config.DATA_DIR, mode='train')
    val_dataset = CellularDataset(val_data, Config.DATA_DIR, mode='train')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, 
                              shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, 
                            shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True)
    
    # Create model
    print(f"Creating model: {Config.MODEL_NAME}...")
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.EPOCHS)
    
    # Training loop
    best_acc = 0
    for epoch in range(Config.EPOCHS):
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_auc, val_f1, val_prec, val_rec = validate(
            model, val_loader, criterion, device
        )
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | "
              f"AUC: {val_auc:.4f} | F1: {val_f1:.4f} | "
              f"Prec: {val_prec:.4f} | Rec: {val_rec:.4f}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50.pth')
            print(f"Saved best model with accuracy: {best_acc:.2f}%")
    
    # Save final model
    torch.save(model.state_dict(), 'final_resnet50.pth')
    print("\nSaved final model: final_resnet50.pth")
    
    print(f"\n{'='*60}")
    print(f"Training Complete! Best Val Acc: {best_acc:.2f}%")
    print(f"{'='*60}")

if __name__ == '__main__':
    main()
