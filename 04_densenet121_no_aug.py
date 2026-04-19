"""
DenseNet121 - No Augmentation (Procedural)
==========================================
Simple procedural version of DenseNet121 for cellular image classification.
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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Config
DATA_DIR = '/kaggle/input/competitions/recursion-cellular-image-classification'
TRAIN_CSV = f'{DATA_DIR}/train.csv'
TEST_CSV = '/kaggle/input/datasets/himanshusardana2/corrected-test-csv-recurrence-cellular/test.csv'

MODEL_NAME = 'densenet121'
IMG_SIZE = 320
BATCH_SIZE = 32
EPOCHS = 20
LR = 3e-4
NUM_WORKERS = 2
SEED = 42
CELL_TYPES = ['HUVEC']

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============== Dataset ==============
class SimpleDataset(Dataset):
    def __init__(self, df, data_dir, mode='train'):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        exp, plate, well = row['experiment'], row['plate'], row['well']
        
        # Path
        prefix = 'test' if self.mode == 'test' else 'train'
        base = f'{self.data_dir}/{prefix}/{exp}/Plate{plate}/{well}_s1_w'
        
        # Load 6 channels
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


# ============== Model ==============
def create_model(num_classes):
    model = timm.create_model(MODEL_NAME, pretrained=True, in_chans=3)
    
    # Modify first conv for 6 channels
    old_conv = model.features.conv0
    
    # Extract values properly (convert to native Python types)
    out_ch = int(old_conv.out_channels)
    kernel = int(old_conv.kernel_size[0]) if hasattr(old_conv.kernel_size, '__len__') else int(old_conv.kernel_size)
    stride = int(old_conv.stride[0]) if hasattr(old_conv.stride, '__len__') else int(old_conv.stride)
    padding = int(old_conv.padding[0]) if hasattr(old_conv.padding, '__len__') else int(old_conv.padding)
    bias = old_conv.bias is not None
    
    model.features.conv0 = nn.Conv2d(6, out_ch, kernel, stride, padding, bias=bias)
    
    with torch.no_grad():
        w = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, 6, 1, 1) / 6
        model.features.conv0.weight.copy_(w)
    
    # Replace classifier
    n_features = model.get_classifier().in_features
    model.reset_classifier(0)
    model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(n_features, num_classes))
    
    return model


# ============== Training ==============
def train_epoch(model, loader, criterion, optimizer):
    model.train()
    loss_sum, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc='Training')
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        out = model(imgs)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()
        
        loss_sum += loss.item()
        correct += (out.argmax(1) == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar with live metrics
        pbar.set_postfix({
            'loss': f'{loss_sum/len(loader):.4f}', 
            'acc': f'{100.*correct/total:.2f}%'
        })
    
    return loss_sum / len(loader), 100 * correct / total


def validate(model, loader, criterion):
    model.eval()
    loss_sum, correct, total = 0, 0, 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val'):
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            
            loss_sum += loss.item()
            correct += (out.argmax(1) == labels).sum().item()
            total += labels.size(0)
    
    return loss_sum / len(loader), 100 * correct / total


# ============== Main ==============
def main():
    print("=" * 60)
    print("DenseNet121 - No Augmentation (Procedural)")
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
    train_data, val_data = train_test_split(train_df, test_size=0.15, 
                                            stratify=train_df['label'], random_state=SEED)
    
    # Dataloaders
    train_loader = DataLoader(SimpleDataset(train_data, DATA_DIR), 
                              batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(SimpleDataset(val_data, DATA_DIR), 
                            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    
    # Model
    print(f"\nCreating {MODEL_NAME}...")
    model = create_model(num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc, patience, counter = 0, 3, 0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'best_densenet121.pth')
            print(f"  → Saved best: {best_acc:.2f}%")
        else:
            counter += 1
            print(f"  → No improvement ({counter}/{patience})")
            if counter >= patience:
                print("Early stopping!")
                break
    
    # Inference
    print("\nInference...")
    model.load_state_dict(torch.load('best_densenet121.pth'))
    model.eval()
    
    test_loader = DataLoader(SimpleDataset(test_df, DATA_DIR, mode='test'),
                             batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    preds, ids = [], []
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Test'):
            out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(img_ids)
    
    # Submission
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    submission = pd.DataFrame({'id_code': ids, 'sirna': [label_to_sirna[p] for p in preds]})
    submission.to_csv('submission_densenet121.csv', index=False)
    
    print(f"\n✓ Done! Best Val Acc: {best_acc:.2f}%")
    print(f"Saved submission_densenet121.csv")


if __name__ == '__main__':
    main()
