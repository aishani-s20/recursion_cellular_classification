"""
Attention Pooling Neural Cellular Automata (aNCA) - Adapted for Recursion Cellular
===============================================================================
Implementation inspired by arXiv:2508.12324 "Attention Pooling Enhances NCA-based 
Classification of Microscopy Images". Adapted to match DenseNet121 code structure.
Uses 64x64 resolution, 6-channel input, NCA steps=64, top-10% attention pooling.
[page:2508.12324][web:17]
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

IMG_SIZE = 32  # Reduced from 64 for OOM
BATCH_SIZE = 8  # Reduced for OOM
EPOCHS = 32
LR = 4e-4
NUM_WORKERS = 2
SEED = 42
CELL_TYPES = ['HUVEC']
N_CHANNELS = 32  # Reduced from 64 for OOM
NCA_STEPS = 16  # Reduced from 32 for OOM
TOP_PCT = 0.1  # Top 10% attention as per paper [page:2508.12324]

# Set seed
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ============== Dataset (same as DenseNet) ==============
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
        
        prefix = 'test' if self.mode == 'test' else 'train'
        base = f'{self.data_dir}/{prefix}/{exp}/Plate{plate}/{well}_s1_w'
        
        channels = []
        for i in range(1, 7):
            img = cv2.imread(f'{base}{i}.png', cv2.IMREAD_GRAYSCALE)
            channels.append(img if img is not None else np.zeros((512, 512), dtype=np.uint8))
        
        img = np.stack(channels, axis=-1)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1)  # C,H,W
        
        if self.mode == 'train':
            return img, row['label']
        return img, row['id_code']

# ============== aNCA Model ==============
class NCAUpdate(nn.Module):
    def __init__(self, c_in, c_out, c_hidden):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.conv(x))

class aNCA(nn.Module):
    def __init__(self, in_channels=6, n_channels=128, steps=64, top_pct=0.1, num_classes=110, hidden_dim=128):
        super().__init__()
        self.n_channels = n_channels
        self.steps = steps
        self.top_pct = top_pct
        
        # Initial projection to features
        self.project = nn.Conv2d(in_channels, n_channels, 1)
        
        # NCA update rule (simplified U-Net like [page:2508.12324])
        self.update = nn.Sequential(
            NCAUpdate(n_channels, hidden_dim, hidden_dim),
            NCAUpdate(hidden_dim, n_channels, hidden_dim)
        )
        
        # Attention pooling weights (per channel)
        self.attn_weights = nn.Parameter(torch.ones(n_channels))
        
        # Classifier - use adaptive pooling to reduce dimensions
        self.pool = nn.AdaptiveAvgPool2d(4)  # Reduce 32x32 -> 4x4
        self.classifier = nn.Sequential(
            nn.Linear(n_channels * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x):
        # Project input to NCA state [B,6,H,W] -> [B,C,H,W]
        state = F.pad(self.project(x), (1,1,1,1))  # Neighborhood padding
        
        # Iterate NCA steps
        for _ in range(self.steps):
            update = self.update(state)
            state = state + update  # Residual update
    
        # Global attention pooling
        B, C, H, W = state.shape
        attn = F.softmax(self.attn_weights, dim=0).view(1, C, 1, 1)
        features = attn * state  # Weighted features
        spatial_attn = torch.sigmoid(features.mean(dim=1, keepdim=True))  # Spatial attention
        attended = features * spatial_attn
        
        # Global pooling instead of top-k to save memory
        pooled = self.pool(attended)  # [B,C,4,4]
        pooled = pooled.view(B, -1)  # [B,C*4*4]
        
        out = self.classifier(pooled)
        return out

def create_model(num_classes):
    return aNCA(num_classes=num_classes).to(device)

# ============== Training Functions (same as DenseNet) ==============
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
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
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    
    print("=" * 80)
    print("aNCA - Attention Pooling Neural Cellular Automata (arXiv:2508.12324)")
    print("=" * 80)
    print(f"Config: IMG_SIZE={IMG_SIZE}, N_CHANNELS={N_CHANNELS}, NCA_STEPS={NCA_STEPS}, BATCH_SIZE={BATCH_SIZE}")
    
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
    print("\nCreating aNCA model...")
    model = create_model(num_classes)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    
    # Training loop
    best_acc, patience, counter = 0, 5, 0
    
    for epoch in range(EPOCHS):
        torch.cuda.empty_cache()
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)
        scheduler.step()
        
        print(f"Train: {train_loss:.4f}, {train_acc:.2f}% | Val: {val_loss:.4f}, {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            counter = 0
            torch.save(model.state_dict(), 'best_aNCA.pth')
            print(f"  → Saved best: {best_acc:.2f}%")
        else:
            counter += 1
            print(f"  → No improvement ({counter}/{patience})")
            if counter >= patience:
                print("Early stopping!")
                break
    
    # Inference
    print("\nInference...")
    torch.cuda.empty_cache()
    model.load_state_dict(torch.load('best_aNCA.pth'))
    model.eval()
    
    test_loader = DataLoader(SimpleDataset(test_df, DATA_DIR, mode='test'),
                             batch_size=8, shuffle=False, num_workers=NUM_WORKERS)
    
    preds, ids = [], []
    with torch.no_grad():
        for imgs, img_ids in tqdm(test_loader, desc='Test'):
            out = model(imgs.to(device))
            preds.extend(out.argmax(1).cpu().numpy())
            ids.extend(img_ids)
    
    # Submission
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    submission = pd.DataFrame({'id_code': ids, 'sirna': [label_to_sirna[p] for p in preds]})
    submission.to_csv('submission_aNCA.csv', index=False)
    
    print(f"\n✓ Done! Best Val Acc: {best_acc:.2f}%")
    print("Saved submission_aNCA.csv")

if __name__ == '__main__':
    main()
