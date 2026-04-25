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
    
    MODEL_NAME = 'resnext50_32x4d'
    IMG_SIZE = 384
    BATCH_SIZE = 24
    EPOCHS = 10
    LR = 3e-4
    LABEL_SMOOTHING = 0.1
    
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)

class CellularDataset(Dataset):
    def __init__(self, df, data_dir, mode='train', site=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.mode = mode
        self.site = site
        
    def __len__(self):
        return len(self.df)
    
    def load_image(self, row, site):
        exp = row['experiment']
        plate = row['plate']
        well = row['well']
        
        if self.mode == 'test':
            path_template = f'{self.data_dir}/test/{exp}/Plate{plate}/{well}_s{site}_w'
        else:
            path_template = f'{self.data_dir}/train/{exp}/Plate{plate}/{well}_s{site}_w'
        
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
        if self.mode != 'train':
            return img
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        if np.random.rand() > 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k=k).copy()
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        if self.mode == 'train':
            site = np.random.choice(['1', '2'])
        else:
            site = self.site if self.site is not None else '1'
            
        img = self.load_image(row, site)
        img = self.augment(img)
        img = torch.from_numpy(img).permute(2, 0, 1)
        
        if self.mode in ['train', 'val']:
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
            with torch.no_grad():
                self.backbone.conv1.weight[:, :3] = old_conv.weight
                self.backbone.conv1.weight[:, 3:] = old_conv.weight
                if old_conv.bias is not None:
                    self.backbone.conv1.bias = old_conv.bias
        
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
        
        pbar.set_postfix({'loss': running_loss/len(loader), 'acc': 100.*correct/total})
    
    return running_loss/len(loader), 100.*correct/total

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
    
    return running_loss/len(loader), 100.*correct/total

def predict_with_tta(model, df, data_dir, device):
    model.eval()
    configs = [('1', None), ('1', 'h'), ('1', 'v'), ('2', None), ('2', 'h'), ('2', 'v')]
    all_probs = []
    ids = None
    
    for site, flip in configs:
        dataset = CellularDataset(df, data_dir, mode='test', site=site)
        loader = DataLoader(dataset, batch_size=Config.BATCH_SIZE,
                          shuffle=False, num_workers=Config.NUM_WORKERS)
        
        probs = []
        batch_ids = []
        with torch.no_grad():
            for imgs, img_ids in tqdm(loader, desc=f'Site{site} {flip or ""}'):
                imgs = imgs.to(device)
                if flip == 'h':
                    imgs = torch.flip(imgs, dims=[3])
                elif flip == 'v':
                    imgs = torch.flip(imgs, dims=[2])
                outputs = model(imgs)
                probs.append(outputs.softmax(dim=1).cpu())
                batch_ids.extend(img_ids)
        
        probs = torch.cat(probs, dim=0)
        all_probs.append(probs)
        if ids is None:
            ids = batch_ids
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    preds = avg_probs.argmax(dim=1).numpy()
    return preds, ids

def main():
    print("Loading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    test_df['cell_type'] = test_df['experiment'].str.split('-').str[0]
    
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Cell types in train: {train_df['cell_type'].unique()}")
    
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {s: i for i, s in enumerate(unique_sirnas)}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    Config.NUM_CLASSES = len(unique_sirnas)
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    train_data, val_data = train_test_split(
        train_df, test_size=0.15, random_state=Config.SEED,
        stratify=train_df['label']
    )
    
    train_loader = DataLoader(
        CellularDataset(train_data, Config.DATA_DIR, mode='train'),
        batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    val_loader = DataLoader(
        CellularDataset(val_data, Config.DATA_DIR, mode='val', site='1'),
        batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    print(f"Creating model: {Config.MODEL_NAME}...")
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTHING)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=Config.EPOCHS, T_mult=1
    )
    
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
            torch.save(model.state_dict(), 'best_resnext50_32x4d.pth')
            print(f"Saved best model: {best_acc:.2f}%")
    
    print("\nInference with TTA (dual site + flips)...")
    model.load_state_dict(torch.load('best_resnext50_32x4d.pth'))
    
    predictions, ids = predict_with_tta(model, test_df, Config.DATA_DIR, device)
    
    label_to_sirna = {v: k for k, v in sirna_to_label.items()}
    predictions = [label_to_sirna[p] for p in predictions]
    
    submission = pd.DataFrame({
        'id_code': ids,
        'sirna': predictions
    })
    submission.to_csv('submission_resnext50_32x4d.csv', index=False)
    print(f"Saved submission_resnext50_32x4d.csv | Best val acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
