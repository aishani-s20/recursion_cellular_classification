# ==========================================
# 0. Imports
# ==========================================
import os
import numpy as np
import pandas as pd
from PIL import Image
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

# ==========================================
# 1. Config
# ==========================================
DATA_DIR = '/kaggle/input/recursion-cellular-image-classification'
IMG_SIZE = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 3e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 2. Load Data
# ==========================================
train_df = pd.read_csv(f'{DATA_DIR}/train.csv')
test_df = pd.read_csv(f'{DATA_DIR}/test.csv')

print("Test rows:", len(test_df))  # should be 19887

# Label encoding
le = LabelEncoder()
train_df['sirna'] = le.fit_transform(train_df['sirna'])
NUM_CLASSES = train_df['sirna'].nunique()

# Group split (no leakage)
gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(train_df, groups=train_df['experiment']))

train_df = train_df.iloc[train_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

# ==========================================
# 3. Dataset
# ==========================================
class RxRxDataset(Dataset):
    def __init__(self, df, mode='train'):
        self.df = df.reset_index(drop=True)
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        imgs = []
        for c in range(1, 7):
            path = os.path.join(
                DATA_DIR,
                self.mode,
                row['experiment'],
                f"Plate{row['plate']}",
                f"{row['well']}_s1_w{c}.png"
            )

            if os.path.exists(path):
                img = Image.open(path).resize((IMG_SIZE, IMG_SIZE))
                img = np.array(img)
            else:
                img = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.uint8)

            imgs.append(img)

        x = np.stack(imgs, axis=0)
        x = torch.tensor(x, dtype=torch.float32) / 255.0

        if self.mode == 'test':
            return x, row['id_code']
        else:
            return x, row['sirna']

# ==========================================
# 4. Model (FAST)
# ==========================================
def get_model():
    model = models.resnet18(pretrained=True)

    # modify input channels
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # init weights
    with torch.no_grad():
        model.conv1.weight[:, :3] = model.conv1.weight[:, :3]
        model.conv1.weight[:, 3:] = model.conv1.weight[:, :3]

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)

model = get_model()

# ==========================================
# 5. Loaders
# ==========================================
train_loader = DataLoader(RxRxDataset(train_df), batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader = DataLoader(RxRxDataset(val_df), batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ==========================================
# 6. Training Setup
# ==========================================
optimizer = optim.AdamW(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()
scaler = torch.cuda.amp.GradScaler()

# ==========================================
# 7. Training Loop
# ==========================================
best_acc = 0

for epoch in range(EPOCHS):
    model.train()
    total, correct = 0, 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        with torch.cuda.amp.autocast():
            out = model(x)
            loss = criterion(out, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        pbar.set_description(f"Loss {loss.item():.4f} Acc {100*correct/total:.2f}%")

    # ===== VALIDATION =====
    model.eval()
    total, correct = 0, 0

    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(1)

            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    print(f"Val Acc: {acc:.2f}%")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best.pth")

# ==========================================
# 8. Inference
# ==========================================
model.load_state_dict(torch.load("best.pth"))
model.eval()

test_loader = DataLoader(RxRxDataset(test_df, mode='test'),
                         batch_size=BATCH_SIZE,
                         shuffle=False)

ids, preds = [], []

with torch.no_grad():
    for x, id_code in tqdm(test_loader):
        x = x.to(DEVICE)
        out = model(x)
        pred = out.argmax(1).cpu().numpy()

        preds.extend(pred)
        ids.extend(id_code)

# convert back to original labels
preds = le.inverse_transform(preds)

# ==========================================
# 9. Submission
# ==========================================
submission = pd.DataFrame({
    "id_code": ids,
    "sirna": preds
})

print("Submission rows:", len(submission))  # MUST be 19887

submission.to_csv("submission.csv", index=False)
print("submission.csv saved!")
