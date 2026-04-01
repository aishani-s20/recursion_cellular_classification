import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from PIL import Image
from tqdm.auto import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold

DATA_DIR = "/kaggle/input/competitions/recursion-cellular-image-classification"
BATCH_SIZE = 16
NUM_EPOCHS = 10
LR = 3e-4
IMG_SIZE = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
test_df = pd.read_csv(f"{DATA_DIR}/test.csv")

if train_df["sirna"].dtype == "O":
    train_df["sirna"] = train_df["sirna"].str.replace("sirna_", "").astype(int)

le = LabelEncoder()
train_df["sirna"] = le.fit_transform(train_df["sirna"])
NUM_CLASSES = len(le.classes_)
np.save("label_mapping.npy", le.classes_)

gkf = GroupKFold(n_splits=5)
train_idx, val_idx = next(gkf.split(train_df, groups=train_df["experiment"]))
train_split = train_df.iloc[train_idx].reset_index(drop=True)
val_split = train_df.iloc[val_idx].reset_index(drop=True)

print(f"Classes: {NUM_CLASSES} | Train: {len(train_split)} | Val: {len(val_split)}")


class RxRxDatasetDualSite(Dataset):
    def __init__(self, df, mode="train"):
        self.df = df
        self.mode = mode

    def __len__(self):
        return len(self.df)

    def _get_img_path(self, experiment, plate, well, site, channel):
        return os.path.join(
            DATA_DIR,
            self.mode,
            experiment,
            f"Plate{plate}",
            f"{well}_s{site}_w{channel}.png",
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        channels = []
        for site in range(1, 3):
            for ch in range(1, 7):
                path = self._get_img_path(
                    row["experiment"], row["plate"], row["well"], site, ch
                )

                if not os.path.exists(path):
                    img = Image.new("L", (IMG_SIZE, IMG_SIZE))
                else:
                    img = Image.open(path).resize((IMG_SIZE, IMG_SIZE))

                channels.append(np.array(img))

        x = np.stack(channels, axis=-1)
        x = torch.from_numpy(x.transpose(2, 0, 1)).float() / 255.0

        if self.mode == "train":
            return x, row["sirna"]
        return x, row["id_code"]


def get_model():
    model = models.resnet50(pretrained=True)

    original_conv = model.conv1
    model.conv1 = nn.Conv2d(12, 64, kernel_size=7, stride=2, padding=3, bias=False)

    with torch.no_grad():
        model.conv1.weight[:, :3] = original_conv.weight.repeat(2, 1, 1, 1) / 2
        model.conv1.weight[:, 3:] = original_conv.weight.repeat(2, 1, 1, 1) / 2

    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    return model.to(DEVICE)


train_loader = DataLoader(
    RxRxDatasetDualSite(train_split), batch_size=BATCH_SIZE, shuffle=True, num_workers=2
)
val_loader = DataLoader(
    RxRxDatasetDualSite(val_split), batch_size=BATCH_SIZE, shuffle=False, num_workers=2
)

model = get_model()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=len(train_loader) * NUM_EPOCHS
)

best_acc = 0.0

for epoch in range(NUM_EPOCHS):
    model.train()
    train_correct = 0
    t_total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1} Train")
    for inputs, labels in pbar:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        scheduler.step()

        _, pred = outputs.max(1)
        train_correct += pred.eq(labels).sum().item()
        t_total += labels.size(0)
        pbar.set_description(f"Acc: {100.0 * train_correct / t_total:.2f}%")

    model.eval()
    val_correct = 0
    v_total = 0
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc=f"Epoch {epoch + 1} Val"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, pred = outputs.max(1)
            val_correct += pred.eq(labels).sum().item()
            v_total += labels.size(0)

    val_acc = 100.0 * val_correct / v_total
    print(f"Validation Accuracy: {val_acc:.2f}%")

    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_model_dual_site.pth")
        print("New best model saved!")

print("\nStarting Prediction on Test Set...")
model.load_state_dict(torch.load("best_model_dual_site.pth"))
model.eval()

test_loader = DataLoader(
    RxRxDatasetDualSite(test_df, mode="test"), batch_size=BATCH_SIZE, shuffle=False
)
id_codes, predictions = [], []

with torch.no_grad():
    for inputs, ids in tqdm(test_loader, desc="Predicting"):
        inputs = inputs.to(DEVICE)
        outputs = model(inputs)
        _, pred = outputs.max(1)

        original_sirna = le.inverse_transform(pred.cpu().numpy())
        predictions.extend(original_sirna)
        id_codes.extend(ids)

submission = pd.DataFrame({"id_code": id_codes, "sirna": predictions})
submission.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' generated!")
