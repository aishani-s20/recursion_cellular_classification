"""
Recursion Cellular Image Classification
========================================
ResNet50 with 6-channel input, per-channel z-score normalization,
mixed precision training, control-well plate normalization,
dual-site TTA, and MSAM (Momentum Sharpness-Aware Minimization) optimizer.

MSAM: Sharpness Aware Minimization without Computational Overhead
Reference: Becker et al., "Momentum-SAM: Sharpness Aware Minimization 
           without Computational Overhead", arXiv:2401.12033
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
import warnings
import math

warnings.filterwarnings('ignore')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


class AdamW_MSAM(torch.optim.Optimizer):
    """
    AdamW with Momentum Sharpness-Aware Minimization (MSAM).
    
    MSAM perturbs parameters in the direction of the accumulated momentum
    (exp_avg in AdamW) instead of the gradient direction. This achieves
    flat loss regions (better generalization) without the 2x computational
    overhead of standard SAM.
    
    Key difference from AdamW:
    - Before computing gradient: p = p + rho * v/||v|| (momentum ascent)
    - After parameter update: p = p - rho * v/||v|| (prepare next ascent)
    
    Args:
        params: model parameters
        lr: learning rate (default: 1e-3)
        betas: coefficients for running averages (default: (0.9, 0.999))
        eps: term added for numerical stability (default: 1e-8)
        weight_decay: weight decay coefficient (default: 1e-2)
        rho: MSAM perturbation radius (default: 0.05-0.3)
               Larger rho = stronger sharpness regularization
    """
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-2,
        rho=0.1,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            rho=rho,
        )
        super(AdamW_MSAM, self).__init__(params, defaults)
        
        # Store norm_factor for ascent direction
        for group in self.param_groups:
            group["norm_factor"] = [0.0]

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single MSAM optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        if len(self.param_groups) > 1:
            raise RuntimeError("MSAM currently supports only one parameter group")
        
        group = self.param_groups[0]
        beta1, beta2 = group['betas']
        
        params_with_grad = []
        grads = []
        exp_avgs = []
        exp_avg_sqs = []
        state_steps = []
        
        for p in group['params']:
            if p.grad is None:
                continue
            if p.grad.is_sparse:
                raise RuntimeError('AdamW_MSAM does not support sparse gradients')
            
            params_with_grad.append(p)
            grads.append(p.grad)
            
            state = self.state[p]
            
            # State initialization
            if len(state) == 0:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            
            exp_avgs.append(state['exp_avg'])
            exp_avg_sqs.append(state['exp_avg_sq'])
            state['step'] += 1
            state_steps.append(state['step'])
        
        # MSAM update
        _adamw_msam_step(
            params_with_grad,
            grads,
            exp_avgs,
            exp_avg_sqs,
            state_steps,
            beta1=beta1,
            beta2=beta2,
            lr=group['lr'],
            weight_decay=group['weight_decay'],
            eps=group['eps'],
            rho=group['rho'],
            norm_factor=group['norm_factor'],
        )
        
        return loss


def _adamw_msam_step(
    params,
    grads,
    exp_avgs,
    exp_avg_sqs,
    state_steps,
    *,
    beta1: float,
    beta2: float,
    lr: float,
    weight_decay: float,
    eps: float,
    rho: float,
    norm_factor: list,
):
    """Functional API for AdamW MSAM update."""
    
    # Step 1: Apply momentum ascent (perturb parameters in momentum direction)
    # p_t = p_t + rho * v_t / ||v_t||
    for i, param in enumerate(params):
        param.add_(exp_avgs[i], alpha=norm_factor[0])
    
    # Step 2: Standard AdamW update at perturbed point
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        
        bias_correction1 = 1 - beta1 ** step
        bias_correction2 = 1 - beta2 ** step
        
        # Update biased first moment estimate: v = beta1*v + (1-beta1)*g
        exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
        
        # Update biased second raw moment estimate
        exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        
        # Compute bias-corrected step size
        step_size = lr / bias_correction1
        
        # Bias-corrected second moment
        denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
        
        # Parameter update: p = p - lr * v / denom
        param.addcdiv_(exp_avg, denom, value=-step_size)
        
        # Weight decay (decoupled from gradient)
        param.mul_(1 - lr * weight_decay)
    
    # Step 3: Calculate ascent step norm for next iteration
    # ||v_{t+1}|| = sqrt(sum(||v_{t+1}^i||^2))
    ascent_norm = torch.norm(
        torch.stack([buf.norm(p=2) for buf in exp_avgs]),
        p=2
    )
    norm_factor[0] = rho / (ascent_norm + 1e-12)
    
    # Step 4: Apply negative momentum ascent (prepare for next step)
    # This effectively does: p_{t+1} = p_{t+1} - rho * v_{t+1} / ||v_{t+1}||
    for i, param in enumerate(params):
        param.sub_(exp_avgs[i], alpha=norm_factor[0])


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    DATA_DIR = '/kaggle/input/competitions/recursion-cellular-image-classification'
    TRAIN_CSV = f'{DATA_DIR}/train.csv'
    TEST_CSV = f'{DATA_DIR}/test.csv'
    PIXEL_STATS = f'{DATA_DIR}/pixel_stats.csv'
    TRAIN_CONTROLS_CSV = f'{DATA_DIR}/train_controls.csv'
    TEST_CONTROLS_CSV = f'{DATA_DIR}/test_controls.csv'
    
    MODEL_NAME = 'resnet50'
    IMG_SIZE = 320
    BATCH_SIZE = 32
    EPOCHS = 50
    WARMUP_EP = 1
    LR = 3e-4
    MIN_LR = 1e-6
    WEIGHT_DECAY = 1e-4
    LABEL_SMOOTH = 0.1
    
    # MSAM specific parameters
    MSAM_RHO = 0.1          # Perturbation radius (larger = stronger sharpness regularization)
                            # Typical values: 0.05 - 0.3 for AdamW
    
    NUM_WORKERS = 2
    SEED = 42
    NUM_CLASSES = 1108
    
    CELL_TYPES = ['HUVEC']
    
    USE_PLATE_NORMALIZATION = True      # Subtract per-plate control mean
    USE_CONTROLS_AS_EXTRA_DATA = True   # Add negative controls as extra training data


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(Config.SEED)


# =============================================================================
# NORMALIZATION STATS
# =============================================================================

def build_stats_lookup(pixel_stats_path):
    """
    Returns a dict keyed by (experiment, plate, well, site, channel)
    with values (mean, std). Falls back to (127.5, 64.0) if key missing.
    """
    stats = pd.read_csv(pixel_stats_path)
    lookup = {}
    for _, row in stats.iterrows():
        key = (row['experiment'], int(row['plate']),
               row['well'], int(row['site']),
               int(row['channel']))
        lookup[key] = (float(row['mean']), float(row['std']))
    return lookup


# =============================================================================
# PLATE CONTROL STATS
# =============================================================================

def compute_plate_control_stats(controls_df, data_dir, mode='train'):
    """
    Compute per-plate mean image from control wells.
    Returns dict: {(experiment, plate): mean_image_array}  (H, W, 6)
    """
    stats = {}
    grouped = controls_df.groupby(['experiment', 'plate'])
    
    print(f"Computing plate control stats from {mode}_controls.csv ...")
    for (exp, plate), group in tqdm(grouped, desc='Plate stats'):
        accum = None
        count = 0
        for _, row in group.iterrows():
            well = row['well']
            path_template = f'{data_dir}/{mode}/{exp}/Plate{plate}/{well}_s1_w'
            channels = []
            for ch in range(1, 7):
                img_path = f'{path_template}{ch}.png'
                if os.path.exists(img_path):
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                else:
                    img = np.zeros((512, 512), dtype=np.uint8)
                channels.append(img)
            img = np.stack(channels, axis=-1).astype(np.float32) / 255.0
            if accum is None:
                accum = img
            else:
                accum += img
            count += 1
        
        if count > 0:
            stats[(exp, plate)] = accum / count
    
    print(f"  Computed stats for {len(stats)} plates")
    return stats


# =============================================================================
# DATASET
# =============================================================================

class CellularDataset(Dataset):
    def __init__(self, df, data_dir, stats_lookup, mode='train',
                 site=None, plate_stats=None):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.stats_lookup = stats_lookup
        self.mode = mode
        self.site = site
        self.plate_stats = plate_stats
    
    def __len__(self):
        return len(self.df)
    
    def _load_channels(self, row, site):
        """Load 6 grayscale channels with z-score normalization."""
        exp = row['experiment']
        plate = int(row['plate'])
        well = row['well']
        split = 'test' if self.mode == 'test' else 'train'
        base = f'{self.data_dir}/{split}/{exp}/Plate{plate}/{well}_s{site}_w'
        
        channels = []
        for ch in range(1, 7):
            path = f'{base}{ch}.png'
            img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) if os.path.exists(path) \
                  else np.zeros((512, 512), dtype=np.uint8)
            img = img.astype(np.float32)
            
            key = (exp, plate, well, int(site), ch)
            mean, std = self.stats_lookup.get(key, (127.5, 64.0))
            img = (img - mean) / (std + 1e-6)
            channels.append(img)
        
        stacked = np.stack(channels, axis=-1)
        stacked = cv2.resize(stacked, (Config.IMG_SIZE, Config.IMG_SIZE))
        
        # Plate-level normalization
        if self.plate_stats is not None:
            ctrl_key = (exp, plate)
            if ctrl_key in self.plate_stats:
                ctrl_mean = cv2.resize(
                    self.plate_stats[ctrl_key],
                    (Config.IMG_SIZE, Config.IMG_SIZE)
                )
                # Scale control mean to z-score space (approximate)
                stacked = stacked - (ctrl_mean * 255.0 - 127.5) / 64.0
        
        return stacked
    
    def _augment(self, img):
        """Spatial augmentations - train only."""
        if self.mode != 'train':
            return img
        if np.random.rand() > 0.5:
            img = np.fliplr(img).copy()
        if np.random.rand() > 0.5:
            img = np.flipud(img).copy()
        return img
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        site = np.random.choice(['1', '2']) if self.mode == 'train' \
               else (self.site or '1')
        
        img = self._load_channels(row, site)
        img = self._augment(img)
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        
        if self.mode in ('train', 'val'):
            return img, int(row['label'])
        else:
            return img, row['id_code']


# =============================================================================
# MODEL
# =============================================================================

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
                    self.backbone.conv1.bias = nn.Parameter(old_conv.bias.clone())
        
        n_features = self.backbone.get_classifier().in_features
        self.backbone.reset_classifier(0)
        
        # Enhanced head with BatchNorm and Dropout
        self.classifier = nn.Sequential(
            nn.BatchNorm1d(n_features),
            nn.Dropout(0.4),
            nn.Linear(n_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    
    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False
    
    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


# =============================================================================
# LR SCHEDULE (linear warmup -> cosine decay)
# =============================================================================

def build_scheduler(optimizer, total_steps, warmup_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
        return max(Config.MIN_LR / Config.LR, cosine)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# =============================================================================
# TRAIN / VALIDATE
# =============================================================================

def train_epoch(model, loader, criterion, optimizer, scheduler, scaler, device):
    model.train()
    total_loss = correct = total = 0
    
    pbar = tqdm(loader, desc='Train', leave=False)
    for imgs, labels in pbar:
        imgs, labels = imgs.to(device), labels.to(device)
        
        # MSAM optimizer handles the perturbation internally
        # Just zero grads and do standard forward-backward
        optimizer.zero_grad()
        
        with autocast():
            out = model(imgs)
            loss = criterion(out, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        
        total_loss += loss.item()
        _, pred = out.max(1)
        correct += pred.eq(labels).sum().item()
        total += labels.size(0)
        pbar.set_postfix(loss=f'{total_loss/len(loader):.4f}',
                        acc=f'{100.*correct/total:.2f}%')
    
    return total_loss / len(loader), 100. * correct / total


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = correct = total = 0
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc='Val  ', leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            with autocast():
                out = model(imgs)
                loss = criterion(out, labels)
            total_loss += loss.item()
            _, pred = out.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)
    
    return total_loss / len(loader), 100. * correct / total


# =============================================================================
# TEST-TIME AUGMENTATION
# =============================================================================

def predict_tta(model, test_df, data_dir, stats_lookup, device, plate_stats=None):
    """
    6 TTA passes: site1/site2 x {original, hflip, vflip}
    """
    model.eval()
    tta_configs = [
        ('1', None), ('1', 'h'), ('1', 'v'),
        ('2', None), ('2', 'h'), ('2', 'v'),
    ]
    all_probs = []
    ids = None
    
    for site, flip in tta_configs:
        ds = CellularDataset(test_df, data_dir, stats_lookup,
                            mode='test', site=site, plate_stats=plate_stats)
        loader = DataLoader(ds, batch_size=Config.BATCH_SIZE,
                           shuffle=False, num_workers=Config.NUM_WORKERS,
                           pin_memory=True)
        
        probs_list = []
        id_list = []
        
        with torch.no_grad():
            for imgs, img_ids in tqdm(loader,
                                     desc=f'TTA site{site} {flip or "orig"}',
                                     leave=False):
                imgs = imgs.to(device)
                if flip == 'h':
                    imgs = torch.flip(imgs, dims=[3])
                elif flip == 'v':
                    imgs = torch.flip(imgs, dims=[2])
                with autocast():
                    out = model(imgs)
                probs_list.append(out.softmax(dim=1).cpu())
                id_list.extend(img_ids)
        
        all_probs.append(torch.cat(probs_list, dim=0))
        if ids is None:
            ids = id_list
    
    avg_probs = torch.stack(all_probs).mean(dim=0)
    preds = avg_probs.argmax(dim=1).numpy()
    return preds, ids


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("ResNet50 with Controls, Plate Normalization, and MSAM")
    print("=" * 60)
    print("\nMSAM (Momentum Sharpness-Aware Minimization):")
    print("- Perturbs parameters in MOMENTUM direction (not gradient)")
    print("- Achieves flat loss regions without 2x computation")
    print(f"- rho = {Config.MSAM_RHO} (perturbation radius)")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv(Config.TRAIN_CSV)
    test_df = pd.read_csv(Config.TEST_CSV)
    train_controls_df = pd.read_csv(Config.TRAIN_CONTROLS_CSV)
    test_controls_df = pd.read_csv(Config.TEST_CONTROLS_CSV)
    
    print(f"  train_controls.csv: {len(train_controls_df)} rows")
    print(f"  test_controls.csv: {len(test_controls_df)} rows")
    
    train_df['cell_type'] = train_df['experiment'].str.split('-').str[0]
    test_df['cell_type'] = test_df['experiment'].str.split('-').str[0]
    
    # Filter HUVEC only
    train_df = train_df[train_df['cell_type'].isin(Config.CELL_TYPES)].reset_index(drop=True)
    
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Cell types: {train_df['cell_type'].unique()}")
    
    # Convert sirna to numeric labels
    train_df['sirna_id'] = train_df['sirna'].str.replace('sirna_', '').astype(int)
    unique_sirnas = sorted(train_df['sirna_id'].unique())
    sirna_to_label = {s: i for i, s in enumerate(unique_sirnas)}
    label_to_sirna = {i: s for s, i in sirna_to_label.items()}
    train_df['label'] = train_df['sirna_id'].map(sirna_to_label)
    
    Config.NUM_CLASSES = len(unique_sirnas)
    print(f"Number of classes: {Config.NUM_CLASSES}")
    
    # ------------------------------------------------------------------
    # Add negative control wells as extra training data
    # ------------------------------------------------------------------
    control_df = pd.DataFrame()
    if Config.USE_CONTROLS_AS_EXTRA_DATA:
        print("\n--- Adding control wells as extra training data ---")
        
        neg_mask = train_controls_df['sirna'].astype(str).str.contains(
            'negative', case=False, na=False
        )
        if neg_mask.sum() == 0:
            neg_mask = pd.Series([True] * len(train_controls_df))
        
        neg_controls = train_controls_df[neg_mask].copy()
        print(f"  Found {len(neg_controls)} negative control wells")
        
        required_cols = ['experiment', 'plate', 'well']
        if all(c in neg_controls.columns for c in required_cols) and len(neg_controls) > 0:
            rng = np.random.default_rng(Config.SEED)
            neg_controls = neg_controls[required_cols].copy()
            
            neg_controls['sirna'] = 'negative_control'
            neg_controls['sirna_id'] = rng.choice(unique_sirnas, size=len(neg_controls))
            neg_controls['label'] = neg_controls['sirna_id'].map(sirna_to_label)
            
            # Limit to avoid overwhelming real data
            frac = min(1.0, len(train_df) / max(1, len(neg_controls)))
            control_df = neg_controls.sample(frac=frac, random_state=Config.SEED)
            print(f"  Adding {len(control_df)} control samples to training")
            
            train_df = pd.concat([train_df, control_df], ignore_index=True)
            train_df = train_df.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
            print(f"  New training size: {len(train_df)}")
    
    # ------------------------------------------------------------------
    # Compute plate-level control statistics
    # ------------------------------------------------------------------
    plate_stats_train = None
    plate_stats_test = None
    
    if Config.USE_PLATE_NORMALIZATION:
        print("\n--- Computing plate control statistics ---")
        plate_stats_train = compute_plate_control_stats(
            train_controls_df, Config.DATA_DIR, mode='train'
        )
        plate_stats_test = compute_plate_control_stats(
            test_controls_df, Config.DATA_DIR, mode='test'
        )
        # Fallback to train stats for shared plates
        for key, val in plate_stats_train.items():
            if key not in plate_stats_test:
                plate_stats_test[key] = val
    
    # ------------------------------------------------------------------
    # Split data (stratified)
    # ------------------------------------------------------------------
    real_train_mask = train_df['sirna'] != 'negative_control'
    real_train_df = train_df[real_train_mask]
    
    train_data, val_data = train_test_split(
        real_train_df, test_size=0.15, random_state=Config.SEED,
        stratify=real_train_df['label']
    )
    
    # Merge controls into training split only
    if len(control_df) > 0:
        train_data = pd.concat([train_data, control_df], ignore_index=True)
        train_data = train_data.sample(frac=1, random_state=Config.SEED).reset_index(drop=True)
    
    print(f"\nTrain: {len(train_data)}, Val: {len(val_data)}")
    
    # ------------------------------------------------------------------
    # Build loaders
    # ------------------------------------------------------------------
    print("\nBuilding pixel-stats lookup...")
    stats_lookup = build_stats_lookup(Config.PIXEL_STATS)
    
    train_loader = DataLoader(
        CellularDataset(train_data, Config.DATA_DIR, stats_lookup,
                       mode='train', plate_stats=plate_stats_train),
        batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(
        CellularDataset(val_data, Config.DATA_DIR, stats_lookup,
                       mode='val', site='1', plate_stats=plate_stats_train),
        batch_size=Config.BATCH_SIZE, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=True
    )
    
    # ------------------------------------------------------------------
    # Create model
    # ------------------------------------------------------------------
    print(f"\nCreating model: {Config.MODEL_NAME}...")
    model = CellularModel(Config.MODEL_NAME, Config.NUM_CLASSES).to(device)
    
    # Loss and MSAM optimizer
    criterion = nn.CrossEntropyLoss(label_smoothing=Config.LABEL_SMOOTH)
    
    # Use AdamW_MSAM instead of standard AdamW
    print(f"\nUsing MSAM optimizer with rho={Config.MSAM_RHO}")
    print("  MSAM provides sharpness-aware minimization without 2x computation")
    optimizer = AdamW_MSAM(
        model.parameters(),
        lr=Config.LR,
        weight_decay=Config.WEIGHT_DECAY,
        rho=Config.MSAM_RHO,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    scaler = GradScaler()
    
    total_steps = Config.EPOCHS * len(train_loader)
    warmup_steps = Config.WARMUP_EP * len(train_loader)
    scheduler = build_scheduler(optimizer, total_steps, warmup_steps)
    
    # Training loop
    best_acc = 0
    
    # Phase 1: freeze backbone for warmup
    print(f"\nPhase 1 - backbone frozen for {Config.WARMUP_EP} epoch(s)")
    model.freeze_backbone()
    
    for epoch in range(Config.EPOCHS):
        if epoch == Config.WARMUP_EP:
            model.unfreeze_backbone()
            print(f"\nPhase 2 - full fine-tune from epoch {epoch+1}")
        
        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}  "
              f"lr={optimizer.param_groups[0]['lr']:.2e}")
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                           optimizer, scheduler, scaler, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        print(f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), 'best_resnet50_msam.pth')
            print(f"Saved best model: {best_acc:.2f}%")
    
    # ------------------------------------------------------------------
    # Inference with TTA
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Loading best weights (val acc={best_acc:.2f}%)...")
    model.load_state_dict(torch.load('best_resnet50_msam.pth', map_location=device))
    
    print("Running TTA inference...")
    preds, ids = predict_tta(model, test_df, Config.DATA_DIR, stats_lookup,
                            device, plate_stats=plate_stats_test)
    
    predictions = [label_to_sirna[int(p)] for p in preds]
    submission = pd.DataFrame({'id_code': ids, 'sirna': predictions})
    submission.to_csv('submission_resnet50_msam.csv', index=False)
    
    print(f"\n{'='*60}")
    print(f"Complete! Best Val Acc: {best_acc:.2f}%")
    print("Saved submission_resnet50_msam.csv")
    print(submission.head())


if __name__ == '__main__':
    main()
