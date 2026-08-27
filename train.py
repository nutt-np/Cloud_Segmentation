import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from dataset import CloudDataset, samples
from model import UNet

# ============================================================
# 1. CONFIGURATION & HYPERPARAMETERS
# ============================================================
BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 1e-4
VAL_SPLIT = 0.2
EARLY_STOPPING_PATIENCE = 4  
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(OUTPUT_DIR, "best_model.pth")
CURVE_PATH = os.path.join(OUTPUT_DIR, "training_curves.png")
# ============================================================
# 2. LOSS FUNCTION & METRICS
# ============================================================
class BCEDiceLoss(nn.Module):
    """
    ใช้ BCE + Dice Loss เพื่อจัดการปัญหา Class Imbalance
    BCE ช่วยเรื่องความน่าจะเป็นโดยรวม และ Dice ช่วยโฟกัสที่การทับซ้อนของพื้นที่เมฆ
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCELoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, preds, targets, smooth=1e-6):
        bce_loss = self.bce(preds, targets)
        
        preds_f = preds.view(-1)
        targets_f = targets.view(-1)
        
        intersection = (preds_f * targets_f).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (preds_f.sum() + targets_f.sum() + smooth)
        
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    """คำนวณค่า IoU และ F1-score"""
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()
    
    preds_f = preds_bin.view(-1)
    targets_f = targets_bin.view(-1)
    
    intersection = (preds_f * targets_f).sum()
    union = preds_f.sum() + targets_f.sum() - intersection
    
    iou = (intersection + smooth) / (union + smooth)
    f1 = (2. * intersection + smooth) / (preds_f.sum() + targets_f.sum() + smooth)
    
    return iou.item(), f1.item()

# ============================================================
# 3. DATA PREPARATION (Safe Split)
# ============================================================
print(f"Total samples found: {len(samples)}")

if len(samples) >= 2:
    train_samples, val_samples = train_test_split(
        samples, 
        test_size=VAL_SPLIT, 
        random_state=42
    )
else:
    print("Warning: มีข้อมูลน้อยเกินไป ข้ามการแบ่ง Train/Val (ใช้ข้อมูลเดียวกันเพื่อเทสต์รันระบบ)")
    train_samples = samples
    val_samples = samples

train_dataset = CloudDataset(train_samples)
val_dataset = CloudDataset(val_samples)

# ปรับ batch_size ให้ไม่เกินจำนวนข้อมูลที่มี
actual_batch_size = min(BATCH_SIZE, len(train_samples))

train_loader = DataLoader(train_dataset, batch_size=actual_batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=actual_batch_size, shuffle=False)

# ============================================================
# 4. MODEL, OPTIMIZER, & SCHEDULER SETUP
# ============================================================
model = UNet(in_channels=4, out_channels=1).to(DEVICE)
criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

# ============================================================
# 5. TRAINING LOOP
# ============================================================
def train_model():
    best_val_iou = 0.0
    best_val_loss = float("inf")
    patience_counter = 0
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_iou': [], 'val_iou': [],
        'train_f1': [], 'val_f1': []}
    
    print(f"Starting training on device: {DEVICE} for {EPOCHS} epochs...\n")

    for epoch in range(EPOCHS):
        # --- Training Phase ---
        model.train()
        train_loss, train_iou, train_f1 = 0.0, 0.0, 0.0

        for images, masks in train_loader:
            images, masks = images.to(DEVICE), masks.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            iou, f1 = calculate_metrics(outputs, masks)
            train_iou += iou
            train_f1 += f1

        train_loss /= len(train_loader)
        train_iou /= len(train_loader)
        train_f1 /= len(train_loader)

        # --- Validation Phase ---
        model.eval()
        val_loss, val_iou, val_f1 = 0.0, 0.0, 0.0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(DEVICE), masks.to(DEVICE)

                outputs = model(images)
                loss = criterion(outputs, masks)

                val_loss += loss.item()
                iou, f1 = calculate_metrics(outputs, masks)
                val_iou += iou
                val_f1 += f1

        val_loss /= len(val_loader)
        val_iou /= len(val_loader)
        val_f1 /= len(val_loader)

        scheduler.step(val_loss)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_f1'].append(train_f1)
        history['val_f1'].append(val_f1)

        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train IoU: {train_iou:.4f} | Val IoU: {val_iou:.4f} | "
              f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f}")

        # --- Checkpointing (based on best val IoU, per requirement) ---
        if val_iou >= best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), CHECKPOINT_PATH)
            print(f"Saved best model checkpoint (Val IoU: {best_val_iou:.4f})")

        # --- Early stopping (based on val LOSS, per requirement) ---
        if val_loss < best_val_loss - 1e-5:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"Early stopping counter: {patience_counter}/{EARLY_STOPPING_PATIENCE}")

        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"\n Early stopping triggered at epoch {epoch+1}")
            break
        
    # ============================================================
    # 6. PLOT TRAINING CURVES
    # ============================================================
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train', marker='o')
    plt.plot(history['val_loss'], label='Val', marker='o')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['train_iou'], label='Train', marker='o')
    plt.plot(history['val_iou'], label='Val', marker='o')
    plt.title('IoU vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('IoU Score')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(CURVE_PATH, dpi=150)
    print(f"\n Training curve plot saved to: {CURVE_PATH}")
    plt.show()

if __name__ == "__main__":
    train_model()