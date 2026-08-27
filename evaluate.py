import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import rasterio
from dataset import CloudDataset, samples
from model import UNet

# ============================================================
# 1. CONFIGURATION
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHECKPOINT_PATH = os.path.join("outputs", "best_model.pth")
PRED_OUTPUT_DIR = os.path.join("outputs", "predictions")
os.makedirs(PRED_OUTPUT_DIR, exist_ok=True)

BATCH_SIZE = 1  # แนะนำให้ใช้ 1 ตอนทำ qualitative visualization เพื่อเซฟทีละรูป

# ============================================================
# 2. METRICS CALCULATION FUNCTION
# ============================================================
def evaluate_metrics(preds, targets, threshold=0.5, smooth=1e-6):
    """
    คำนวณ Pixel Accuracy, IoU, F1/Dice, Precision, และ Recall
    """
    preds_bin = (preds > threshold).float()
    targets_bin = (targets > threshold).float()

    preds_f = preds_bin.view(-1)
    targets_f = targets_bin.view(-1)

    # True Positives, False Positives, False Negatives, True Negatives
    tp = (preds_f * targets_f).sum()
    fp = (preds_f * (1 - targets_f)).sum()
    fn = ((1 - preds_f) * targets_f).sum()
    tn = ((1 - preds_f) * (1 - targets_f)).sum()

    # Pixel Accuracy
    pixel_acc = (tp + tn) / (tp + tn + fp + fn + smooth)

    # Precision & Recall
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)

    # IoU (Jaccard) & F1 / Dice
    intersection = tp
    union = tp + fp + fn
    iou = (intersection + smooth) / (union + smooth)
    f1 = (2. * intersection + smooth) / (preds_f.sum() + targets_f.sum() + smooth)

    return {
        "Pixel Accuracy": pixel_acc.item(),
        "IoU": iou.item(),
        "F1 / Dice": f1.item(),
        "Precision": precision.item(),
        "Recall": recall.item()
    }


# ============================================================
# 3. EVALUATION & VISUALIZATION SCRIPT
# ============================================================
def run_evaluation():
    print(f"Loading model from {CHECKPOINT_PATH}...")
    model = UNet(in_channels=4, out_channels=1).to(DEVICE)
    
    if os.path.exists(CHECKPOINT_PATH):
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    else:
        print("Error: ไม่พบไฟล์น้ำหนักโมเดล (best_model.pth) กรุณารัน train.py ก่อน")
        return

    model.eval()

    # สมมติใช้ samples ชุดทดสอบ (หรือเปลี่ยนเป็น test_loader ของคุณ)
    test_dataset = CloudDataset(samples)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    total_metrics = {"Pixel Accuracy": 0, "IoU": 0, "F1 / Dice": 0, "Precision": 0, "Recall": 0}
    num_batches = len(test_loader)

    print(f"Starting evaluation on {num_batches} samples...\n")

    visualization_count = 0

    with torch.no_grad():
        for idx, (images, masks) in enumerate(test_loader):
            images_gpu = images.to(DEVICE)
            masks_gpu = masks.to(DEVICE)

            outputs = model(images_gpu)

            # คำนวณเมตริกสำหรับ batch นี้
            batch_metrics = evaluate_metrics(outputs, masks_gpu)
            for k in total_metrics:
                total_metrics[k] += batch_metrics[k]

            # --- Qualitative Visualizations (อย่างน้อย 5 ภาพ) ---
            if visualization_count < 5:
                # แปลง Tensor กลับเป็น Numpy เพื่อพล็อตภาพ
                img_np = images[0].cpu().numpy()  # [4, H, W]
                mask_np = masks[0].cpu().numpy().squeeze()  # [H, W]
                pred_np = (outputs[0].cpu().numpy().squeeze() > 0.5).astype(np.float32)

                # ดึงแบนด์ R, G, B มาทำ RGB True Color Image
                rgb = np.transpose(img_np[:3], (1, 2, 0))
                rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min() + 1e-8)  # Normalization สำหรับแสดงผล

                # สร้างภาพเปรียบเทียบ Side-by-Side
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                axes[0].imshow(rgb)
                axes[0].set_title("Input RGB Image")
                axes[0].axis("off")

                axes[1].imshow(mask_np, cmap="gray")
                axes[1].set_title("Ground-Truth Mask")
                axes[1].axis("off")

                axes[2].imshow(pred_np, cmap="gray")
                axes[2].set_title("Predicted Mask")
                axes[2].axis("off")

                plt.tight_layout()
                save_path = os.path.join(PRED_OUTPUT_DIR, f"prediction_sample_{visualization_count + 1}.png")
                plt.savefig(save_path, dpi=150)
                plt.close(fig)
                print(f"Saved qualitative visualization to {save_path}")
                
                visualization_count += 1

    # หาค่าเฉลี่ยของทุกเมตริก
    for k in total_metrics:
        total_metrics[k] /= num_batches

    # --- Print Evaluation Report ---
    print("\n" + "=" * 40)
    print("      FINAL EVALUATION REPORT")
    print("=" * 40)
    for k, v in total_metrics.items():
        print(f"| {k:<16} | {v:.4f} ({(v*100):.2f}%) {'':<10}|")
    print("=" * 40)
    print(f"บันทึกภาพ Qualitative Visualizations ทั้งหมดลงในโฟลเดอร์ '{PRED_OUTPUT_DIR}' เรียบร้อยแล้ว!")

if __name__ == "__main__":
    run_evaluation()