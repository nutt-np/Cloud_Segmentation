# Cloud Segmentation — 38-Cloud Dataset

Binary semantic segmentation (cloud / clear) บนภาพดาวเทียม Landsat-8
4 band (R, G, B, NIR) ด้วย U-Net ที่ implement เอง (PyTorch)

## โครงสร้างไฟล์

```
.
├── dataset.py      # Task 1: PyTorch Dataset + DataLoader
├── model.py        # Task 2: U-Net architecture
├── train.py        # Task 3: training loop (loss, optimizer, early stop, checkpoint)
├── evaluate.py      # Task 4: evaluation + qualitative visualization
├── notebooks/
│   └── eda.ipynb    # EDA: RGB/NIR visualization, class distribution
├── REPORT.md         # Task 5: รายงานสรุปผล
├── requirements.txt
└── outputs/          # ผลลัพธ์จากการรัน (สร้างอัตโนมัติ)
    ├── best_model.pth
    ├── training_curves.png
    ├── metrics.json
    └── predictions/*.png
```

## วิธีติดตั้ง

```bash
pip install -r requirements.txt
```

## วิธีโหลด Dataset

**ไม่ commit ตัวข้อมูลไว้ใน repo** เพราะไฟล์ใหญ่เกินไป ให้โหลดแยกดังนี้:

1. โคลนโค้ด/README ต้นฉบับของ dataset (ไม่ใช่ตัวข้อมูลจริง):
   ```bash
   git clone https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset.git
   ```
2. โหลดตัวข้อมูลจริง (.TIF ทั้งหมด 8,400 train / 9,201 test patches) จาก Kaggle:
   https://www.kaggle.com/sorour/38cloud-cloud-segmentation-in-satellite-images
3. แตกไฟล์ให้ได้โครงสร้าง:
   ```
   38-Cloud_training/
       train_red/  train_green/  train_blue/  train_nir/  train_gt/
   38-Cloud_test/
       test_red/  test_green/  test_blue/  test_nir/
   ```
4. แก้ path `DATA_DIR` / `TRAIN_ROOT` ใน `dataset.py` ให้ชี้ไปยังโฟลเดอร์ที่แตกไว้

## วิธีรัน

```bash
python dataset.py     # ทดสอบโหลดข้อมูล
python train.py       # เทรนโมเดล (จะได้ outputs/best_model.pth, training_curves.png)
python evaluate.py     # ประเมินผล (จะได้ outputs/metrics.json, outputs/predictions/*.png)
```

เปิด `notebooks/eda.ipynb` เพื่อดู EDA (sample images, class distribution)

## หมายเหตุ

โมเดลรับ input 4 channel (R,G,B,NIR) ขนาด 384×384 และ output เป็น
probability map ขนาดเท่ากัน (sigmoid, 1 channel) รายละเอียดเหตุผลการเลือก
สถาปัตยกรรมและผลลัพธ์ดูได้ใน `REPORT.md`
