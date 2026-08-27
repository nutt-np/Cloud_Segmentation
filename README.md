# Cloud Segmentation — 38-Cloud Dataset

Binary semantic segmentation (cloud / clear) บนภาพดาวเทียม Landsat-8
4 band (R, G, B, NIR) ด้วย U-Net ที่ implement เอง (PyTorch)

## โครงสร้างไฟล์

```
.
├── dataset.py       # Task 1: PyTorch Dataset + DataLoader
├── model.py         # Task 2: U-Net architecture
├── train.py         # Task 3: training loop
├── evaluate.py       # Task 4: evaluation + qualitative visualization
├── notebooks/
│   └── eda.ipynb     # EDA: RGB/NIR visualization, class distribution
├── README.md
├── REPORT.md          # Task 5: รายงานสรุปผล
├── requirements.txt
└── outputs/
    ├── training_curves.png
    ├── metrics.json
    └── predictions/*.png
```

## วิธีติดตั้ง

```bash
pip install -r requirements.txt
```

## วิธีโหลด Dataset

ไม่ commit ตัวข้อมูลไว้ใน repo ให้โหลดแยกดังนี้:

1. โคลนโค้ด/README ต้นฉบับของ dataset:
   ```bash
   git clone https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset.git
   ```
2. โหลดตัวข้อมูลจริง (.TIF ทั้งหมด) จาก Kaggle:
   https://www.kaggle.com/sorour/38cloud-cloud-segmentation-in-satellite-images
3. แก้ path ข้อมูลใน `dataset.py` ให้ชี้ไปยังโฟลเดอร์ที่แตกไว้

## วิธีรัน

```bash
python dataset.py
python train.py
python evaluate.py
```

เปิด `notebooks/eda.ipynb` เพื่อดู EDA

รายละเอียดสถาปัตยกรรมและผลลัพธ์ดูได้ใน `REPORT.md`
