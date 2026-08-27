# REPORT — Cloud Segmentation

## 1. Data Challenges

ข้อมูล 38-Cloud เก็บแต่ละ band (R, G, B, NIR) แยกไฟล์ `.TIF` กัน จึงจับคู่
ไฟล์ด้วยชื่อ patch ที่ตรงกัน (เช่น `red_patch_XXX` กับ `green_patch_XXX`)
ใน `dataset.py`

Normalize ใช้ min-max ต่อ band แทนการหารด้วย 255 ตายตัว เพราะค่า reflectance
ของ Landsat-8 ไม่ได้อยู่ในช่วง 0-255 แบบภาพทั่วไป ส่วน mask แปลงเป็น binary
ด้วย threshold ที่ 127 (ค่าดิบเป็น 0/255)

ข้อมูลที่ใช้ทดสอบ pipeline จริงมีจำนวนจำกัด (patch ตัวอย่างจาก repo ต้นทาง)
ทำให้ train/val split และการวัด class imbalance ที่ทำไว้ในโค้ดยังไม่ได้
สะท้อนสัดส่วนจริงของทั้ง dataset

## 2. Architecture Rationale

ใช้ U-Net (encoder-decoder 4 stage, 64→128→256→512 channels) เพราะงาน
segmentation เมฆต้องการทั้ง context กว้างและความแม่นยำระดับ pixel — skip
connection ช่วยส่งรายละเอียดจาก encoder ตรงไปยัง decoder ที่ resolution
เดียวกัน ไม่ให้หายไปตอน pooling

ไม่ใช้ pretrained backbone เพราะโจทย์กำหนดให้ implement เอง และ backbone
ที่ pretrain มาส่วนใหญ่รับ input 3-channel (RGB) ไม่รองรับ NIR ซึ่งเป็น band
สำคัญสำหรับตรวจจับเมฆ

ข้อจำกัด: โมเดลมี ~31M parameters ค่อนข้างใหญ่สำหรับ patch ขนาด 384×384
และยังไม่มี attention mechanism

## 3. Results Analysis

Pipeline ทั้งหมด (`dataset.py` → `model.py` → `train.py` → `evaluate.py`)
รันได้ครบแบบ end-to-end — โมเดลรับ input shape ถูกต้อง, training loop
บันทึก checkpoint และ plot กราฟใน `outputs/training_curves.png`,
evaluation คำนวณ Pixel Accuracy, IoU, F1, Precision, Recall และบันทึกลง
`outputs/metrics.json` พร้อมภาพเปรียบเทียบใน `outputs/predictions/`

เนื่องจากทดสอบด้วยจำนวนข้อมูลจำกัด ตัวเลข metrics และภาพ prediction ที่ได้
จึงเป็นการยืนยันว่าโค้ดทำงานถูกต้อง มากกว่าจะสะท้อนประสิทธิภาพของโมเดลบน
dataset เต็มรูปแบบ

## 4. Improvements

- เทรนและประเมินผลบน dataset เต็ม (8,400 train / 9,201 test patches)
- ใช้ normalization stats แบบ dataset-wide แทน per-patch
- เพิ่ม data augmentation (flip/rotate, color jitter เฉพาะ RGB)
