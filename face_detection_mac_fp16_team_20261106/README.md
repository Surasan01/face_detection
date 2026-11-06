# Face detection — ชุดส่งทีมสำหรับ Mac

TFLite FP16 ใช้ input **480×640 (กว้าง×สูง) แนวตั้งเท่านั้น**
เก็บ 3 โมเดลที่เลือกหลังเทรน: **MTCNN (epoch 4), YuNet (epoch 12), RetinaFace MNV1 0.25 (epoch 9)**
ไม่รวม RetinaFace MNV2/MNV3/MNV4 หรือไฟล์โมเดลขนาดอื่น
หลักฐานที่มาของน้ำหนักอยู่ใน model_manifest.json น้ำหนักเหมือนแพ็กเต็มเดิม

## ติดตั้งบน Apple Silicon

ใช้ Python 3.12 arm64 และ macOS 12 ขึ้นไป เปิด Terminal ในโฟลเดอร์นี้:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-mac.txt
python inference.py --check
```

ใช้ TFLite CPU ไม่ต้องติดตั้ง PyTorch, CUDA หรือ tensorflow-metal
Dependencies โดยตรงล็อกเวอร์ชันไว้ ส่วน dependencies ทางอ้อมใช้ pip resolve
TensorFlow 2.16.1 มี wheel Python 3.12 macOS arm64: https://pypi.org/project/tensorflow/2.16.1/
ทดสอบแพ็กบน Windows CPU แล้ว ยังไม่ได้ทดสอบบน Mac จริง ให้รัน --check หลังติดตั้ง

## รันภาพ

ใส่ JPG/PNG ใน images/ แล้วรัน:

```bash
python inference.py --model all --input images --output outputs/demo
open outputs/demo/gallery.html
```

รันเฉพาะ RetinaFace:

```bash
python inference.py --model retinaface_mnv1_025 --input images
```

ชื่อโมเดลอื่น: mtcnn, yunet เลือก --model all เพื่อรันทั้งสามต่อกัน
ค่าเริ่มต้น confidence 0.5 และ CPU threads 2 ปรับด้วย --confidence / --threads
MTCNN ยังใช้ thresholds ภายใน cascade 0.6/0.7/0.7
รอบใหม่ให้ใช้ output folder ใหม่ หรือไม่ใส่ --output เพื่อสร้างโฟลเดอร์ตามเวลา
รองรับ JPG/PNG/WebP/TIFF/BMP; HEIC ให้แปลงเป็น JPG ก่อน

## โค้ดเหลือ 2 ไฟล์

- **inference.py**: โหลดภาพ → letterbox → TFLite → คืนพิกัด → วาดและเซฟผล พร้อม CLI
- **decoders.py**: สูตรแปลง output ของ 3 โมเดล, NMS และ MTCNN cascade แบ่งหัวข้อแยกชัดเจน

ใช้งานจาก Python:

```python
from inference import Detector, load_image
model = Detector('retinaface_mnv1_025')
boxes, landmarks = model(load_image('images/photo.jpg'))
```

ภาพรักษาสัดส่วน เติมขอบดำให้ครบ 480×640 ไม่ยืดภาพและไม่ขยายกลับเป็นจัตุรัส
YuNet/RetinaFace รับ RGB float32 /255, [1,640,480,3]; preprocessing ของต้นฉบับส่วนที่เหลืออยู่ในกราฟ
MTCNN ใช้ image pyramid และ crop 24/48 ภายใน น้ำหนักเป็น FP16 แต่ input/output เป็น FP32

## ดูผลและ Notebook

gallery.html แสดงทุกรูป; JSON เก็บ boxes_xyxy_score [N,5] และ landmarks_xy [N,5,2]
พิกัดอยู่บนภาพต้นฉบับหลังจัด EXIF จุดเรียง: ตาซ้ายภาพ, ตาขวาภาพ, จมูก, มุมปากซ้ายภาพ, มุมปากขวาภาพ
กรอบเขียว/จุดส้ม/เลขเหลืองในภาพ inference ปกติ คือกรอบ/5จุด/confidence
summary.csv มีจำนวนหน้าและเวลา ms ต่อภาพ ไม่รวมอ่าน/วาด/เซฟ มี first-call warm-up จึงไม่ใช่ controlled benchmark

```bash
python -m pip install -r requirements-notebook.txt
python -m ipykernel install --user --name face-team --display-name "Face Team FP16"
```

เปิด Notebook ใน VS Code และเลือก kernel นี้:

1. **01_inference_demo.ipynb** — รันทั้ง 3 โมเดลและ plot ผลทุกรูป
2. **02_mtcnn_retinaface_overlay_demo.ipynb** — MTCNN ฟ้า/วงกลม กับ RetinaFace ส้ม/กากบาท ซ้อนภาพเดียวกัน พร้อมซูมทุกหน้า

ทั้งสองมีผลตัวอย่างที่รันไว้ด้วยภาพ WIDER FACE 3 ภาพบน Windows แต่ไม่รวมไฟล์ dataset
ใส่รูปของคุณใน images/ แล้ว Run All เพื่อแทนที่ผลตัวอย่าง
Overlay ไม่มี ground truth ใช้ดูด้วยสายตา ไม่ตัดสินว่าแม่นกว่าจาก confidence

ข้อมูลตรวจสอบแพ็กอยู่ใน verification.json และ SHA256SUMS.txt

## ???????? demo ???????????????

- `images/`: ??????????????? 7 ?????? WIDER FACE ???????????? `sample_sources.json`
- `outputs/demo_480x640/gallery.html`: ????? `.py` ???? 3 ????? ??? 21 ???
- `outputs/demo_480x640/summary.csv`: ????????????????????? demo
- ?? PNG ??? JSON ?????????????????????????????????

????????? `python inference.py --model all --input images` ???????????????????????????????
