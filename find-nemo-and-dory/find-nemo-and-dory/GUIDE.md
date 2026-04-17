# 🐠 Hướng Dẫn Xây Dựng Dự Án Nhận Dạng Nemo & Dory Từ Đầu

> **Hướng dẫn từng bước chi tiết** cho người mới bắt đầu học Machine Learning và Computer Vision

---

## 🎯 Mục Tiêu Dự Án

Xây dựng một hệ thống **nhận dạng cá Nemo và Dory** trong ảnh sử dụng **3 mô hình AI khác nhau**:

- **Faster R-CNN** (Two-stage detector)
- **YOLOv5** (One-stage detector)
- **YOLOv8** (Phiên bản mới nhất của YOLO)

**Kết quả cuối:** Có cả giao diện dòng lệnh và web app để người dùng upload ảnh và xem kết quả nhận dạng.

---

## 📚 Kiến Thức Cần Có

### Mức Tối Thiểu

- **Python cơ bản**: variables, functions, classes, import modules
- **Sử dụng terminal/command line** cơ bản
- **Cài đặt thư viện** với pip

### Sẽ Học Được Trong Quá Trình

- Computer Vision (xử lý ảnh)
- Deep Learning (mạng nơ-ron sâu)
- Object Detection (phát hiện vật thể)
- PyTorch framework
- Flask web development
- YOLO và Faster R-CNN architectures

---

## 🛠️ Chuẩn Bị Môi Trường

### Bước 1: Cài Đặt Python

**Windows:**

```bash
# Tải Python 3.9-3.11 từ https://python.org
# Chọn "Add Python to PATH" khi cài đặt
python --version  # Kiểm tra
```

**Linux/Mac:**

```bash
sudo apt update && sudo apt install python3 python3-pip  # Ubuntu
brew install python                                      # macOS
```

### Bước 2: Tạo Thư Mục Dự Án

```bash
# Tạo thư mục chính
mkdir find-nemo-and-dory
cd find-nemo-and-dory

# Tạo cấu trúc thư mục
mkdir data
mkdir data/fasterrcnn_runs
mkdir data/yolo5_results
mkdir data/yolo5_results/weights
mkdir data/yolo8_results
mkdir data/yolo8_results/weights
mkdir templates
mkdir static
mkdir output

# Cấu trúc sẽ như thế này:
# find-nemo-and-dory/
# ├── data/                    # Chứa model đã train
# ├── templates/              # HTML templates cho web
# ├── static/                 # CSS, JS, images
# ├── output/                 # Kết quả nhận dạng
# ├── app.py                  # Web server
# ├── detect.py               # Script nhận dạng chính
# ├── requirements.txt        # Danh sách thư viện
# └── GUIDE.md               # File hướng dẫn này
```

### Bước 3: Tạo Virtual Environment (Khuyến Nghị)

```bash
# Tạo môi trường ảo (tránh xung đột thư viện)
python -m venv venv

# Kích hoạt
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Kiểm tra (sẽ thấy (venv) ở đầu dòng)
which python  # Linux/Mac
where python  # Windows
```

---

## 📦 Cài Đặt Thư Viện

### Bước 4: Tạo File requirements.txt

Tạo file `requirements.txt` với nội dung:

```txt
# PyTorch & TorchVision (CPU - xem README nếu cần GPU)
torch>=2.0.0
torchvision>=0.15.0

# Ultralytics (YOLOv5 + YOLOv8)
ultralytics>=8.0.0

# OpenCV
opencv-python>=4.8.0

# Các thư viện hỗ trợ
numpy>=1.24.0
Pillow>=9.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
requests>=2.31.0
scipy>=1.10.0

# Web server (giao diện web)
flask>=3.0.0
```

### Bước 5: Cài Đặt Các Thư Viện

```bash
# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt tất cả thư viện
pip install -r requirements.txt

# Kiểm tra cài đặt thành công
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics installed!')"
```

**Lưu ý:** Nếu có GPU NVIDIA, dùng script tự động hoặc cài thủ công:

```bash
# Cách 1: Tự động detect GPU và cài đúng phiên bản (khuyến nghị)
python setup_gpu.py

# Cách 2: Chỉ kiểm tra, không cài
python setup_gpu.py --check

# Cách 3: Cài thủ công (chọn CUDA phù hợp với nvidia-smi)
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu118  # CUDA 11.8
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128  # CUDA 12.8
```

> **Quan trọng:** Dùng `--force-reinstall` nếu đã cài bản CPU trước đó. Pip không tự thay thế vì cùng version number.
>
> **Cách chọn CUDA version:** Chạy `nvidia-smi` → xem "CUDA Version" → chọn phiên bản PyTorch CUDA **≤** version đó.
> Ví dụ: Driver CUDA 13.1 → cài được cu128, cu126, cu124, cu121, cu118.

```bash
# Kiểm tra GPU đã được nhận chưa
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

---

## 🧠 Hiểu Về Object Detection

### Khái Niệm Cơ Bản

**Object Detection** = **Phát hiện** + **Phân loại** vật thể trong ảnh

- **Input**: Ảnh
- **Output**: Vị trí (bounding box) + nhãn (class) + độ tin cậy (confidence)

### 3 Mô Hình Trong Dự Án

| Mô Hình          | Loại      | Đặc Điểm                     | Tốc Độ | Độ Chính Xác |
| ---------------- | --------- | ---------------------------- | ------ | ------------ |
| **Faster R-CNN** | Two-stage | Chậm nhưng chính xác cao     | 🐌     | ⭐⭐⭐⭐⭐   |
| **YOLOv5**       | One-stage | Cân bằng tốc độ/độ chính xác | 🏃‍♂️     | ⭐⭐⭐⭐     |
| **YOLOv8**       | One-stage | Phiên bản mới nhất của YOLO  | 🏃‍♂️💨   | ⭐⭐⭐⭐⭐   |

**Two-stage vs One-stage:**

- **Two-stage** (Faster R-CNN): Đề xuất vùng → Phân loại
- **One-stage** (YOLO): Phát hiện trực tiếp trong 1 lần forward

---

## 💾 Chuẩn Bị Dữ Liệu và Model

### Bước 6: Hiểu Về Dataset

Để train được các model, bạn cần:

**Dataset gốc:** Hình ảnh cá Nemo và Dory với **annotations** (bounding box + labels)

- `images/`: Thư mục chứa ảnh (.jpg, .png)
- `labels/`: Thư mục chứa file annotation (.txt cho YOLO, .xml cho Faster R-CNN)

**Format annotation YOLO:**

```txt
# Mỗi dòng: class_id x_center y_center width height (tất cả normalized [0,1])
0 0.5 0.3 0.2 0.4  # nemo ở giữa ảnh
1 0.8 0.7 0.15 0.25  # dory ở góc dưới phải
```

**Class mapping:**

```txt
0: nemo  # Cá hề (Clownfish)
1: dory  # Cá đuôi xanh (Blue Tang)
```

### Bước 7: Tải Pre-trained Weights

Vì việc train từ đầu mất rất nhiều thời gian và dữ liệu, dự án này sử dụng **pre-trained weights** có sẵn:

```bash
# Tạo file dummy weights để test (trong thực tế bạn cần weights đã train thật)
# Đặt vào đúng thư mục:
# data/fasterrcnn_runs/fasterrcnn_best.pth
# data/yolo5_results/weights/best.pt
# data/yolo8_results/weights/best.pt

# Hoặc download pre-trained COCO weights và fine-tune:
curl -L https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt -o data/yolo5_results/weights/yolov5s.pt
curl -L https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt -o data/yolo8_results/weights/yolov8s.pt
```

---

## 🔨 Xây Dựng Code Từng Phần

### Bước 8: Tạo File detect.py Cơ Bản

Bắt đầu với skeleton cơ bản:

```python
"""
Nhận dạng cá Nemo và Dory sử dụng 3 mô hình:
  - Faster R-CNN, YOLOv5, YOLOv8
"""

import argparse
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# ---------------------------------------------------------------------------
# Cấu hình chung
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

CLASS_NAMES = ["nemo", "dory"]
COLORS = {
    "nemo": (0, 128, 255),    # Màu cam (BGR)
    "dory": (255, 200, 0),    # Màu xanh dương (BGR)
}
DEFAULT_CONF = 0.5
IMG_SIZE = 640

print("🐠 Khởi tạo dự án nhận dạng Nemo & Dory")
print(f"📁 Thư mục dự án: {BASE_DIR}")
```

### Bước 9: Thêm Hàm Load Model

```python
# ---------------------------------------------------------------------------
# 1. Faster R-CNN
# ---------------------------------------------------------------------------

def load_fasterrcnn(weights_path, num_classes=4, device="cpu"):
    """Load mô hình Faster R-CNN ResNet50-FPN"""
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    print(f"📦 Đang load Faster R-CNN từ {weights_path}")

    # Tạo model architecture
    model = fasterrcnn_resnet50_fpn(weights=None)

    # Thay đổi số classes
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load weights
    if os.path.exists(weights_path):
        state = torch.load(weights_path, map_location=device, weights_only=False)

        # Handle different checkpoint formats
        if isinstance(state, dict) and "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif isinstance(state, dict) and "state_dict" in state:
            model.load_state_dict(state["state_dict"])
        else:
            model.load_state_dict(state)
        print("✅ Faster R-CNN weights loaded!")
    else:
        print("⚠️ Weights file không tồn tại, sử dụng random weights")

    model.to(device)
    model.eval()
    return model

# ---------------------------------------------------------------------------
# 2. YOLOv5
# ---------------------------------------------------------------------------

def load_yolov5(weights_path, device="cpu"):
    """Load mô hình YOLOv5"""
    print(f"📦 Đang load YOLOv5 từ {weights_path}")

    try:
        # Sử dụng ultralytics YOLOv5
        from ultralytics import YOLO
        model = YOLO(weights_path if os.path.exists(weights_path) else "yolov5s.pt")
        print("✅ YOLOv5 loaded!")
        return model
    except Exception as e:
        print(f"❌ Lỗi load YOLOv5: {e}")
        return None

# ---------------------------------------------------------------------------
# 3. YOLOv8
# ---------------------------------------------------------------------------

def load_yolov8(weights_path, device="cpu"):
    """Load mô hình YOLOv8"""
    print(f"📦 Đang load YOLOv8 từ {weights_path}")

    try:
        from ultralytics import YOLO
        model = YOLO(weights_path if os.path.exists(weights_path) else "yolov8s.pt")
        print("✅ YOLOv8 loaded!")
        return model
    except Exception as e:
        print(f"❌ Lỗi load YOLOv8: {e}")
        return None
```

### Bước 10: Thêm Hàm Prediction

```python
# ---------------------------------------------------------------------------
# Các hàm prediction
# ---------------------------------------------------------------------------

def predict_fasterrcnn(model, image_bgr, device="cpu", conf_thres=DEFAULT_CONF):
    """Dự đoán bằng Faster R-CNN"""
    import torchvision.transforms.functional as F

    # Chuyển BGR → RGB
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    # Chuyển thành tensor [0, 1]
    img_tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    with torch.no_grad():
        predictions = model(img_tensor)

    # Xử lý kết quả
    results = []
    if len(predictions) > 0:
        pred = predictions[0]
        boxes = pred['boxes'].cpu().numpy()
        scores = pred['scores'].cpu().numpy()
        labels = pred['labels'].cpu().numpy()

        for i, score in enumerate(scores):
            if score >= conf_thres:
                label_id = labels[i]
                if label_id in [2, 3]:  # Dory=2, Nemo=3
                    class_name = "dory" if label_id == 2 else "nemo"
                    box = boxes[i].astype(int)
                    results.append((class_name, score, tuple(box)))

    return results

def predict_yolo(model, image_bgr, conf_thres=DEFAULT_CONF):
    """Dự đoán bằng YOLO (v5 hoặc v8)"""
    if model is None:
        return []

    try:
        # YOLO inference
        results = model(image_bgr, conf=conf_thres, imgsz=IMG_SIZE, verbose=False)

        detections = []
        if len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                for box in result.boxes:
                    # Extract thông tin
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])

                    if cls_id < len(CLASS_NAMES):  # nemo=0, dory=1
                        class_name = CLASS_NAMES[cls_id]
                        xyxy = box.xyxy[0].cpu().numpy().astype(int)
                        detections.append((class_name, conf, tuple(xyxy)))

        return detections
    except Exception as e:
        print(f"❌ YOLO prediction error: {e}")
        return []

# Wrapper functions
def predict_yolov5(model, image_bgr, device="cpu", conf_thres=DEFAULT_CONF):
    return predict_yolo(model, image_bgr, conf_thres)

def predict_yolov8(model, image_bgr, device="cpu", conf_thres=DEFAULT_CONF):
    return predict_yolo(model, image_bgr, conf_thres)
```

### Bước 11: Thêm Hàm Vẽ Kết Quả

```python
# ---------------------------------------------------------------------------
# Hàm vẽ và hiển thị kết quả
# ---------------------------------------------------------------------------

def draw_results(image_bgr, predictions, model_name=""):
    """Vẽ bounding boxes và labels lên ảnh"""
    img_result = image_bgr.copy()

    for class_name, conf, (x1, y1, x2, y2) in predictions:
        color = COLORS.get(class_name, (0, 255, 0))

        # Vẽ bounding box
        cv2.rectangle(img_result, (x1, y1), (x2, y2), color, 2)

        # Vẽ label và confidence
        label = f"{class_name} {conf:.2f}"
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

        # Background cho text
        cv2.rectangle(img_result, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img_result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img_result

def draw_time_on_image(image_bgr, inference_time_ms, model_name=""):
    """Vẽ thời gian xử lý lên ảnh"""
    img_result = image_bgr.copy()

    time_text = f"{model_name} - {inference_time_ms:.1f}ms"
    cv2.putText(img_result, time_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    return img_result
```

### Bước 12: Thêm Hàm Main

```python
# ---------------------------------------------------------------------------
# Hàm chính
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="🐠 Nhận dạng cá Nemo và Dory")
    parser.add_argument("--image", "-i", required=True, help="Đường dẫn ảnh hoặc thư mục")
    parser.add_argument("--model", "-m", default="yolov8", choices=["fasterrcnn", "yolov5", "yolov8"], help="Lựa chọn mô hình")
    parser.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF, help="Ngưỡng confidence")
    parser.add_argument("--device", "-d", default="auto", help="Device: cpu, cuda:0, hoặc auto")
    parser.add_argument("--output", "-o", default="output", help="Thư mục lưu kết quả")
    parser.add_argument("--show", action="store_true", help="Hiển thị kết quả")

    args = parser.parse_args()

    # Auto detect device
    if args.device == "auto":
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"🖥️ Sử dụng device: {device}")
    print(f"🧠 Mô hình: {args.model}")
    print(f"🎯 Confidence threshold: {args.conf}")

    # Load model
    model = None
    load_start = time.time()

    if args.model == "fasterrcnn":
        weights_path = str(DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth")
        model = load_fasterrcnn(weights_path, device=device)
        predict_fn = predict_fasterrcnn
    elif args.model == "yolov5":
        weights_path = str(DATA_DIR / "yolo5_results" / "weights" / "best.pt")
        model = load_yolov5(weights_path, device=device)
        predict_fn = predict_yolov5
    elif args.model == "yolov8":
        weights_path = str(DATA_DIR / "yolo8_results" / "weights" / "best.pt")
        model = load_yolov8(weights_path, device=device)
        predict_fn = predict_yolov8

    load_time = (time.time() - load_start) * 1000
    print(f"⏱️ Model loading time: {load_time:.1f}ms")

    # Tạo output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)

    # Xử lý ảnh
    image_path = Path(args.image)

    if image_path.is_file():
        # Single image
        process_single_image(image_path, model, predict_fn, args, output_dir)
    elif image_path.is_dir():
        # Directory
        image_files = []
        for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
            image_files.extend(image_path.glob(f"*{ext}"))
            image_files.extend(image_path.glob(f"*{ext.upper()}"))

        print(f"📁 Tìm thấy {len(image_files)} ảnh trong thư mục")

        for img_path in image_files:
            process_single_image(img_path, model, predict_fn, args, output_dir)
    else:
        print("❌ Đường dẫn không hợp lệ!")
        return

def process_single_image(image_path, model, predict_fn, args, output_dir):
    """Xử lý một ảnh đơn"""
    print(f"\n📷 Đang xử lý: {image_path.name}")

    # Đọc ảnh
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print(f"❌ Không thể đọc ảnh: {image_path}")
        return

    # Inference
    start_time = time.time()
    predictions = predict_fn(model, image_bgr, device=args.device, conf_thres=args.conf)
    inference_time = (time.time() - start_time) * 1000

    # In kết quả
    print(f"⏱️ {inference_time:.1f}ms:")
    if predictions:
        for class_name, conf, (x1, y1, x2, y2) in predictions:
            print(f"    ✅ {class_name}  conf={conf:.3f}  box=[{x1},{y1},{x2},{y2}]")
    else:
        print("    ❌ Không phát hiện Nemo hoặc Dory")

    # Vẽ kết quả
    result_image = draw_results(image_bgr, predictions, args.model)
    result_image = draw_time_on_image(result_image, inference_time, args.model)

    # Lưu ảnh
    output_path = output_dir / f"{args.model}_{image_path.name}"
    cv2.imwrite(str(output_path), result_image)
    print(f"💾 Đã lưu: {output_path}")

    # Hiển thị (nếu được yêu cầu)
    if args.show:
        cv2.imshow(f"Result - {args.model}", result_image)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            cv2.destroyAllWindows()
            sys.exit(0)
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
```

### Bước 13: Test Script Cơ Bản

```bash
# Đảm bảo có ảnh test
# Download một ảnh Nemo/Dory từ internet hoặc sử dụng ảnh có sẵn

# Test với YOLOv8 (default)
python detect.py --image "test_image.jpg" --show

# Test các model khác
python detect.py --image "test_image.jpg" --model yolov5 --conf 0.3
python detect.py --image "test_image.jpg" --model fasterrcnn --conf 0.5
```

---

## 🌐 Xây Dựng Web Interface

### Bước 14: Tạo File app.py

```python
"""
🐠 Web App - Nhận dạng cá Nemo & Dory
"""

import base64
import io
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from flask import Flask, render_template, request, jsonify
from PIL import Image

# Import từ detect.py
from detect import (
    BASE_DIR, DATA_DIR, CLASS_NAMES, COLORS, DEFAULT_CONF,
    load_fasterrcnn, predict_fasterrcnn,
    load_yolov5, predict_yolov5,
    load_yolov8, predict_yolov8,
    draw_results, draw_time_on_image,
)

# ---------------------------------------------------------------------------
# Khởi tạo Flask
# ---------------------------------------------------------------------------

app = Flask(
    __name__,
    template_folder=str(BASE_DIR / "templates"),
    static_folder=str(BASE_DIR / "static"),
)

app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024  # 100MB

# Global variables để cache models
models_cache = {}
device = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"🌐 Flask Web App started on {device}")

# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_models():
    """Load tất cả models vào cache"""
    global models_cache

    print("📦 Đang load models...")

    # YOLOv8
    try:
        weights_path = str(DATA_DIR / "yolo8_results" / "weights" / "best.pt")
        models_cache["yolov8"] = load_yolov8(weights_path, device)
    except:
        print("⚠️ YOLOv8 load failed")

    # YOLOv5
    try:
        weights_path = str(DATA_DIR / "yolo5_results" / "weights" / "best.pt")
        models_cache["yolov5"] = load_yolov5(weights_path, device)
    except:
        print("⚠️ YOLOv5 load failed")

    # Faster R-CNN
    try:
        weights_path = str(DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth")
        models_cache["fasterrcnn"] = load_fasterrcnn(weights_path, device=device)
    except:
        print("⚠️ Faster R-CNN load failed")

    print(f"✅ Loaded {len(models_cache)} models")

def image_to_base64(image_bgr):
    """Chuyển OpenCV image thành base64 string"""
    _, buffer = cv2.imencode('.jpg', image_bgr)
    img_str = base64.b64encode(buffer).decode()
    return f"data:image/jpeg;base64,{img_str}"

def base64_to_image(base64_str):
    """Chuyển base64 string thành OpenCV image"""
    # Remove header
    img_data = base64_str.split(',')[1]
    img_bytes = base64.b64decode(img_data)

    # Chuyển thành numpy array
    nparr = np.frombuffer(img_bytes, np.uint8)
    image_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image_bgr

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Trang chủ"""
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    """API endpoint để nhận dạng ảnh"""
    try:
        data = request.get_json()

        # Parse request
        image_base64 = data.get("image")
        model_name = data.get("model", "yolov8")
        confidence = float(data.get("confidence", DEFAULT_CONF))
        compare_all = data.get("compare", False)

        if not image_base64:
            return jsonify({"error": "No image provided"}), 400

        # Chuyển base64 thành image
        image_bgr = base64_to_image(image_base64)
        if image_bgr is None:
            return jsonify({"error": "Invalid image format"}), 400

        results = {}

        if compare_all:
            # So sánh tất cả models
            for model_key in ["yolov8", "yolov5", "fasterrcnn"]:
                if model_key in models_cache:
                    result = process_with_model(image_bgr, model_key, confidence)
                    results[model_key] = result
        else:
            # Chỉ dùng 1 model
            if model_name in models_cache:
                result = process_with_model(image_bgr, model_name, confidence)
                results[model_name] = result

        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

def process_with_model(image_bgr, model_name, confidence):
    """Xử lý ảnh với 1 model cụ thể"""
    model = models_cache.get(model_name)
    if model is None:
        return {"error": f"Model {model_name} not loaded"}

    # Chọn predict function
    predict_functions = {
        "yolov8": predict_yolov8,
        "yolov5": predict_yolov5,
        "fasterrcnn": predict_fasterrcnn,
    }
    predict_fn = predict_functions.get(model_name)

    # Inference
    start_time = time.time()
    predictions = predict_fn(model, image_bgr, device=device, conf_thres=confidence)
    inference_time = (time.time() - start_time) * 1000

    # Vẽ kết quả
    result_image = draw_results(image_bgr, predictions, model_name)
    result_image = draw_time_on_image(result_image, inference_time, model_name.upper())

    # Chuyển kết quả sang format JSON
    detections = []
    for class_name, conf, (x1, y1, x2, y2) in predictions:
        detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": [x1, y1, x2, y2]
        })

    return {
        "detections": detections,
        "inference_time": inference_time,
        "result_image": image_to_base64(result_image)
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    load_models()
    app.run(debug=True, host="0.0.0.0", port=5000)
```

### Bước 15: Tạo HTML Template

Tạo file `templates/index.html`:

```html
<!DOCTYPE html>
<html lang="vi">
  <head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>🐠 Find Marine Friends</title>
    <style>
      * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
      }

      body {
        font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        padding: 20px;
      }

      .container {
        max-width: 1200px;
        margin: 0 auto;
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
      }

      h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 30px;
        font-size: 2.5em;
      }

      .upload-area {
        border: 3px dashed #3498db;
        border-radius: 15px;
        padding: 50px;
        text-align: center;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 30px;
      }

      .upload-area:hover {
        border-color: #2980b9;
        background: rgba(52, 152, 219, 0.05);
      }

      .upload-area.dragover {
        border-color: #27ae60;
        background: rgba(39, 174, 96, 0.1);
      }

      .controls {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
        gap: 20px;
        margin-bottom: 30px;
      }

      .control-group {
        background: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
      }

      .control-group label {
        display: block;
        font-weight: bold;
        margin-bottom: 10px;
        color: #2c3e50;
      }

      select,
      input[type="range"] {
        width: 100%;
        padding: 10px;
        border: 2px solid #ddd;
        border-radius: 5px;
        font-size: 16px;
      }

      .confidence-value {
        text-align: center;
        font-weight: bold;
        color: #3498db;
      }

      .buttons {
        display: flex;
        gap: 15px;
        justify-content: center;
        flex-wrap: wrap;
      }

      button {
        background: linear-gradient(135deg, #3498db, #2980b9);
        color: white;
        border: none;
        padding: 15px 30px;
        border-radius: 25px;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.3s ease;
      }

      button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(52, 152, 219, 0.4);
      }

      button:disabled {
        opacity: 0.6;
        cursor: not-allowed;
      }

      .results {
        margin-top: 30px;
      }

      .result-item {
        background: #f8f9fa;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
      }

      .result-header {
        display: flex;
        justify-content: between;
        align-items: center;
        margin-bottom: 15px;
      }

      .result-image {
        max-width: 100%;
        border-radius: 10px;
        box-shadow: 0 5px 15px rgba(0, 0, 0, 0.2);
      }

      .detection-info {
        margin-top: 15px;
        padding: 15px;
        background: #e8f4fd;
        border-radius: 8px;
      }

      .detection-item {
        display: flex;
        justify-content: space-between;
        padding: 8px 0;
        border-bottom: 1px solid #ddd;
      }

      .detection-item:last-child {
        border-bottom: none;
      }

      .loading {
        text-align: center;
        padding: 50px;
      }

      .spinner {
        border: 4px solid #f3f3f3;
        border-top: 4px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 0 auto 20px;
      }

      @keyframes spin {
        0% {
          transform: rotate(0deg);
        }
        100% {
          transform: rotate(360deg);
        }
      }
    </style>
  </head>
  <body>
    <div class="container">
      <h1>🐠 Find Marine Friends</h1>
      <p
        style="text-align: center; margin-bottom: 30px; color: #666; font-size: 18px;"
      >
        Upload ảnh để nhận dạng cá Nemo, Dory hoặc các loại sao biển bằng AI
      </p>

      <!-- Upload Area -->
      <div class="upload-area" id="uploadArea">
        <h3>📥 Kéo thả ảnh vào đây hoặc click để chọn</h3>
        <p>Hỗ trợ: JPG, PNG, BMP (tối đa 100MB)</p>
        <input
          type="file"
          id="fileInput"
          accept="image/*"
          style="display: none;"
        />
      </div>

      <!-- Controls -->
      <div class="controls">
        <div class="control-group">
          <label for="modelSelect">🧠 Chọn Model AI:</label>
          <select id="modelSelect">
            <option value="yolov8">YOLOv8 (Fastest)</option>
            <option value="yolov5">YOLOv5 (Balanced)</option>
            <option value="fasterrcnn">Faster R-CNN (Accurate)</option>
          </select>
        </div>

        <div class="control-group">
          <label for="confidenceSlider">🎯 Confidence Threshold:</label>
          <input
            type="range"
            id="confidenceSlider"
            min="0.1"
            max="1.0"
            step="0.05"
            value="0.5"
          />
          <div class="confidence-value" id="confidenceValue">0.5</div>
        </div>
      </div>

      <!-- Buttons -->
      <div class="buttons">
        <button id="detectBtn" disabled>🔍 Nhận Dạng</button>
        <button id="compareBtn" disabled>📊 So Sánh Tất Cả Models</button>
        <button id="clearBtn">🗑️ Xóa Kết Quả</button>
      </div>

      <!-- Results -->
      <div class="results" id="results"></div>
    </div>

    <script>
      // Global variables
      let selectedImage = null;
      let selectedImageName = "";

      // Elements
      const uploadArea = document.getElementById("uploadArea");
      const fileInput = document.getElementById("fileInput");
      const modelSelect = document.getElementById("modelSelect");
      const confidenceSlider = document.getElementById("confidenceSlider");
      const confidenceValue = document.getElementById("confidenceValue");
      const detectBtn = document.getElementById("detectBtn");
      const compareBtn = document.getElementById("compareBtn");
      const clearBtn = document.getElementById("clearBtn");
      const results = document.getElementById("results");

      // Event listeners
      uploadArea.addEventListener("click", () => fileInput.click());
      uploadArea.addEventListener("dragover", handleDragOver);
      uploadArea.addEventListener("dragleave", handleDragLeave);
      uploadArea.addEventListener("drop", handleDrop);
      fileInput.addEventListener("change", handleFileSelect);
      confidenceSlider.addEventListener("input", updateConfidenceValue);
      detectBtn.addEventListener("click", detectImage);
      compareBtn.addEventListener("click", compareModels);
      clearBtn.addEventListener("click", clearResults);

      // Functions
      function handleDragOver(e) {
        e.preventDefault();
        uploadArea.classList.add("dragover");
      }

      function handleDragLeave(e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");
      }

      function handleDrop(e) {
        e.preventDefault();
        uploadArea.classList.remove("dragover");

        const files = e.dataTransfer.files;
        if (files.length > 0) {
          processFile(files[0]);
        }
      }

      function handleFileSelect(e) {
        if (e.target.files.length > 0) {
          processFile(e.target.files[0]);
        }
      }

      function processFile(file) {
        if (!file.type.startsWith("image/")) {
          alert("Vui lòng chọn file ảnh!");
          return;
        }

        selectedImageName = file.name;

        const reader = new FileReader();
        reader.onload = function (e) {
          selectedImage = e.target.result;
          uploadArea.innerHTML = `
                    <h3>✅ Đã chọn: ${file.name}</h3>
                    <p>Click để chọn ảnh khác</p>
                `;
          detectBtn.disabled = false;
          compareBtn.disabled = false;
        };
        reader.readAsDataURL(file);
      }

      function updateConfidenceValue() {
        confidenceValue.textContent = confidenceSlider.value;
      }

      async function detectImage() {
        if (!selectedImage) return;

        showLoading();

        try {
          const response = await fetch("/detect", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              image: selectedImage,
              model: modelSelect.value,
              confidence: parseFloat(confidenceSlider.value),
              compare: false,
            }),
          });

          const data = await response.json();
          displayResults(data);
        } catch (error) {
          console.error("Error:", error);
          results.innerHTML =
            '<div class="result-item"><h3>❌ Lỗi: ' +
            error.message +
            "</h3></div>";
        }
      }

      async function compareModels() {
        if (!selectedImage) return;

        showLoading("Đang so sánh tất cả models...");

        try {
          const response = await fetch("/detect", {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
            },
            body: JSON.stringify({
              image: selectedImage,
              confidence: parseFloat(confidenceSlider.value),
              compare: true,
            }),
          });

          const data = await response.json();
          displayResults(data);
        } catch (error) {
          console.error("Error:", error);
          results.innerHTML =
            '<div class="result-item"><h3>❌ Lỗi: ' +
            error.message +
            "</h3></div>";
        }
      }

      function showLoading(message = "Đang xử lý...") {
        results.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <h3>${message}</h3>
                </div>
            `;
      }

      function displayResults(data) {
        let html = "";

        for (const [modelName, result] of Object.entries(data)) {
          if (result.error) {
            html += `
                        <div class="result-item">
                            <h3>❌ ${modelName.toUpperCase()}: ${result.error}</h3>
                        </div>
                    `;
            continue;
          }

          html += `
                    <div class="result-item">
                        <div class="result-header">
                            <h3>🤖 ${modelName.toUpperCase()}</h3>
                            <span style="font-weight: bold; color: #3498db;">
                                ⏱️ ${result.inference_time.toFixed(1)}ms
                            </span>
                        </div>
                        
                        <img src="${result.result_image}" alt="Result" class="result-image">
                        
                        <div class="detection-info">
                            <h4>📋 Kết quả phát hiện:</h4>
                `;

          if (result.detections.length > 0) {
            for (const det of result.detections) {
              const emoji = det.class === "nemo" ? "🐠" : "🐟";
              html += `
                            <div class="detection-item">
                                <span>${emoji} <strong>${det.class}</strong></span>
                                <span>Confidence: <strong>${(det.confidence * 100).toFixed(1)}%</strong></span>
                            </div>
                        `;
            }
          } else {
            html +=
              '<div class="detection-item">❌ Không phát hiện Nemo hoặc Dory</div>';
          }

          html += `
                        </div>
                    </div>
                `;
        }

        results.innerHTML = html;
      }

      function clearResults() {
        results.innerHTML = "";
        selectedImage = null;
        uploadArea.innerHTML = `
                <h3>📥 Kéo thả ảnh vào đây hoặc click để chọn</h3>
                <p>Hỗ trợ: JPG, PNG, BMP (tối đa 100MB)</p>
            `;
        detectBtn.disabled = true;
        compareBtn.disabled = true;
        fileInput.value = "";
      }
    </script>
  </body>
</html>
```

### Bước 16: Test Web App

```bash
# Chạy web server
python app.py

# Mở trình duyệt tại: http://localhost:5000
```

---

## 🚀 Test và Debug

### Bước 17: Tạo Test Cases

```bash
# Tạo thư mục test
mkdir test_images

# Download một vài ảnh test từ internet:
# - Ảnh có chứa cá Nemo (cá hề)
# - Ảnh có chứa cá Dory (cá xanh)
# - Ảnh có cả hai
# - Ảnh không có cá nào
```

### Bước 18: Debug Common Issues

**1. Lỗi model loading:**

```python
# Thêm vào detect.py để debug
def debug_model_loading():
    print("🔍 Checking model files...")

    frcnn_path = DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth"
    yolo5_path = DATA_DIR / "yolo5_results" / "weights" / "best.pt"
    yolo8_path = DATA_DIR / "yolo8_results" / "weights" / "best.pt"

    for name, path in [("Faster R-CNN", frcnn_path), ("YOLOv5", yolo5_path), ("YOLOv8", yolo8_path)]:
        if path.exists():
            size = path.stat().st_size / 1024 / 1024  # MB
            print(f"✅ {name}: {path} ({size:.1f}MB)")
        else:
            print(f"❌ {name}: {path} - NOT FOUND")

# Gọi debug function
if __name__ == "__main__":
    debug_model_loading()
    main()
```

**2. Lỗi inference:**

```python
# Thêm error handling
def safe_predict(predict_fn, model, image_bgr, **kwargs):
    """Predict với error handling"""
    try:
        return predict_fn(model, image_bgr, **kwargs)
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return []
```

**3. Lỗi performance:**

```python
# Monitor memory usage
import psutil
import gc

def monitor_performance():
    process = psutil.Process()
    memory_mb = process.memory_info().rss / 1024 / 1024
    print(f"🧠 Memory usage: {memory_mb:.1f}MB")

    if torch.cuda.is_available():
        gpu_mb = torch.cuda.memory_allocated() / 1024 / 1024
        print(f"🎮 GPU memory: {gpu_mb:.1f}MB")

# Gọi sau mỗi prediction
# monitor_performance()
```

---

## 📈 Cải Tiến và Tối Ưu

### Bước 19: Thêm Tính Năng Mở Rộng

**1. Thêm thống kê chi tiết:**

```python
def get_training_metrics(model_name):
    """Đọc metrics từ file CSV"""
    csv_path = TRAINING_METRICS_CSV.get(model_name)
    if csv_path and csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            # Tìm epoch tốt nhất
            best_row = df.loc[df['metrics/mAP_0.5'].idxmax()]
            return {
                "best_epoch": int(best_row['epoch']),
                "precision": float(best_row['metrics/precision']),
                "recall": float(best_row['metrics/recall']),
                "mAP_50": float(best_row['metrics/mAP_0.5']),
                "mAP_50_95": float(best_row['metrics/mAP_0.5:0.95'])
            }
        except:
            return None
    return None
```

**2. Batch processing:**

```python
def process_batch_images(image_dir, output_dir, model_name="yolov8"):
    """Xử lý hàng loạt ảnh"""
    from tqdm import tqdm

    # Load model một lần
    model = load_model_by_name(model_name)

    image_files = get_image_files(image_dir)
    results = []

    for img_path in tqdm(image_files, desc="Processing"):
        # Xử lý từng ảnh
        result = process_single_image(img_path, model, ...)
        results.append(result)

    # Lưu summary
    save_batch_results(results, output_dir)
```

**3. Video support:**

```python
def process_video(video_path, model_name="yolov8"):
    """Xử lý video"""
    cap = cv2.VideoCapture(video_path)

    # Video writer setup
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))

    model = load_model_by_name(model_name)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process frame
        predictions = predict_function(model, frame)
        result_frame = draw_results(frame, predictions)

        out.write(result_frame)

    cap.release()
    out.release()
```

### Bước 20: Deployment

**1. Tạo Docker container:**

```dockerfile
# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000
CMD ["python", "app.py"]
```

**2. Production setup:**

```bash
# Sử dụng gunicorn thay vì development server
pip install gunicorn

# Chạy production server
gunicorn --bind 0.0.0.0:5000 --workers 4 app:app
```

**3. Environment variables:**

```python
# config.py
import os

class Config:
    MODEL_DIR = os.getenv('MODEL_DIR', 'data')
    CONFIDENCE_THRESHOLD = float(os.getenv('CONF_THRESH', 0.5))
    DEVICE = os.getenv('DEVICE', 'auto')
    MAX_UPLOAD_SIZE = int(os.getenv('MAX_UPLOAD_MB', 100)) * 1024 * 1024
```

---

## ⚡ GPU Setup & Tối Ưu Tốc Độ

### Bước 21: Tự Động Detect GPU & Cài PyTorch CUDA

Dự án có sẵn script `setup_gpu.py` giúp tự động:

1. Phát hiện GPU NVIDIA qua `nvidia-smi`
2. Kiểm tra PyTorch hiện tại (CPU-only hay CUDA)
3. Chọn phiên bản CUDA cao nhất mà driver hỗ trợ
4. Cài đặt PyTorch CUDA phù hợp

```bash
# Kiểm tra trước (không cài)
python setup_gpu.py --check

# Output ví dụ:
# ============================================================
#   🔍 KIỂM TRA GPU & PYTORCH
# ============================================================
#   GPU:          NVIDIA GeForce RTX 3080 Ti
#   Driver CUDA:  13.1
#   PyTorch:      2.10.0+cpu
#   CUDA dùng:    Không (CPU-only)
#   Khuyến nghị:  PyTorch cu128 (CUDA 12.8)
# ⚠️  PyTorch hiện tại là CPU-only → Cần cài lại với CUDA cu128

# Tự động cài đặt
python setup_gpu.py
```

**Cách hoạt động của setup_gpu.py:**

```python
# Script parse nvidia-smi để lấy Driver CUDA version
# Driver CUDA version là version TỐI ĐA mà driver hỗ trợ (backward compatible)
# VD: Driver CUDA 13.1 chạy được cu128, cu126, cu124, cu121, cu118

# Mapping các phiên bản CUDA mà PyTorch hỗ trợ:
cuda_options = [
    (12.8, "cu128"),  # Mới nhất, khuyến nghị
    (12.6, "cu126"),
    (12.4, "cu124"),
    (12.1, "cu121"),
    (11.8, "cu118"),  # Tương thích rộng nhất
]
# Chọn version cao nhất <= Driver CUDA version
```

### Bước 22: Chuyển Đổi CPU/GPU Trên Web App

Web app hỗ trợ toggle CPU ↔ GPU realtime trên giao diện:

```python
# Backend (app.py) - Device state management
_current_device = "cuda:0" if torch.cuda.is_available() else "cpu"

# API endpoint chuyển đổi device
@app.route("/set-device", methods=["POST"])
def set_device():
    global _current_device
    data = request.get_json()
    new_device = data.get("device", "cpu")  # "cpu" hoặc "cuda:0"
    _current_device = new_device
    return jsonify({"device": _current_device})

# Model cache theo device - key = (model_name, device)
# Khi chuyển device, model được load lại trên device mới
_model_cache = {}  # {("yolov8", "cuda:0"): (model, predict_fn), ...}
```

```javascript
// Frontend (index.html) - Toggle switch
async function toggleDevice(checkbox) {
  const newDevice = checkbox.checked ? "cuda:0" : "cpu";
  const resp = await fetch("/set-device", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ device: newDevice }),
  });
  // Cập nhật UI hiển thị device hiện tại
}
```

### Bước 23: Benchmark CPU vs GPU

Kết quả đo trên ảnh 480×640, `imgsz=416` (RTX 3080 Ti):

| Mô hình          | CPU               | GPU (CUDA)         | Tăng tốc |
| ---------------- | ----------------- | ------------------ | -------- |
| **YOLOv8**       | 102 ms (10 FPS)   | **11 ms (91 FPS)** | ~9×      |
| **YOLOv5**       | 63 ms (16 FPS)    | **12 ms (87 FPS)** | ~5×      |
| **Faster R-CNN** | 1506 ms (0.7 FPS) | **38 ms (26 FPS)** | ~39×     |

**Phân tích:**

- **YOLOv8 + GPU**: 91 FPS — real-time, xử lý video mượt mà
- **YOLOv5 + GPU**: 87 FPS — tương đương YOLOv8
- **Faster R-CNN + GPU**: 26 FPS — đã đủ real-time nhờ GPU
- **Faster R-CNN + CPU**: 0.7 FPS — quá chậm, cần GPU bắt buộc
- Video 1m30s (30fps, ~2700 frames) với skip_frames=3:
  - CPU: ~3 phút | GPU: ~15 giây

### Tối ưu tốc độ video

```python
# 1. Giảm imgsz cho video (416 thay vì 640)
VIDEO_IMGSZ = 416  # Nhanh hơn ~2.4x so với 640

# 2. Skip frames - chỉ detect 1/N frame
skip_frames = 3  # Default: detect 1 trong 3 frame

# 3. Dùng grab() thay vì read() cho frame bị skip
while True:
    ret = cap.grab()  # Chỉ nhảy tới frame, không decode
    if frame_count % skip_frames == 0:
        ret2, frame = cap.retrieve()  # Decode frame cần xử lý
        detections = predict_fn(frame, conf)
        writer.write(draw_results(frame, detections))

# 4. Resize frame trước inference
max_infer_dim = 480  # Resize frame lớn trước khi đưa vào model
```

---

## 🎯 Kết Luận

### Tóm Tắt Những Gì Đã Xây Dựng

1. **Object Detection System** hoàn chỉnh với 3 mô hình state-of-the-art
2. **Command-line interface** cho batch processing
3. **Web application** thân thiện với người dùng
4. **Performance monitoring** và error handling
5. **Modular code structure** dễ mở rộng
6. **GPU auto-setup** tự detect GPU và cài đúng PyTorch CUDA
7. **CPU/GPU toggle** chuyển đổi device realtime trên web

### Kiến Thức Đã Học

✅ **Computer Vision**: Image processing, object detection, bounding boxes  
✅ **Deep Learning**: PyTorch, model loading, inference  
✅ **YOLO Architecture**: One-stage detection, anchor-based vs anchor-free  
✅ **Faster R-CNN**: Two-stage detection, RPN + detection head  
✅ **Web Development**: Flask, REST APIs, file upload  
✅ **GPU Acceleration**: CUDA setup, CPU/GPU toggle, performance benchmarking  
✅ **Production Skills**: Error handling, performance optimization, deployment

### Bước Tiếp Theo

🔥 **Cải tiến model:**

- Fine-tune trên dataset riêng
- Data augmentation
- Ensemble methods

🔥 **Mở rộng tính năng:**

- Real-time camera detection
- Mobile app deployment
- Cloud-based inference API

🔥 **Tối ưu performance:**

- Model quantization
- TensorRT optimization
- GPU cluster deployment

---

## 📚 Tài Liệu Tham Khảo

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Ultralytics YOLOv5/v8 Docs](https://docs.ultralytics.com/)
- [Faster R-CNN Paper](https://arxiv.org/abs/1506.01497)
- [Flask Documentation](https://flask.palletsprojects.com/)
- [OpenCV Python Tutorials](https://opencv-python-tutroals.readthedocs.io/)

---

**🎉 Chúc mừng bạn đã hoàn thành dự án!**

Đây là một foundation tuyệt vời để tiến tới các dự án Computer Vision phức tạp hơn. Hãy tiếp tục thử nghiệm và khám phá! 🚀
