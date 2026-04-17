"""
Nhận dạng cá Nemo và Dory sử dụng 3 mô hình:
  - Faster R-CNN  (torchvision)
  - YOLOv5        (ultralytics/yolov5)
  - YOLOv8        (ultralytics)

Sử dụng:
  python detect.py --image path/to/image.jpg --model yolov8
  python detect.py --image path/to/image.jpg --model yolov5
  python detect.py --image path/to/image.jpg --model fasterrcnn
  python detect.py --image path/to/folder   --model yolov8      # xử lý cả thư mục
"""

# ---------------------------------------------------------------------------
# Import thư viện
# ---------------------------------------------------------------------------
import argparse          # Xử lý tham số dòng lệnh (--image, --model, ...)
import os                # Thao tác với hệ điều hành (đường dẫn, thư mục)
import sys               # Thoát chương trình khi gặp lỗi
import time              # Đo thời gian xử lý (load model, inference)
from pathlib import Path # Thao tác đường dẫn file dễ dàng hơn os.path

import cv2               # OpenCV - đọc/ghi/vẽ ảnh, xử lý hình ảnh
import numpy as np       # Thao tác mảng số (ảnh = mảng numpy)
import pandas as pd      # Đọc file CSV chứa metrics huấn luyện
import torch             # PyTorch - framework deep learning chính
from PIL import Image    # Pillow - đọc ảnh (hỗ trợ thêm cho một số model)

# ---------------------------------------------------------------------------
# Cấu hình chung
# ---------------------------------------------------------------------------

# Thư mục gốc của dự án (nơi chứa file detect.py)
BASE_DIR = Path(__file__).resolve().parent
# Thư mục chứa trọng số các mô hình đã train
DATA_DIR = BASE_DIR / "data"

# Tên 7 lớp (class) mà mô hình nhận dạng
CLASS_NAMES = [
    "bat sea star",
    "blue sea star",
    "crown of thorn starfish",
    "dory",
    "nemo",
    "red cushion sea star",
    "royal starfish",
]

# Màu sắc BGR (OpenCV dùng BGR thay vì RGB) để vẽ bounding box cho mỗi lớp
COLORS = {
    "bat sea star":             (0, 165, 255),   # Cam
    "blue sea star":            (255, 200, 0),   # Xanh dương
    "crown of thorn starfish":  (0, 0, 220),     # Đỏ
    "dory":                     (255, 255, 0),   # Cyan
    "nemo":                     (0, 128, 255),   # Cam đậm
    "red cushion sea star":     (80, 80, 255),   # Đỏ nhạt
    "royal starfish":           (200, 0, 200),   # Tím
}

# Ngưỡng confidence mặc định: chỉ giữ detection có độ tin cậy >= 50%
# Giá trị từ 0.0 (giữ tất cả) đến 1.0 (rất chặt). Thường dùng 0.25-0.7
DEFAULT_CONF = 0.5

# Mapping label index → tên lớp cho Faster R-CNN
# Checkpoint được train với 7 lớp + background (index 0)
FRCNN_LABEL_MAP = {
    1: "bat sea star",
    2: "blue sea star",
    3: "crown of thorn starfish",
    4: "dory",
    5: "nemo",
    6: "red cushion sea star",
    7: "royal starfish",
}

# Kích thước ảnh đầu vào cho YOLO (resize về 640x640 pixel trước khi inference)
# Giá trị này phải khớp với kích thước khi train (imgsz=640 trong config)
IMG_SIZE = 640

# Đường dẫn tới file CSV lưu metrics trong quá trình huấn luyện
# Mỗi dòng = 1 epoch, chứa precision, recall, mAP, loss, v.v.
TRAINING_METRICS_CSV = {
    "yolov5": DATA_DIR / "yolo5_results" / "results.csv",
    "yolov8": DATA_DIR / "yolo8_results" / "results.csv",
}

# ---------------------------------------------------------------------------
# 1. Faster R-CNN
# ---------------------------------------------------------------------------
def load_fasterrcnn(weights_path: str, num_classes: int = 8, device: str = "cpu"):
    """
    Load mô hình Faster R-CNN ResNet50-FPN đã fine-tune.
    
    Faster R-CNN là mô hình object detection 2 giai đoạn:
      Giai đoạn 1 (RPN): Đề xuất vùng có thể chứa đối tượng (Region Proposals)
      Giai đoạn 2 (Head): Phân loại + tinh chỉnh bounding box cho mỗi vùng
    
    Backbone: ResNet50 + Feature Pyramid Network (FPN) để trích xuất đặc trưng
    đa tỉ lệ, giúp phát hiện cả đối tượng nhỏ và lớn.
    
    Args:
        weights_path: Đường dẫn tới file .pth chứa trọng số đã train
        num_classes:  Số lớp đối tượng (3 = background + nemo + dory)
                      Faster R-CNN luôn có thêm 1 lớp background (index 0)
        device:       'cpu' hoặc 'cuda:0' (GPU)
    Returns:
        model: Mô hình Faster R-CNN sẵn sàng inference
    """
    import torchvision
    from torchvision.models.detection import fasterrcnn_resnet50_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Tạo mô hình Faster R-CNN với backbone ResNet50-FPN
    # weights=None vì ta sẽ load trọng số riêng (đã fine-tune)
    model = fasterrcnn_resnet50_fpn(weights=None)
    
    # Thay thế đầu phân loại (box predictor) cho phù hợp số lớp
    # Mặc định Faster R-CNN có 91 lớp (COCO), ta cần 8 (bg + 7 lớp)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Load trọng số đã train từ file .pth
    # map_location=device: chuyển trọng số về đúng device (CPU/GPU)
    state = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Hỗ trợ nhiều cách lưu trọng số phổ biến:
    #   - {'model_state_dict': ...}  (cách lưu chuẩn khi train)
    #   - {'model_state': ...}       (cách lưu khác khi train)
    #   - {'state_dict': ...}        (cách lưu của PyTorch Lightning)
    #   - state_dict trực tiếp       (torch.save(model.state_dict(), ...))
    if isinstance(state, dict) and "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif isinstance(state, dict) and "model_state" in state:
        model.load_state_dict(state["model_state"])
    elif isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.to(device)   # Chuyển mô hình sang CPU/GPU
    model.eval()       # Chuyển sang chế độ inference (tắt dropout, batch norm)
    return model


def predict_fasterrcnn(model, image_bgr: np.ndarray, device: str = "cpu",
                       conf_thres: float = DEFAULT_CONF):
    """
    Dự đoán bằng Faster R-CNN.
    
    Quy trình:
      1. Chuyển ảnh BGR → RGB (Faster R-CNN nhận RGB)
      2. Chuyển ảnh thành tensor [0,1] và thêm batch dimension
      3. Chạy forward pass (inference)
      4. Lọc kết quả theo ngưỡng confidence
    
    Args:
        model:      Mô hình Faster R-CNN đã load
        image_bgr:  Ảnh đầu vào dạng numpy array (BGR - OpenCV format)
        device:     'cpu' hoặc 'cuda:0'
        conf_thres: Ngưỡng confidence, bỏ qua detection thấp hơn giá trị này
    Returns:
        list of (label, score, (x1, y1, x2, y2))
    """
    import torchvision.transforms.functional as F

    # Chuyển BGR (OpenCV) → RGB (PyTorch/torchvision)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # F.to_tensor(): chuyển ảnh HxWxC [0-255] → tensor CxHxW [0.0-1.0]
    # unsqueeze(0): thêm batch dimension → 1xCxHxW (batch size = 1)
    tensor = F.to_tensor(img_rgb).unsqueeze(0).to(device)

    # Tắt gradient vì chỉ inference (không cần backpropagation)
    with torch.no_grad():
        # model() trả về list dict, mỗi dict = 1 ảnh trong batch
        # outputs chứa: 'boxes' (tọa độ), 'labels' (lớp), 'scores' (confidence)
        outputs = model(tensor)[0]

    results = []
    for box, label, score in zip(outputs["boxes"], outputs["labels"], outputs["scores"]):
        # Bỏ qua detection có confidence thấp hơn ngưỡng
        if score.item() < conf_thres:
            continue
        # Trong Faster R-CNN (checkpoint này):
        #   label 0 = background, label 1 = objects (bỏ qua)
        #   label 2 = dory,       label 3 = nemo
        label_idx = label.item()
        if label_idx not in FRCNN_LABEL_MAP:
            continue
        cls_name = FRCNN_LABEL_MAP[label_idx]
        # Chuyển tọa độ bounding box từ tensor sang int
        # box = [x1, y1, x2, y2] = [góc trên-trái, góc dưới-phải]
        x1, y1, x2, y2 = box.int().tolist()
        results.append((cls_name, score.item(), (x1, y1, x2, y2)))
    return results

# ---------------------------------------------------------------------------
# 2. YOLOv5
# ---------------------------------------------------------------------------
def load_yolov5(weights_path: str, device: str = "cpu"):
    """
    Load mô hình YOLOv5 qua torch.hub.
    
    YOLOv5 là mô hình object detection 1 giai đoạn (single-stage):
      - Chia ảnh thành lưới (grid), mỗi ô dự đoán bounding box + class
      - Nhanh hơn Faster R-CNN nhưng có thể kém chính xác hơn ở object nhỏ
    
    Cách load: torch.hub tải code YOLOv5 từ GitHub (ultralytics/yolov5)
    rồi load trọng số custom (.pt) đã train trên dataset cá Nemo/Dory.
    Lần đầu chạy cần internet để tải code; sau đó cache lại.
    
    Args:
        weights_path: Đường dẫn tới file .pt (trọng số YOLOv5 đã train)
        device:       'cpu' hoặc 'cuda:0'
    Returns:
        model: Mô hình YOLOv5 sẵn sàng inference
    """
    # Sửa lỗi PosixPath trên Windows:
    # File .pt train trên Linux chứa PosixPath, Windows không thể khởi tạo.
    # Workaround: tạm gán PosixPath = WindowsPath khi load trên Windows.
    import pathlib
    _posix_backup = getattr(pathlib, "PosixPath")
    if os.name == "nt":  # Windows
        pathlib.PosixPath = pathlib.WindowsPath

    try:
        # torch.hub.load(): tải model từ GitHub repo ultralytics/yolov5
        # "custom": cho biết ta dùng trọng số tự train (không phải pretrained COCO)
        # force_reload=False: dùng cache nếu đã tải trước đó
        model = torch.hub.load("ultralytics/yolov5", "custom", path=weights_path,
                               force_reload=False, device=device)
    finally:
        pathlib.PosixPath = _posix_backup  # Khôi phục lại để không ảnh hưởng code khác

    model.conf = DEFAULT_CONF  # Đặt ngưỡng confidence mặc định
    return model


def predict_yolov5(model, image_bgr: np.ndarray, conf_thres: float = DEFAULT_CONF,
                   imgsz: int = IMG_SIZE):
    """
    Dự đoán bằng YOLOv5.
    
    Quy trình:
      1. Đặt ngưỡng confidence cho model
      2. Chuyển ảnh BGR → RGB
      3. Gọi model() với kích thước ảnh imgsz
      4. Parse kết quả từ tensor xyxy (x1,y1,x2,y2,conf,class)
    
    Args:
        model:      Mô hình YOLOv5 đã load
        image_bgr:  Ảnh đầu vào (BGR, numpy array)
        conf_thres: Ngưỡng confidence
        imgsz:      Kích thước ảnh inference (mặc định 640)
    Returns:
        list of (label, score, (x1, y1, x2, y2))
    """
    model.conf = conf_thres  # Cập nhật ngưỡng confidence
    
    # YOLOv5 nhận ảnh RGB (không phải BGR như OpenCV)
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # Chạy inference, size=imgsz để resize ảnh
    det = model(img_rgb, size=imgsz)
    
    results = []
    # det.xyxy[0]: tensor chứa tất cả detection của ảnh đầu tiên (batch=1)
    # Mỗi hàng: [x1, y1, x2, y2, confidence, class_index]
    for *xyxy, conf, cls in det.xyxy[0].cpu().numpy():
        cls_idx = int(cls)
        # Lấy tên lớp từ model.names (dict {0: 'nemo', 1: 'dory'})
        label = model.names[cls_idx] if cls_idx < len(model.names) else CLASS_NAMES[cls_idx]
        x1, y1, x2, y2 = map(int, xyxy)  # Chuyển tọa độ sang int
        results.append((label.lower(), float(conf), (x1, y1, x2, y2)))
    return results

# ---------------------------------------------------------------------------
# 3. YOLOv8
# ---------------------------------------------------------------------------
def load_yolov8(weights_path: str, device: str = "cpu"):
    """
    Load mô hình YOLOv8 qua thư viện ultralytics.
    
    YOLOv8 là phiên bản mới nhất của YOLO (2023), cải tiến:
      - Kiến trúc backbone tốt hơn (CSPDarknet cải tiến)
      - Anchor-free detection (không cần định nghĩa anchor trước)
      - API đơn giản hơn YOLOv5
    
    Args:
        weights_path: Đường dẫn tới file .pt (trọng số YOLOv8 đã train)
        device:       'cpu' hoặc 'cuda:0'
    Returns:
        model: Mô hình YOLOv8 (ultralytics.YOLO object)
    """
    from ultralytics import YOLO
    # YOLO() tự động nhận dạng loại model từ file .pt
    model = YOLO(weights_path)
    return model


def predict_yolov8(model, image_bgr: np.ndarray, device: str = "cpu",
                   conf_thres: float = DEFAULT_CONF, imgsz: int = IMG_SIZE):
    """
    Dự đoán bằng YOLOv8.
    
    Quy trình:
      1. Gọi model.predict() với ảnh BGR trực tiếp (YOLOv8 tự xử lý)
      2. Kết quả trả về list Results, mỗi Results chứa .boxes
      3. Mỗi box có: .xyxy (tọa độ), .conf (confidence), .cls (class index)
    
    Args:
        model:      Mô hình YOLOv8 đã load
        image_bgr:  Ảnh đầu vào (BGR, numpy array)
        device:     'cpu' hoặc 'cuda:0'
        conf_thres: Ngưỡng confidence
        imgsz:      Kích thước ảnh inference (mặc định 640)
    Returns:
        list of (label, score, (x1, y1, x2, y2))
    """
    # model.predict(): chạy inference
    #   source: ảnh đầu vào (numpy array, path, URL, ...)
    #   imgsz:  resize ảnh về kích thước này trước khi detect
    #   conf:   ngưỡng confidence (bỏ detection thấp hơn)
    #   verbose=False: không in log ra console
    preds = model.predict(source=image_bgr, imgsz=imgsz, conf=conf_thres,
                          device=device, verbose=False)
    
    results = []
    for r in preds:           # Duyệt qua từng ảnh (ở đây chỉ 1 ảnh)
        for box in r.boxes:   # Duyệt qua từng bounding box phát hiện được
            cls_idx = int(box.cls.item())    # Index lớp đối tượng (0=nemo, 1=dory)
            # Lấy tên lớp từ model.names (dict tự động load từ file .pt)
            label = model.names[cls_idx] if cls_idx in model.names else CLASS_NAMES[cls_idx]
            # box.xyxy[0]: tensor [x1, y1, x2, y2] - tọa độ bounding box
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf.item())   # Confidence score (0.0 - 1.0)
            results.append((label.lower(), score, (x1, y1, x2, y2)))
    return results

# ---------------------------------------------------------------------------
# Metrics huấn luyện
# ---------------------------------------------------------------------------
def get_training_metrics(model_name: str) -> dict | None:
    """
    Đọc metrics tốt nhất từ file CSV ghi lại trong quá trình huấn luyện.
    
    Các metrics quan trọng:
      - Precision: Trong các detection model đưa ra, bao nhiêu % là đúng
                   Precision cao = ít false positive (ít nhận nhầm)
      - Recall:    Trong tất cả đối tượng thật, model phát hiện được bao nhiêu %
                   Recall cao = ít false negative (ít bỏ sót)
      - mAP@0.5:   Mean Average Precision tại IoU=0.5
                   (bounding box dự đoán trùng >= 50% với ground truth)
      - mAP@0.5:0.95: Trung bình mAP tại nhiều mức IoU (0.5, 0.55, ..., 0.95)
                       Đây là metric khắt khe nhất, đánh giá tổng thể
    
    IoU (Intersection over Union): Tỉ lệ diện tích giao / diện tích hợp
    giữa box dự đoán và box thật. IoU=1.0 = trùng hoàn toàn.
    
    Args:
        model_name: 'yolov5' hoặc 'yolov8' (fasterrcnn không có CSV)
    Returns:
        dict chứa metrics hoặc None nếu không đọc được
    """
    csv_path = TRAINING_METRICS_CSV.get(model_name)
    if csv_path is None or not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
        # Xóa khoảng trắng thừa trong tên cột (CSV từ YOLO hay có space thừa)
        df.columns = df.columns.str.strip()

        if model_name == "yolov5":
            # Tìm epoch có mAP@0.5 cao nhất (= model tốt nhất)
            best_idx = df["metrics/mAP_0.5"].idxmax()
            row = df.iloc[best_idx]
            return {
                "epoch":     int(row["epoch"]),
                "precision":  row["metrics/precision"],
                "recall":     row["metrics/recall"],
                "mAP@0.5":    row["metrics/mAP_0.5"],
                "mAP@0.5:0.95": row["metrics/mAP_0.5:0.95"],
            }
        elif model_name == "yolov8":
            # YOLOv8 dùng tên cột khác: metrics/mAP50(B), metrics/precision(B),...
            # (B) = Box detection metrics
            best_idx = df["metrics/mAP50(B)"].idxmax()
            row = df.iloc[best_idx]
            return {
                "epoch":     int(row["epoch"]),
                "precision":  row["metrics/precision(B)"],
                "recall":     row["metrics/recall(B)"],
                "mAP@0.5":    row["metrics/mAP50(B)"],
                "mAP@0.5:0.95": row["metrics/mAP50-95(B)"],
            }
    except Exception as e:
        print(f"  [CẢNH BÁO] Không đọc được metrics: {e}")
    return None


def print_training_metrics(model_name: str):
    """In metrics huấn luyện tốt nhất của mô hình."""
    metrics = get_training_metrics(model_name)
    if metrics is None:
        if model_name == "fasterrcnn":
            print("  (Faster R-CNN không có file CSV metrics huấn luyện)")
        return

    print(f"\n{'='*60}")
    print(f"  📊 ĐỘ CHÍNH XÁC HUẤN LUYỆN TỐT NHẤT - {model_name.upper()}")
    print(f"{'='*60}")
    print(f"  Epoch tốt nhất :  {metrics['epoch']}")
    print(f"  Precision      :  {metrics['precision']:.4f}  ({metrics['precision']*100:.1f}%)")
    print(f"  Recall         :  {metrics['recall']:.4f}  ({metrics['recall']*100:.1f}%)")
    print(f"  mAP@0.5        :  {metrics['mAP@0.5']:.4f}  ({metrics['mAP@0.5']*100:.1f}%)")
    print(f"  mAP@0.5:0.95   :  {metrics['mAP@0.5:0.95']:.4f}  ({metrics['mAP@0.5:0.95']*100:.1f}%)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Vẽ kết quả lên ảnh
# ---------------------------------------------------------------------------
def draw_results(image_bgr: np.ndarray, detections: list) -> np.ndarray:
    """
    Vẽ bounding box + nhãn + confidence lên ảnh.
    
    Mỗi detection được vẽ:
      - Hình chữ nhật (rectangle) bao quanh đối tượng
      - Nền nhãn (filled rectangle) phía trên bounding box
      - Chữ nhãn: "nemo 0.95" hoặc "dory 0.88"
    
    Args:
        image_bgr:  Ảnh gốc (BGR)
        detections: List các (label, score, (x1, y1, x2, y2))
    Returns:
        Ảnh mới đã vẽ kết quả (không thay đổi ảnh gốc)
    """
    img = image_bgr.copy()  # Copy để không sửa ảnh gốc
    for label, score, (x1, y1, x2, y2) in detections:
        # Chọn màu theo loại cá (cam cho Nemo, xanh cho Dory)
        color = COLORS.get(label, (0, 255, 0))  # Mặc định xanh lá nếu lớp lạ
        
        # Vẽ bounding box (hình chữ nhật viền, độ dày 2px)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Tạo text hiển thị: "nemo 0.95"
        text = f"{label} {score:.2f}"
        # Tính kích thước text để vẽ nền phía sau
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        # Vẽ nền nhãn (filled rectangle) ngay phía trên bounding box
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        # Vẽ text nhãn lên nền (chữ trắng)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def draw_time_on_image(image_bgr: np.ndarray, elapsed_ms: float,
                       model_name: str) -> np.ndarray:
    """
    Vẽ thông tin thời gian xử lý lên góc trên-trái ảnh.
    Hiển thị dạng: "yolov8 | 45.2 ms"
    
    Args:
        image_bgr:  Ảnh đã vẽ bounding box
        elapsed_ms: Thời gian inference (mili giây)
        model_name: Tên mô hình (hiển thị trên ảnh)
    Returns:
        Ảnh mới có thêm thông tin thời gian
    """
    img = image_bgr.copy()
    text = f"{model_name} | {elapsed_ms:.1f} ms"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    # Vẽ nền đen cho text (dễ đọc trên ảnh bất kỳ)
    cv2.rectangle(img, (5, 5), (tw + 14, th + 16), (0, 0, 0), -1)
    # Vẽ text màu vàng (0, 255, 255 = cyan/vàng trong BGR)
    cv2.putText(img, text, (10, th + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)
    return img

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def get_image_paths(path: str) -> list:
    """Trả về danh sách ảnh từ file hoặc thư mục."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        return sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    else:
        print(f"[LỖI] Không tìm thấy: {path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Find Marine Friends - Nhận dạng sinh vật biển bằng Faster R-CNN / YOLOv5 / YOLOv8")
    parser.add_argument("--image", "-i", required=True,
                        help="Đường dẫn ảnh hoặc thư mục chứa ảnh")
    parser.add_argument("--model", "-m", default="yolov8",
                        choices=["fasterrcnn", "yolov5", "yolov8"],
                        help="Chọn mô hình (mặc định: yolov8)")
    parser.add_argument("--weights", "-w", default=None,
                        help="Đường dẫn tới file trọng số (tùy chọn, mặc định dùng best)")
    parser.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF,
                        help=f"Ngưỡng confidence (mặc định: {DEFAULT_CONF})")
    parser.add_argument("--device", "-d", default="",
                        help="Device: cpu hoặc 0,1,... cho GPU (mặc định: tự động)")
    parser.add_argument("--output", "-o", default="output",
                        help="Thư mục lưu kết quả (mặc định: output/)")
    parser.add_argument("--show", action="store_true",
                        help="Hiển thị ảnh kết quả (nhấn phím bất kỳ để tiếp)")
    parser.add_argument("--compare", action="store_true",
                        help="Chạy cả 3 mô hình trên cùng ảnh và so sánh")
    args = parser.parse_args()

    # --- Xác định device ---
    if args.device:
        device = args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Device: {device}")

    # --- Xác định đường dẫn trọng số ---
    weight_map = {
        "fasterrcnn": DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth",
        "yolov5":     DATA_DIR / "yolo5_results" / "weights" / "best.pt",
        "yolov8":     DATA_DIR / "yolo8_results" / "weights" / "best.pt",
    }

    if not args.compare:
        weights = Path(args.weights) if args.weights else weight_map[args.model]
        if not weights.exists():
            print(f"[LỖI] Không tìm thấy trọng số: {weights}")
            sys.exit(1)
        print(f"[INFO] Mô hình: {args.model}  |  Trọng số: {weights}")

    # --- Danh sách mô hình cần chạy ---
    if args.compare:
        model_list = ["fasterrcnn", "yolov5", "yolov8"]
    else:
        model_list = [args.model]

    # --- Tạo thư mục output ---
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Chạy từng mô hình ---
    for current_model in model_list:
        print(f"\n{'#'*60}")
        print(f"  MÔ HÌNH: {current_model.upper()}")
        print(f"{'#'*60}")

        # Xác định trọng số
        cur_weights = Path(args.weights) if args.weights and not args.compare else weight_map[current_model]
        if not cur_weights.exists():
            print(f"  [LỖI] Không tìm thấy trọng số: {cur_weights}")
            continue
        print(f"  Trọng số: {cur_weights}")

        # --- Load mô hình (đo thời gian) ---
        t_load_start = time.perf_counter()
        if current_model == "fasterrcnn":
            model = load_fasterrcnn(str(cur_weights), num_classes=9, device=device)
            predict_fn = lambda img, m=model: predict_fasterrcnn(m, img, device, args.conf)
        elif current_model == "yolov5":
            model = load_yolov5(str(cur_weights), device=device)
            predict_fn = lambda img, m=model: predict_yolov5(m, img, args.conf)
        else:  # yolov8
            model = load_yolov8(str(cur_weights), device=device)
            predict_fn = lambda img, m=model: predict_yolov8(m, img, device, args.conf)
        t_load = (time.perf_counter() - t_load_start) * 1000
        print(f"  ⏱️  Thời gian load mô hình: {t_load:.1f} ms ({t_load/1000:.2f} s)")

        # --- Hiển thị metrics huấn luyện ---
        print_training_metrics(current_model)

        # --- Chạy nhận dạng ---
        image_paths = get_image_paths(args.image)
        print(f"  [INFO] Tìm thấy {len(image_paths)} ảnh. Bắt đầu nhận dạng...\n")

        all_times_ms = []          # thời gian inference mỗi ảnh
        all_confidences = []       # tất cả confidence scores
        total_detections = 0

        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                print(f"  [CẢNH BÁO] Không đọc được ảnh: {img_path}")
                continue

            # --- Đo thời gian inference ---
            t_start = time.perf_counter()
            detections = predict_fn(img)
            t_infer = (time.perf_counter() - t_start) * 1000  # ms
            all_times_ms.append(t_infer)

            # --- Thống kê ---
            total_detections += len(detections)
            all_confidences.extend([s for _, s, _ in detections])

            # --- In kết quả ---
            print(f"  📷 {img_path.name}  ⏱️ {t_infer:.1f} ms:")
            if not detections:
                print("      Không phát hiện đối tượng nào.")
            for label, score, (x1, y1, x2, y2) in detections:
                print(f"      ✅ {label:>5s}  conf={score:.3f}  box=[{x1},{y1},{x2},{y2}]")

            # --- Vẽ và lưu ảnh ---
            img_out = draw_results(img, detections)
            img_out = draw_time_on_image(img_out, t_infer, current_model)
            save_path = out_dir / f"{current_model}_{img_path.name}"
            cv2.imwrite(str(save_path), img_out)

            if args.show:
                cv2.imshow(f"Detect - {current_model}", img_out)
                key = cv2.waitKey(0) & 0xFF
                if key == 27:  # ESC để thoát
                    break

        # --- Tổng kết cho mô hình này ---
        if all_times_ms:
            avg_time = sum(all_times_ms) / len(all_times_ms)
            total_time = sum(all_times_ms)
            avg_conf = sum(all_confidences) / len(all_confidences) if all_confidences else 0
            fps = 1000.0 / avg_time if avg_time > 0 else 0

            print(f"\n  {'─'*50}")
            print(f"  📊 TỔNG KẾT - {current_model.upper()}")
            print(f"  {'─'*50}")
            print(f"  Tổng ảnh xử lý       :  {len(all_times_ms)}")
            print(f"  Tổng đối tượng        :  {total_detections}")
            print(f"  Thời gian load model  :  {t_load:.1f} ms")
            print(f"  Tổng thời gian infer  :  {total_time:.1f} ms ({total_time/1000:.2f} s)")
            print(f"  TB thời gian/ảnh      :  {avg_time:.1f} ms")
            print(f"  Tốc độ (FPS)          :  {fps:.1f}")
            print(f"  Thời gian nhanh nhất  :  {min(all_times_ms):.1f} ms")
            print(f"  Thời gian chậm nhất   :  {max(all_times_ms):.1f} ms")
            if all_confidences:
                print(f"  TB confidence         :  {avg_conf:.4f} ({avg_conf*100:.1f}%)")
                print(f"  Confidence cao nhất   :  {max(all_confidences):.4f}")
                print(f"  Confidence thấp nhất  :  {min(all_confidences):.4f}")
            print(f"  {'─'*50}")

    cv2.destroyAllWindows()
    print(f"\n[INFO] Hoàn tất! Kết quả được lưu tại: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
