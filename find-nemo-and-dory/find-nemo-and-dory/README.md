# 🐠 Find Marine Friends – Nhận dạng sinh vật biển

Dự án sử dụng **3 mô hình Deep Learning** (Faster R-CNN, YOLOv5, YOLOv8) để phát hiện và nhận dạng **7 loài sinh vật biển** trong ảnh và video.

### 🎯 Các lớp nhận dạng

| #   | Lớp                         | Mô tả                    |
| --- | --------------------------- | ------------------------ |
| 1   | **Bat Sea Star**            | Sao biển dơi             |
| 2   | **Blue Sea Star**           | Sao biển xanh            |
| 3   | **Crown Of Thorn Starfish** | Sao biển gai             |
| 4   | **Dory**                    | Cá đuôi vàng (Blue Tang) |
| 5   | **Nemo**                    | Cá hề (Clownfish)        |
| 6   | **Red Cushion Sea Star**    | Sao biển đệm đỏ          |
| 7   | **Royal Starfish**          | Sao biển hoàng gia       |

---

## 📁 Cấu trúc dự án

```
find-nemo-and-dory/
├── data/
│   ├── fasterrcnn_runs/
│   │   ├── fasterrcnn_best.pth      # Trọng số Faster R-CNN
│   │   └── fasterrcnn_last.pth
│   ├── yolo5_results/
│   │   └── weights/
│   │       ├── best.pt               # Trọng số YOLOv5
│   │       └── last.pt
│   └── yolo8_results/
│       └── weights/
│           ├── best.pt               # Trọng số YOLOv8
│           └── last.pt
├── templates/
│   └── index.html                    # Giao diện web
├── static/                           # CSS, JS, ảnh tĩnh
├── nemo_dory_example/                # (tùy chọn) ảnh/video mẫu thử — cục bộ, không đẩy Git
├── detect.py                         # Script nhận dạng (dòng lệnh)
├── app.py                            # Web server (Flask)
├── setup_gpu.py                      # Tự detect GPU & cài PyTorch CUDA
├── requirements.txt                  # Danh sách thư viện
├── GUIDE.md                          # Hướng dẫn xây dựng từ đầu
└── README.md
```

---

## ⚙️ Yêu cầu hệ thống

| Thành phần      | Phiên bản khuyến nghị       |
| --------------- | --------------------------- |
| Python          | **3.9 – 3.12**              |
| CUDA (tùy chọn) | 11.8+ (nếu dùng GPU NVIDIA) |
| RAM             | ≥ 8 GB                      |

---

## 🚀 Cài đặt

### 1. Tạo môi trường ảo

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux / macOS
source venv/bin/activate
```

### 2. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3. (Tùy chọn) Cài PyTorch hỗ trợ GPU

Dùng script tự động detect GPU và cài đúng phiên bản:

```bash
# Tự detect GPU → chọn CUDA phù hợp → cài đặt
python setup_gpu.py

# Chỉ kiểm tra, không cài
python setup_gpu.py --check
```

Hoặc cài thủ công:

```bash
# CUDA 11.8
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu121

# CUDA 12.8
pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

> **Lưu ý:** Dùng `--force-reinstall` nếu đã cài bản CPU trước đó, vì pip sẽ không tự thay thế.

Xem thêm: [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)

---

## 🎯 Sử dụng dòng lệnh

### Cú pháp

```bash
python detect.py --image <ảnh/thư_mục> --model <tên_mô_hình> [tuỳ chọn]
```

### Tham số

| Tham số     | Viết tắt | Mô tả                                      | Mặc định     |
| ----------- | -------- | ------------------------------------------ | ------------ |
| `--image`   | `-i`     | Đường dẫn tới ảnh hoặc thư mục chứa ảnh    | _(bắt buộc)_ |
| `--model`   | `-m`     | Mô hình: `fasterrcnn`, `yolov5`, `yolov8`  | `yolov8`     |
| `--weights` | `-w`     | Đường dẫn file trọng số (ghi đè mặc định)  | best weights |
| `--conf`    | `-c`     | Ngưỡng confidence (0.0 – 1.0)              | `0.5`        |
| `--device`  | `-d`     | Device: `cpu` hoặc `0` (GPU)               | tự động      |
| `--output`  | `-o`     | Thư mục lưu kết quả                        | `output/`    |
| `--show`    |          | Hiển thị ảnh kết quả (ESC để thoát)        | tắt          |
| `--compare` |          | Chạy cả 3 mô hình trên cùng ảnh và so sánh | tắt          |

### Ví dụ

```bash
# Nhận dạng 1 ảnh bằng YOLOv8
python detect.py --image test.jpg --model yolov8

# Nhận dạng cả thư mục, ngưỡng 0.3
python detect.py --image ./test_images/ --model yolov5 --conf 0.3

# Dùng GPU
python detect.py --image test.jpg --model yolov8 --device 0

# So sánh cả 3 mô hình
python detect.py --image test.jpg --compare
```

---

## 📊 Kết quả dòng lệnh

Ảnh kết quả (có bounding box) được lưu trong `output/`. Terminal hiển thị chi tiết:

```
  📷 reef.jpg  ⏱️ 45.2 ms:
      ✅  nemo  conf=0.921  box=[120,85,340,290]
      ✅  dory  conf=0.874  box=[400,100,600,320]
      ✅  royal starfish  conf=0.812  box=[50,200,180,350]
```

Bảng tổng kết sau khi xử lý xong:

```
  ──────────────────────────────────────────────────
  📊 TỔNG KẾT - YOLOV8
  ──────────────────────────────────────────────────
  Tổng ảnh xử lý       :  10
  Tổng đối tượng        :  15
  Thời gian load model  :  320.5 ms
  TB thời gian/ảnh      :  45.2 ms
  Tốc độ (FPS)          :  22.1
  TB confidence         :  0.8742 (87.4%)
  ──────────────────────────────────────────────────
```

---

## 🌐 Giao diện Web

### Chạy web server

```bash
python app.py
```

Mở trình duyệt tại: **http://localhost:5000**

### Tính năng

| Tính năng                | Mô tả                                               |
| ------------------------ | --------------------------------------------------- |
| 🖼️ **Ảnh đơn**           | Upload 1 ảnh để nhận dạng (kéo thả hoặc click)      |
| 📂 **Thư mục**           | Upload cả thư mục ảnh, duyệt kết quả từng ảnh       |
| 🎬 **Video**             | Upload video MP4/AVI/MOV, detect từng frame         |
| 🤖 **Chọn model**        | YOLOv8, YOLOv5, Faster R-CNN, hoặc so sánh cả 3     |
| 🎯 **Confidence slider** | Điều chỉnh ngưỡng 0.10 – 0.95                       |
| ⏭️ **Skip frames**       | Bỏ qua frame để tăng tốc xử lý video (1–10)         |
| 📊 **Progress bar**      | Hiển thị %, frame, FPS, ETA khi xử lý video (SSE)   |
| ⏱️ **Đồng hồ đếm**       | Thời gian đã chạy realtime                          |
| 💾 **Cache model**       | Model cache sau lần load đầu, lần sau nhanh hơn     |
| 📊 **Training metrics**  | Bảng so sánh Precision, Recall, mAP các model       |
| 🔄 **Chuyển CPU/GPU**    | Toggle chuyển đổi CPU ↔ GPU realtime trên giao diện |

### Video Detection

Khi detect video, hệ thống sẽ:

1. Upload video lên server
2. Xử lý từng frame với **progress bar realtime** (%, Frame X/Y, FPS, ETA)
3. Re-encode kết quả sang **H.264** để browser play được
4. Hiển thị video kết quả cùng thống kê chi tiết

**Tùy chọn Skip Frames:** Đặt giá trị 2–5 để tăng tốc (chỉ detect 1 trong N frame, dùng kết quả frame trước cho các frame bỏ qua).

---

## 📝 Thông tin mô hình

| Mô hình      | Base model   | Input size | Số lớp |
| ------------ | ------------ | ---------- | ------ |
| Faster R-CNN | ResNet50-FPN | tùy ý      | 7 + bg |
| YOLOv5       | YOLOv5       | 640×640    | 7      |
| YOLOv8       | YOLOv8       | 640×640    | 7      |

**7 lớp:** Bat Sea Star, Blue Sea Star, Crown Of Thorn Starfish, Dory, Nemo, Red Cushion Sea Star, Royal Starfish

### ⚡ Benchmark CPU vs GPU

Đo trên ảnh 480×640, `imgsz=416` (RTX 3080 Ti):

| Mô hình      | CPU               | GPU (CUDA)         | Tăng tốc |
| ------------ | ----------------- | ------------------ | -------- |
| YOLOv8       | 102 ms (10 FPS)   | **11 ms (91 FPS)** | ~9×      |
| YOLOv5       | 63 ms (16 FPS)    | **12 ms (87 FPS)** | ~5×      |
| Faster R-CNN | 1506 ms (0.7 FPS) | **38 ms (26 FPS)** | ~39×     |

> **Kết luận:** GPU tăng tốc 5–39 lần. YOLOv8 và YOLOv5 đạt real-time (>30 FPS) trên GPU.
> Trên CPU, YOLOv5/v8 vẫn chạy tốt (~10–16 FPS). Faster R-CNN cần GPU để đạt hiệu suất tốt.

---

## ❓ Xử lý sự cố

| Vấn đề                              | Giải pháp                                          |
| ----------------------------------- | -------------------------------------------------- |
| `No module named 'ultralytics'`     | `pip install ultralytics`                          |
| `No module named 'cv2'`             | `pip install opencv-python`                        |
| `No module named 'seaborn'`         | `pip install seaborn`                              |
| Video kết quả không play được       | Kiểm tra đã cài `imageio-ffmpeg`                   |
| CUDA out of memory                  | Dùng `--device cpu` hoặc giảm kích thước ảnh       |
| Có GPU nhưng PyTorch không nhận     | Chạy `python setup_gpu.py` để cài đúng CUDA        |
| `torch.cuda.is_available()` = False | PyTorch đang là bản CPU, cài lại bản CUDA          |
| Không phát hiện được gì             | Giảm `--conf` xuống (ví dụ: 0.25)                  |
| YOLOv5 lần đầu chạy chậm            | Cần internet để tải code từ GitHub qua `torch.hub` |

---

## 🔄 Chuyển sang máy khác

```bash
# 1. Copy toàn bộ thư mục (bao gồm data/ với file trọng số)
# 2. Tạo lại môi trường ảo
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt

# 3. (Tùy chọn) Cài GPU
python setup_gpu.py

# 4. Chạy
python app.py
```

**Lưu ý:**

- **Không copy** thư mục `venv/` và `__pycache__/` — tạo lại trên máy mới
- **Bắt buộc có** thư mục `data/` với đầy đủ file trọng số (.pth, .pt)
- Lần đầu chạy YOLOv5 cần **kết nối internet**
- Nếu máy mới có GPU, chạy `python setup_gpu.py` để tự cài PyTorch CUDA
