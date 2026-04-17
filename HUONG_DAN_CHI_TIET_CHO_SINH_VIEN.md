# 🎓 HƯỚNG DẪN CHI TIẾT CHO SINH VIÊN
## Hoàn thành Đồ án Tốt nghiệp: Hệ thống Nhận dạng Sinh vật Biển

> **Dành cho:** Sinh viên K64 CNTT - Đại học Thủy Lợi  
> **Mục tiêu:** Hướng dẫn từng bước để hoàn thành ĐATN ngay cả khi chưa biết nhiều về AI/ML  
> **Thời gian:** 8-10 tuần (Tháng 4 - Tháng 6/2026)

---

## 📋 MỤC LỤC

- [PHẦN 1: CHUẨN BỊ (Tuần 1)](#phần-1-chuẩn-bị-tuần-1)
- [PHẦN 2: HỌC LÝ THUYẾT CƠ BẢN (Tuần 2)](#phần-2-học-lý-thuyết-cơ-bản-tuần-2)
- [PHẦN 3: CHUẨN BỊ DỮ LIỆU (Tuần 3)](#phần-3-chuẩn-bị-dữ-liệu-tuần-3)
- [PHẦN 4: HUẤN LUYỆN MÔ HÌNH (Tuần 4-5)](#phần-4-huấn-luyện-mô-hình-tuần-4-5)
- [PHẦN 5: XÂY DỰNG ỨNG DỤNG (Tuần 6-7)](#phần-5-xây-dựng-ứng-dụng-tuần-6-7)
- [PHẦN 6: VIẾT BÁO CÁO (Tuần 8-9)](#phần-6-viết-báo-cáo-tuần-8-9)
- [PHẦN 7: CHUẨN BỊ BỔ VỆ (Tuần 10)](#phần-7-chuẩn-bị-bảo-vệ-tuần-10)

---

## 🎯 TỔNG QUAN ROADMAP

```
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 1: Chuẩn bị                                           │
│  ✓ Cài đặt môi trường                                       │
│  ✓ Hiểu tổng quan về đồ án                                  │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 2: Học lý thuyết                                      │
│  ✓ Hiểu Object Detection là gì                              │
│  ✓ Hiểu 3 mô hình: Faster R-CNN, YOLOv5, YOLOv8            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 3: Chuẩn bị dữ liệu                                   │
│  ✓ Tải dataset từ Roboflow                                  │
│  ✓ Kiểm tra và thống kê dữ liệu                             │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 4-5: Huấn luyện mô hình                               │
│  ✓ Train YOLOv8 trên Google Colab                           │
│  ✓ Train YOLOv5 trên Google Colab                           │
│  ✓ Train Faster R-CNN trên Google Colab                     │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 6-7: Xây dựng ứng dụng                                │
│  ✓ Tạo script detect.py                                     │
│  ✓ Tạo web app với Flask                                    │
│  ✓ Test và debug                                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 8-9: Viết báo cáo                                     │
│  ✓ Viết 5 chương theo template                              │
│  ✓ Chèn hình ảnh, bảng biểu                                 │
│  ✓ Hoàn thiện tài liệu tham khảo                            │
└─────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────┐
│  TUẦN 10: Chuẩn bị bảo vệ                                   │
│  ✓ Làm slide PowerPoint                                     │
│  ✓ Chuẩn bị demo                                            │
│  ✓ Luyện tập thuyết trình                                   │
└─────────────────────────────────────────────────────────────┘
```

---

## PHẦN 1: CHUẨN BỊ (Tuần 1)

### 📅 Mục tiêu tuần 1
- ✅ Cài đặt đầy đủ môi trường làm việc
- ✅ Hiểu tổng quan về đồ án
- ✅ Có thể chạy được code mẫu

### Bước 1.1: Cài đặt Python và các công cụ cơ bản

#### 🪟 Trên Windows:

**1. Cài Python:**
```bash
# Tải Python 3.10 từ: https://www.python.org/downloads/
# ⚠️ QUAN TRỌNG: Tick vào "Add Python to PATH" khi cài đặt

# Kiểm tra cài đặt thành công:
python --version
# Kết quả mong đợi: Python 3.10.x
```

**2. Cài Git:**
```bash
# Tải Git từ: https://git-scm.com/download/win
# Cài đặt với các tùy chọn mặc định

# Kiểm tra:
git --version
```

**3. Cài Visual Studio Code (khuyến nghị):**
```bash
# Tải từ: https://code.visualstudio.com/
# Extensions nên cài:
# - Python (Microsoft)
# - Jupyter (Microsoft)
# - GitLens
```

#### 🐧 Trên Linux/Mac:

```bash
# Ubuntu/Debian:
sudo apt update
sudo apt install python3.10 python3-pip git

# macOS (dùng Homebrew):
brew install python@3.10 git

# Kiểm tra:
python3 --version
git --version
```

### Bước 1.2: Tạo thư mục dự án

```bash
# Mở Terminal (hoặc Command Prompt trên Windows)

# Tạo thư mục chính
mkdir DATN_NhanDangSinhVatBien
cd DATN_NhanDangSinhVatBien

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
mkdir notebooks

# Kiểm tra cấu trúc:
# Windows:
tree /F

# Linux/Mac:
tree
```

**Cấu trúc mong đợi:**
```
DATN_NhanDangSinhVatBien/
├── data/
│   ├── fasterrcnn_runs/
│   ├── yolo5_results/
│   │   └── weights/
│   └── yolo8_results/
│       └── weights/
├── templates/
├── static/
├── output/
└── notebooks/
```

### Bước 1.3: Tạo Virtual Environment

**Tại sao cần Virtual Environment?**
- Tránh xung đột giữa các thư viện
- Dễ dàng quản lý dependencies
- Có thể tạo lại môi trường trên máy khác

```bash
# Tạo virtual environment
python -m venv venv

# Kích hoạt:
# Windows:
venv\Scripts\activate

# Linux/Mac:
source venv/bin/activate

# Sau khi kích hoạt, bạn sẽ thấy (venv) ở đầu dòng:
# (venv) C:\Users\YourName\DATN_NhanDangSinhVatBien>
```

### Bước 1.4: Cài đặt thư viện cơ bản

**Tạo file requirements.txt:**

```bash
# Tạo file requirements.txt với nội dung sau:
```

```txt
# PyTorch & TorchVision (CPU version - sẽ upgrade GPU sau)
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

# Web server
flask>=3.0.0

# Jupyter (để chạy notebook)
jupyter>=1.0.0
ipykernel>=6.25.0
```

**Cài đặt:**

```bash
# Nâng cấp pip
python -m pip install --upgrade pip

# Cài đặt tất cả thư viện
pip install -r requirements.txt

# ⏳ Quá trình này mất 5-10 phút
# Đợi cho đến khi thấy "Successfully installed..."
```

**Kiểm tra cài đặt:**

```bash
# Test Python
python -c "import torch; print('PyTorch:', torch.__version__)"
python -c "import cv2; print('OpenCV:', cv2.__version__)"
python -c "import ultralytics; print('Ultralytics OK')"

# Nếu tất cả đều in ra version → Thành công! ✅
```

### Bước 1.5: Đăng ký Google Colab

**Tại sao cần Google Colab?**
- Miễn phí GPU để train mô hình
- Không cần máy tính mạnh
- Có sẵn môi trường Python

**Các bước:**

1. Truy cập: https://colab.research.google.com/
2. Đăng nhập bằng Gmail (dùng Gmail trường hoặc cá nhân)
3. Tạo notebook mới: File → New notebook
4. Test GPU:

```python
# Chạy cell này trong Colab:
import torch
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Kết quả mong đợi:
# CUDA available: True
# GPU: Tesla T4 (hoặc V100, P100)
```

5. Kết nối Google Drive:

```python
from google.colab import drive
drive.mount('/content/drive')

# Click vào link → Chọn tài khoản → Copy code → Paste vào ô
# Thấy "Mounted at /content/drive" → Thành công!
```

### Bước 1.6: Hiểu tổng quan về đồ án

**Đọc và hiểu các file sau:**

1. **README.md** - Tổng quan dự án
2. **GUIDE.md** - Hướng dẫn xây dựng từ đầu
3. **DATN_NoiDung_MVP.md** - Nội dung báo cáo

**Câu hỏi tự kiểm tra:**
- [ ] Đồ án này làm gì? (Nhận dạng sinh vật biển)
- [ ] Có bao nhiêu mô hình? (3: Faster R-CNN, YOLOv5, YOLOv8)
- [ ] Nhận dạng bao nhiêu loài? (7 loài)
- [ ] Kết quả cuối cùng là gì? (Web app + Báo cáo)

### ✅ Checklist Tuần 1

- [ ] Cài đặt Python 3.10
- [ ] Cài đặt Git và VS Code
- [ ] Tạo thư mục dự án với cấu trúc đúng
- [ ] Tạo và kích hoạt virtual environment
- [ ] Cài đặt thành công tất cả thư viện
- [ ] Đăng ký và test Google Colab
- [ ] Đọc hiểu tổng quan về đồ án

**🎉 Nếu hoàn thành tất cả → Chuyển sang Tuần 2!**

---

## PHẦN 2: HỌC LÝ THUYẾT CƠ BẢN (Tuần 2)

### 📅 Mục tiêu tuần 2
- ✅ Hiểu Object Detection là gì
- ✅ Hiểu sự khác biệt giữa 3 mô hình
- ✅ Hiểu các metrics đánh giá (mAP, Precision, Recall)

### Bước 2.1: Hiểu Object Detection

#### 🤔 Object Detection là gì?

**Định nghĩa đơn giản:**
> Object Detection = Tìm vị trí + Nhận dạng đối tượng trong ảnh

**Ví dụ thực tế:**
```
Input: Ảnh một bể cá
Output: 
  - Nemo ở vị trí (120, 85, 340, 290) với độ tin cậy 92%
  - Dory ở vị trí (400, 100, 600, 320) với độ tin cậy 87%
```

**So sánh với các bài toán khác:**

| Bài toán | Input | Output | Ví dụ |
|----------|-------|--------|-------|
| **Image Classification** | Ảnh | Nhãn | "Đây là ảnh con mèo" |
| **Object Detection** | Ảnh | Vị trí + Nhãn | "Con mèo ở góc trái trên" |
| **Segmentation** | Ảnh | Mask từng pixel | "Tô màu chính xác con mèo" |

#### 📦 Bounding Box là gì?

**Định nghĩa:**
- Hình chữ nhật bao quanh đối tượng
- Được biểu diễn bằng 4 số: (x1, y1, x2, y2) hoặc (x, y, width, height)

**Ví dụ:**
```
Ảnh kích thước: 640×480
Nemo ở vị trí: (100, 50, 250, 200)
  → x1=100, y1=50 (góc trên trái)
  → x2=250, y2=200 (góc dưới phải)
  → width = 250-100 = 150
  → height = 200-50 = 150
```

#### 🎯 Confidence Score là gì?

**Định nghĩa:**
- Độ tin cậy của dự đoán (0-1 hoặc 0-100%)
- Càng cao → Mô hình càng chắc chắn

**Ví dụ:**
```
Nemo: 0.92 (92%) → Rất chắc chắn
Dory: 0.65 (65%) → Khá chắc chắn
Star: 0.45 (45%) → Không chắc lắm (có thể loại bỏ)
```

### Bước 2.2: Hiểu 3 mô hình

#### 🐢 Faster R-CNN (Chậm nhưng chính xác)

**Đặc điểm:**
- **Two-stage detector** (2 giai đoạn)
- Giai đoạn 1: Đề xuất vùng có thể chứa đối tượng (~2000 vùng)
- Giai đoạn 2: Phân loại từng vùng

**Ưu điểm:**
- ✅ Độ chính xác cao nhất
- ✅ Tốt với vật thể nhỏ

**Nhược điểm:**
- ❌ Chậm (~10 FPS)
- ❌ Phức tạp, khó triển khai

**Khi nào dùng:**
- Khi cần độ chính xác tuyệt đối
- Không quan tâm tốc độ
- Ví dụ: Chẩn đoán y tế, kiểm tra chất lượng sản phẩm

#### 🐇 YOLOv5 (Nhanh và cân bằng)

**Đặc điểm:**
- **One-stage detector** (1 giai đoạn)
- Dự đoán trực tiếp bounding box và class

**Ưu điểm:**
- ✅ Nhanh (~80 FPS)
- ✅ Dễ sử dụng
- ✅ Cân bằng tốc độ/độ chính xác

**Nhược điểm:**
- ❌ Độ chính xác thấp hơn Faster R-CNN
- ❌ Kém với vật thể nhỏ

**Khi nào dùng:**
- Ứng dụng real-time
- Cần cân bằng tốc độ và độ chính xác
- Ví dụ: Camera giám sát, xe tự lái

#### 🚀 YOLOv8 (Mới nhất, tốt nhất)

**Đặc điểm:**
- Phiên bản mới nhất của YOLO (2023)
- Cải tiến nhiều so với YOLOv5

**Ưu điểm:**
- ✅ Nhanh hơn YOLOv5 (~90 FPS)
- ✅ Chính xác hơn YOLOv5
- ✅ Dễ sử dụng nhất

**Nhược điểm:**
- ❌ Mới, ít tài liệu hơn YOLOv5

**Khi nào dùng:**
- Dự án mới, muốn dùng công nghệ mới nhất
- Cần cả tốc độ và độ chính xác

#### 📊 So sánh 3 mô hình

| Tiêu chí | Faster R-CNN | YOLOv5 | YOLOv8 |
|----------|--------------|---------|---------|
| **Tốc độ** | 🐢 Chậm (10 FPS) | 🐇 Nhanh (80 FPS) | 🚀 Rất nhanh (90 FPS) |
| **Độ chính xác** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Dễ sử dụng** | ❌ Khó | ✅ Dễ | ✅ Rất dễ |
| **Khi nào dùng** | Cần chính xác tuyệt đối | Cần cân bằng | Dự án mới |

### Bước 2.3: Hiểu các Metrics đánh giá

#### 🎯 Precision (Độ chính xác)

**Định nghĩa:**
> Trong số những gì mô hình dự đoán là "Nemo", có bao nhiêu % thực sự là Nemo?

**Công thức:**
```
Precision = TP / (TP + FP)

TP (True Positive): Dự đoán đúng
FP (False Positive): Dự đoán sai (nhầm Dory thành Nemo)
```

**Ví dụ:**
```
Mô hình dự đoán 100 con là "Nemo"
Trong đó:
  - 90 con thực sự là Nemo (TP = 90)
  - 10 con thực ra là Dory (FP = 10)

Precision = 90 / (90 + 10) = 0.9 = 90%
```

**Khi nào quan trọng:**
- Khi không muốn dự đoán sai (False Positive tốn kém)
- Ví dụ: Lọc spam email (không muốn email quan trọng bị đánh là spam)

#### 🔍 Recall (Độ phủ)

**Định nghĩa:**
> Trong số tất cả con Nemo thực tế, mô hình phát hiện được bao nhiêu %?

**Công thức:**
```
Recall = TP / (TP + FN)

TP (True Positive): Dự đoán đúng
FN (False Negative): Bỏ sót (có Nemo nhưng không phát hiện)
```

**Ví dụ:**
```
Trong ảnh có 100 con Nemo thực tế
Mô hình phát hiện được:
  - 85 con (TP = 85)
  - Bỏ sót 15 con (FN = 15)

Recall = 85 / (85 + 15) = 0.85 = 85%
```

**Khi nào quan trọng:**
- Khi không muốn bỏ sót (False Negative nguy hiểm)
- Ví dụ: Phát hiện bệnh ung thư (không được bỏ sót bệnh nhân)

#### ⚖️ Trade-off giữa Precision và Recall

**Mối quan hệ:**
- Tăng Precision → Giảm Recall
- Tăng Recall → Giảm Precision

**Ví dụ:**
```
Confidence threshold = 0.9 (cao)
  → Chỉ dự đoán khi rất chắc chắn
  → Precision cao (ít sai)
  → Recall thấp (bỏ sót nhiều)

Confidence threshold = 0.3 (thấp)
  → Dự đoán ngay cả khi không chắc
  → Precision thấp (nhiều sai)
  → Recall cao (ít bỏ sót)
```

#### 📈 mAP (mean Average Precision)

**Định nghĩa:**
> Metric tổng hợp, cân bằng giữa Precision và Recall

**Cách tính (đơn giản hóa):**
1. Vẽ đường Precision-Recall curve
2. Tính diện tích dưới đường cong (AP)
3. Lấy trung bình AP của tất cả classes (mAP)

**Các biến thể:**
- **mAP@0.5:** Tính tại IoU threshold = 0.5 (dễ hơn)
- **mAP@0.5:0.95:** Trung bình từ IoU 0.5 đến 0.95 (khắt khe hơn)

**Ví dụ:**
```
mAP@0.5 = 0.912 (91.2%)
  → Mô hình rất tốt!

mAP@0.5:0.95 = 0.806 (80.6%)
  → Vẫn tốt, nhưng kém hơn khi yêu cầu khắt khe
```

#### ⚡ FPS (Frames Per Second)

**Định nghĩa:**
> Số khung hình xử lý được trong 1 giây

**Phân loại:**
- FPS < 10: Không real-time
- 10 ≤ FPS < 30: Gần real-time
- FPS ≥ 30: Real-time (mượt mà)

**Ví dụ:**
```
YOLOv8: 90 FPS → Rất nhanh, real-time
Faster R-CNN: 10 FPS → Chậm, không real-time
```

### Bước 2.4: Xem video và đọc tài liệu

**Video khuyến nghị (YouTube):**

1. **"Object Detection in 5 Minutes"** - Sentdex
   - Link: https://www.youtube.com/watch?v=GSwYGkTfOKk
   - Thời lượng: 5 phút
   - Nội dung: Giới thiệu cơ bản về Object Detection

2. **"YOLO Object Detection Explained"** - Computerphile
   - Link: https://www.youtube.com/watch?v=9s_FpMpdYW8
   - Thời lượng: 10 phút
   - Nội dung: Giải thích YOLO hoạt động như thế nào

3. **"Faster R-CNN Explained"** - deeplizard
   - Link: https://www.youtube.com/watch?v=XGi-Mz3do2s
   - Thời lượng: 15 phút
   - Nội dung: Giải thích Faster R-CNN chi tiết

**Bài viết khuyến nghị:**

1. **"A Gentle Introduction to Object Detection"** - Machine Learning Mastery
   - Link: https://machinelearningmastery.com/object-recognition-with-deep-learning/

2. **"Understanding mAP"** - Jonathan Hui
   - Link: https://jonathan-hui.medium.com/map-mean-average-precision-for-object-detection-45c121a31173

### ✅ Checklist Tuần 2

- [ ] Hiểu Object Detection là gì
- [ ] Hiểu Bounding Box và Confidence Score
- [ ] Hiểu sự khác biệt giữa 3 mô hình
- [ ] Hiểu Precision, Recall, mAP, FPS
- [ ] Xem ít nhất 2 video về Object Detection
- [ ] Đọc ít nhất 1 bài viết về mAP

**🎓 Tự kiểm tra:**
- [ ] Có thể giải thích Object Detection cho người không biết AI
- [ ] Có thể so sánh 3 mô hình và nói ưu/nhược điểm
- [ ] Có thể tính Precision và Recall từ ví dụ đơn giản

**🎉 Nếu hoàn thành → Chuyển sang Tuần 3!**

---


## PHẦN 3: CHUẨN BỊ DỮ LIỆU (Tuần 3)

### 📅 Mục tiêu tuần 3
- ✅ Tải dataset từ Roboflow
- ✅ Kiểm tra chất lượng dữ liệu
- ✅ Hiểu cấu trúc dataset YOLO

### Bước 3.1: Đăng ký và tải dataset từ Roboflow

#### 🌐 Đăng ký Roboflow

1. Truy cập: https://roboflow.com/
2. Click "Sign Up" → Đăng ký bằng Gmail
3. Chọn plan "Free" (đủ cho đồ án)

#### 🔍 Tìm dataset

**Option 1: Sử dụng dataset có sẵn**

1. Truy cập Roboflow Universe: https://universe.roboflow.com/
2. Tìm kiếm: "marine fish detection" hoặc "underwater object detection"
3. Chọn dataset có:
   - ✅ Ít nhất 1000+ ảnh
   - ✅ Có classes: nemo, dory, starfish
   - ✅ Đã được annotate (gán nhãn)

**Option 2: Tạo dataset riêng (nếu muốn)**

1. Tạo project mới trên Roboflow
2. Upload ảnh từ Google Images hoặc Kaggle
3. Annotate (gán nhãn) bằng công cụ của Roboflow
   - ⚠️ Mất nhiều thời gian (5-10 giây/ảnh)
   - Cần ít nhất 500-1000 ảnh

**💡 Khuyến nghị:** Dùng Option 1 để tiết kiệm thời gian

#### 📥 Export dataset

1. Chọn dataset → Click "Download"
2. Chọn format: **YOLOv8**
3. Chọn split:
   - Train: 70%
   - Valid: 20%
   - Test: 10%
4. Click "Download ZIP"
5. Giải nén vào thư mục `DATN_NhanDangSinhVatBien/dataset/`

**Cấu trúc sau khi giải nén:**
```
dataset/
├── data.yaml          # File cấu hình
├── train/
│   ├── images/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── labels/
│       ├── img001.txt
│       ├── img002.txt
│       └── ...
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

### Bước 3.2: Hiểu cấu trúc dataset YOLO

#### 📄 File data.yaml

**Mở file `dataset/data.yaml`:**

```yaml
path: /path/to/dataset  # Đường dẫn gốc
train: train/images     # Thư mục ảnh train
val: valid/images       # Thư mục ảnh validation
test: test/images       # Thư mục ảnh test

nc: 7  # Số lượng classes
names:
  0: bat_sea_star
  1: blue_sea_star
  2: crown_of_thorn_starfish
  3: dory
  4: nemo
  5: red_cushion_sea_star
  6: royal_starfish
```

**Giải thích:**
- `nc`: Number of classes (số lượng loài)
- `names`: Tên các classes (0-indexed)

#### 📝 File label (.txt)

**Mở file `dataset/train/labels/img001.txt`:**

```
4 0.512 0.345 0.234 0.456
3 0.789 0.567 0.123 0.234
```

**Format:**
```
class_id x_center y_center width height
```

**Giải thích:**
- `class_id`: ID của class (4 = nemo, 3 = dory)
- `x_center, y_center`: Tọa độ tâm bounding box (normalized 0-1)
- `width, height`: Kích thước bounding box (normalized 0-1)

**Ví dụ cụ thể:**
```
Ảnh kích thước: 640×480
Label: 4 0.5 0.5 0.2 0.3

Chuyển sang pixel:
  x_center = 0.5 × 640 = 320
  y_center = 0.5 × 480 = 240
  width = 0.2 × 640 = 128
  height = 0.3 × 480 = 144

Bounding box:
  x1 = 320 - 128/2 = 256
  y1 = 240 - 144/2 = 168
  x2 = 320 + 128/2 = 384
  y2 = 240 + 144/2 = 312
```

### Bước 3.3: Kiểm tra chất lượng dataset

#### 🔧 Sử dụng script check_dataset.py

**Copy file từ project mẫu:**

```bash
# Copy file check_dataset.py vào thư mục dự án
# Hoặc tạo file mới với nội dung từ DATN_NoiDung_MVP.md
```

**Chạy script:**

```bash
# Kích hoạt virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Chạy kiểm tra
python check_dataset.py --data dataset/data.yaml --visualize --num_samples 10

# ⏳ Đợi 1-2 phút để script chạy
```

**Kết quả mong đợi:**

```
  📁 KIỂM TRA CẤU TRÚC DATASET
  ══════════════════════════════════════════════════════
  File config : dataset/data.yaml
  Số lớp      : 7
  Tên lớp     : ['bat_sea_star', 'blue_sea_star', ...]

  ✅ Split 'train':
      Số ảnh       : 840
      Số nhãn      : 840
      ✅ Ảnh và nhãn khớp hoàn toàn

  📊 THỐNG KÊ PHÂN PHỐI NHÃN
  ══════════════════════════════════════════════════════
  Split: TRAIN (tổng 1456 bounding boxes)
  Lớp                              Số BB    Tỉ lệ
  ─────────────────────────────────────────────────────
  nemo                               312   21.4%  ██████████
  dory                               289   19.8%  █████████
  ...

  Tỉ lệ chênh lệch max/min: 2.2x
  ✅ Phân phối dữ liệu tương đối cân bằng
```

#### 🖼️ Kiểm tra ảnh trực quan

**Mở file `dataset_check/label_check_train.png`:**

- Xem 10 ảnh mẫu với bounding boxes
- Kiểm tra:
  - [ ] Bounding box có khớp với đối tượng không?
  - [ ] Nhãn có đúng không? (Nemo là Nemo, không phải Dory)
  - [ ] Có ảnh nào bị lỗi không? (ảnh đen, ảnh mờ)

**Nếu phát hiện lỗi:**
1. Ghi chú lại tên file ảnh bị lỗi
2. Xóa ảnh đó khỏi dataset (cả file .jpg và .txt)
3. Chạy lại script kiểm tra

### Bước 3.4: Thống kê dataset

#### 📊 Tạo bảng thống kê

**Tạo file Excel hoặc Google Sheets:**

| Metric | Train | Valid | Test | Tổng |
|--------|-------|-------|------|------|
| Số ảnh | 840 | 240 | 120 | 1200 |
| Số BB | 1456 | 416 | 208 | 2080 |
| **Phân phối classes:** |
| nemo | 312 | 89 | 45 | 446 |
| dory | 289 | 83 | 41 | 413 |
| bat_sea_star | 198 | 57 | 28 | 283 |
| ... | ... | ... | ... | ... |

**Tính toán:**
- Tỉ lệ train/val/test: 70%/20%/10% ✅
- Tỉ lệ max/min classes: 2.2x ✅ (< 10x là OK)
- Trung bình BB/ảnh: 1456/840 = 1.73 ✅

#### 📸 Chụp screenshots

**Chụp các ảnh sau để dùng trong báo cáo:**

1. **Biểu đồ phân phối classes** (`dataset_check/class_distribution.png`)
2. **Ảnh mẫu với bounding boxes** (`dataset_check/label_check_train.png`)
3. **Bảng thống kê** (screenshot từ Excel)

**Lưu vào thư mục:** `DATN_NhanDangSinhVatBien/report_images/chapter3/`

### Bước 3.5: Upload dataset lên Google Drive

**Tại sao cần upload?**
- Google Colab không có dữ liệu local
- Cần truy cập dataset từ Colab để train

**Các bước:**

1. **Nén dataset:**
```bash
# Windows (dùng 7-Zip hoặc WinRAR):
# Click phải dataset/ → Send to → Compressed (zipped) folder

# Linux/Mac:
zip -r dataset.zip dataset/
```

2. **Upload lên Google Drive:**
   - Mở Google Drive: https://drive.google.com/
   - Tạo thư mục: `DATN_Marine_Detection`
   - Upload file `dataset.zip` vào thư mục này
   - ⏳ Đợi upload hoàn tất (5-10 phút tùy tốc độ mạng)

3. **Kiểm tra:**
   - Mở Google Colab
   - Mount Drive:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   - Kiểm tra file:
   ```python
   !ls /content/drive/MyDrive/DATN_Marine_Detection/
   # Kết quả: dataset.zip
   ```

### ✅ Checklist Tuần 3

- [ ] Đăng ký Roboflow thành công
- [ ] Tải dataset về (1000+ ảnh)
- [ ] Hiểu cấu trúc dataset YOLO (data.yaml, labels)
- [ ] Chạy script check_dataset.py thành công
- [ ] Kiểm tra trực quan ảnh và bounding boxes
- [ ] Tạo bảng thống kê dataset
- [ ] Chụp screenshots cho báo cáo
- [ ] Upload dataset lên Google Drive

**🎓 Tự kiểm tra:**
- [ ] Có thể giải thích format label YOLO
- [ ] Có thể tính toán tọa độ bounding box từ label
- [ ] Biết cách kiểm tra chất lượng dataset

**🎉 Nếu hoàn thành → Chuyển sang Tuần 4!**

---

## PHẦN 4: HUẤN LUYỆN MÔ HÌNH (Tuần 4-5)

### 📅 Mục tiêu tuần 4-5
- ✅ Train YOLOv8 trên Google Colab
- ✅ Train YOLOv5 trên Google Colab
- ✅ Train Faster R-CNN trên Google Colab
- ✅ Lưu trọng số và kết quả

### Bước 4.1: Chuẩn bị Google Colab

#### 📓 Tạo Notebook mới

1. Truy cập: https://colab.research.google.com/
2. File → New notebook
3. Đổi tên: "DATN_Train_YOLOv8.ipynb"
4. Chọn Runtime → Change runtime type → GPU (T4)

#### 🔗 Mount Google Drive

```python
# Cell 1: Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Kiểm tra
!ls /content/drive/MyDrive/DATN_Marine_Detection/
```

#### 📦 Giải nén dataset

```python
# Cell 2: Giải nén dataset
!unzip -q /content/drive/MyDrive/DATN_Marine_Detection/dataset.zip -d /content/

# Kiểm tra
!ls /content/dataset/
# Kết quả: data.yaml  train  valid  test
```

### Bước 4.2: Train YOLOv8 (Dễ nhất)

#### 📝 Notebook: Train YOLOv8

**Cell 1: Cài đặt thư viện**

```python
# Cài Ultralytics
!pip install ultralytics -q

# Import
from ultralytics import YOLO
import torch

# Kiểm tra GPU
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))
```

**Cell 2: Load pretrained model**

```python
# Load YOLOv8 pretrained trên COCO
model = YOLO('yolov8s.pt')  # s = small (11M params)

# Có thể thử các size khác:
# yolov8n.pt - nano (3M params, nhanh nhất)
# yolov8s.pt - small (11M params, cân bằng)
# yolov8m.pt - medium (26M params, chính xác hơn)
```

**Cell 3: Train**

```python
# Train model
results = model.train(
    data='/content/dataset/data.yaml',
    epochs=100,              # Số epoch
    imgsz=640,               # Kích thước ảnh
    batch=16,                # Batch size (giảm nếu out of memory)
    device=0,                # GPU
    patience=20,             # Early stopping
    save=True,               # Lưu weights
    project='runs/yolov8',   # Thư mục lưu kết quả
    name='marine_friends',   # Tên experiment
    
    # Optimizer
    optimizer='AdamW',
    lr0=0.001,               # Learning rate ban đầu
    lrf=0.01,                # Learning rate cuối
    
    # Augmentation (tự động)
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=15.0,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    mosaic=1.0,
)

# ⏳ Quá trình train mất 2-3 giờ
# Colab sẽ hiển thị progress bar và metrics
```

**Cell 4: Xem kết quả**

```python
# Xem metrics
from IPython.display import Image, display

# Loss curves
display(Image('/content/runs/yolov8/marine_friends/results.png'))

# Confusion matrix
display(Image('/content/runs/yolov8/marine_friends/confusion_matrix.png'))

# PR curve
display(Image('/content/runs/yolov8/marine_friends/PR_curve.png'))
```

**Cell 5: Test model**

```python
# Load best model
best_model = YOLO('/content/runs/yolov8/marine_friends/weights/best.pt')

# Test trên 1 ảnh
results = best_model('/content/dataset/test/images/img001.jpg')

# Hiển thị kết quả
results[0].show()

# In metrics
print(f"mAP@0.5: {results[0].boxes.map50}")
print(f"mAP@0.5:0.95: {results[0].boxes.map}")
```

**Cell 6: Lưu kết quả về Drive**

```python
# Copy toàn bộ kết quả về Drive
!cp -r /content/runs/yolov8/marine_friends /content/drive/MyDrive/DATN_Marine_Detection/yolov8_results

print("✅ Đã lưu kết quả về Google Drive!")
```

#### 📊 Đọc kết quả training

**Mở file `results.csv`:**

```python
import pandas as pd

df = pd.read_csv('/content/runs/yolov8/marine_friends/results.csv')
print(df.tail(10))  # 10 epoch cuối
```

**Các metrics quan trọng:**
- `metrics/mAP50(B)`: mAP@0.5 (càng cao càng tốt, >0.8 là tốt)
- `metrics/mAP50-95(B)`: mAP@0.5:0.95 (>0.6 là tốt)
- `metrics/precision(B)`: Precision (>0.85 là tốt)
- `metrics/recall(B)`: Recall (>0.85 là tốt)
- `train/box_loss`: Loss (càng thấp càng tốt)

**Ví dụ kết quả tốt:**
```
Epoch 95:
  mAP@0.5: 0.912
  mAP@0.5:0.95: 0.806
  Precision: 0.906
  Recall: 0.897
```

### Bước 4.3: Train YOLOv5

**Tạo notebook mới: "DATN_Train_YOLOv5.ipynb"**

**Cell 1: Clone YOLOv5 repo**

```python
# Clone YOLOv5
!git clone https://github.com/ultralytics/yolov5
%cd yolov5
!pip install -r requirements.txt -q
```

**Cell 2: Train**

```python
# Train YOLOv5
!python train.py \
  --img 640 \
  --batch 16 \
  --epochs 100 \
  --data /content/dataset/data.yaml \
  --weights yolov5s.pt \
  --device 0 \
  --project /content/runs/yolov5 \
  --name marine_friends \
  --patience 20

# ⏳ Mất 2-3 giờ
```

**Cell 3: Lưu kết quả**

```python
# Copy về Drive
!cp -r /content/runs/yolov5/marine_friends /content/drive/MyDrive/DATN_Marine_Detection/yolov5_results

print("✅ Đã lưu YOLOv5 results!")
```

### Bước 4.4: Train Faster R-CNN

**Tạo notebook mới: "DATN_Train_FasterRCNN.ipynb"**

**⚠️ Lưu ý:** Faster R-CNN phức tạp hơn, cần code nhiều hơn

**Cell 1: Cài đặt**

```python
!pip install torch torchvision -q
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
```

**Cell 2: Tạo Dataset class**

```python
import os
from PIL import Image
from torch.utils.data import Dataset

class MarineDataset(Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = sorted(os.listdir(os.path.join(root, "images")))
        
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.root, "images", self.imgs[idx])
        img = Image.open(img_path).convert("RGB")
        
        # Load labels
        label_path = os.path.join(self.root, "labels", 
                                  self.imgs[idx].replace('.jpg', '.txt'))
        
        boxes = []
        labels = []
        
        with open(label_path, 'r') as f:
            for line in f:
                cls, x, y, w, h = map(float, line.split())
                
                # Convert YOLO format to [x1, y1, x2, y2]
                img_w, img_h = img.size
                x1 = (x - w/2) * img_w
                y1 = (y - h/2) * img_h
                x2 = (x + w/2) * img_w
                y2 = (y + h/2) * img_h
                
                boxes.append([x1, y1, x2, y2])
                labels.append(int(cls) + 1)  # +1 vì 0 là background
        
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        
        if self.transforms:
            img = self.transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.imgs)
```

**Cell 3: Train loop**

```python
# Load model
model = fasterrcnn_resnet50_fpn(weights='DEFAULT')

# Replace head
num_classes = 8  # 7 classes + background
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# Move to GPU
device = torch.device('cuda')
model.to(device)

# Optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

# Training loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    
    for images, targets in train_loader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {losses.item():.4f}")
    
    # Save checkpoint
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'fasterrcnn_epoch{epoch+1}.pth')

# ⏳ Mất 4-5 giờ
```

**💡 Tip:** Faster R-CNN phức tạp, nếu gặp khó khăn:
- Có thể bỏ qua và chỉ train YOLOv8 + YOLOv5
- Hoặc dùng pretrained weights có sẵn từ project mẫu

### Bước 4.5: So sánh kết quả

#### 📊 Tạo bảng so sánh

**Tạo file Excel:**

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | Training Time |
|---------|---------|--------------|-----------|--------|---------------|
| YOLOv8 | 0.912 | 0.806 | 0.906 | 0.897 | 2.3h |
| YOLOv5 | 0.876 | 0.723 | 0.895 | 0.887 | 2.1h |
| Faster R-CNN | 0.923 | 0.812 | 0.908 | 0.901 | 4.5h |

**Nhận xét:**
- Faster R-CNN chính xác nhất nhưng train lâu nhất
- YOLOv8 cân bằng tốt giữa độ chính xác và thời gian
- YOLOv5 nhanh nhất nhưng độ chính xác thấp hơn

### ✅ Checklist Tuần 4-5

- [ ] Train YOLOv8 thành công (mAP@0.5 > 0.8)
- [ ] Train YOLOv5 thành công (mAP@0.5 > 0.8)
- [ ] Train Faster R-CNN thành công (hoặc dùng pretrained)
- [ ] Lưu tất cả weights về Google Drive
- [ ] Lưu tất cả biểu đồ (loss, mAP, confusion matrix)
- [ ] Tạo bảng so sánh kết quả
- [ ] Chụp screenshots cho báo cáo

**🎓 Tự kiểm tra:**
- [ ] Hiểu quá trình training (epochs, loss, metrics)
- [ ] Biết cách đọc kết quả training
- [ ] Có thể giải thích tại sao mô hình này tốt hơn mô hình kia

**🎉 Nếu hoàn thành → Chuyển sang Tuần 6!**

---

