# 📚 LÝ THUYẾT CƠ BẢN VỀ DEEP LEARNING VÀ OBJECT DETECTION
## Dành cho sinh viên tự học từ đầu

> **Mục tiêu:** Hiểu rõ lý thuyết đằng sau đồ án, có thể giải thích cho hội đồng  
> **Thời gian:** 1-2 tuần đọc hiểu  
> **Yêu cầu:** Biết toán cơ bản (đại số, ma trận)

---

## 📋 MỤC LỤC

1. [Machine Learning là gì?](#1-machine-learning-là-gì)
2. [Neural Networks cơ bản](#2-neural-networks-cơ-bản)
3. [Convolutional Neural Networks (CNN)](#3-convolutional-neural-networks-cnn)
4. [Object Detection](#4-object-detection)
5. [Faster R-CNN chi tiết](#5-faster-r-cnn-chi-tiết)
6. [YOLO chi tiết](#6-yolo-chi-tiết)
7. [Training và Optimization](#7-training-và-optimization)
8. [Metrics đánh giá](#8-metrics-đánh-giá)

---

## 1. MACHINE LEARNING LÀ GÌ?

### 1.1. Định nghĩa đơn giản

**Machine Learning (ML)** = Dạy máy tính học từ dữ liệu, không cần lập trình cụ thể

**Ví dụ thực tế:**

**Cách truyền thống (lập trình):**
```python
def nhan_dien_nemo(anh):
    if mau_cam(anh) and co_van_trang(anh):
        return "Nemo"
    else:
        return "Không phải Nemo"
```
❌ **Vấn đề:** Phải viết rule cho mọi trường hợp (ánh sáng khác nhau, góc nhìn khác nhau...)

**Cách Machine Learning:**
```python
# Cho máy xem 1000 ảnh Nemo
# Máy tự học đặc điểm của Nemo
model = train(1000_anh_nemo)

# Dùng để nhận dạng
ket_qua = model.predict(anh_moi)
```
✅ **Ưu điểm:** Máy tự học, không cần viết rule

### 1.2. Ba loại Machine Learning

#### 🎓 Supervised Learning (Học có giám sát)

**Định nghĩa:** Học từ dữ liệu đã có nhãn

**Ví dụ:**
```
Input: Ảnh con mèo → Label: "Mèo"
Input: Ảnh con chó → Label: "Chó"

Máy học: "Ảnh có tai nhọn, râu dài → Mèo"
         "Ảnh có tai cụp, lưỡi dài → Chó"
```

**Ứng dụng:** Phân loại ảnh, nhận dạng giọng nói, dự đoán giá nhà

#### 🤖 Unsupervised Learning (Học không giám sát)

**Định nghĩa:** Học từ dữ liệu không có nhãn, tự tìm pattern

**Ví dụ:**
```
Input: 1000 ảnh động vật (không có nhãn)

Máy tự phân nhóm:
  - Nhóm 1: Động vật có 4 chân, lông
  - Nhóm 2: Động vật có vây, sống dưới nước
  - Nhóm 3: Động vật có cánh, bay được
```

**Ứng dụng:** Phân nhóm khách hàng, phát hiện bất thường

#### 🎮 Reinforcement Learning (Học tăng cường)

**Định nghĩa:** Học bằng cách thử và nhận phần thưởng/hình phạt

**Ví dụ:** Dạy máy chơi game
```
Hành động: Nhảy qua hố → Phần thưởng: +10 điểm
Hành động: Rơi xuống hố → Hình phạt: -10 điểm

Máy học: "Nhảy qua hố = tốt, rơi xuống = xấu"
```

**Ứng dụng:** Game AI, xe tự lái, robot

**🎯 Đồ án này dùng:** Supervised Learning (có nhãn: Nemo, Dory, Starfish)

---

## 2. NEURAL NETWORKS CƠ BẢN

### 2.1. Neuron - Đơn vị cơ bản

**Neuron nhân tạo** mô phỏng neuron trong não người

**Cấu trúc:**
```
Input → [Neuron] → Output

Chi tiết:
x1 ──w1──┐
x2 ──w2──┤
x3 ──w3──┼──→ Σ ──→ Activation ──→ Output
   ...   │
xn ──wn──┘
    +b
```

**Công thức:**
```
Output = Activation(w1*x1 + w2*x2 + ... + wn*xn + b)

w: weights (trọng số)
b: bias (độ lệch)
Activation: hàm kích hoạt
```

**Ví dụ cụ thể:**

Nhận dạng "Ảnh có phải Nemo không?"

```
Input:
  x1 = độ cam của ảnh (0-1)
  x2 = có vân trắng không (0 hoặc 1)
  x3 = kích thước (0-1)

Weights (máy học được):
  w1 = 0.8  (màu cam quan trọng)
  w2 = 0.6  (vân trắng quan trọng)
  w3 = 0.2  (kích thước ít quan trọng)
  b = -0.5

Tính toán:
  z = 0.8*0.9 + 0.6*1 + 0.2*0.5 + (-0.5)
    = 0.72 + 0.6 + 0.1 - 0.5
    = 0.92

Activation (Sigmoid):
  Output = 1 / (1 + e^(-0.92))
         = 0.715 (71.5%)

Kết luận: 71.5% là Nemo → Có thể là Nemo!
```

### 2.2. Activation Functions (Hàm kích hoạt)

**Tại sao cần Activation Function?**
- Thêm tính phi tuyến (non-linearity)
- Giúp mạng học được các pattern phức tạp

#### 📊 Các hàm phổ biến

**1. Sigmoid**
```
f(x) = 1 / (1 + e^(-x))

Đặc điểm:
  - Output: 0 đến 1
  - Dùng cho: Binary classification (2 classes)
  
Ví dụ:
  f(-5) = 0.007 (gần 0)
  f(0)  = 0.5   (giữa)
  f(5)  = 0.993 (gần 1)
```

**2. ReLU (Rectified Linear Unit)**
```
f(x) = max(0, x)

Đặc điểm:
  - Output: 0 đến ∞
  - Đơn giản, nhanh
  - Phổ biến nhất trong CNN
  
Ví dụ:
  f(-5) = 0
  f(0)  = 0
  f(5)  = 5
```

**3. Softmax**
```
f(xi) = e^xi / Σ(e^xj)

Đặc điểm:
  - Output: Tổng = 1 (xác suất)
  - Dùng cho: Multi-class classification
  
Ví dụ:
  Input: [2.0, 1.0, 0.1]
  Output: [0.659, 0.242, 0.099]
  → Class 0 có xác suất cao nhất (65.9%)
```

### 2.3. Neural Network (Mạng nhiều lớp)

**Cấu trúc:**
```
Input Layer → Hidden Layers → Output Layer

Ví dụ: Nhận dạng chữ số viết tay (0-9)

Input (784 neurons):        Hidden (128 neurons):      Output (10 neurons):
[Pixel 1]                   [Neuron 1]                 [0: 0.01]
[Pixel 2]  ──────────────→  [Neuron 2]  ────────────→  [1: 0.02]
[Pixel 3]                   [...]                      [2: 0.05]
[...]                       [Neuron 128]               [...]
[Pixel 784]                                            [9: 0.85] ← Dự đoán: 9
```

**Quá trình Forward Pass:**
```
1. Input → Hidden Layer 1
   h1 = ReLU(W1 * input + b1)

2. Hidden Layer 1 → Hidden Layer 2
   h2 = ReLU(W2 * h1 + b2)

3. Hidden Layer 2 → Output
   output = Softmax(W3 * h2 + b3)
```

### 2.4. Backpropagation (Lan truyền ngược)

**Mục đích:** Cập nhật weights để giảm lỗi

**Quy trình:**

**1. Forward Pass (Tính toán)**
```
Input → Hidden → Output → Dự đoán
```

**2. Tính Loss (Lỗi)**
```
Loss = (Dự đoán - Thực tế)²

Ví dụ:
  Thực tế: Nemo (1)
  Dự đoán: 0.7
  Loss = (0.7 - 1)² = 0.09
```

**3. Backward Pass (Cập nhật)**
```
Tính gradient (đạo hàm) của Loss theo từng weight
Cập nhật weight:
  w_new = w_old - learning_rate * gradient

Ví dụ:
  w_old = 0.5
  gradient = 0.2
  learning_rate = 0.01
  w_new = 0.5 - 0.01 * 0.2 = 0.498
```

**Ví dụ trực quan:**

```
Epoch 1:
  Dự đoán: 0.3 → Loss: 0.49 (cao)
  Cập nhật weights...

Epoch 2:
  Dự đoán: 0.5 → Loss: 0.25 (giảm)
  Cập nhật weights...

Epoch 10:
  Dự đoán: 0.9 → Loss: 0.01 (thấp)
  → Mô hình đã học tốt!
```

---

## 3. CONVOLUTIONAL NEURAL NETWORKS (CNN)

### 3.1. Tại sao cần CNN?

**Vấn đề với Neural Network thông thường:**

```
Ảnh 640×480 RGB = 640 × 480 × 3 = 921,600 pixels
→ Cần 921,600 neurons ở input layer
→ Nếu hidden layer có 1000 neurons
→ Cần 921,600 × 1000 = 921 triệu weights!

❌ Quá nhiều parameters
❌ Dễ overfitting
❌ Chậm
```

**Giải pháp: CNN**
- Sử dụng filters (bộ lọc) nhỏ
- Chia sẻ weights (weight sharing)
- Giảm số parameters xuống hàng triệu

### 3.2. Convolutional Layer (Lớp tích chập)

**Khái niệm:** Áp dụng filter trượt trên ảnh để trích xuất đặc trưng

**Ví dụ trực quan:**

**Input Image (5×5):**
```
1  2  3  4  5
6  7  8  9  10
11 12 13 14 15
16 17 18 19 20
21 22 23 24 25
```

**Filter (3×3) - Phát hiện cạnh dọc:**
```
-1  0  1
-1  0  1
-1  0  1
```

**Convolution Operation:**
```
Vị trí (0,0):
┌─────────┐
│ 1  2  3 │
│ 6  7  8 │
│11 12 13 │
└─────────┘

Tính toán:
= (-1×1) + (0×2) + (1×3) +
  (-1×6) + (0×7) + (1×8) +
  (-1×11) + (0×12) + (1×13)
= -1 + 0 + 3 - 6 + 0 + 8 - 11 + 0 + 13
= 6

Trượt sang phải, lặp lại...
```

**Output Feature Map (3×3):**
```
6   6   6
6   6   6
6   6   6
```

**Ý nghĩa:**
- Giá trị cao → Có cạnh dọc tại vị trí đó
- Giá trị thấp → Không có cạnh dọc

### 3.3. Các loại Filters

**1. Edge Detection (Phát hiện cạnh)**

**Vertical Edge (Cạnh dọc):**
```
-1  0  1
-1  0  1
-1  0  1
```

**Horizontal Edge (Cạnh ngang):**
```
-1 -1 -1
 0  0  0
 1  1  1
```

**2. Blur (Làm mờ):**
```
1/9  1/9  1/9
1/9  1/9  1/9
1/9  1/9  1/9
```

**3. Sharpen (Làm sắc nét):**
```
 0 -1  0
-1  5 -1
 0 -1  0
```

**💡 Trong CNN:** Máy tự học filters tối ưu, không cần định nghĩa trước!

### 3.4. Pooling Layer (Lớp gộp)

**Mục đích:**
- Giảm kích thước feature map
- Giảm số parameters
- Tăng tính bất biến với vị trí (translation invariance)

**Max Pooling (Phổ biến nhất):**

**Input (4×4):**
```
1  3  2  4
5  6  7  8
3  2  1  0
1  2  3  4
```

**Max Pooling 2×2 (stride=2):**
```
Vùng 1:        Vùng 2:
┌─────┐        ┌─────┐
│ 1  3│        │ 2  4│
│ 5  6│        │ 7  8│
└─────┘        └─────┘
Max = 6        Max = 8

Vùng 3:        Vùng 4:
┌─────┐        ┌─────┐
│ 3  2│        │ 1  0│
│ 1  2│        │ 3  4│
└─────┘        └─────┘
Max = 3        Max = 4
```

**Output (2×2):**
```
6  8
3  4
```

**Giảm kích thước:** 4×4 = 16 → 2×2 = 4 (giảm 75%)

### 3.5. Kiến trúc CNN điển hình

**LeNet-5 (1998) - CNN đầu tiên:**

```
Input (32×32×1)
    ↓
Conv1 (6 filters, 5×5) → (28×28×6)
    ↓
MaxPool (2×2) → (14×14×6)
    ↓
Conv2 (16 filters, 5×5) → (10×10×16)
    ↓
MaxPool (2×2) → (5×5×16)
    ↓
Flatten → (400)
    ↓
FC1 (120 neurons)
    ↓
FC2 (84 neurons)
    ↓
Output (10 classes)
```

**Giải thích từng bước:**

1. **Conv1:** Trích xuất đặc trưng cơ bản (cạnh, góc)
2. **MaxPool:** Giảm kích thước, giữ thông tin quan trọng
3. **Conv2:** Trích xuất đặc trưng phức tạp hơn (hình dạng)
4. **MaxPool:** Giảm tiếp
5. **Flatten:** Chuyển 2D → 1D
6. **FC (Fully Connected):** Kết hợp đặc trưng để phân loại

### 3.6. Tại sao CNN hiệu quả với ảnh?

**1. Local Connectivity (Kết nối cục bộ)**
- Mỗi neuron chỉ nhìn một vùng nhỏ
- Giống như mắt người: nhìn từng phần rồi ghép lại

**2. Weight Sharing (Chia sẻ trọng số)**
- Cùng 1 filter dùng cho toàn bộ ảnh
- Giảm số parameters từ hàng tỷ xuống hàng triệu

**3. Translation Invariance (Bất biến vị trí)**
- Nhận dạng được đối tượng dù ở đâu trong ảnh
- Nemo ở góc trái hay góc phải đều nhận dạng được

**Ví dụ so sánh:**

```
Neural Network thông thường:
  640×480×3 ảnh → 921,600 inputs
  Hidden 1000 neurons → 921 triệu weights

CNN:
  Conv1: 32 filters 3×3 → 32 × 3 × 3 = 288 weights
  Conv2: 64 filters 3×3 → 64 × 32 × 3 × 3 = 18,432 weights
  ...
  Tổng: ~1-2 triệu weights (giảm 1000 lần!)
```

---

## 4. OBJECT DETECTION

### 4.1. Sự khác biệt với Image Classification

**Image Classification:**
```
Input: Ảnh
Output: 1 nhãn

Ví dụ:
  Input: [Ảnh con mèo]
  Output: "Mèo"
```

**Object Detection:**
```
Input: Ảnh
Output: Nhiều objects, mỗi object có:
  - Bounding box (x, y, w, h)
  - Class label
  - Confidence score

Ví dụ:
  Input: [Ảnh có mèo và chó]
  Output:
    - Object 1: "Mèo" tại (100, 50, 200, 150), conf=0.95
    - Object 2: "Chó" tại (300, 100, 400, 250), conf=0.87
```

### 4.2. Các thách thức

**1. Nhiều objects trong 1 ảnh**
```
Ảnh có 10 con cá → Phải phát hiện cả 10
```

**2. Objects có kích thước khác nhau**
```
Cá lớn: 300×200 pixels
Cá nhỏ: 50×30 pixels
→ Cần phát hiện cả 2
```

**3. Objects bị che khuất**
```
Nemo bị rạn san hô che 50%
→ Vẫn phải nhận dạng được
```

**4. Tốc độ xử lý**
```
Video 30 FPS → Phải xử lý 30 ảnh/giây
→ Mỗi ảnh < 33ms
```

### 4.3. Hai hướng tiếp cận chính

#### 🐢 Two-Stage Detectors (Chậm, chính xác)

**Quy trình:**
```
Stage 1: Region Proposal
  - Đề xuất ~2000 vùng có thể chứa object
  - Dùng: Selective Search hoặc RPN

Stage 2: Classification
  - Phân loại từng vùng
  - Tinh chỉnh bounding box
```

**Ví dụ:** R-CNN, Fast R-CNN, **Faster R-CNN**

**Ưu điểm:**
- ✅ Độ chính xác cao
- ✅ Tốt với objects nhỏ

**Nhược điểm:**
- ❌ Chậm (~5-10 FPS)
- ❌ Phức tạp

#### 🐇 One-Stage Detectors (Nhanh, ít chính xác hơn)

**Quy trình:**
```
Single Stage:
  - Dự đoán trực tiếp bounding box và class
  - Không cần region proposal
```

**Ví dụ:** SSD, RetinaNet, **YOLO**

**Ưu điểm:**
- ✅ Nhanh (~30-100 FPS)
- ✅ Đơn giản
- ✅ Real-time

**Nhược điểm:**
- ❌ Độ chính xác thấp hơn
- ❌ Kém với objects nhỏ

### 4.4. Bounding Box Representation

**Có 2 cách biểu diễn:**

**1. Corner Format (x1, y1, x2, y2)**
```
(x1, y1): Góc trên trái
(x2, y2): Góc dưới phải

Ví dụ: (100, 50, 300, 200)
  x1=100, y1=50  ┌─────────┐
                 │         │
                 │ Object  │
                 │         │
  x2=300, y2=200 └─────────┘
```

**2. Center Format (x, y, w, h)**
```
(x, y): Tâm bounding box
(w, h): Width, Height

Ví dụ: (200, 125, 200, 150)
  x=200, y=125 (tâm)
  w=200, h=150

Chuyển đổi:
  x1 = x - w/2 = 200 - 100 = 100
  y1 = y - h/2 = 125 - 75 = 50
  x2 = x + w/2 = 200 + 100 = 300
  y2 = y + h/2 = 125 + 75 = 200
```

**YOLO dùng:** Center Format (normalized 0-1)

### 4.5. Anchor Boxes

**Khái niệm:** Bounding boxes mẫu được định nghĩa trước

**Tại sao cần Anchor Boxes?**
- Objects có nhiều hình dạng khác nhau
- Anchor boxes giúp mô hình học dễ hơn

**Ví dụ:**

```
3 Anchor boxes với aspect ratios khác nhau:

Anchor 1 (1:1):    Anchor 2 (1:2):    Anchor 3 (2:1):
┌────┐             ┌──┐               ┌────────┐
│    │             │  │               │        │
│    │             │  │               └────────┘
│    │             │  │
└────┘             │  │
                   └──┘

Dùng cho:          Dùng cho:          Dùng cho:
- Sao biển         - Cá đứng          - Cá nằm ngang
- Objects vuông    - Objects cao      - Objects rộng
```

**Quy trình sử dụng:**

```
1. Định nghĩa anchor boxes (K-means clustering trên training data)
2. Với mỗi vị trí trên feature map:
   - Dự đoán offset từ anchor box
   - Dự đoán class
   - Dự đoán objectness score
3. Chọn predictions tốt nhất (NMS)
```

---


## 5. FASTER R-CNN CHI TIẾT

### 5.1. Lịch sử phát triển R-CNN

**R-CNN (2014):**
```
1. Selective Search → 2000 region proposals
2. Resize mỗi region về 227×227
3. Chạy CNN cho từng region (2000 lần!)
4. SVM classifier

❌ Rất chậm: 47 giây/ảnh
```

**Fast R-CNN (2015):**
```
1. Chạy CNN 1 lần cho toàn bộ ảnh
2. Selective Search → 2000 regions
3. RoI Pooling để extract features
4. FC layers để classify

✅ Nhanh hơn: 2 giây/ảnh
❌ Vẫn chậm vì Selective Search
```

**Faster R-CNN (2015):**
```
1. CNN → Feature maps
2. RPN (Region Proposal Network) → Proposals
3. RoI Pooling
4. Classification + Bbox Regression

✅ Nhanh hơn nhiều: 0.2 giây/ảnh (5 FPS)
✅ End-to-end training
```

### 5.2. Kiến trúc Faster R-CNN

**Tổng quan:**

```
Input Image (H×W×3)
        ↓
┌───────────────────────────────────┐
│   Backbone (ResNet50-FPN)         │
│   - Conv layers                   │
│   - Feature Pyramid Network       │
└───────────────────────────────────┘
        ↓
Feature Maps (H/16×W/16×C)
        ↓
    ┌───┴───┐
    ↓       ↓
┌───────┐ ┌──────────────────────┐
│  RPN  │ │  RoI Pooling         │
│       │ │  + Detection Head    │
└───┬───┘ └──────────────────────┘
    │              ↓
    │     ┌────────┴────────┐
    │     ↓                 ↓
    │  Classification   Bbox Regression
    │     ↓                 ↓
    └──→ NMS ──→ Final Detections
```

### 5.3. Backbone: ResNet50-FPN

**ResNet50:**

**Vấn đề với Deep Networks:**
```
Network càng sâu → Càng khó train
Lý do: Vanishing Gradient
  - Gradient nhỏ dần khi backprop
  - Layers đầu không học được
```

**Giải pháp: Residual Connection (Skip Connection)**

```
Input (x)
    ↓
    ├─────────────────┐
    ↓                 │
Conv + ReLU          │
    ↓                 │
Conv                 │
    ↓                 │
    +  ←──────────────┘
    ↓
  ReLU
    ↓
Output (F(x) + x)
```

**Công thức:**
```
Output = F(x) + x

F(x): Learned residual
x: Identity (skip connection)

Nếu F(x) = 0 → Output = x (identity mapping)
→ Dễ học hơn!
```

**Feature Pyramid Network (FPN):**

**Vấn đề:** Objects có nhiều scales khác nhau
```
Cá lớn: 300×200 pixels
Cá nhỏ: 30×20 pixels
→ Cần feature maps ở nhiều scales
```

**Giải pháp: FPN**

```
Bottom-up pathway (ResNet):
C1 (H/2×W/2)
    ↓
C2 (H/4×W/4)
    ↓
C3 (H/8×W/8)
    ↓
C4 (H/16×W/16)
    ↓
C5 (H/32×W/32)

Top-down pathway + Lateral connections:
C5 ──→ P5 (H/32×W/32) ──┐
       ↑                 │
C4 ──→ P4 (H/16×W/16) ──┤
       ↑                 ├─→ Multi-scale features
C3 ──→ P3 (H/8×W/8)   ──┤
       ↑                 │
C2 ──→ P2 (H/4×W/4)   ──┘
```

**Ý nghĩa:**
- P2: Phát hiện objects nhỏ (high resolution)
- P3, P4: Phát hiện objects trung bình
- P5: Phát hiện objects lớn (low resolution)

### 5.4. Region Proposal Network (RPN)

**Mục đích:** Đề xuất vùng có thể chứa objects

**Kiến trúc:**

```
Feature Map (H×W×C)
        ↓
    3×3 Conv
        ↓
    ┌───┴───┐
    ↓       ↓
1×1 Conv  1×1 Conv
    ↓       ↓
Objectness  Bbox Regression
(2k scores) (4k coords)

k = số anchor boxes (thường k=9)
```

**Anchor Boxes:**

```
Tại mỗi vị trí (i, j) trên feature map:
  - 3 scales: 128², 256², 512²
  - 3 aspect ratios: 1:1, 1:2, 2:1
  → 3 × 3 = 9 anchor boxes

Ví dụ tại vị trí (10, 15):
  Anchor 1: (160, 240, 128, 128)  # Scale 128, ratio 1:1
  Anchor 2: (160, 240, 91, 181)   # Scale 128, ratio 1:2
  Anchor 3: (160, 240, 181, 91)   # Scale 128, ratio 2:1
  ...
  Anchor 9: (160, 240, 512, 512)  # Scale 512, ratio 1:1
```

**Objectness Score:**
```
Cho mỗi anchor box:
  - Score 0: Không có object (background)
  - Score 1: Có object (foreground)

Ví dụ:
  Anchor 1: 0.95 → Rất có thể có object
  Anchor 2: 0.12 → Không có object
  Anchor 3: 0.78 → Có thể có object
```

**Bbox Regression:**
```
Dự đoán offset từ anchor box đến ground truth box

Anchor box: (xa, ya, wa, ha)
Ground truth: (x, y, w, h)

Dự đoán:
  tx = (x - xa) / wa
  ty = (y - ya) / ha
  tw = log(w / wa)
  th = log(h / ha)

Khi inference:
  x = tx * wa + xa
  y = ty * ha + ya
  w = wa * exp(tw)
  h = ha * exp(th)
```

**Training RPN:**

**Loss Function:**
```
L = L_cls + λ * L_reg

L_cls: Binary cross-entropy (object/background)
L_reg: Smooth L1 loss (bbox regression)
λ: Balance weight (thường = 1)
```

**Positive/Negative Anchors:**
```
Positive (có object):
  - IoU với ground truth > 0.7
  - Hoặc IoU cao nhất với ground truth

Negative (background):
  - IoU với ground truth < 0.3

Ignore:
  - 0.3 ≤ IoU ≤ 0.7
```

### 5.5. RoI Pooling

**Vấn đề:** Region proposals có kích thước khác nhau
```
Proposal 1: 200×150
Proposal 2: 80×60
Proposal 3: 300×400

→ Cần chuẩn hóa về cùng kích thước
```

**Giải pháp: RoI Pooling**

**Ví dụ cụ thể:**

```
Input: Region 8×8, Output: 2×2

┌─────────────────┐
│ 1  2│ 3  4│     │
│ 5  6│ 7  8│     │
├─────┼─────┤     │
│ 9 10│11 12│     │
│13 14│15 16│     │
├─────┴─────┴─────┤
│                 │
│                 │
└─────────────────┘

Chia thành 2×2 = 4 vùng:
  Vùng 1: [1,2,5,6] → Max = 6
  Vùng 2: [3,4,7,8] → Max = 8
  Vùng 3: [9,10,13,14] → Max = 14
  Vùng 4: [11,12,15,16] → Max = 16

Output (2×2):
┌────┬────┐
│ 6  │ 8  │
├────┼────┤
│ 14 │ 16 │
└────┴────┘
```

**RoI Align (Cải tiến):**
- Không làm tròn tọa độ
- Dùng bilinear interpolation
- Chính xác hơn RoI Pooling

### 5.6. Detection Head

**Sau RoI Pooling:**

```
RoI Features (7×7×C)
        ↓
    Flatten
        ↓
    FC (4096)
        ↓
    ReLU
        ↓
    FC (4096)
        ↓
    ReLU
        ↓
    ┌───┴───┐
    ↓       ↓
FC (N+1)  FC (4×N)
    ↓       ↓
Classes  Bbox Deltas

N = số classes (không tính background)
```

**Classification:**
```
Output: Xác suất cho mỗi class

Ví dụ (N=7):
  Background: 0.02
  Nemo: 0.85
  Dory: 0.08
  Bat Star: 0.03
  ...

→ Dự đoán: Nemo (0.85)
```

**Bbox Regression:**
```
Tinh chỉnh bounding box từ proposal

Proposal: (100, 50, 200, 150)
Deltas: (0.1, -0.05, 0.2, 0.15)

Final bbox:
  x = 100 + 0.1 * 200 = 120
  y = 50 - 0.05 * 150 = 42.5
  w = 200 * exp(0.2) = 244
  h = 150 * exp(0.15) = 174
```

### 5.7. Non-Maximum Suppression (NMS)

**Vấn đề:** Nhiều proposals cho cùng 1 object

```
Object: Nemo

Proposal 1: (100, 50, 200, 150), conf=0.95
Proposal 2: (105, 52, 198, 148), conf=0.92
Proposal 3: (98, 48, 202, 152), conf=0.88

→ Cần chọn 1 proposal tốt nhất
```

**Giải pháp: NMS**

**Thuật toán:**
```
1. Sắp xếp proposals theo confidence giảm dần
2. Chọn proposal có confidence cao nhất
3. Loại bỏ các proposals có IoU > threshold (0.5) với proposal đã chọn
4. Lặp lại cho đến khi hết proposals
```

**Ví dụ:**

```
Input:
  P1: conf=0.95, bbox=(100,50,200,150)
  P2: conf=0.92, bbox=(105,52,198,148)  # IoU với P1 = 0.85
  P3: conf=0.88, bbox=(98,48,202,152)   # IoU với P1 = 0.82
  P4: conf=0.75, bbox=(300,100,400,200) # IoU với P1 = 0.0

Bước 1: Chọn P1 (conf cao nhất)
Bước 2: Loại P2, P3 (IoU > 0.5 với P1)
Bước 3: Giữ P4 (IoU = 0 với P1)

Output:
  P1: (100,50,200,150), conf=0.95
  P4: (300,100,400,200), conf=0.75
```

### 5.8. Training Faster R-CNN

**Loss Function:**

```
L_total = L_rpn + L_detection

L_rpn = L_rpn_cls + λ1 * L_rpn_reg
L_detection = L_det_cls + λ2 * L_det_reg

Thường: λ1 = λ2 = 1
```

**Training Strategy:**

**4-Step Alternating Training (cũ):**
```
1. Train RPN
2. Train Detection Head (fix RPN)
3. Fine-tune RPN (fix Detection Head)
4. Fine-tune Detection Head (fix RPN)
```

**End-to-End Training (mới):**
```
Train cả RPN và Detection Head cùng lúc
→ Đơn giản hơn, hiệu quả hơn
```

**Hyperparameters:**
```
Optimizer: SGD
Learning rate: 0.001 (giảm 10× sau mỗi 30 epochs)
Momentum: 0.9
Weight decay: 0.0001
Batch size: 2 (mỗi GPU)
Epochs: 50-100
```

---

## 6. YOLO CHI TIẾT

### 6.1. Ý tưởng chính của YOLO

**"You Only Look Once"** = Chỉ nhìn 1 lần

**So với Faster R-CNN:**
```
Faster R-CNN:
  1. Đề xuất regions (~2000)
  2. Classify từng region
  → Nhìn nhiều lần

YOLO:
  1. Chia ảnh thành grid
  2. Mỗi cell dự đoán trực tiếp
  → Chỉ nhìn 1 lần
```

### 6.2. YOLO v1 (2016) - Nền tảng

**Kiến trúc:**

```
Input Image (448×448×3)
        ↓
Darknet-19 (CNN Backbone)
  - 19 Conv layers
  - 5 MaxPool layers
        ↓
Feature Map (7×7×1024)
        ↓
FC Layers
        ↓
Output (7×7×30)

7×7: Grid size
30 = B×5 + C
  B=2: Số bounding boxes/cell
  5: (x, y, w, h, confidence)
  C=20: Số classes (COCO)
```

**Grid System:**

```
Chia ảnh thành 7×7 = 49 cells

┌─┬─┬─┬─┬─┬─┬─┐
├─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┤
├─┼─┼─┼─┼─┼─┼─┤  Mỗi cell dự đoán:
├─┼─┼─┼─┼─┼─┼─┤  - 2 bounding boxes
├─┼─┼─┼─┼─┼─┼─┤  - 20 class probabilities
├─┼─┼─┼─┼─┼─┼─┤
└─┴─┴─┴─┴─┴─┴─┘

Cell chứa tâm object → Chịu trách nhiệm detect object đó
```

**Prediction Format:**

```
Mỗi cell dự đoán:
  - Bbox 1: (x1, y1, w1, h1, conf1)
  - Bbox 2: (x2, y2, w2, h2, conf2)
  - Class probs: [p1, p2, ..., p20]

x, y: Tọa độ tâm (relative to cell, 0-1)
w, h: Width, height (relative to image, 0-1)
conf: Confidence = Pr(Object) × IoU
```

**Ví dụ cụ thể:**

```
Cell (3, 4) dự đoán:
  Bbox 1:
    x = 0.6 (60% từ trái cell)
    y = 0.4 (40% từ trên cell)
    w = 0.3 (30% chiều rộng ảnh)
    h = 0.2 (20% chiều cao ảnh)
    conf = 0.85

  Class probs:
    Nemo: 0.92
    Dory: 0.05
    ...

Final confidence:
  Nemo: 0.85 × 0.92 = 0.782
```

**Loss Function:**

```
L = λ_coord × L_coord + L_conf + L_class

L_coord: Localization loss (bbox)
L_conf: Confidence loss
L_class: Classification loss

λ_coord = 5 (tăng trọng số cho localization)
```

**Chi tiết:**

```
L_coord = Σ 1_obj [ (x - x̂)² + (y - ŷ)² + (√w - √ŵ)² + (√h - √ĥ)² ]

Dùng √w, √h vì:
  - Lỗi nhỏ ở bbox lớn ít quan trọng hơn
  - Lỗi nhỏ ở bbox nhỏ quan trọng hơn

L_conf = Σ 1_obj (C - Ĉ)² + λ_noobj × Σ 1_noobj (C - Ĉ)²

λ_noobj = 0.5 (giảm trọng số cho background)

L_class = Σ 1_obj Σ_classes (p - p̂)²
```

**Hạn chế YOLOv1:**
- ❌ Mỗi cell chỉ dự đoán 1 class → Không tốt với objects gần nhau
- ❌ Không tốt với objects nhỏ
- ❌ Không tốt với objects có aspect ratio lạ

### 6.3. YOLOv5 (2020)

**Cải tiến chính:**

**1. Anchor Boxes**
```
Không dùng grid cells cố định
Dùng anchor boxes như Faster R-CNN

3 scales × 3 anchors = 9 anchor boxes
Anchor sizes được tính tự động từ dataset (K-means)
```

**2. Multi-Scale Predictions**
```
Dự đoán ở 3 scales:
  - P3 (80×80): Objects nhỏ
  - P4 (40×40): Objects trung bình
  - P5 (20×20): Objects lớn
```

**3. CSPDarknet Backbone**
```
Cross Stage Partial Network
  - Chia feature map thành 2 phần
  - 1 phần qua Conv layers
  - 1 phần skip
  - Concat lại

Ưu điểm:
  - Giảm tính toán
  - Giảm memory
  - Tăng tốc độ
```

**Kiến trúc YOLOv5:**

```
Input (640×640×3)
        ↓
┌─────────────────────────┐
│  Backbone: CSPDarknet   │
│  - Focus layer          │
│  - CSP blocks           │
│  - SPPF                 │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Neck: PANet            │
│  - FPN (top-down)       │
│  - PAN (bottom-up)      │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Head: Detection        │
│  - 3 detection layers   │
│  - P3, P4, P5           │
└─────────────────────────┘
        ↓
    Predictions
```

**Focus Layer:**
```
Mục đích: Giảm kích thước ảnh mà không mất thông tin

Input (640×640×3)
        ↓
Slice thành 4 phần:
  - Top-left (320×320)
  - Top-right (320×320)
  - Bottom-left (320×320)
  - Bottom-right (320×320)
        ↓
Concat theo channel
        ↓
Output (320×320×12)
        ↓
Conv 3×3
        ↓
Output (320×320×64)
```

**SPPF (Spatial Pyramid Pooling Fast):**
```
Mục đích: Tăng receptive field

Input
    ↓
MaxPool 5×5 ──┐
    ↓         │
MaxPool 5×5 ──┤
    ↓         ├─→ Concat
MaxPool 5×5 ──┤
    ↓         │
Identity ─────┘
    ↓
Output (4× channels)
```

**Loss Function YOLOv5:**

```
L = L_box + L_obj + L_cls

L_box: CIoU loss (Complete IoU)
L_obj: BCE loss (objectness)
L_cls: BCE loss (classification)
```

**CIoU Loss:**
```
CIoU = 1 - IoU + ρ²(b, b_gt)/c² + αv

IoU: Intersection over Union
ρ: Khoảng cách Euclidean giữa tâm 2 boxes
c: Đường chéo của smallest enclosing box
v: Đo sự tương đồng về aspect ratio
α: Trọng số cân bằng

Ưu điểm so với IoU:
  - Xét cả khoảng cách tâm
  - Xét cả aspect ratio
  - Hội tụ nhanh hơn
```

### 6.4. YOLOv8 (2023)

**Cải tiến chính:**

**1. Anchor-Free Detection**
```
Không dùng anchor boxes nữa
Dự đoán trực tiếp từ center point

Ưu điểm:
  - Đơn giản hơn
  - Không cần K-means clustering
  - Generalize tốt hơn
```

**2. Decoupled Head**
```
YOLOv5:
  Shared Conv → Classification + Localization

YOLOv8:
  ┌─→ Classification Conv → Class probs
  │
Input Conv
  │
  └─→ Localization Conv → Bbox coords

Ưu điểm:
  - Mỗi task có riêng features
  - Cải thiện cả 2 tasks
```

**3. New Loss Functions**

**VFL (Varifocal Loss) cho Classification:**
```
VFL = -q × (q - p)^γ × log(p)

q: Target quality (IoU với ground truth)
p: Predicted probability
γ: Focusing parameter (thường = 2)

Ý nghĩa:
  - Focus vào high-quality predictions
  - Giảm ảnh hưởng của low-quality predictions
```

**DFL (Distribution Focal Loss) cho Bbox:**
```
Thay vì dự đoán 1 giá trị cho mỗi coordinate:
  Dự đoán phân phối xác suất

Ví dụ:
  x ∈ [10, 11, 12, 13, 14]
  Probs: [0.1, 0.2, 0.4, 0.2, 0.1]
  → x = Σ (value × prob) = 12.0

Ưu điểm:
  - Mô hình uncertainty
  - Chính xác hơn
```

**Kiến trúc YOLOv8:**

```
Input (640×640×3)
        ↓
┌─────────────────────────┐
│  Backbone: CSPDarknet v8│
│  - C2f modules          │
│  - SPPF                 │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Neck: PANet v8         │
│  - C2f modules          │
│  - Upsample + Concat    │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│  Head: Decoupled        │
│  - Classification head  │
│  - Localization head    │
└─────────────────────────┘
        ↓
    Predictions
```

**C2f Module:**
```
Cải tiến từ CSP:

Input
    ↓
Split
    ├─→ Branch 1 (Conv)
    │       ↓
    │   Bottleneck 1
    │       ↓
    │   Bottleneck 2
    │       ↓
    └─→ Branch 2 (Identity)
        ↓
    Concat
        ↓
    Conv
        ↓
    Output

Ưu điểm:
  - Gradient flow tốt hơn
  - Nhanh hơn CSP
```

---

