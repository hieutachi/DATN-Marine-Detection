Chào bạn Hải Anh, một "đồng nghiệp" từ K64 khoa CNTT trường Đại học Thủy Lợi[cite: 1, 4]. [cite_start]Với đề tài nhận diện sinh vật biển (Nemo, Dory, Sao biển) sử dụng YOLOv8, đây là một lựa chọn rất hợp thời và phù hợp cho mức độ Đồ án tốt nghiệp (ĐATN)[cite: 12, 24].

Dưới đây là **Master Plan** được thiết kế theo dạng "mảnh ghép" (Modular Approach). [cite_start]Mỗi mảnh ghép sẽ đi kèm một Prompt cụ thể để bạn có thể làm việc hiệu quả với AI, giúp bạn không chỉ có kết quả mà còn hiểu sâu để bảo vệ trước Hội đồng[cite: 28].

---

## 📅 Master Plan: Hệ thống Nhận dạng Sinh vật biển (MVP Level)

[cite_start]Kế hoạch này tập trung vào tính thực dụng, giúp bạn hoàn thành đúng tiến độ từ tháng 4 đến tháng 6/2026[cite: 58, 100].

| Giai đoạn | Tên mảnh ghép | Mục tiêu chính | Công cụ |
| :--- | :--- | :--- | :--- |
| **P1: Data** | **Hạt nhân Dữ liệu** | [cite_start]Chuẩn hóa Dataset từ Roboflow, kiểm tra nhãn[cite: 13, 19]. | Roboflow, Python |
| **P2: Train** | **Huấn luyện YOLOv8** | [cite_start]Train mô hình Baseline trên Colab, lưu trọng số[cite: 16, 44]. | Google Colab, Ultralytics |
| **P3: Compare**| **Đối sánh Mô hình** | [cite_start]Chạy Faster R-CNN & YOLOv5 để lấy số liệu so sánh[cite: 25, 48]. | TorchVision, YOLOv5 |
| **P4: Web** | **Giao diện Flask** | [cite_start]Xây dựng Web cho phép Upload ảnh/video và hiển thị kết quả[cite: 18, 47]. | Flask, OpenCV |
| **P5: Report** | **Hoàn thiện Báo cáo** | [cite_start]Viết nội dung dựa trên kết quả thực nghiệm[cite: 57, 102]. | Word, LaTeX |

---

## 🧩 Mảnh ghép số 1: Hạt nhân Dữ liệu & Tiền xử lý

Đây là bước quan trọng nhất. [cite_start]Nếu dữ liệu sai, mô hình sẽ không bao giờ chính xác[cite: 32].

### 🤖 Prompt gợi ý cho mảnh ghép này:
> [cite_start]"Tôi đang làm đồ án tốt nghiệp về nhận diện sinh vật biển (Nemo, Dory, sao biển)[cite: 12]. [cite_start]Tôi đã xuất dữ liệu định dạng YOLOv8 từ Roboflow[cite: 14]. Hãy viết một script Python giúp tôi: 1. Kiểm tra cấu trúc thư mục xem đã đúng chuẩn (train/val/test) chưa. 2. Thống kê số lượng ảnh và số lượng nhãn (bounding box) của từng lớp để đảm bảo dữ liệu không bị lệch (imbalance). 3. Vẽ thử một vài ảnh kèm bounding box để xác nhận nhãn đã khớp với ảnh. [cite_start]Code cần có comment chi tiết để tôi giải thích trong báo cáo[cite: 57]."

### 💻 Code thực thi mẫu (MVP):
[cite_start]Bạn có thể chạy đoạn code này trực tiếp trên máy local hoặc Colab để kiểm tra dataset sau khi tải về từ Roboflow[cite: 15].

```python
import os
import cv2
import matplotlib.pyplot as plt
import yaml
from glob import glob

def check_dataset_logic(data_yaml_path):
    """
    Kiểm tra logic của bộ dữ liệu dựa trên file data.yaml từ Roboflow.
    """
    # 1. Đọc file cấu hình yaml
    with open(data_yaml_path, 'r') as f:
        data_config = yaml.safe_load(f)
    
    classes = data_config['names']
    [cite_start]print(f"--- Danh sách các loài: {classes} ---") [cite: 30]

    # 2. Thống kê số lượng file
    for split in ['train', 'val', 'test']:
        img_path = os.path.join(os.path.dirname(data_yaml_path), data_config[split])
        images = glob(f"{img_path}/*.jpg") + glob(f"{img_path}/*.png")
        print(f"Số lượng ảnh tập {split}: {len(images)}")

def visualize_sample(image_path, label_path, class_names):
    """
    Vẽ thử ảnh và nhãn để kiểm tra độ khớp (Validation trực quan).
    """
    img = cv2.imread(image_path)
    h, w, _ = img.shape
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
        
    for line in lines:
        cls, x, y, nw, nh = map(float, line.split())
        # Chuyển đổi từ format YOLO (0-1) sang tọa độ pixel
        x1 = int((x - nw/2) * w)
        y1 = int((y - nh/2) * h)
        x2 = int((x + nw/2) * w)
        y2 = int((y + nh/2) * h)
        
        # Vẽ bounding box và tên lớp
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, class_names[int(cls)], (x1, y1-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Kiểm tra nhãn dữ liệu thực tế")
    plt.show()

# Lưu ý: Thay đổi đường dẫn tới file của bạn
# check_dataset_logic('path/to/data.yaml')
```

---

## 🧩 Mảnh ghép số 2: Huấn luyện & Lưu trữ (Colab + Drive)

[cite_start]Sau khi dữ liệu sạch, chúng ta sẽ đưa lên "lò luyện" Google Colab[cite: 15, 33].

### 🤖 Prompt gợi ý:
> [cite_start]"Viết script Python sử dụng thư viện `ultralytics` để huấn luyện mô hình YOLOv8n (bản nano để chạy nhanh, phù hợp MVP) trên Google Colab[cite: 16, 24]. [cite_start]Yêu cầu: 1. Kết nối với Google Drive để lưu kết quả huấn luyện (best.pt) nhằm tránh mất dữ liệu khi ngắt kết nối[cite: 33, 44]. 2. Thiết lập các thông số cơ bản: epochs=50, imgsz=640. 3. [cite_start]Viết code để sau khi train xong, hiển thị biểu đồ loss và mAP từ thư mục kết quả[cite: 36, 48]."

---

## 🧩 Mảnh ghép số 3: Web App Flask (Trình diễn kết quả)

[cite_start]Đây là phần giúp hội đồng thấy được sản phẩm thực tế thay vì chỉ là các con số khô khan[cite: 26, 35].

### 🤖 Prompt gợi ý:
> [cite_start]"Xây dựng một ứng dụng Web đơn giản bằng Flask (Python)[cite: 18, 47]. [cite_start]Giao diện bao gồm: 1. Một nút 'Upload' để người dùng tải ảnh cá Nemo hoặc Dory lên[cite: 55]. 2. Backend sử dụng mô hình YOLOv8 đã train để dự đoán. 3. [cite_start]Hiển thị ảnh kết quả đã được vẽ bounding box và độ tin cậy (confidence score) ngay trên trang web[cite: 53]. [cite_start]Code cần tối giản, dễ cài đặt cho máy cá nhân[cite: 23]."

---

## 💡 Lời khuyên để "giải thích cho sinh viên dễ hiểu" (Traceability):

[cite_start]Khi bảo vệ đồ án, thầy cô thường hỏi: **"Tại sao em chọn YOLOv8 mà không phải cái khác?"** [cite: 42]
* [cite_start]**Câu trả lời (vết lưu):** "Em chọn YOLOv8 vì nó là mô hình SOTA (State-of-the-art) tại thời điểm thực hiện, hỗ trợ tốt cả CLI và Python API, giúp việc triển khai lên Flask dễ dàng hơn so với các dòng cũ[cite: 24, 27]. [cite_start]Tuy nhiên, để khách quan, em vẫn đối sánh với Faster R-CNN (đại diện cho two-stage) để thấy sự chênh lệch về tốc độ suy luận"[cite: 25, 34].

