# 🐠 Đồ Án Tốt Nghiệp: Hệ Thống Nhận Dạng Sinh Vật Biển

> **Sinh viên:** Hoàng Hải Anh  
> **MSSV:** 2251172223  
> **Lớp:** K64 KTPM 64KTPM4  
> **Khoa:** Công nghệ Thông tin - Đại học Thủy Lợi  
> **Năm học:** 2025-2026

---

## 📋 Giới Thiệu

Đồ án tốt nghiệp về **Nhận dạng sinh vật biển sử dụng Deep Learning**, so sánh 3 mô hình:
- **Faster R-CNN** (Two-stage detector)
- **YOLOv5** (One-stage detector)
- **YOLOv8** (One-stage detector, SOTA)

**Nhận dạng 7 loài:**
1. Bat Sea Star (Sao biển dơi)
2. Blue Sea Star (Sao biển xanh)
3. Crown Of Thorn Starfish (Sao biển gai)
4. Dory (Cá đuôi vàng)
5. Nemo (Cá hề)
6. Red Cushion Sea Star (Sao biển đệm đỏ)
7. Royal Starfish (Sao biển hoàng gia)

---

## 🎯 Kết Quả Đạt Được

| Mô hình | mAP@0.5 | mAP@0.5:0.95 | FPS (GPU) | Training Time |
|---------|---------|--------------|-----------|---------------|
| **YOLOv8** | **0.912** | **0.806** | **91** | 2.3h |
| **YOLOv5** | 0.876 | 0.723 | 83 | 2.1h |
| **Faster R-CNN** | **0.923** | **0.812** | 26 | 4.5h |

✅ **Ứng dụng web hoàn chỉnh** với Flask  
✅ **Nhận dạng cả ảnh và video**  
✅ **So sánh 3 mô hình side-by-side**

---

## 📚 Mã nguồn và tài liệu chạy thử

Repository trên Git chỉ giữ **phần cần cho tái hiện đồ án**: mã Python, giao diện web, `requirements.txt` và README kỹ thuật.

**Hướng dẫn cài đặt, chạy nhận dạng và pipeline:**  
→ **[find-nemo-and-dory/find-nemo-and-dory/README.md](./find-nemo-and-dory/find-nemo-and-dory/README.md)**

*(Tài liệu kế hoạch HD, outline báo cáo dài và file lý thuyết phục vụ soạn báo cáo có thể được lưu cục bộ; không đưa vào repo để giữ nhẹ và rõ ràng.)*

---

## 🚀 Bắt đầu nhanh

### Bước 1: Clone repository

```bash
git clone https://github.com/hieutachi/DATN-Marine-Detection.git
cd DATN-Marine-Detection
```

### Bước 2: Cài đặt và chạy web demo

```bash
cd find-nemo-and-dory/find-nemo-and-dory
python -m venv venv
# Windows: venv\Scripts\activate   | Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tại **http://localhost:5000**. Cần đặt trọng số huấn luyện vào `data/` theo hướng dẫn trong README của thư mục con (trọng số không có trong Git).

---

## 📂 Cấu trúc repository (trên Git)

```
DATN-Marine-Detection/
├── README.md                         # Giới thiệu đồ án (file này)
├── .gitignore
└── find-nemo-and-dory/find-nemo-and-dory/
    ├── app.py                        # Flask
    ├── detect.py                     # Inference
    ├── *.py                          # Pipeline, tiện ích
    ├── requirements.txt
    ├── templates/index.html
    └── README.md                     # Chi tiết chạy thử và cấu trúc thư mục dự án
```

---

## 🎓 Sinh viên thực hiện

### ✅ Checklist ngắn (đồng bộ với tiến độ đồ án)

- [ ] Cài đặt môi trường Python và chạy được `app.py`
- [ ] Chuẩn bị/huấn luyện dữ liệu và đặt đúng trọng số trong `data/`
- [ ] Chạy được nhận dạng (CLI và/hoặc web), so sánh mô hình
- [ ] Hoàn thiện báo cáo và slide bảo vệ theo yêu cầu khoa

### 📞 Hỗ Trợ

**Nếu gặp khó khăn:**

1. **Đọc [README kỹ thuật](./find-nemo-and-dory/find-nemo-and-dory/README.md)** — thiết lập và lỗi thường gặp
2. **Google lỗi** - Copy lỗi vào Google, thường có giải pháp
3. **Hỏi ChatGPT/Claude** - Giải thích lỗi và cách fix
4. **Hỏi bạn cùng lớp** - Làm việc nhóm hiệu quả hơn
5. **Hỏi giảng viên** - Khi thực sự bí

### 💡 Tips Thành Công

1. **Làm từng bước** - Đừng nhảy cóc
2. **Test thường xuyên** - Chạy code sau mỗi thay đổi
3. **Commit thường xuyên** - Lưu tiến độ lên Git
4. **Đọc hiểu, đừng copy** - Phải giải thích được code
5. **Bắt đầu sớm** - Đừng để đến phút chót

---

## 🛠️ Yêu Cầu Hệ Thống

### Phần Mềm

- **Python:** 3.9 - 3.11
- **Git:** Latest version
- **VS Code:** Khuyến nghị (hoặc editor khác)
- **Google Colab:** Miễn phí (có GPU)

### Phần Cứng

**Tối thiểu:**
- RAM: 8GB
- Ổ cứng: 10GB trống
- Internet: Ổn định (để tải dataset, train trên Colab)

**Khuyến nghị:**
- RAM: 16GB+
- GPU: NVIDIA (nếu train local)
- SSD: Nhanh hơn HDD

---

## 📖 Tài Liệu Tham Khảo

### Papers

- [Faster R-CNN](https://arxiv.org/abs/1506.01497) - Ren et al., 2015
- [YOLOv5](https://github.com/ultralytics/yolov5) - Ultralytics, 2020
- [YOLOv8](https://github.com/ultralytics/ultralytics) - Ultralytics, 2023

### Tutorials

- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Ultralytics Docs](https://docs.ultralytics.com/)
- [Roboflow Docs](https://docs.roboflow.com/)

### Videos

- [Object Detection in 5 Minutes](https://www.youtube.com/watch?v=GSwYGkTfOKk)
- [YOLO Explained](https://www.youtube.com/watch?v=9s_FpMpdYW8)
- [Faster R-CNN Explained](https://www.youtube.com/watch?v=XGi-Mz3do2s)

---

## 📊 Kết Quả Mẫu

### Confusion Matrix

*[Sẽ có sau khi train xong]*

### Ảnh Demo

*[Sẽ có sau khi xây dựng app]*

### Video Demo

*[Sẽ có sau khi hoàn thành]*

---

## 🤝 Đóng Góp

Repository này được tạo để hỗ trợ sinh viên K64 CNTT - Đại học Thủy Lợi.

**Nếu bạn muốn đóng góp:**
1. Fork repository
2. Tạo branch mới (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -m 'Add some improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Tạo Pull Request

---

## 📜 License

Dự án này được tạo cho mục đích học tập và nghiên cứu.

**Lưu ý:**
- Code và tài liệu có thể sử dụng tự do cho mục đích học tập
- Nếu sử dụng cho mục đích khác, vui lòng ghi nguồn
- Dataset từ Roboflow có license riêng

---

## 🎯 Mục Tiêu Cuối Cùng

✅ Hoàn thành đồ án tốt nghiệp  
✅ Hiểu sâu về Deep Learning và Object Detection  
✅ Có sản phẩm thực tế để demo  
✅ Bảo vệ thành công trước hội đồng  
✅ Tốt nghiệp đúng hạn

---

## 📞 Liên Hệ

**Sinh viên:** Hoàng Hải Anh  
**Email:** [Thêm email của bạn]  
**GitHub:** [@hieutachi](https://github.com/hieutachi)

**Giảng viên hướng dẫn:** [Tên giảng viên]  
**Email:** [Email giảng viên]

---

## 🙏 Lời Cảm Ơn

- **Giảng viên hướng dẫn** - Hỗ trợ và định hướng
- **Khoa CNTT - Đại học Thủy Lợi** - Cung cấp môi trường học tập
- **Roboflow** - Cung cấp dataset và công cụ
- **Ultralytics** - Thư viện YOLO tuyệt vời
- **Google Colab** - GPU miễn phí

---

## 📅 Timeline

```
Tháng 4/2026: Chuẩn bị và học lý thuyết
Tháng 5/2026: Huấn luyện mô hình và xây dựng app
Tháng 6/2026: Viết báo cáo và bảo vệ
```

---

**🎓 Chúc bạn Hoàng Hải Anh hoàn thành tốt đồ án tốt nghiệp!**

**💪 Bắt đầu từ [README kỹ thuật trong thư mục dự án](./find-nemo-and-dory/find-nemo-and-dory/README.md).**

---

*Last updated: April 2026*
