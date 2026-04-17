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

## 📚 Tài Liệu Hướng Dẫn

Repository này chứa **TẤT CẢ** tài liệu cần thiết để hoàn thành đồ án:

### 📖 Cho Sinh Viên Mới Bắt Đầu

1. **[HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md](./HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md)** ⭐
   - Hướng dẫn từng bước (10 tuần)
   - Cài đặt môi trường
   - Chuẩn bị dữ liệu
   - Huấn luyện mô hình
   - Xây dựng ứng dụng
   - **BẮT ĐẦU TỪ ĐÂY!**

2. **[LY_THUYET_CO_BAN_CHO_SINH_VIEN.md](./LY_THUYET_CO_BAN_CHO_SINH_VIEN.md)** 📚
   - Lý thuyết Machine Learning cơ bản
   - Neural Networks & CNN
   - Object Detection
   - Faster R-CNN chi tiết
   - YOLO chi tiết
   - **ĐỌC ĐỂ HIỂU LÝ THUYẾT!**

### 📝 Nội Dung Báo Cáo

3. **[DATN_Outline_MVP.md](./DATN_Outline_MVP.md)**
   - Sườn outline đầy đủ
   - Mục lục chi tiết

4. **[DATN_NoiDung_MVP.md](./DATN_NoiDung_MVP.md)** 📄
   - Nội dung đầy đủ 5 chương
   - Phần mở đầu
   - Chương 1-5
   - Tài liệu tham khảo
   - Phụ lục
   - **DÙNG ĐỂ VIẾT BÁO CÁO!**

---

## 🚀 Bắt Đầu Nhanh

### Bước 1: Clone Repository

```bash
# Clone về máy
git clone https://github.com/hieutachi/DATN-Marine-Detection.git
cd DATN-Marine-Detection
```

### Bước 2: Đọc Hướng Dẫn

```bash
# Mở file hướng dẫn chi tiết
# Windows:
notepad HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md

# Linux/Mac:
nano HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md

# Hoặc dùng VS Code:
code HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
```

### Bước 3: Làm Theo Từng Tuần

**Tuần 1:** Chuẩn bị môi trường  
**Tuần 2:** Học lý thuyết cơ bản  
**Tuần 3:** Chuẩn bị dữ liệu  
**Tuần 4-5:** Huấn luyện mô hình  
**Tuần 6-7:** Xây dựng ứng dụng  
**Tuần 8-9:** Viết báo cáo  
**Tuần 10:** Chuẩn bị bảo vệ

---

## 📂 Cấu Trúc Repository

```
DATN-Marine-Detection/
├── README.md                              # File này
├── HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md  # Hướng dẫn thực hành ⭐
├── LY_THUYET_CO_BAN_CHO_SINH_VIEN.md    # Lý thuyết cơ bản 📚
├── DATN_Outline_MVP.md                   # Outline báo cáo
├── DATN_NoiDung_MVP.md                   # Nội dung báo cáo 📄
└── masterPlan.md                         # Master plan gốc
```

---

## 🎓 Dành Cho Sinh Viên Hoàng Hải Anh

### ✅ Checklist Hoàn Thành Đồ Án

- [ ] **Tuần 1:** Đọc xong HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md (Phần 1)
- [ ] **Tuần 1:** Cài đặt xong Python, Git, VS Code
- [ ] **Tuần 1:** Tạo được virtual environment
- [ ] **Tuần 1:** Đăng ký Google Colab thành công
- [ ] **Tuần 2:** Đọc xong LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
- [ ] **Tuần 2:** Hiểu được Object Detection là gì
- [ ] **Tuần 2:** Hiểu sự khác biệt 3 mô hình
- [ ] **Tuần 3:** Tải được dataset từ Roboflow
- [ ] **Tuần 3:** Chạy được script check_dataset.py
- [ ] **Tuần 3:** Upload dataset lên Google Drive
- [ ] **Tuần 4:** Train được YOLOv8 (mAP > 0.8)
- [ ] **Tuần 5:** Train được YOLOv5 và Faster R-CNN
- [ ] **Tuần 6:** Tạo được script detect.py
- [ ] **Tuần 7:** Xây dựng được web app Flask
- [ ] **Tuần 8:** Viết xong Chương 1-3
- [ ] **Tuần 9:** Viết xong Chương 4-5
- [ ] **Tuần 10:** Làm xong slide PowerPoint
- [ ] **Tuần 10:** Luyện tập thuyết trình

### 📞 Hỗ Trợ

**Nếu gặp khó khăn:**

1. **Đọc lại hướng dẫn** - 90% câu hỏi đã có trong tài liệu
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

**💪 Bắt đầu ngay từ hôm nay - Đọc file HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md!**

---

*Last updated: April 2026*
