# 🚀 BẮT ĐẦU NGAY - HƯỚNG DẪN NHANH

> **Dành cho:** Sinh viên Hoàng Hải Anh  
> **Thời gian:** 10 phút  
> **Mục tiêu:** Upload tài liệu lên GitHub và bắt đầu học

---

## ⚡ 3 BƯỚC ĐƠN GIẢN

### BƯỚC 1: Cấu hình Git (Chỉ làm 1 lần)

```bash
# Mở Terminal (hoặc Git Bash)

# Cấu hình tên (thay "Hoang Hai Anh" bằng tên bạn)
git config --global user.name "Hoang Hai Anh"

# Cấu hình email (thay bằng email GitHub của bạn)
git config --global user.email "your-email@example.com"

# Kiểm tra
git config --list
```

### BƯỚC 2: Tạo Repository trên GitHub

1. Đăng nhập GitHub: https://github.com/hieutachi
2. Click nút **"New"** (góc trên phải)
3. Điền:
   - **Repository name:** `DATN-Marine-Detection`
   - **Description:** `Đồ án tốt nghiệp: Nhận dạng sinh vật biển`
   - Chọn **Public**
   - **KHÔNG** tick "Add a README file"
4. Click **"Create repository"**
5. **COPY** URL hiện ra (dạng: `https://github.com/hieutachi/DATN-Marine-Detection.git`)

### BƯỚC 3: Upload Files

```bash
# Trong thư mục chứa các file tài liệu

# Thêm remote (paste URL vừa copy)
git remote add origin https://github.com/hieutachi/DATN-Marine-Detection.git

# Commit
git commit -m "Initial commit: Tài liệu đồ án"

# Đổi branch thành main
git branch -M main

# Push lên GitHub
git push -u origin main

# Nhập username: hieutachi
# Nhập password: [Personal Access Token - xem hướng dẫn bên dưới]
```

---

## 🔑 Tạo Personal Access Token (PAT)

GitHub không dùng password nữa, cần PAT:

1. Truy cập: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Điền:
   - **Note:** `DATN Upload`
   - **Expiration:** 90 days
   - **Scopes:** Tick ✅ **repo**
4. Click **"Generate token"**
5. **COPY TOKEN** (chỉ hiện 1 lần, lưu vào Notepad!)
6. Dùng token này thay cho password khi push

---

## ✅ Kiểm Tra Thành Công

1. Truy cập: https://github.com/hieutachi/DATN-Marine-Detection
2. Thấy các file:
   - ✅ README.md
   - ✅ HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
   - ✅ LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
   - ✅ DATN_Outline_MVP.md
   - ✅ DATN_NoiDung_MVP.md
   - ✅ masterPlan.md
   - ✅ .gitignore

---

## 📚 BẮT ĐẦU HỌC

### Bước 1: Đọc README.md

Mở file README.md trên GitHub để hiểu tổng quan

### Bước 2: Đọc Hướng Dẫn Thực Hành

```bash
# Mở file hướng dẫn
code HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md

# Hoặc xem trên GitHub:
# https://github.com/hieutachi/DATN-Marine-Detection/blob/main/HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
```

**Làm theo từng tuần:**
- Tuần 1: Chuẩn bị môi trường
- Tuần 2: Học lý thuyết
- Tuần 3: Chuẩn bị dữ liệu
- ...

### Bước 3: Đọc Lý Thuyết

```bash
# Mở file lý thuyết
code LY_THUYET_CO_BAN_CHO_SINH_VIEN.md

# Hoặc xem trên GitHub:
# https://github.com/hieutachi/DATN-Marine-Detection/blob/main/LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
```

**Đọc để hiểu:**
- Machine Learning cơ bản
- Neural Networks & CNN
- Object Detection
- Faster R-CNN & YOLO

---

## 🎯 Lộ Trình 10 Tuần

```
┌─────────────────────────────────────────┐
│ TUẦN 1: Chuẩn bị                        │
│ - Cài Python, Git, VS Code              │
│ - Đọc tổng quan                          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 2: Học lý thuyết                   │
│ - Đọc LY_THUYET_CO_BAN_CHO_SINH_VIEN.md│
│ - Xem video về Object Detection         │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 3: Chuẩn bị dữ liệu                │
│ - Tải dataset từ Roboflow               │
│ - Kiểm tra chất lượng                   │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 4-5: Huấn luyện mô hình            │
│ - Train YOLOv8 trên Colab               │
│ - Train YOLOv5 và Faster R-CNN          │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 6-7: Xây dựng ứng dụng             │
│ - Tạo script detect.py                  │
│ - Xây dựng web app Flask                │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 8-9: Viết báo cáo                  │
│ - Dùng DATN_NoiDung_MVP.md làm mẫu     │
│ - Chèn hình ảnh, bảng biểu              │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ TUẦN 10: Chuẩn bị bảo vệ                │
│ - Làm slide PowerPoint                  │
│ - Luyện tập thuyết trình                │
└─────────────────────────────────────────┘
```

---

## 📞 Cần Giúp Đỡ?

### Gặp lỗi khi upload GitHub?

1. **Đọc:** `HUONG_DAN_UPLOAD_GITHUB.md` (chi tiết hơn)
2. **Google:** Copy lỗi vào Google
3. **Hỏi ChatGPT:** Paste lỗi và hỏi cách fix

### Không hiểu lý thuyết?

1. **Đọc lại:** `LY_THUYET_CO_BAN_CHO_SINH_VIEN.md`
2. **Xem video:** Links trong README.md
3. **Hỏi bạn:** Học nhóm hiệu quả hơn

### Không biết làm gì tiếp theo?

1. **Đọc:** `HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md`
2. **Làm theo checklist:** Tick từng mục một
3. **Đừng vội:** Làm từng bước, đừng nhảy cóc

---

## 💡 Tips Quan Trọng

### ✅ NÊN:
- Đọc kỹ hướng dẫn trước khi làm
- Test code sau mỗi thay đổi
- Commit thường xuyên lên Git
- Hỏi khi không hiểu
- Bắt đầu sớm

### ❌ KHÔNG NÊN:
- Copy code không hiểu
- Làm tất cả trong 1 ngày
- Bỏ qua phần lý thuyết
- Để đến phút chót
- Làm một mình (hãy hỏi bạn bè)

---

## 🎓 Mục Tiêu Cuối Cùng

- ✅ Hiểu sâu về Deep Learning
- ✅ Train được 3 mô hình
- ✅ Xây dựng được ứng dụng web
- ✅ Viết được báo cáo đầy đủ
- ✅ Bảo vệ thành công
- ✅ **TỐT NGHIỆP ĐÚNG HẠN!**

---

## 🚀 BẮT ĐẦU NGAY!

**Bước tiếp theo:**

1. ✅ Upload lên GitHub (làm xong rồi!)
2. 📖 Đọc `HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md` - Phần 1
3. 💻 Cài đặt Python, Git, VS Code
4. 🎓 Đọc `LY_THUYET_CO_BAN_CHO_SINH_VIEN.md` - Phần 1-2

**Thời gian:** Bắt đầu từ hôm nay!

---

**🎉 Chúc bạn Hoàng Hải Anh thành công!**

**💪 Hãy bắt đầu ngay - Đừng trì hoãn!**

---

## 📋 Quick Links

- **Repository:** https://github.com/hieutachi/DATN-Marine-Detection
- **Hướng dẫn thực hành:** [HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md](./HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md)
- **Lý thuyết:** [LY_THUYET_CO_BAN_CHO_SINH_VIEN.md](./LY_THUYET_CO_BAN_CHO_SINH_VIEN.md)
- **Nội dung báo cáo:** [DATN_NoiDung_MVP.md](./DATN_NoiDung_MVP.md)
- **Upload GitHub:** [HUONG_DAN_UPLOAD_GITHUB.md](./HUONG_DAN_UPLOAD_GITHUB.md)

---

*Cập nhật: Tháng 4/2026*
