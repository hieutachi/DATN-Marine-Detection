# 📤 HƯỚNG DẪN UPLOAD LÊN GITHUB

> **Dành cho:** Sinh viên Hoàng Hải Anh  
> **Mục tiêu:** Upload toàn bộ tài liệu đồ án lên GitHub cá nhân

---

## 🎯 Mục Tiêu

Upload các file sau lên GitHub repository `DATN-Marine-Detection`:
- ✅ README.md
- ✅ HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
- ✅ LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
- ✅ DATN_Outline_MVP.md
- ✅ DATN_NoiDung_MVP.md
- ✅ masterPlan.md

---

## 📋 CÁCH 1: Sử Dụng Git Command Line (Khuyến nghị)

### Bước 1: Kiểm tra Git đã cài chưa

```bash
# Mở Terminal (hoặc Git Bash trên Windows)
git --version

# Kết quả mong đợi:
# git version 2.x.x

# Nếu chưa có, tải tại: https://git-scm.com/
```

### Bước 2: Cấu hình Git (Lần đầu tiên)

```bash
# Cấu hình tên
git config --global user.name "Hoang Hai Anh"

# Cấu hình email (dùng email GitHub)
git config --global user.email "your-email@example.com"

# Kiểm tra
git config --list
```

### Bước 3: Tạo Repository trên GitHub

1. Truy cập: https://github.com/hieutachi
2. Click nút **"New"** (hoặc dấu + góc trên phải → New repository)
3. Điền thông tin:
   - **Repository name:** `DATN-Marine-Detection`
   - **Description:** `Đồ án tốt nghiệp: Hệ thống nhận dạng sinh vật biển sử dụng Deep Learning`
   - **Public** (để mọi người có thể xem)
   - ✅ **Add a README file** (BỎ TICK - vì ta đã có README.md)
   - ✅ **Add .gitignore** → Chọn **Python**
   - **License:** MIT License (hoặc None)
4. Click **"Create repository"**

### Bước 4: Initialize Git trong thư mục hiện tại

```bash
# Di chuyển đến thư mục chứa các file
cd /path/to/your/folder

# Ví dụ Windows:
# cd C:\Users\YourName\Documents\DATN

# Ví dụ Linux/Mac:
# cd ~/Documents/DATN

# Khởi tạo Git repository
git init

# Kết quả:
# Initialized empty Git repository in ...
```

### Bước 5: Thêm remote repository

```bash
# Thêm remote (thay YOUR_USERNAME bằng hieutachi)
git remote add origin https://github.com/hieutachi/DATN-Marine-Detection.git

# Kiểm tra
git remote -v

# Kết quả:
# origin  https://github.com/hieutachi/DATN-Marine-Detection.git (fetch)
# origin  https://github.com/hieutachi/DATN-Marine-Detection.git (push)
```

### Bước 6: Thêm tất cả files

```bash
# Xem trạng thái
git status

# Thêm tất cả files
git add .

# Hoặc thêm từng file cụ thể:
git add README.md
git add HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
git add LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
git add DATN_Outline_MVP.md
git add DATN_NoiDung_MVP.md
git add masterPlan.md

# Kiểm tra lại
git status
# Kết quả: Các file màu xanh (staged)
```

### Bước 7: Commit

```bash
# Commit với message
git commit -m "Initial commit: Thêm tài liệu đồ án tốt nghiệp"

# Kết quả:
# [main (root-commit) abc1234] Initial commit: Thêm tài liệu đồ án tốt nghiệp
# 6 files changed, 5000 insertions(+)
```

### Bước 8: Push lên GitHub

```bash
# Push lên branch main
git push -u origin main

# Nếu gặp lỗi "branch main doesn't exist", dùng:
git branch -M main
git push -u origin main

# Nhập username và password (hoặc Personal Access Token)
```

**⚠️ Lưu ý về Authentication:**

GitHub không còn hỗ trợ password, cần dùng **Personal Access Token (PAT)**:

1. Truy cập: https://github.com/settings/tokens
2. Click **"Generate new token"** → **"Generate new token (classic)"**
3. Điền:
   - **Note:** `DATN Upload`
   - **Expiration:** 90 days
   - **Scopes:** Tick ✅ **repo** (full control)
4. Click **"Generate token"**
5. **COPY TOKEN** (chỉ hiện 1 lần!)
6. Khi push, dùng token thay cho password

### Bước 9: Kiểm tra trên GitHub

1. Truy cập: https://github.com/hieutachi/DATN-Marine-Detection
2. Kiểm tra:
   - ✅ Có 6 files
   - ✅ README.md hiển thị đẹp
   - ✅ Commit message đúng

---

## 📋 CÁCH 2: Sử Dụng GitHub Desktop (Dễ hơn)

### Bước 1: Tải GitHub Desktop

1. Truy cập: https://desktop.github.com/
2. Tải và cài đặt
3. Đăng nhập bằng tài khoản GitHub

### Bước 2: Tạo Repository

1. Mở GitHub Desktop
2. File → **New Repository**
3. Điền:
   - **Name:** `DATN-Marine-Detection`
   - **Description:** `Đồ án tốt nghiệp: Hệ thống nhận dạng sinh vật biển`
   - **Local Path:** Chọn thư mục chứa files
   - **Git Ignore:** Python
   - **License:** MIT
4. Click **"Create Repository"**

### Bước 3: Copy Files vào Repository

1. Mở thư mục repository (GitHub Desktop → Repository → Show in Explorer)
2. Copy 6 files vào thư mục này:
   - README.md
   - HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
   - LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
   - DATN_Outline_MVP.md
   - DATN_NoiDung_MVP.md
   - masterPlan.md

### Bước 4: Commit và Push

1. Quay lại GitHub Desktop
2. Thấy danh sách files ở bên trái
3. Điền **Summary:** `Initial commit: Thêm tài liệu đồ án`
4. Click **"Commit to main"**
5. Click **"Publish repository"**
6. Chọn:
   - ✅ **Public** (để mọi người xem được)
   - ✅ **Keep this code private** (BỎ TICK)
7. Click **"Publish Repository"**

### Bước 5: Kiểm tra

1. GitHub Desktop → Repository → **View on GitHub**
2. Kiểm tra files đã upload

---

## 📋 CÁCH 3: Upload Trực Tiếp trên GitHub (Nhanh nhất)

### Bước 1: Tạo Repository

1. Truy cập: https://github.com/hieutachi
2. Click **"New"**
3. Điền thông tin như Cách 1
4. Click **"Create repository"**

### Bước 2: Upload Files

1. Trong repository mới tạo, click **"uploading an existing file"**
2. Kéo thả 6 files vào:
   - README.md
   - HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
   - LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
   - DATN_Outline_MVP.md
   - DATN_NoiDung_MVP.md
   - masterPlan.md
3. Điền **Commit message:** `Initial commit: Thêm tài liệu đồ án`
4. Click **"Commit changes"**

**⚠️ Hạn chế:** Không có lịch sử Git, khó quản lý sau này

---

## ✅ Kiểm Tra Sau Khi Upload

### 1. Kiểm tra Repository

Truy cập: https://github.com/hieutachi/DATN-Marine-Detection

**Checklist:**
- [ ] README.md hiển thị đẹp (có emoji, bảng, format đúng)
- [ ] Có đầy đủ 6 files
- [ ] Mô tả repository rõ ràng
- [ ] Repository là Public (mọi người xem được)

### 2. Kiểm tra README.md

- [ ] Tiêu đề hiển thị đúng
- [ ] Bảng kết quả hiển thị đẹp
- [ ] Links đến các file khác hoạt động
- [ ] Emoji hiển thị đúng

### 3. Kiểm tra các file Markdown

Click vào từng file:
- [ ] HUONG_DAN_CHI_TIET_CHO_SINH_VIEN.md
- [ ] LY_THUYET_CO_BAN_CHO_SINH_VIEN.md
- [ ] DATN_Outline_MVP.md
- [ ] DATN_NoiDung_MVP.md

Kiểm tra:
- [ ] Format đúng (heading, list, code block)
- [ ] Không bị lỗi hiển thị
- [ ] Dễ đọc

---

## 🔄 Cập Nhật Sau Này

### Thêm/Sửa File

```bash
# 1. Sửa file (dùng editor)

# 2. Xem thay đổi
git status

# 3. Add file đã sửa
git add filename.md

# Hoặc add tất cả:
git add .

# 4. Commit
git commit -m "Update: Thêm phần X vào file Y"

# 5. Push
git push
```

### Xóa File

```bash
# Xóa file
git rm filename.md

# Commit
git commit -m "Remove: Xóa file không cần thiết"

# Push
git push
```

### Đổi Tên File

```bash
# Đổi tên
git mv old_name.md new_name.md

# Commit
git commit -m "Rename: Đổi tên file"

# Push
git push
```

---

## 🎨 Tùy Chỉnh Repository

### 1. Thêm Topics (Tags)

1. Vào repository trên GitHub
2. Click ⚙️ **Settings** (góc phải)
3. Phần **Topics**, thêm:
   - `deep-learning`
   - `object-detection`
   - `yolo`
   - `faster-rcnn`
   - `computer-vision`
   - `graduation-project`
   - `vietnam`

### 2. Thêm Description

1. Vào repository
2. Click ⚙️ (bên cạnh About)
3. Điền:
   - **Description:** `Đồ án tốt nghiệp: Hệ thống nhận dạng sinh vật biển sử dụng Deep Learning (Faster R-CNN, YOLOv5, YOLOv8)`
   - **Website:** (để trống hoặc link demo nếu có)
   - **Topics:** (đã thêm ở trên)

### 3. Thêm .gitignore

Tạo file `.gitignore`:

```bash
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# Jupyter Notebook
.ipynb_checkpoints

# PyTorch
*.pth
*.pt

# Data
dataset/
data/
*.zip

# IDE
.vscode/
.idea/
*.swp

# OS
.DS_Store
Thumbs.db

# Output
output/
runs/
```

Upload file này lên GitHub.

---

## 📱 Chia Sẻ Repository

### Link để chia sẻ:

```
https://github.com/hieutachi/DATN-Marine-Detection
```

**Chia sẻ với:**
- ✅ Giảng viên hướng dẫn
- ✅ Bạn cùng lớp
- ✅ Hội đồng bảo vệ (trong slide)

### Tạo QR Code (Optional)

1. Truy cập: https://www.qr-code-generator.com/
2. Chọn **URL**
3. Paste: `https://github.com/hieutachi/DATN-Marine-Detection`
4. Download QR code
5. Đưa vào slide PowerPoint

---

## 🐛 Xử Lý Lỗi Thường Gặp

### Lỗi 1: "Permission denied"

**Nguyên nhân:** Chưa xác thực

**Giải pháp:**
```bash
# Dùng Personal Access Token thay vì password
# Xem hướng dẫn ở Bước 8 - Cách 1
```

### Lỗi 2: "Repository not found"

**Nguyên nhân:** URL sai hoặc repository chưa tạo

**Giải pháp:**
```bash
# Kiểm tra URL
git remote -v

# Sửa URL nếu sai
git remote set-url origin https://github.com/hieutachi/DATN-Marine-Detection.git
```

### Lỗi 3: "Failed to push"

**Nguyên nhân:** Có thay đổi trên GitHub chưa pull về

**Giải pháp:**
```bash
# Pull về trước
git pull origin main

# Giải quyết conflict (nếu có)

# Push lại
git push
```

### Lỗi 4: "Large files"

**Nguyên nhân:** File > 100MB

**Giải pháp:**
```bash
# Không upload file lớn (weights, dataset)
# Thêm vào .gitignore:
*.pth
*.pt
dataset/
```

---

## 📚 Tài Liệu Tham Khảo

- [Git Documentation](https://git-scm.com/doc)
- [GitHub Guides](https://guides.github.com/)
- [Markdown Guide](https://www.markdownguide.org/)

---

## ✅ Checklist Hoàn Thành

- [ ] Cài đặt Git
- [ ] Cấu hình Git (name, email)
- [ ] Tạo repository trên GitHub
- [ ] Upload 6 files
- [ ] Kiểm tra README.md hiển thị đúng
- [ ] Thêm Topics và Description
- [ ] Tạo .gitignore
- [ ] Chia sẻ link với giảng viên

---

**🎉 Chúc mừng! Repository của bạn đã sẵn sàng!**

**🔗 Link:** https://github.com/hieutachi/DATN-Marine-Detection

---

*Nếu gặp vấn đề, hãy Google lỗi hoặc hỏi ChatGPT/Claude!*
