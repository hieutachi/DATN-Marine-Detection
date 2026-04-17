"""
🧩 Mảnh ghép số 1: Kiểm tra & Trực quan hóa Dataset
=====================================================
Script này thực hiện 3 nhiệm vụ:
  1. Kiểm tra cấu trúc thư mục dataset (train/val/test)
  2. Thống kê số lượng ảnh và bounding box từng lớp (phát hiện imbalance)
  3. Vẽ thử ảnh kèm bounding box để xác nhận nhãn khớp với ảnh

Cách chạy:
    python check_dataset.py --data path/to/data.yaml
    python check_dataset.py --data path/to/data.yaml --visualize --num_samples 5

Ví dụ:
    python check_dataset.py --data "C:/datasets/marine-friends/data.yaml"
"""

import argparse
import os
import sys
from pathlib import Path
from glob import glob
from collections import defaultdict

import cv2
import matplotlib
matplotlib.use("Agg")  # Không cần màn hình (headless), lưu file thay vì hiển thị
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import yaml


# ---------------------------------------------------------------------------
# 1. Kiểm tra cấu trúc thư mục
# ---------------------------------------------------------------------------

def check_folder_structure(data_yaml_path: str) -> dict:
    """
    Kiểm tra cấu trúc thư mục dataset theo chuẩn YOLO.

    Chuẩn YOLO yêu cầu:
      - File data.yaml chứa đường dẫn train/val/test và danh sách lớp
      - Mỗi split có thư mục images/ và labels/ tương ứng
      - File nhãn .txt cùng tên với file ảnh

    Args:
        data_yaml_path: Đường dẫn tới file data.yaml từ Roboflow
    Returns:
        dict chứa thông tin cấu trúc và số lượng file
    """
    yaml_path = Path(data_yaml_path)
    if not yaml_path.exists():
        print(f"[LỖI] Không tìm thấy file: {data_yaml_path}")
        sys.exit(1)

    # Đọc file cấu hình YAML
    with open(yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n" + "=" * 60)
    print("  📁 KIỂM TRA CẤU TRÚC DATASET")
    print("=" * 60)
    print(f"  File config : {yaml_path}")
    print(f"  Số lớp      : {config.get('nc', '?')}")
    print(f"  Tên lớp     : {config.get('names', [])}")
    print()

    classes = config.get("names", [])
    base_dir = yaml_path.parent  # Thư mục chứa data.yaml

    result = {
        "classes": classes,
        "splits": {},
        "config": config,
        "base_dir": str(base_dir),
    }

    # Kiểm tra từng split: train, val, test
    for split in ["train", "val", "test"]:
        split_path_raw = config.get(split, None)
        if split_path_raw is None:
            print(f"  ⚠️  Split '{split}': Không có trong data.yaml (bỏ qua)")
            continue

        # Xử lý đường dẫn tương đối hoặc tuyệt đối
        split_path = Path(split_path_raw)
        if not split_path.is_absolute():
            split_path = base_dir / split_path

        # Tìm thư mục ảnh
        img_dir = split_path if split_path.is_dir() else split_path.parent / "images"
        lbl_dir = str(img_dir).replace("images", "labels")
        lbl_dir = Path(lbl_dir)

        # Đếm số ảnh
        img_exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"]
        images = []
        for ext in img_exts:
            images.extend(glob(str(img_dir / ext)))
            images.extend(glob(str(img_dir / ext.upper())))

        # Đếm số file nhãn
        labels = glob(str(lbl_dir / "*.txt")) if lbl_dir.exists() else []

        # Kiểm tra tính nhất quán ảnh ↔ nhãn
        img_stems = {Path(p).stem for p in images}
        lbl_stems = {Path(p).stem for p in labels}
        missing_labels = img_stems - lbl_stems  # Ảnh không có nhãn
        missing_images = lbl_stems - img_stems  # Nhãn không có ảnh

        status = "✅" if img_dir.exists() else "❌"
        print(f"  {status} Split '{split}':")
        print(f"      Thư mục ảnh  : {img_dir}")
        print(f"      Thư mục nhãn : {lbl_dir}")
        print(f"      Số ảnh       : {len(images)}")
        print(f"      Số nhãn      : {len(labels)}")
        if missing_labels:
            print(f"      ⚠️  Ảnh thiếu nhãn: {len(missing_labels)} file")
        if missing_images:
            print(f"      ⚠️  Nhãn thiếu ảnh: {len(missing_images)} file")
        if not missing_labels and not missing_images and len(images) > 0:
            print(f"      ✅ Ảnh và nhãn khớp hoàn toàn")
        print()

        result["splits"][split] = {
            "img_dir": str(img_dir),
            "lbl_dir": str(lbl_dir),
            "num_images": len(images),
            "num_labels": len(labels),
            "images": images,
            "labels": labels,
        }

    return result


# ---------------------------------------------------------------------------
# 2. Thống kê phân phối nhãn (phát hiện class imbalance)
# ---------------------------------------------------------------------------

def count_class_distribution(dataset_info: dict) -> dict:
    """
    Thống kê số lượng bounding box của từng lớp trong toàn bộ dataset.

    Class Imbalance là vấn đề phổ biến trong Object Detection:
      - Nếu một lớp có quá ít mẫu, model sẽ học kém lớp đó
      - Tỉ lệ chênh lệch > 10:1 thường cần xử lý (augmentation, oversampling)

    Format nhãn YOLO: mỗi dòng = "class_id x_center y_center width height"
    Tất cả giá trị được normalize về [0, 1] so với kích thước ảnh.

    Args:
        dataset_info: Kết quả từ check_folder_structure()
    Returns:
        dict {split: {class_name: count}}
    """
    classes = dataset_info["classes"]
    distribution = {}

    print("=" * 60)
    print("  📊 THỐNG KÊ PHÂN PHỐI NHÃN (CLASS DISTRIBUTION)")
    print("=" * 60)

    total_per_class = defaultdict(int)  # Tổng toàn bộ dataset

    for split, info in dataset_info["splits"].items():
        lbl_dir = Path(info["lbl_dir"])
        if not lbl_dir.exists():
            continue

        class_counts = defaultdict(int)
        total_boxes = 0

        # Đọc từng file nhãn và đếm số bounding box mỗi lớp
        for lbl_file in info["labels"]:
            with open(lbl_file, "r") as f:
                lines = f.readlines()
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) >= 5:
                    cls_id = int(parts[0])
                    if cls_id < len(classes):
                        class_counts[classes[cls_id]] += 1
                        total_per_class[classes[cls_id]] += 1
                        total_boxes += 1

        distribution[split] = dict(class_counts)

        print(f"\n  Split: {split.upper()} (tổng {total_boxes} bounding boxes)")
        print(f"  {'Lớp':<35} {'Số BB':>8} {'Tỉ lệ':>8}")
        print(f"  {'-'*55}")
        for cls_name in classes:
            count = class_counts.get(cls_name, 0)
            ratio = count / total_boxes * 100 if total_boxes > 0 else 0
            bar = "█" * int(ratio / 2)  # Thanh trực quan
            print(f"  {cls_name:<35} {count:>8}  {ratio:>6.1f}%  {bar}")

    # Kiểm tra imbalance
    if total_per_class:
        max_count = max(total_per_class.values())
        min_count = min(total_per_class.values())
        ratio = max_count / min_count if min_count > 0 else float("inf")

        print(f"\n  {'─'*55}")
        print(f"  Tổng toàn bộ dataset:")
        for cls_name in classes:
            print(f"    {cls_name:<35} {total_per_class.get(cls_name, 0):>6}")
        print(f"\n  Tỉ lệ chênh lệch max/min: {ratio:.1f}x")
        if ratio > 10:
            print(f"  ⚠️  CẢNH BÁO: Dữ liệu bị lệch nghiêm trọng (>{ratio:.0f}x)!")
            print(f"     → Cần augmentation hoặc oversampling cho lớp thiểu số")
        elif ratio > 3:
            print(f"  ⚠️  Dữ liệu hơi lệch ({ratio:.1f}x) - cần theo dõi")
        else:
            print(f"  ✅ Phân phối dữ liệu tương đối cân bằng")

    return distribution


# ---------------------------------------------------------------------------
# 3. Vẽ ảnh kèm bounding box để kiểm tra trực quan
# ---------------------------------------------------------------------------

def visualize_samples(dataset_info: dict, num_samples: int = 4,
                      split: str = "train", output_dir: str = "dataset_check"):
    """
    Vẽ thử một số ảnh kèm bounding box để xác nhận nhãn khớp với ảnh.

    Đây là bước quan trọng để phát hiện:
      - Nhãn bị lệch (offset) so với vị trí thật của đối tượng
      - Nhãn sai lớp (nhầm Nemo thành Dory)
      - Bounding box quá lớn hoặc quá nhỏ

    Args:
        dataset_info: Kết quả từ check_folder_structure()
        num_samples:  Số ảnh muốn vẽ
        split:        Tập dữ liệu cần kiểm tra ('train', 'val', 'test')
        output_dir:   Thư mục lưu ảnh kết quả
    """
    classes = dataset_info["classes"]
    split_info = dataset_info["splits"].get(split)

    if split_info is None or len(split_info["images"]) == 0:
        print(f"  [CẢNH BÁO] Không có ảnh trong split '{split}'")
        return

    # Màu sắc cho từng lớp (tự động tạo từ colormap)
    cmap = plt.cm.get_cmap("tab10", len(classes))
    colors = [cmap(i) for i in range(len(classes))]

    # Chọn ngẫu nhiên num_samples ảnh
    import random
    random.seed(42)  # Seed cố định để kết quả tái lập được
    sample_images = random.sample(
        split_info["images"],
        min(num_samples, len(split_info["images"]))
    )

    # Tạo thư mục output
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  🖼️  TRỰC QUAN HÓA NHÃN - Split: {split.upper()}")
    print(f"{'='*60}")

    # Vẽ từng ảnh
    cols = min(2, len(sample_images))
    rows = (len(sample_images) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 6))
    if len(sample_images) == 1:
        axes = [[axes]]
    elif rows == 1:
        axes = [axes]

    for idx, img_path in enumerate(sample_images):
        row, col = divmod(idx, cols)
        ax = axes[row][col] if rows > 1 else axes[0][col]

        # Đọc ảnh
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            ax.set_title(f"Không đọc được: {Path(img_path).name}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        ax.imshow(img_rgb)
        ax.set_title(f"{Path(img_path).name}\n({w}×{h}px)", fontsize=9)
        ax.axis("off")

        # Tìm file nhãn tương ứng
        lbl_path = Path(split_info["lbl_dir"]) / (Path(img_path).stem + ".txt")
        if not lbl_path.exists():
            ax.set_title(ax.get_title() + "\n[Không có nhãn]", fontsize=9, color="red")
            continue

        # Đọc và vẽ từng bounding box
        with open(lbl_path, "r") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue

            cls_id = int(parts[0])
            # Chuyển từ YOLO format (normalized) sang tọa độ pixel
            # YOLO: x_center, y_center, width, height (tất cả [0,1])
            x_c, y_c, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            x1 = (x_c - bw / 2) * w
            y1 = (y_c - bh / 2) * h
            box_w = bw * w
            box_h = bh * h

            color = colors[cls_id % len(colors)]
            cls_name = classes[cls_id] if cls_id < len(classes) else f"cls_{cls_id}"

            # Vẽ bounding box
            rect = patches.Rectangle(
                (x1, y1), box_w, box_h,
                linewidth=2, edgecolor=color, facecolor="none"
            )
            ax.add_patch(rect)

            # Vẽ nhãn
            ax.text(
                x1, y1 - 4, cls_name,
                fontsize=8, color="white",
                bbox=dict(boxstyle="round,pad=0.2", facecolor=color, alpha=0.8)
            )

        print(f"  ✅ {Path(img_path).name}: {len(lines)} bounding box(es)")

    # Ẩn các subplot trống
    for idx in range(len(sample_images), rows * cols):
        row, col = divmod(idx, cols)
        ax = axes[row][col] if rows > 1 else axes[0][col]
        ax.axis("off")

    plt.suptitle(f"Kiểm tra nhãn dataset - Split: {split.upper()}", fontsize=14, y=1.02)
    plt.tight_layout()

    # Lưu ảnh
    save_path = out_dir / f"label_check_{split}.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\n  💾 Đã lưu ảnh kiểm tra: {save_path}")


# ---------------------------------------------------------------------------
# 4. Vẽ biểu đồ phân phối lớp
# ---------------------------------------------------------------------------

def plot_class_distribution(distribution: dict, classes: list, output_dir: str = "dataset_check"):
    """
    Vẽ biểu đồ cột so sánh số lượng bounding box mỗi lớp giữa các split.

    Biểu đồ này giúp:
      - Nhìn thấy ngay lớp nào bị thiếu dữ liệu
      - So sánh tỉ lệ train/val/test có hợp lý không (thường 70/20/10)

    Args:
        distribution: Kết quả từ count_class_distribution()
        classes:      Danh sách tên lớp
        output_dir:   Thư mục lưu biểu đồ
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = list(distribution.keys())
    x = np.arange(len(classes))
    width = 0.8 / max(len(splits), 1)

    fig, ax = plt.subplots(figsize=(max(12, len(classes) * 1.5), 6))

    colors_bar = ["#2196F3", "#4CAF50", "#FF9800"]
    for i, split in enumerate(splits):
        counts = [distribution[split].get(cls, 0) for cls in classes]
        offset = (i - len(splits) / 2 + 0.5) * width
        bars = ax.bar(x + offset, counts, width, label=split.capitalize(),
                      color=colors_bar[i % len(colors_bar)], alpha=0.85)
        # Ghi số lên đầu mỗi cột
        for bar, count in zip(bars, counts):
            if count > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        str(count), ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("Lớp đối tượng", fontsize=12)
    ax.set_ylabel("Số lượng Bounding Box", fontsize=12)
    ax.set_title("Phân phối nhãn theo lớp và split\n(Class Distribution)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    save_path = out_dir / "class_distribution.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu biểu đồ phân phối: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🧩 P1 - Kiểm tra và trực quan hóa dataset YOLO"
    )
    parser.add_argument(
        "--data", "-d", required=True,
        help="Đường dẫn tới file data.yaml (từ Roboflow)"
    )
    parser.add_argument(
        "--visualize", "-v", action="store_true",
        help="Vẽ ảnh mẫu kèm bounding box để kiểm tra trực quan"
    )
    parser.add_argument(
        "--num_samples", "-n", type=int, default=4,
        help="Số ảnh mẫu cần vẽ (mặc định: 4)"
    )
    parser.add_argument(
        "--split", "-s", default="train",
        choices=["train", "val", "test"],
        help="Split cần trực quan hóa (mặc định: train)"
    )
    parser.add_argument(
        "--output", "-o", default="dataset_check",
        help="Thư mục lưu kết quả (mặc định: dataset_check/)"
    )
    args = parser.parse_args()

    print("\n🐠 KIỂM TRA DATASET - Find Marine Friends")
    print("=" * 60)

    # Bước 1: Kiểm tra cấu trúc thư mục
    dataset_info = check_folder_structure(args.data)

    # Bước 2: Thống kê phân phối nhãn
    distribution = count_class_distribution(dataset_info)

    # Vẽ biểu đồ phân phối
    plot_class_distribution(distribution, dataset_info["classes"], args.output)

    # Bước 3: Trực quan hóa nhãn (nếu được yêu cầu)
    if args.visualize:
        visualize_samples(
            dataset_info,
            num_samples=args.num_samples,
            split=args.split,
            output_dir=args.output
        )

    print(f"\n✅ Hoàn tất kiểm tra dataset! Kết quả lưu tại: {Path(args.output).resolve()}")
    print("\nGợi ý bước tiếp theo:")
    print("  → Nếu dataset OK: chạy train trên Colab (xem Colab Notebooks/)")
    print("  → Nếu có imbalance: thêm augmentation trong data.yaml")
    print("  → Nếu nhãn sai: sửa lại trên Roboflow rồi export lại\n")


if __name__ == "__main__":
    main()
