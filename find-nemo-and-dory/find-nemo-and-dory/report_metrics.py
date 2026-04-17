"""
🧩 Mảnh ghép số 5: Tạo Báo cáo Số liệu Thực nghiệm
====================================================
Script này đọc kết quả training từ các file CSV (YOLOv5, YOLOv8)
và tạo ra các biểu đồ, bảng số liệu phục vụ viết báo cáo ĐATN.

Kết quả tạo ra:
  - Biểu đồ Loss theo epoch (train/val)
  - Biểu đồ mAP theo epoch
  - Biểu đồ Precision-Recall theo epoch
  - Bảng so sánh metrics tốt nhất (LaTeX + CSV)
  - Confusion matrix (nếu có file ảnh)

Cách chạy:
    python report_metrics.py
    python report_metrics.py --output report_figures/
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Thư mục gốc dự án
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

# Đường dẫn file CSV kết quả training
RESULTS_CSV = {
    "yolov5": DATA_DIR / "yolo5_results" / "results.csv",
    "yolov8": DATA_DIR / "yolo8_results" / "results.csv",
}

# Đường dẫn ảnh confusion matrix
CONFUSION_MATRIX = {
    "yolov5": DATA_DIR / "yolo5_results" / "confusion_matrix.png",
    "yolov8": DATA_DIR / "yolo8_results" / "confusion_matrix.png",
    "fasterrcnn": DATA_DIR / "fasterrcnn_runs" / "quick_infer.jpg",
}


# ---------------------------------------------------------------------------
# Đọc và chuẩn hóa CSV
# ---------------------------------------------------------------------------

def load_training_csv(model_name: str) -> pd.DataFrame | None:
    """
    Đọc file CSV kết quả training của YOLO.

    YOLOv5 và YOLOv8 lưu metrics theo từng epoch vào results.csv.
    Tên cột có thể khác nhau giữa 2 phiên bản nên cần chuẩn hóa.

    Cột quan trọng:
      - epoch: Số thứ tự epoch
      - train/box_loss, train/cls_loss, train/dfl_loss: Loss khi train
      - val/box_loss, val/cls_loss: Loss khi validate
      - metrics/precision, metrics/recall: Precision & Recall
      - metrics/mAP_0.5, metrics/mAP_0.5:0.95: mAP

    Args:
        model_name: 'yolov5' hoặc 'yolov8'
    Returns:
        DataFrame đã chuẩn hóa hoặc None nếu không đọc được
    """
    csv_path = RESULTS_CSV.get(model_name)
    if csv_path is None or not csv_path.exists():
        print(f"  ⚠️  Không tìm thấy CSV cho {model_name}: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()  # Xóa khoảng trắng thừa

    # Chuẩn hóa tên cột về dạng thống nhất
    if model_name == "yolov5":
        rename_map = {
            "epoch":                    "epoch",
            "train/box_loss":           "train_box_loss",
            "train/obj_loss":           "train_obj_loss",
            "train/cls_loss":           "train_cls_loss",
            "val/box_loss":             "val_box_loss",
            "val/obj_loss":             "val_obj_loss",
            "val/cls_loss":             "val_cls_loss",
            "metrics/precision":        "precision",
            "metrics/recall":           "recall",
            "metrics/mAP_0.5":          "mAP50",
            "metrics/mAP_0.5:0.95":     "mAP50_95",
        }
    else:  # yolov8
        rename_map = {
            "epoch":                    "epoch",
            "train/box_loss":           "train_box_loss",
            "train/cls_loss":           "train_cls_loss",
            "train/dfl_loss":           "train_dfl_loss",
            "val/box_loss":             "val_box_loss",
            "val/cls_loss":             "val_cls_loss",
            "val/dfl_loss":             "val_dfl_loss",
            "metrics/precision(B)":     "precision",
            "metrics/recall(B)":        "recall",
            "metrics/mAP50(B)":         "mAP50",
            "metrics/mAP50-95(B)":      "mAP50_95",
        }

    # Chỉ rename các cột tồn tại
    existing_rename = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=existing_rename)

    print(f"  ✅ Đọc CSV {model_name}: {len(df)} epochs")
    return df


# ---------------------------------------------------------------------------
# Vẽ biểu đồ Loss
# ---------------------------------------------------------------------------

def plot_loss_curves(dfs: dict, output_dir: Path):
    """
    Vẽ biểu đồ Loss theo epoch cho từng model.

    Loss là hàm mục tiêu mà model cố gắng tối thiểu hóa trong quá trình train.
    Có 3 loại loss trong YOLO:
      - Box loss: Sai số vị trí bounding box
      - Cls loss: Sai số phân loại (classification)
      - Obj/DFL loss: Sai số objectness / distribution focal loss

    Biểu đồ tốt: Train loss và Val loss đều giảm dần và hội tụ.
    Dấu hiệu overfitting: Train loss tiếp tục giảm nhưng Val loss tăng.

    Args:
        dfs:        Dict {model_name: DataFrame}
        output_dir: Thư mục lưu biểu đồ
    """
    colors = {"yolov5": "#3498DB", "yolov8": "#2ECC71"}

    for model_name, df in dfs.items():
        if df is None:
            continue

        # Xác định các cột loss có trong CSV
        train_loss_cols = [c for c in ["train_box_loss", "train_cls_loss",
                                        "train_obj_loss", "train_dfl_loss"] if c in df.columns]
        val_loss_cols   = [c for c in ["val_box_loss", "val_cls_loss",
                                        "val_obj_loss", "val_dfl_loss"] if c in df.columns]

        n_plots = max(len(train_loss_cols), 1)
        fig, axes = plt.subplots(1, n_plots, figsize=(n_plots * 5, 4))
        if n_plots == 1:
            axes = [axes]

        color = colors.get(model_name, "#E74C3C")

        for i, (t_col, v_col) in enumerate(zip(train_loss_cols, val_loss_cols)):
            ax = axes[i]
            epochs = df["epoch"] if "epoch" in df.columns else range(len(df))

            ax.plot(epochs, df[t_col], color=color, linewidth=2,
                    label="Train loss", alpha=0.9)
            ax.plot(epochs, df[v_col], color=color, linewidth=2,
                    linestyle="--", label="Val loss", alpha=0.7)

            # Đánh dấu epoch tốt nhất (val loss thấp nhất)
            best_epoch = df[v_col].idxmin()
            ax.axvline(x=df.loc[best_epoch, "epoch"] if "epoch" in df.columns else best_epoch,
                       color="red", linestyle=":", alpha=0.6, label=f"Best epoch")

            loss_type = t_col.replace("train_", "").replace("_loss", "").upper()
            ax.set_title(f"{loss_type} Loss", fontsize=11)
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=9)
            ax.grid(alpha=0.3)

        plt.suptitle(f"Training & Validation Loss - {model_name.upper()}", fontsize=13, y=1.02)
        plt.tight_layout()

        save_path = output_dir / f"loss_curves_{model_name}.png"
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  💾 Biểu đồ loss: {save_path}")


# ---------------------------------------------------------------------------
# Vẽ biểu đồ mAP và Precision/Recall
# ---------------------------------------------------------------------------

def plot_map_curves(dfs: dict, output_dir: Path):
    """
    Vẽ biểu đồ mAP, Precision, Recall theo epoch.

    mAP (mean Average Precision) là metric chính để đánh giá model:
      - mAP@0.5: Tính tại IoU threshold = 0.5 (dễ hơn)
      - mAP@0.5:0.95: Trung bình tại nhiều IoU (0.5 đến 0.95, khắt khe hơn)

    Precision-Recall tradeoff:
      - Tăng confidence threshold → Precision tăng, Recall giảm
      - Giảm confidence threshold → Precision giảm, Recall tăng

    Args:
        dfs:        Dict {model_name: DataFrame}
        output_dir: Thư mục lưu biểu đồ
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    colors = {"yolov5": "#3498DB", "yolov8": "#2ECC71"}
    linestyles = {"yolov5": "-", "yolov8": "--"}

    metrics_to_plot = [
        ("mAP50",    "mAP@0.5",       axes[0]),
        ("mAP50_95", "mAP@0.5:0.95",  axes[1]),
        ("precision","Precision",      axes[2]),
    ]

    for col, title, ax in metrics_to_plot:
        for model_name, df in dfs.items():
            if df is None or col not in df.columns:
                continue
            epochs = df["epoch"] if "epoch" in df.columns else range(len(df))
            color = colors.get(model_name, "#E74C3C")
            ls = linestyles.get(model_name, "-")
            ax.plot(epochs, df[col], color=color, linewidth=2,
                    linestyle=ls, label=model_name.upper(), alpha=0.9)

            # Đánh dấu giá trị tốt nhất
            best_val = df[col].max()
            best_idx = df[col].idxmax()
            best_ep = df.loc[best_idx, "epoch"] if "epoch" in df.columns else best_idx
            ax.scatter([best_ep], [best_val], color=color, s=80, zorder=5)
            ax.annotate(f"{best_val:.3f}", (best_ep, best_val),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=8, color=color)

        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.suptitle("Metrics theo Epoch: YOLOv5 vs YOLOv8", fontsize=14, y=1.02)
    plt.tight_layout()

    save_path = output_dir / "metrics_curves.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Biểu đồ metrics: {save_path}")


# ---------------------------------------------------------------------------
# Bảng so sánh tổng hợp
# ---------------------------------------------------------------------------

def generate_comparison_table(dfs: dict, output_dir: Path):
    """
    Tạo bảng so sánh metrics tốt nhất của các model.

    Bảng này dùng trực tiếp trong báo cáo ĐATN, chương Thực nghiệm.
    Xuất ra cả CSV (để chỉnh sửa) và LaTeX (để paste vào báo cáo).

    Args:
        dfs:        Dict {model_name: DataFrame}
        output_dir: Thư mục lưu bảng
    """
    rows = []

    for model_name, df in dfs.items():
        if df is None:
            continue

        # Tìm epoch có mAP@0.5 tốt nhất
        if "mAP50" not in df.columns:
            continue
        best_idx = df["mAP50"].idxmax()
        row = df.iloc[best_idx]

        rows.append({
            "Mô hình":          model_name.upper(),
            "Epoch tốt nhất":   int(row.get("epoch", best_idx)),
            "Precision":        f"{row.get('precision', 0):.4f}",
            "Recall":           f"{row.get('recall', 0):.4f}",
            "mAP@0.5":          f"{row.get('mAP50', 0):.4f}",
            "mAP@0.5:0.95":     f"{row.get('mAP50_95', 0):.4f}",
        })

    # Thêm Faster R-CNN (không có CSV, điền thủ công nếu có)
    rows.append({
        "Mô hình":          "FASTER R-CNN",
        "Epoch tốt nhất":   "N/A",
        "Precision":        "N/A",
        "Recall":           "N/A",
        "mAP@0.5":          "N/A",
        "mAP@0.5:0.95":     "N/A",
    })

    if not rows:
        print("  ⚠️  Không có dữ liệu để tạo bảng so sánh")
        return

    comparison_df = pd.DataFrame(rows)

    # In bảng ra terminal
    print(f"\n{'='*70}")
    print(f"  📋 BẢNG SO SÁNH METRICS TỐT NHẤT")
    print(f"{'='*70}")
    print(comparison_df.to_string(index=False))

    # Lưu CSV
    csv_path = output_dir / "best_metrics_comparison.csv"
    comparison_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n  💾 Bảng CSV: {csv_path}")

    # Xuất LaTeX (dùng paste vào báo cáo Word/LaTeX)
    latex_path = output_dir / "best_metrics_comparison.tex"
    latex_content = _df_to_latex(comparison_df)
    with open(latex_path, "w", encoding="utf-8") as f:
        f.write(latex_content)
    print(f"  💾 Bảng LaTeX: {latex_path}")

    return comparison_df


def _df_to_latex(df: pd.DataFrame) -> str:
    """Chuyển DataFrame thành bảng LaTeX."""
    cols = list(df.columns)
    col_fmt = "l" + "c" * (len(cols) - 1)

    lines = [
        "% Bảng so sánh metrics - Tự động tạo bởi report_metrics.py",
        "\\begin{table}[h]",
        "\\centering",
        "\\caption{So sánh kết quả huấn luyện các mô hình}",
        "\\label{tab:model_comparison}",
        f"\\begin{{tabular}}{{{col_fmt}}}",
        "\\hline",
        " & ".join([f"\\textbf{{{c}}}" for c in cols]) + " \\\\",
        "\\hline",
    ]

    for _, row in df.iterrows():
        lines.append(" & ".join([str(v) for v in row.values]) + " \\\\")

    lines += [
        "\\hline",
        "\\end{tabular}",
        "\\end{table}",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ghép confusion matrix
# ---------------------------------------------------------------------------

def compile_confusion_matrices(output_dir: Path):
    """
    Ghép các ảnh confusion matrix của các model thành 1 ảnh so sánh.

    Confusion matrix cho thấy:
      - Mô hình nhầm lẫn giữa các lớp nào nhiều nhất
      - Lớp nào được nhận dạng tốt (đường chéo cao)
      - Lớp nào hay bị nhầm (ngoài đường chéo cao)

    Args:
        output_dir: Thư mục lưu ảnh ghép
    """
    import cv2

    available = {}
    for model_name, img_path in CONFUSION_MATRIX.items():
        if img_path.exists():
            # Dùng np.fromfile để tránh lỗi Unicode trong đường dẫn (Windows)
            img_array = np.fromfile(str(img_path), dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img is not None:
                available[model_name] = img

    if not available:
        print("  ⚠️  Không tìm thấy ảnh confusion matrix")
        return

    # Resize về cùng chiều cao
    target_h = 400
    resized = {}
    for name, img in available.items():
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized[name] = cv2.resize(img, (new_w, target_h))

    # Ghép ngang
    n = len(resized)
    fig, axes = plt.subplots(1, n, figsize=(n * 6, 5))
    if n == 1:
        axes = [axes]

    for i, (name, img) in enumerate(resized.items()):
        axes[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        axes[i].set_title(f"Confusion Matrix\n{name.upper()}", fontsize=11)
        axes[i].axis("off")

    plt.suptitle("Confusion Matrix - So sánh các mô hình", fontsize=13)
    plt.tight_layout()

    save_path = output_dir / "confusion_matrices_combined.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Confusion matrices: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="🧩 P5 - Tạo báo cáo số liệu thực nghiệm cho ĐATN"
    )
    parser.add_argument(
        "--output", "-o", default="report_figures",
        help="Thư mục lưu biểu đồ và bảng (mặc định: report_figures/)"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n🐠 TẠO BÁO CÁO SỐ LIỆU - Find Marine Friends")
    print("=" * 60)

    # Đọc CSV training
    print("\n  📂 Đọc dữ liệu training...")
    dfs = {}
    for model_name in ["yolov5", "yolov8"]:
        dfs[model_name] = load_training_csv(model_name)

    # Vẽ biểu đồ loss
    print("\n  📈 Vẽ biểu đồ Loss...")
    plot_loss_curves(dfs, output_dir)

    # Vẽ biểu đồ mAP
    print("\n  📈 Vẽ biểu đồ mAP & Metrics...")
    plot_map_curves(dfs, output_dir)

    # Tạo bảng so sánh
    print("\n  📋 Tạo bảng so sánh...")
    generate_comparison_table(dfs, output_dir)

    # Ghép confusion matrix
    print("\n  🔲 Ghép Confusion Matrix...")
    compile_confusion_matrices(output_dir)

    print(f"\n✅ Hoàn tất! Tất cả biểu đồ và bảng lưu tại: {output_dir.resolve()}")
    print("\nCác file đã tạo:")
    for f in sorted(output_dir.iterdir()):
        print(f"  📄 {f.name}")
    print()
    print("Gợi ý sử dụng trong báo cáo ĐATN:")
    print("  - loss_curves_*.png      → Chương 4: Quá trình huấn luyện")
    print("  - metrics_curves.png     → Chương 4: Kết quả training")
    print("  - best_metrics_comparison.csv → Bảng 4.x: So sánh mô hình")
    print("  - best_metrics_comparison.tex → Paste vào LaTeX")
    print("  - confusion_matrices_combined.png → Phân tích lỗi\n")


if __name__ == "__main__":
    main()
