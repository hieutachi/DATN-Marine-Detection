"""
🧩 Mảnh ghép số 3: So sánh 3 mô hình (Faster R-CNN vs YOLOv5 vs YOLOv8)
=========================================================================
Script này chạy cả 3 mô hình trên cùng một bộ ảnh test và tạo báo cáo
so sánh toàn diện về:
  - Tốc độ inference (ms/ảnh, FPS)
  - Số lượng detection
  - Confidence trung bình
  - Metrics huấn luyện (Precision, Recall, mAP)

Kết quả được lưu dưới dạng:
  - Bảng CSV (comparison_results.csv)
  - Biểu đồ so sánh (comparison_chart.png)
  - Ảnh kết quả side-by-side (comparison_images/)

Cách chạy:
    python compare_models.py --image path/to/image.jpg
    python compare_models.py --image path/to/folder/ --conf 0.3
    python compare_models.py --image nemo_dory_example/ --output comparison_output/
"""

import argparse
import os
import sys
import time
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch

# Import từ detect.py (cùng thư mục)
from detect import (
    BASE_DIR, DATA_DIR, CLASS_NAMES, COLORS, DEFAULT_CONF,
    load_fasterrcnn, predict_fasterrcnn,
    load_yolov5, predict_yolov5,
    load_yolov8, predict_yolov8,
    draw_results, draw_time_on_image,
    get_training_metrics,
)


# ---------------------------------------------------------------------------
# Cấu hình
# ---------------------------------------------------------------------------

WEIGHT_MAP = {
    "fasterrcnn": DATA_DIR / "fasterrcnn_runs" / "fasterrcnn_best.pth",
    "yolov5":     DATA_DIR / "yolo5_results"   / "weights" / "best.pt",
    "yolov8":     DATA_DIR / "yolo8_results"   / "weights" / "best.pt",
}

MODEL_LABELS = {
    "fasterrcnn": "Faster R-CNN\n(Two-stage)",
    "yolov5":     "YOLOv5\n(One-stage)",
    "yolov8":     "YOLOv8\n(One-stage, SOTA)",
}


# ---------------------------------------------------------------------------
# Load tất cả model
# ---------------------------------------------------------------------------

def load_all_models(device: str) -> dict:
    """
    Load cả 3 mô hình vào bộ nhớ.

    Tại sao load trước?
      - Tránh tính thời gian load vào thời gian inference
      - Đảm bảo so sánh công bằng (chỉ đo thời gian predict)

    Args:
        device: 'cpu' hoặc 'cuda:0'
    Returns:
        dict {model_name: (model, predict_fn, load_time_ms)}
    """
    models = {}
    print(f"\n{'='*60}")
    print(f"  📦 LOADING CÁC MÔ HÌNH (device: {device})")
    print(f"{'='*60}")

    for name, weights in WEIGHT_MAP.items():
        if not weights.exists():
            print(f"  ⚠️  {name}: Không tìm thấy trọng số tại {weights}")
            continue

        print(f"  ⏳ Đang load {name}...", end=" ", flush=True)
        t_start = time.perf_counter()

        try:
            if name == "fasterrcnn":
                model = load_fasterrcnn(str(weights), num_classes=9, device=device)
                predict_fn = lambda img, conf, m=model: predict_fasterrcnn(m, img, device, conf)
            elif name == "yolov5":
                model = load_yolov5(str(weights), device=device)
                predict_fn = lambda img, conf, m=model: predict_yolov5(m, img, conf)
            else:  # yolov8
                model = load_yolov8(str(weights), device=device)
                predict_fn = lambda img, conf, m=model: predict_yolov8(m, img, device, conf)

            load_ms = (time.perf_counter() - t_start) * 1000
            models[name] = (model, predict_fn, load_ms)
            print(f"✅ ({load_ms:.0f} ms)")

        except Exception as e:
            print(f"❌ Lỗi: {e}")

    return models


# ---------------------------------------------------------------------------
# Chạy inference và thu thập số liệu
# ---------------------------------------------------------------------------

def run_comparison(image_paths: list, models: dict, conf_thres: float,
                   output_dir: Path) -> pd.DataFrame:
    """
    Chạy cả 3 mô hình trên từng ảnh và thu thập số liệu so sánh.

    Số liệu thu thập cho mỗi ảnh × mỗi model:
      - inference_ms: Thời gian xử lý (ms)
      - num_detections: Số đối tượng phát hiện được
      - avg_confidence: Confidence trung bình
      - detections_per_class: Số detection theo từng lớp

    Args:
        image_paths: Danh sách đường dẫn ảnh
        models:      Dict model từ load_all_models()
        conf_thres:  Ngưỡng confidence
        output_dir:  Thư mục lưu ảnh kết quả
    Returns:
        DataFrame chứa toàn bộ số liệu so sánh
    """
    img_out_dir = output_dir / "comparison_images"
    img_out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    model_names = list(models.keys())

    print(f"\n{'='*60}")
    print(f"  🔬 CHẠY SO SÁNH 3 MÔ HÌNH")
    print(f"  Số ảnh: {len(image_paths)} | Conf: {conf_thres} | Models: {model_names}")
    print(f"{'='*60}")

    for img_idx, img_path in enumerate(image_paths):
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  ⚠️  Không đọc được: {img_path}")
            continue

        print(f"\n  [{img_idx+1}/{len(image_paths)}] {Path(img_path).name}")

        # Lưu ảnh kết quả của từng model để ghép side-by-side
        result_imgs = {}

        for m_name, (model, predict_fn, load_ms) in models.items():
            # Đo thời gian inference (chạy 1 lần warm-up + 1 lần đo thật)
            _ = predict_fn(img, conf_thres)  # Warm-up (bỏ qua kết quả)
            t_start = time.perf_counter()
            detections = predict_fn(img, conf_thres)
            infer_ms = (time.perf_counter() - t_start) * 1000

            # Thống kê
            num_det = len(detections)
            confidences = [s for _, s, _ in detections]
            avg_conf = sum(confidences) / len(confidences) if confidences else 0.0
            max_conf = max(confidences) if confidences else 0.0

            # Đếm theo lớp
            class_counts = defaultdict(int)
            for label, _, _ in detections:
                class_counts[label] += 1

            # Vẽ kết quả
            img_result = draw_results(img, detections)
            img_result = draw_time_on_image(img_result, infer_ms, m_name)
            result_imgs[m_name] = img_result

            # Ghi số liệu
            record = {
                "image":          Path(img_path).name,
                "model":          m_name,
                "inference_ms":   round(infer_ms, 2),
                "fps":            round(1000 / infer_ms, 1) if infer_ms > 0 else 0,
                "load_ms":        round(load_ms, 1),
                "num_detections": num_det,
                "avg_confidence": round(avg_conf, 4),
                "max_confidence": round(max_conf, 4),
            }
            # Thêm số detection theo từng lớp
            for cls in CLASS_NAMES:
                record[f"det_{cls.replace(' ', '_')}"] = class_counts.get(cls, 0)

            records.append(record)

            print(f"    {m_name:<12}: {infer_ms:>7.1f} ms | {num_det} detections | conf={avg_conf:.3f}")

        # Tạo ảnh so sánh side-by-side
        if result_imgs:
            _save_comparison_image(img, result_imgs, img_path, img_out_dir)

    df = pd.DataFrame(records)
    return df


def _save_comparison_image(original: np.ndarray, result_imgs: dict,
                            img_path, output_dir: Path):
    """Ghép ảnh gốc + kết quả 3 model thành 1 ảnh side-by-side."""
    n_models = len(result_imgs)
    n_cols = n_models + 1  # +1 cho ảnh gốc
    fig, axes = plt.subplots(1, n_cols, figsize=(n_cols * 5, 5))

    # Ảnh gốc
    axes[0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Ảnh gốc", fontsize=11, fontweight="bold")
    axes[0].axis("off")

    # Kết quả từng model
    for i, (m_name, img_result) in enumerate(result_imgs.items()):
        axes[i + 1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
        axes[i + 1].set_title(MODEL_LABELS.get(m_name, m_name), fontsize=10)
        axes[i + 1].axis("off")

    plt.suptitle(f"So sánh kết quả: {Path(img_path).name}", fontsize=13, y=1.02)
    plt.tight_layout()

    save_path = output_dir / f"compare_{Path(img_path).stem}.png"
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close()


# ---------------------------------------------------------------------------
# Tạo báo cáo và biểu đồ
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, models: dict, output_dir: Path):
    """
    Tạo báo cáo tổng hợp và biểu đồ so sánh.

    Báo cáo bao gồm:
      1. Bảng tổng kết (CSV)
      2. Biểu đồ tốc độ inference
      3. Biểu đồ số lượng detection
      4. Bảng metrics huấn luyện (Precision, Recall, mAP)

    Args:
        df:         DataFrame từ run_comparison()
        models:     Dict model đã load
        output_dir: Thư mục lưu kết quả
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Tính tổng kết theo model ---
    summary = df.groupby("model").agg(
        avg_inference_ms=("inference_ms", "mean"),
        min_inference_ms=("inference_ms", "min"),
        max_inference_ms=("inference_ms", "max"),
        avg_fps=("fps", "mean"),
        total_detections=("num_detections", "sum"),
        avg_detections=("num_detections", "mean"),
        avg_confidence=("avg_confidence", "mean"),
    ).round(2)

    # Thêm thông tin load time
    for m_name, (_, _, load_ms) in models.items():
        if m_name in summary.index:
            summary.loc[m_name, "load_ms"] = round(load_ms, 1)

    # Thêm training metrics
    for m_name in summary.index:
        metrics = get_training_metrics(m_name)
        if metrics:
            summary.loc[m_name, "train_precision"] = round(metrics["precision"], 4)
            summary.loc[m_name, "train_recall"]    = round(metrics["recall"], 4)
            summary.loc[m_name, "train_mAP@0.5"]   = round(metrics["mAP@0.5"], 4)
            summary.loc[m_name, "train_mAP@0.5:0.95"] = round(metrics["mAP@0.5:0.95"], 4)
            summary.loc[m_name, "best_epoch"]      = metrics["epoch"]

    # In bảng tổng kết
    print(f"\n{'='*70}")
    print(f"  📊 BẢNG TỔNG KẾT SO SÁNH 3 MÔ HÌNH")
    print(f"{'='*70}")
    print(f"\n  {'Chỉ số':<30} {'Faster R-CNN':>14} {'YOLOv5':>14} {'YOLOv8':>14}")
    print(f"  {'-'*72}")

    metrics_to_show = [
        ("TB thời gian inference (ms)", "avg_inference_ms"),
        ("Nhanh nhất (ms)",             "min_inference_ms"),
        ("Chậm nhất (ms)",              "max_inference_ms"),
        ("Tốc độ TB (FPS)",             "avg_fps"),
        ("Thời gian load model (ms)",   "load_ms"),
        ("TB số detection/ảnh",         "avg_detections"),
        ("TB confidence",               "avg_confidence"),
        ("Precision (train)",           "train_precision"),
        ("Recall (train)",              "train_recall"),
        ("mAP@0.5 (train)",             "train_mAP@0.5"),
        ("mAP@0.5:0.95 (train)",        "train_mAP@0.5:0.95"),
    ]

    for label, col in metrics_to_show:
        row_vals = []
        for m in ["fasterrcnn", "yolov5", "yolov8"]:
            val = summary.loc[m, col] if m in summary.index and col in summary.columns else "N/A"
            row_vals.append(f"{val:>14}" if val != "N/A" else f"{'N/A':>14}")
        print(f"  {label:<30} {''.join(row_vals)}")

    # Lưu CSV
    csv_path = output_dir / "comparison_results.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    summary_path = output_dir / "comparison_summary.csv"
    summary.to_csv(summary_path, encoding="utf-8-sig")
    print(f"\n  💾 Đã lưu CSV: {csv_path}")
    print(f"  💾 Đã lưu tổng kết: {summary_path}")

    # --- Vẽ biểu đồ so sánh ---
    _plot_comparison_charts(summary, output_dir)


def _plot_comparison_charts(summary: pd.DataFrame, output_dir: Path):
    """Vẽ biểu đồ so sánh tốc độ, accuracy và training metrics."""
    model_order = [m for m in ["fasterrcnn", "yolov5", "yolov8"] if m in summary.index]
    labels = [MODEL_LABELS.get(m, m) for m in model_order]
    colors = ["#E74C3C", "#3498DB", "#2ECC71"]

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # --- Biểu đồ 1: Thời gian inference ---
    ax1 = fig.add_subplot(gs[0, 0])
    vals = [summary.loc[m, "avg_inference_ms"] if m in summary.index else 0 for m in model_order]
    bars = ax1.bar(labels, vals, color=colors[:len(model_order)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{val:.1f}ms", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax1.set_title("⏱️ Thời gian Inference TB", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Milliseconds (ms)")
    ax1.grid(axis="y", alpha=0.3)

    # --- Biểu đồ 2: FPS ---
    ax2 = fig.add_subplot(gs[0, 1])
    vals = [summary.loc[m, "avg_fps"] if m in summary.index else 0 for m in model_order]
    bars = ax2.bar(labels, vals, color=colors[:len(model_order)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.axhline(y=30, color="red", linestyle="--", alpha=0.5, label="Real-time (30 FPS)")
    ax2.set_title("🚀 Tốc độ xử lý (FPS)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Frames Per Second")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    # --- Biểu đồ 3: Số detection TB ---
    ax3 = fig.add_subplot(gs[0, 2])
    vals = [summary.loc[m, "avg_detections"] if m in summary.index else 0 for m in model_order]
    bars = ax3.bar(labels, vals, color=colors[:len(model_order)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax3.set_title("🎯 Số Detection TB/ảnh", fontsize=11, fontweight="bold")
    ax3.set_ylabel("Số lượng detection")
    ax3.grid(axis="y", alpha=0.3)

    # --- Biểu đồ 4: mAP@0.5 ---
    ax4 = fig.add_subplot(gs[1, 0])
    vals = []
    for m in model_order:
        v = summary.loc[m, "train_mAP@0.5"] if (m in summary.index and "train_mAP@0.5" in summary.columns) else 0
        vals.append(float(v) if v != "N/A" else 0)
    bars = ax4.bar(labels, vals, color=colors[:len(model_order)], alpha=0.85, edgecolor="white")
    for bar, val in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                 f"{val:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax4.set_ylim(0, 1.05)
    ax4.set_title("📈 mAP@0.5 (Training)", fontsize=11, fontweight="bold")
    ax4.set_ylabel("mAP@0.5")
    ax4.grid(axis="y", alpha=0.3)

    # --- Biểu đồ 5: Precision & Recall grouped ---
    ax5 = fig.add_subplot(gs[1, 1])
    x = np.arange(len(model_order))
    width = 0.35
    prec_vals = []
    rec_vals = []
    for m in model_order:
        p = summary.loc[m, "train_precision"] if (m in summary.index and "train_precision" in summary.columns) else 0
        r = summary.loc[m, "train_recall"] if (m in summary.index and "train_recall" in summary.columns) else 0
        prec_vals.append(float(p) if p != "N/A" else 0)
        rec_vals.append(float(r) if r != "N/A" else 0)
    ax5.bar(x - width/2, prec_vals, width, label="Precision", color="#9B59B6", alpha=0.85)
    ax5.bar(x + width/2, rec_vals, width, label="Recall", color="#E67E22", alpha=0.85)
    ax5.set_xticks(x)
    ax5.set_xticklabels(labels, fontsize=9)
    ax5.set_ylim(0, 1.05)
    ax5.set_title("🎯 Precision & Recall (Training)", fontsize=11, fontweight="bold")
    ax5.legend(fontsize=9)
    ax5.grid(axis="y", alpha=0.3)

    # --- Biểu đồ 6: Radar chart tổng hợp ---
    ax6 = fig.add_subplot(gs[1, 2], polar=True)
    categories = ["mAP@0.5", "Precision", "Recall", "FPS\n(norm)", "Conf TB"]
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    # Normalize FPS về [0,1] (max 100 FPS)
    max_fps = 100.0

    for i, m_name in enumerate(model_order):
        if m_name not in summary.index:
            continue
        row = summary.loc[m_name]
        vals_radar = [
            float(row.get("train_mAP@0.5", 0) or 0),
            float(row.get("train_precision", 0) or 0),
            float(row.get("train_recall", 0) or 0),
            min(float(row.get("avg_fps", 0) or 0) / max_fps, 1.0),
            float(row.get("avg_confidence", 0) or 0),
        ]
        vals_radar += vals_radar[:1]
        ax6.plot(angles, vals_radar, "o-", linewidth=2,
                 color=colors[i], label=m_name, alpha=0.85)
        ax6.fill(angles, vals_radar, alpha=0.1, color=colors[i])

    ax6.set_xticks(angles[:-1])
    ax6.set_xticklabels(categories, fontsize=8)
    ax6.set_ylim(0, 1)
    ax6.set_title("🕸️ Tổng hợp (Radar)", fontsize=11, fontweight="bold", pad=15)
    ax6.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

    plt.suptitle("So sánh 3 Mô hình: Faster R-CNN vs YOLOv5 vs YOLOv8",
                 fontsize=14, fontweight="bold", y=1.01)

    save_path = output_dir / "comparison_chart.png"
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  💾 Đã lưu biểu đồ so sánh: {save_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def get_image_paths(path: str) -> list:
    """Lấy danh sách ảnh từ file hoặc thư mục."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    p = Path(path)
    if p.is_file():
        return [p]
    elif p.is_dir():
        imgs = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
        return imgs
    else:
        print(f"[LỖI] Không tìm thấy: {path}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="🧩 P3 - So sánh 3 mô hình: Faster R-CNN vs YOLOv5 vs YOLOv8"
    )
    parser.add_argument("--image", "-i", required=True,
                        help="Đường dẫn ảnh hoặc thư mục ảnh test")
    parser.add_argument("--conf", "-c", type=float, default=DEFAULT_CONF,
                        help=f"Ngưỡng confidence (mặc định: {DEFAULT_CONF})")
    parser.add_argument("--device", "-d", default="",
                        help="Device: cpu hoặc 0 (GPU). Mặc định: tự động")
    parser.add_argument("--output", "-o", default="comparison_output",
                        help="Thư mục lưu kết quả (mặc định: comparison_output/)")
    parser.add_argument("--models", "-m", nargs="+",
                        default=["fasterrcnn", "yolov5", "yolov8"],
                        choices=["fasterrcnn", "yolov5", "yolov8"],
                        help="Chọn model cần so sánh (mặc định: cả 3)")
    args = parser.parse_args()

    # Xác định device
    if args.device:
        device = f"cuda:{args.device}" if args.device.isdigit() else args.device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print(f"\n🐠 SO SÁNH MÔ HÌNH - Find Marine Friends")
    print(f"  Device : {device}")
    print(f"  Models : {args.models}")
    print(f"  Conf   : {args.conf}")

    # Lấy danh sách ảnh
    image_paths = get_image_paths(args.image)
    print(f"  Ảnh    : {len(image_paths)} file(s)")

    # Load models
    all_models = load_all_models(device)
    # Lọc chỉ giữ model được chọn
    selected_models = {k: v for k, v in all_models.items() if k in args.models}

    if not selected_models:
        print("[LỖI] Không load được model nào!")
        sys.exit(1)

    # Chạy so sánh
    output_dir = Path(args.output)
    df = run_comparison(image_paths, selected_models, args.conf, output_dir)

    # Tạo báo cáo
    generate_report(df, selected_models, output_dir)

    print(f"\n✅ Hoàn tất so sánh! Kết quả lưu tại: {output_dir.resolve()}")
    print("\nCác file đã tạo:")
    print(f"  📊 {output_dir}/comparison_results.csv   - Số liệu chi tiết từng ảnh")
    print(f"  📊 {output_dir}/comparison_summary.csv   - Tổng kết theo model")
    print(f"  📈 {output_dir}/comparison_chart.png     - Biểu đồ so sánh")
    print(f"  🖼️  {output_dir}/comparison_images/      - Ảnh kết quả side-by-side\n")


if __name__ == "__main__":
    main()
