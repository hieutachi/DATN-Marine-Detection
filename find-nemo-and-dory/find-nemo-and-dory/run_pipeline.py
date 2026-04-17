"""
🚀 Pipeline Runner - Chạy toàn bộ Master Plan
==============================================
Script tiện lợi để chạy từng bước hoặc toàn bộ pipeline.

Cách dùng:
    python run_pipeline.py --step all          # Chạy toàn bộ
    python run_pipeline.py --step check        # P1: Kiểm tra dataset
    python run_pipeline.py --step compare      # P3: So sánh mô hình
    python run_pipeline.py --step report       # P5: Tạo báo cáo
    python run_pipeline.py --step web          # P4: Chạy web app

    # Với dataset path:
    python run_pipeline.py --step check --data "C:/datasets/marine-friends/data.yaml"
"""

import argparse
import subprocess
import sys
import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

# Ảnh mẫu để test (dùng thư mục example có sẵn)
EXAMPLE_DIR = BASE_DIR / "nemo_dory_example"


def run_cmd(cmd: list, desc: str):
    """Chạy lệnh và in kết quả."""
    print(f"\n{'='*60}")
    print(f"  ▶ {desc}")
    print(f"  Lệnh: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, cwd=str(BASE_DIR))
    return result.returncode == 0


def step_check(data_yaml: str = None):
    """P1: Kiểm tra dataset."""
    if data_yaml is None:
        print("\n⚠️  Bước P1 cần đường dẫn data.yaml từ Roboflow.")
        print("   Ví dụ: python run_pipeline.py --step check --data C:/datasets/marine-friends/data.yaml")
        print("   Bỏ qua bước này...\n")
        return True

    cmd = [sys.executable, "check_dataset.py",
           "--data", data_yaml,
           "--visualize",
           "--num_samples", "6",
           "--output", "output/dataset_check"]
    return run_cmd(cmd, "P1: Kiểm tra Dataset")


def step_compare():
    """P3: So sánh 3 mô hình."""
    if not EXAMPLE_DIR.exists() or not any(EXAMPLE_DIR.iterdir()):
        print(f"\n⚠️  Không tìm thấy ảnh mẫu tại: {EXAMPLE_DIR}")
        return False

    cmd = [sys.executable, "compare_models.py",
           "--image", str(EXAMPLE_DIR),
           "--conf", "0.3",
           "--output", "output/comparison"]
    return run_cmd(cmd, "P3: So sánh 3 mô hình")


def step_report():
    """P5: Tạo báo cáo số liệu."""
    cmd = [sys.executable, "report_metrics.py",
           "--output", "output/report_figures"]
    return run_cmd(cmd, "P5: Tạo báo cáo số liệu")


def step_detect():
    """Chạy detect nhanh trên ảnh mẫu."""
    if not EXAMPLE_DIR.exists():
        print(f"\n⚠️  Không tìm thấy thư mục: {EXAMPLE_DIR}")
        return False

    cmd = [sys.executable, "detect.py",
           "--image", str(EXAMPLE_DIR),
           "--model", "yolov8",
           "--conf", "0.3",
           "--output", "output/detect_results",
           "--compare"]
    return run_cmd(cmd, "Detect: Nhận dạng ảnh mẫu (cả 3 model)")


def step_web():
    """P4: Chạy web app."""
    print(f"\n{'='*60}")
    print(f"  ▶ P4: Khởi động Web App Flask")
    print(f"{'='*60}")
    print(f"\n  Mở trình duyệt tại: http://localhost:5000")
    print(f"  Nhấn Ctrl+C để dừng\n")
    os.system(f"{sys.executable} app.py")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="🚀 Pipeline Runner - Find Marine Friends"
    )
    parser.add_argument(
        "--step", "-s",
        choices=["all", "check", "compare", "report", "detect", "web"],
        default="all",
        help="Bước cần chạy (mặc định: all)"
    )
    parser.add_argument(
        "--data", "-d", default=None,
        help="Đường dẫn data.yaml (cần cho bước 'check')"
    )
    args = parser.parse_args()

    # Tạo thư mục output
    (BASE_DIR / "output").mkdir(exist_ok=True)

    print("\n🐠 FIND MARINE FRIENDS - PIPELINE RUNNER")
    print("=" * 60)

    results = {}

    if args.step in ("all", "check"):
        results["P1: Check Dataset"] = step_check(args.data)

    if args.step in ("all", "compare"):
        results["P3: Compare Models"] = step_compare()

    if args.step in ("all", "report"):
        results["P5: Report Metrics"] = step_report()

    if args.step in ("all", "detect"):
        results["Detect Sample"] = step_detect()

    if args.step == "web":
        step_web()
        return

    # Tổng kết
    if results:
        print(f"\n{'='*60}")
        print(f"  📋 KẾT QUẢ PIPELINE")
        print(f"{'='*60}")
        for step_name, success in results.items():
            status = "✅" if success else "⚠️ "
            print(f"  {status} {step_name}")

        print(f"\n  📁 Tất cả kết quả lưu tại: {(BASE_DIR / 'output').resolve()}")
        print(f"\n  Để chạy web app: python run_pipeline.py --step web\n")


if __name__ == "__main__":
    main()
