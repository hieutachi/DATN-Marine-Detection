"""
Tự động phát hiện GPU NVIDIA và cài đặt PyTorch CUDA phù hợp.

Cách dùng:
    python setup_gpu.py          # Tự detect và cài
    python setup_gpu.py --check  # Chỉ kiểm tra, không cài
"""
import subprocess
import sys
import re
import argparse


def get_nvidia_cuda_version():
    """Lấy CUDA version từ nvidia-smi."""
    try:
        result = subprocess.run(
            ["nvidia-smi"], capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return None, None
        
        output = result.stdout
        # Parse GPU name
        gpu_name = None
        for line in output.splitlines():
            match = re.search(r"\|\s+\d+\s+(.+?)\s+(WDDM|TCC)", line)
            if match:
                gpu_name = match.group(1).strip()
                break
        
        # Parse CUDA version
        cuda_match = re.search(r"CUDA Version:\s+(\d+\.\d+)", output)
        cuda_version = float(cuda_match.group(1)) if cuda_match else None
        
        return gpu_name, cuda_version
    except FileNotFoundError:
        return None, None
    except Exception as e:
        print(f"[LỖI] nvidia-smi: {e}")
        return None, None


def get_pytorch_info():
    """Lấy thông tin PyTorch hiện tại."""
    try:
        import torch
        return {
            "version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except ImportError:
        return None


def pick_pytorch_cuda(driver_cuda_version):
    """
    Chọn phiên bản CUDA phù hợp cho PyTorch dựa trên driver CUDA version.
    
    Driver CUDA version >= PyTorch CUDA version là OK (backward compatible).
    VD: Driver CUDA 13.1 chạy được PyTorch cu118, cu121, cu124, cu128.
    """
    # Các phiên bản CUDA mà PyTorch hỗ trợ (từ mới → cũ)
    cuda_options = [
        (12.8, "cu128"),
        (12.6, "cu126"),
        (12.4, "cu124"),
        (12.1, "cu121"),
        (11.8, "cu118"),
    ]
    
    for min_ver, tag in cuda_options:
        if driver_cuda_version >= min_ver:
            return tag, min_ver
    
    return None, None


def install_pytorch(cuda_tag):
    """Cài đặt PyTorch với CUDA tag chỉ định."""
    url = f"https://download.pytorch.org/whl/{cuda_tag}"
    cmd = [
        sys.executable, "-m", "pip", "install",
        "torch", "torchvision",
        "--force-reinstall",
        "--index-url", url,
    ]
    print(f"\n[CÀI ĐẶT] Đang chạy:")
    print(f"  pip install torch torchvision --force-reinstall --index-url {url}")
    print(f"  (Có thể mất vài phút...)\n")
    
    result = subprocess.run(cmd, timeout=600)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Tự động cài PyTorch CUDA phù hợp")
    parser.add_argument("--check", action="store_true", help="Chỉ kiểm tra, không cài đặt")
    args = parser.parse_args()
    
    print("=" * 60)
    print("  🔍 KIỂM TRA GPU & PYTORCH")
    print("=" * 60)
    
    # 1. Check nvidia-smi
    gpu_name, driver_cuda = get_nvidia_cuda_version()
    
    if gpu_name is None:
        print("\n❌ Không tìm thấy GPU NVIDIA (nvidia-smi không hoạt động).")
        print("   → Dùng PyTorch CPU: pip install torch torchvision")
        return
    
    print(f"\n  GPU:          {gpu_name}")
    print(f"  Driver CUDA:  {driver_cuda}")
    
    # 2. Check PyTorch hiện tại
    pt_info = get_pytorch_info()
    if pt_info:
        is_cpu = "+cpu" in pt_info["version"] or not pt_info["cuda_available"]
        print(f"\n  PyTorch:      {pt_info['version']}")
        print(f"  CUDA dùng:    {pt_info['cuda_version'] or 'Không (CPU-only)'}")
        if pt_info["cuda_available"]:
            print(f"  GPU PyTorch:  {pt_info['gpu_name']}")
    else:
        print("\n  PyTorch:      Chưa cài đặt")
        is_cpu = True
    
    # 3. Chọn CUDA version phù hợp
    cuda_tag, cuda_ver = pick_pytorch_cuda(driver_cuda)
    
    if cuda_tag is None:
        print(f"\n❌ Driver CUDA {driver_cuda} quá cũ. Cần ít nhất CUDA 11.8.")
        print("   → Cập nhật driver NVIDIA: https://www.nvidia.com/drivers")
        return
    
    print(f"\n  Khuyến nghị:  PyTorch {cuda_tag} (CUDA {cuda_ver})")
    
    # 4. Check nếu đã OK
    if pt_info and pt_info["cuda_available"] and f"+{cuda_tag}" in pt_info["version"]:
        print(f"\n✅ PyTorch đã cài đúng phiên bản CUDA ({cuda_tag}). Không cần thay đổi.")
        return
    
    if is_cpu:
        print(f"\n⚠️  PyTorch hiện tại là CPU-only → Cần cài lại với CUDA {cuda_tag}")
    elif pt_info and pt_info["cuda_available"]:
        print(f"\n⚠️  PyTorch hiện tại dùng CUDA {pt_info['cuda_version']} → Có thể nâng lên {cuda_tag}")
    
    print("=" * 60)
    
    if args.check:
        print(f"\n📋 Để cài đặt, chạy:")
        print(f"   pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/{cuda_tag}")
        return
    
    # 5. Hỏi user
    answer = input(f"\nCài PyTorch {cuda_tag}? (y/n): ").strip().lower()
    if answer != "y":
        print("Đã bỏ qua.")
        return
    
    # 6. Cài đặt
    success = install_pytorch(cuda_tag)
    
    if success:
        print("\n✅ Cài đặt thành công!")
        # Verify
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; print(f'PyTorch {torch.__version__}'); "
             "print(f'CUDA: {torch.cuda.is_available()}'); "
             "print(f'GPU: {torch.cuda.get_device_name(0)}' if torch.cuda.is_available() else '')"],
            capture_output=True, text=True
        )
        print(result.stdout)
    else:
        print("\n❌ Cài đặt thất bại. Thử chạy thủ công:")
        print(f"   pip install torch torchvision --force-reinstall --index-url https://download.pytorch.org/whl/{cuda_tag}")


if __name__ == "__main__":
    main()
