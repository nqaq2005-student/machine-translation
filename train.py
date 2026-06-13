import sys
import subprocess
import torch

def main():
    print("=" * 50)
    print("🤖 TRẠM ĐIỀU PHỐI HUẤN LUYỆN (TRAINING LAUNCHER)")
    print("=" * 50)

    # 1. Nhận diện số lượng GPU
    num_gpus = torch.cuda.device_count()

    if num_gpus == 0:
        print("⚠️ Cảnh báo: Không tìm thấy GPU nào! Hệ thống sẽ chạy trên CPU.")
        num_gpus = 1  # Coi như chạy single pipeline trên CPU

    else:
        gpu_names = [torch.cuda.get_device_name(i) for i in range(num_gpus)]
        print(f"🔍 Tìm thấy {num_gpus} GPU(s):")
        for i, name in enumerate(gpu_names):
            print(f"   - GPU {i}: {name}")

    print("-" * 50)

    # 2. Thu thập các tham số truyền thêm từ Terminal (nếu có)
    # Ví dụ: python run.py --batch_size 64 -> forwarded_args = ['--batch_size', '64']
    forwarded_args = sys.argv[1:]

    # 3. Rẽ nhánh logic xây dựng câu lệnh
    if num_gpus <= 1:
        print("🚀 KÍCH HOẠT CHẾ ĐỘ: ĐƠN GPU (Single-GPU)")
        # Lệnh: python train_single_gpu.py [args]
        cmd = [sys.executable, "train_single_gpu.py"] + forwarded_args
    else:
        print(f"🚀 KÍCH HOẠT CHẾ ĐỘ: ĐA GPU (Multi-GPU Distributed Data Parallel)")
        # Lệnh: torchrun --nproc_per_node=X train_multi_gpu.py [args]
        cmd = [
                  sys.executable, "-m", "torch.distributed.run",  # Tương đương với lệnh torchrun
                  f"--nproc_per_node={num_gpus}",
                  "train_multi_gpu.py"
              ] + forwarded_args

    print(f"⚙️ Lệnh thực thi: {' '.join(cmd)}")
    print("=" * 50 + "\n")

    # 4. Thực thi file con và truyền luồng hiển thị (stdout/stderr) ra màn hình
    try:
        # subprocess.run sẽ khóa Terminal ở đây cho đến khi train xong hoặc sập
        subprocess.run(cmd, check=True)

    except subprocess.CalledProcessError as e:
        print(f"\n❌ Lỗi: Tiến trình huấn luyện bị sập (Mã lỗi: {e.returncode})")
        sys.exit(e.returncode)

    except KeyboardInterrupt:
        print("\n🛑 Người dùng chủ động dừng huấn luyện (Ctrl+C). Đang dọn dẹp...")
        sys.exit(0)

if __name__ == "__main__":
    main()