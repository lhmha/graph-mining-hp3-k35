# Graph Mining HP3 K35

Repository quản lý mã nguồn thực nghiệm của nhóm.

## PCMEA reproduction

Mã nguồn và hướng dẫn chạy lại nằm tại [`PCMEA/`](PCMEA/README.md).

- Phần cứng đã kiểm thử: 1 NVIDIA RTX 3090 24 GB.
- Môi trường đã kiểm thử: Python 3.13.13, PyTorch 2.14.0, CUDA 13.0.
- Dữ liệu và checkpoint không được commit vào Git.

Sau khi hoàn tất các bước cài môi trường và chuẩn bị dữ liệu trong `PCMEA/README.md`, chạy một cấu hình bằng:

```bash
cd PCMEA
GPU_ID=0 bash scripts/run_experiment.sh FB15K_DB15K 0.2 42
```

Hoặc chạy tuần tự toàn bộ ma trận cấu hình:

```bash
cd PCMEA
GPU_ID=0 bash scripts/run_all.sh 42
```

Các câu lệnh chuẩn bị PLM features, kiểm tra dữ liệu, checkpoint/resume, cấu trúc output và toàn bộ dependency được ghi trong [`PCMEA/README.md`](PCMEA/README.md).

## Nội dung khác

- `MSPT/`: notebook do nhóm quản lý từ trước.

Repository này không lưu kết quả thực nghiệm; kết quả và phân tích được trình bày trong báo cáo.
