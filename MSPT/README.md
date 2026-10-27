# Chạy thực nghiệm MSPT trên IMRE

## 1. Chuẩn bị

Import một trong các notebook `sep5.ipynb`, `sep7.ipynb` hoặc `sep10.ipynb` vào Kaggle. Bật **GPU** và **Internet**.

Trong **Add Input**, thêm hai dataset:

- [Mã nguồn ContinueMKGC](https://www.kaggle.com/datasets/nguyenmanhha2502/continuemkgc).
- [Dữ liệu IMRE](https://www.kaggle.com/datasets/nguyenmanhha2502/mspt-imre-official-data).

Thêm các Secrets và cấp quyền truy cập cho notebook:

- `KAGGLE_USERNAME`
- `KAGGLE_KEY`
- `WANDB_API_KEY`

## 2. Cấu hình và chạy

Chọn notebook tương ứng với số nhiệm vụ: sep5, sep7 hoặc sep10. Trong phần cấu hình của notebook sep10:

```python
TASK_CONFIGS = [10]  # Chọn một cấu hình: [5], [7] hoặc [10]
EPOCHS = 10
BATCH_SIZE = 2
RUN_VERSION = "task-metrics-v2"
RESUME_FROM_CHECKPOINT = False
```

- Chạy mới: giữ `RESUME_FROM_CHECKPOINT = False`.
- Lượt chạy độc lập mới: đổi `RUN_VERSION` để lưu checkpoint và log riêng.
- Nếu dùng notebook sep5/sep7, kiểm tra phần cấu hình và tên dataset checkpoint tương ứng trước khi chạy.

Chạy các cell từ trên xuống. Kiểm tra cell dữ liệu hiển thị `DATA READY`, sau đó để notebook chạy lần lượt các nhiệm vụ. Kết thúc sep10 thành công khi log hiển thị:

```text
PASS: sep10 hoàn thành 10/10
```

## 3. Checkpoint và tiếp tục chạy

[Dataset checkpoint sep10](https://www.kaggle.com/datasets/hlhongm/mspt-imre-checkpoints-sep10-task-metrics-v2/data)

Để tiếp tục lượt sep10 cũ, thêm dataset checkpoint vào Input, giữ cấu hình và `RUN_VERSION` của lượt cũ, rồi đặt:

```python
RESUME_FROM_CHECKPOINT = True
```

Phần khôi phục trong notebook tìm checkpoint dạng ZIP. Nếu Kaggle chỉ hiển thị các file đã giải nén, cần điều chỉnh phần tìm checkpoint hoặc cung cấp ZIP có thư mục `MRE_4_1e-05_sep10` trước khi chạy tiếp.

Checkpoint sau T10 gồm `model_9.pth`, `share_9.pth` và `resume_task_9.pth`. Các checkpoint trước nằm trong lịch sử phiên bản dataset; kiểm tra mô tả nhiệm vụ của từng phiên bản.

## 4. Theo dõi và lấy kết quả

[Theo dõi thực nghiệm trên W&B](https://forge.coreweave.com/wandb/lhm_ha/mspt-imre-reproduction)

Để báo cáo A_k, lấy `current test f1 score` ở lần đánh giá cuối mỗi nhiệm vụ, sau replay nếu có; không lấy `best test f1`.

Với sep10, file `ours_test_0.txt` đến `ours_test_9.txt` đã tích lũy dữ liệu T1 đến Tk. Không nối các file này với nhau vì sẽ lặp mẫu.