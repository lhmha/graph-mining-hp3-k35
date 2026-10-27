# Hướng dẫn chạy thực nghiệm MSPT trên IMRE

Tài liệu này hướng dẫn chạy các notebook `sep5.ipynb`, `sep7.ipynb`, `sep10.ipynb` trên Kaggle, lưu checkpoint và trích kết quả đánh giá tích lũy. Notebook được cung cấp riêng cùng báo cáo; trên máy hiện tại các file nằm trong thư mục Downloads.

Nhóm chỉ đánh giá IMRE vì không tìm thấy dữ liệu IMNER để triển khai thực nghiệm. Ba cấu hình sep5/sep7/sep10 tương ứng 5/7/10 nhiệm vụ trên cùng benchmark.

## 1. Chuẩn bị Kaggle

1. Tạo notebook Kaggle và import file `sep10.ipynb` để chạy thiết lập 10 nhiệm vụ. Dùng notebook tương ứng nếu chạy sep5 hoặc sep7.
2. Bật GPU và Internet. Lần chạy đã ghi nhận môi trường có hai Tesla T4; điều này không khẳng định mô hình dùng đồng thời hai GPU.
3. Thêm các nguồn dữ liệu bằng Add Input:
   - Dataset chứa repo `ContinueMKGC`, bao gồm `MRE/run.py` và các thư mục chia nhiệm vụ.
   - Dataset `mspt-imre-official-data`, bao gồm thư mục `data` chứa ảnh và các file ánh xạ.
   - Dataset checkpoint của đúng cấu hình nếu muốn tiếp tục lượt chạy cũ.
4. Thêm và cấp quyền truy cập các Kaggle Secrets mà notebook yêu cầu:

   | Secret | Mục đích |
   |---|---|
   | `KAGGLE_USERNAME` | Tài khoản sở hữu dataset checkpoint |
   | `KAGGLE_KEY` | Xác thực để tạo/cập nhật dataset checkpoint |
   | `WANDB_API_KEY` | Ghi kết quả lên Weights & Biases |

Không điền khóa trực tiếp vào code. Notebook hiện gọi cả ba secret nên cần khai báo đầy đủ nếu giữ nguyên notebook.

## 2. Kiểm tra dữ liệu

Notebook tìm mã nguồn dưới `/kaggle/input` rồi sao chép thành:

```text
/kaggle/working/ContinueMKGC/
```

Dữ liệu sau khi liên kết phải có cấu trúc:

```text
ContinueMKGC/MRE/data/
├── img_org/
├── img_vg/
├── img_detect/
├── ours_rel2id.json
└── txt/
    ├── mre_train_dict.pth
    ├── mre_dev_dict.pth
    ├── mre_test_dict.pth
    ├── sep5/
    ├── sep7/
    └── sep10/
        ├── ours_train_0.txt ... ours_train_9.txt
        ├── ours_val_0.txt   ... ours_val_9.txt
        └── ours_test_0.txt  ... ours_test_9.txt
```

Cell chuẩn bị dữ liệu phải kết thúc bằng `DATA READY`. Notebook này yêu cầu cả ảnh gốc, ảnh crop và file ánh xạ; chỉ có các file văn bản chưa đủ để chạy.

Cell dữ liệu tìm đường dẫn có dạng `mspt-imre-official-data/data`. Nếu dataset được đổi tên hoặc có thêm thư mục lồng, sửa `DATA_SRC` theo đường dẫn thực tế trong Input.

## 3. Cấu hình lượt chạy

Trong notebook sep10, tìm phần `CẤU HÌNH CHÍNH`:

```python
TASK_CONFIGS = [10]
EPOCHS = 10
BATCH_SIZE = 2
RUN_VERSION = "task-metrics-v2"
RESUME_FROM_CHECKPOINT = False
```

- Chạy mới: đặt `RESUME_FROM_CHECKPOINT = False`.
- Tiếp tục sau nhiệm vụ đã hoàn thành: đặt `RESUME_FROM_CHECKPOINT = True` và gắn dataset checkpoint đúng sep.
- Chạy một thí nghiệm độc lập mới: đổi `RUN_VERSION` để có tên dataset checkpoint và W&B run riêng.
- Chỉ chọn một cấu hình trong mỗi lượt: `[5]`, `[7]` hoặc `[10]`. Kiểm tra cả tên dataset checkpoint nếu chỉnh notebook khác sep10, vì cách đặt tên giữa các notebook có thể khác nhau.

Chế độ chạy mới có thể xóa state package của lượt chạy hiện tại trong `/kaggle/working`. Không dùng chế độ này khi muốn tiếp tục checkpoint cũ.

Các tham số đã dùng:

| Tham số | Giá trị |
|---|---|
| Text encoder | `bert-base-uncased` |
| Vision encoder | `openai/clip-vit-base-patch32` |
| Learning rate | `1e-5` |
| Key learning rate | `1e-6` |
| Bias learning rate | `1e-5` |
| Seed | `1234` |
| Max sequence length | `80` |
| Prompt length | `4` |
| Buffer size | `6` |
| Alpha / gamma | `0.1` / `1.0` |
| Gradient balance / replay / random | Bật |

`EPOCHS = 10` được dùng cho giai đoạn học nhiệm vụ và cho giai đoạn replay khi có; không hiểu là tổng cộng chỉ có 10 epoch cho toàn chuỗi.

## 4. Chạy notebook

Chạy các cell từ trên xuống, hoặc dùng Run All sau khi hoàn thành cấu hình và Secrets. Notebook thực hiện:

1. Tạo môi trường Python riêng và cài các thư viện tương thích.
2. Sao chép repo, áp dụng bản vá tương thích và ghi log W&B.
3. Liên kết dữ liệu, tải BERT và CLIP về bộ nhớ cục bộ.
4. Chạy từng nhiệm vụ, khôi phục trạng thái từ nhiệm vụ trước.
5. Lưu checkpoint và upload phiên bản dataset sau mỗi nhiệm vụ.

Môi trường đã ghi nhận: Python 3.9.25, PyTorch 1.11.0+cu113, CUDA 11.3, Transformers 4.11.3. Ưu tiên các cell tạo môi trường của notebook để giữ cấu hình tương thích.

Các dấu hiệu kết thúc thành công của sep10:

```text
Upload successful: bert.zip (...)
SAVED: sep10 task 10/10
PASS: sep10 hoàn thành 10/10
```

Kiểm tra thêm trang dataset để xác nhận các file thực sự xuất hiện sau khi Kaggle xử lý phiên bản mới.

## 5. Checkpoint và tiếp tục chạy

Lượt sep10 đã chạy sử dụng dataset:

[mspt-imre-checkpoints-sep10-task-metrics-v2](https://www.kaggle.com/datasets/hlhongm/mspt-imre-checkpoints-sep10-task-metrics-v2)

Đây là dataset checkpoint riêng, khác với tab Output của notebook. Output trống không đủ để kết luận checkpoint đã mất.

Thư mục checkpoint cuối có dạng:

```text
MRE_4_1e-05_sep10/
├── model_9.pth
├── share_9.pth
├── resume_task_9.pth
└── LAST_COMPLETED_TASK
```

- `model_9.pth`: trọng số mô hình sau T10.
- `share_9.pth`: các tham số chia sẻ và trạng thái liên quan.
- `resume_task_9.pth`: optimizer, bộ nhớ replay và các trạng thái cần để tiếp tục chạy.
- Chỉ số file bắt đầu từ 0: `model_0.pth` tương ứng T1, `model_9.pth` tương ứng T10.

Notebook xóa checkpoint cục bộ của nhiệm vụ trước sau khi khôi phục thành công. Muốn lấy checkpoint cũ, mở History của dataset và chọn phiên bản tương ứng; kiểm tra thông báo `completed task k/10`, không chỉ dựa vào số phiên bản.

Khi resume, giữ nguyên cấu hình và `RUN_VERSION`, gắn dataset đúng lượt chạy, rồi đặt `RESUME_FROM_CHECKPOINT = True`. Code tìm checkpoint gắn trong Input dưới dạng ZIP; nếu Input chỉ có các file `.pth` đã giải nén, cần đóng gói hoặc điều chỉnh phần tìm/khôi phục checkpoint trước khi chạy. Không mặc định rằng chỉ bật cờ resume là đủ trong mọi cách bố trí Input.

## 6. Trích chỉ số tích lũy A_k

`A_k` là F1 trên test của tất cả nhiệm vụ T1 đến Tk sau khi hoàn tất học Tk. Với sep10 đã kiểm tra, các file test tích lũy sẵn:

| Nhiệm vụ | File test | Số mẫu |
|---|---|---:|
| T1 | `ours_test_0.txt` | 255 |
| T2 | `ours_test_1.txt` | 464 |
| T3 | `ours_test_2.txt` | 662 |
| T4 | `ours_test_3.txt` | 836 |
| T5 | `ours_test_4.txt` | 984 |
| T6 | `ours_test_5.txt` | 1115 |
| T7 | `ours_test_6.txt` | 1241 |
| T8 | `ours_test_7.txt` | 1369 |
| T9 | `ours_test_8.txt` | 1495 |
| T10 | `ours_test_9.txt` | 1614 |

Không nối các file test sep10 với nhau vì sẽ lặp dữ liệu. Không tự suy rộng kết quả kiểm tra sep10 sang cách chia khác nếu chưa kiểm tra nội dung.

Để lấy kết quả cuối nhiệm vụ từ notebook:

1. Xác định đoạn log `===== sep10: task k/10 =====`.
2. Tìm lần đánh giá test cuối đoạn đó, sau replay nếu có.
3. Lấy `current test f1 score`, không lấy trường `best test f1` trong cùng dòng.
4. Ghi kết quả vào hàng Tk.

Ví dụ cuối T10:

```text
best test f1: 49.4536, ... current test f1 score: 43.5702
```

Trong ví dụ này, điểm cuối nhiệm vụ là **43.5702%**. Điểm **49.4536%** là cực đại trong quá trình chạy, không phải điểm cuối nhiệm vụ.

Mã nguồn công khai lưu trạng thái cuối nhiệm vụ; phần lưu checkpoint theo validation trong hàm `evaluate()` bị vô hiệu hóa. Dòng thông báo `Save best model` không có nghĩa checkpoint đã được chọn theo validation. Đánh giá lại checkpoint là bước xác nhận thêm; README này không cung cấp một script inference độc lập và không giả định `--do_test` tự động thực hiện toàn bộ quy trình A1-A10.

## 7. Kết quả sep10 trích từ lần đánh giá cuối nhiệm vụ

| Nhiệm vụ | Nhóm (%) | Tác giả (%) | Chênh lệch (điểm phần trăm) |
|---|---:|---:|---:|
| T1 | 92.02 | 91.70 | +0.32 |
| T2 | 75.28 | 76.20 | -0.92 |
| T3 | 66.93 | 64.70 | +2.23 |
| T4 | 61.75 | 65.10 | -3.35 |
| T5 | 58.68 | 62.30 | -3.62 |
| T6 | 61.45 | 60.90 | +0.55 |
| T7 | 56.48 | 58.80 | -2.32 |
| T8 | 57.18 | 55.80 | +1.38 |
| T9 | 51.17 | 49.80 | +1.37 |
| T10 | 43.57 | 46.00 | -2.43 |

Chênh lệch tính từ số đã làm tròn. Tại T10, mức giảm tương đối là khoảng 5.28%. Đây là một lượt chạy với seed 1234, chưa phải trung bình nhiều seed.

## 8. Xử lý lỗi thường gặp

| Hiện tượng | Kiểm tra |
|---|---|
| Không tìm thấy `MRE/run.py` | Dataset mã nguồn đã được gắn và giải nén đúng cấu trúc chưa |
| Không tìm thấy `DATA_SRC` | Đường dẫn thực tế có khớp `mspt-imre-official-data/data` không |
| Thiếu secret | Tên secret và quyền cho notebook truy cập |
| Thiếu ảnh hoặc mapping | Các thư mục ảnh và ba file `mre_*_dict.pth` |
| Tải BERT/CLIP lỗi | Internet và log của cell tải mô hình |
| Resume không tìm thấy checkpoint | Đúng sep, đúng tên thư mục, có ZIP phù hợp hoặc đã sửa bộ tìm checkpoint |
| Output có `bert` trống | Kiểm tra dataset checkpoint riêng và lịch sử phiên bản |
| Muốn lấy A_k nhưng chỉ thấy best F1 | Tìm `current test f1 score` ở lần test cuối từng nhiệm vụ |

## Tài liệu tham khảo

- [Mã nguồn tác giả](https://github.com/zjunlp/ContinueMKGC)
- [Bài báo và định nghĩa A_k](https://arxiv.org/html/2305.08698v3)
- Notebook thực nghiệm của nhóm: `sep5.ipynb`, `sep7.ipynb`, `sep10.ipynb`.
