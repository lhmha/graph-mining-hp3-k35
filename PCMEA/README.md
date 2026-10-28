# Hướng dẫn chạy tái lập thực nghiệm PCMEA

Thư mục này chứa mã nguồn đã dùng để tái lập PCMEA trên một GPU. Từ commit `77b5ec2ddc3e35f26e5b5f32e312977efd563afa` của [repo PCMEA chính thức](https://github.com/Bubble-bubble77/PCMEA), nhóm đã điều chỉnh code để chạy trên một GPU và lưu checkpoint. README này chỉ hướng dẫn cách chạy; kết quả và phân tích nằm trong báo cáo.

Các lệnh bên dưới chạy từ thư mục `PCMEA`:

```bash
git clone https://github.com/lhmha/graph-mining-hp3-k35.git
cd graph-mining-hp3-k35/PCMEA
```

## 1. Cài môi trường

Phần cứng nhóm dùng để tái lập: **1 NVIDIA RTX 3090 24 GB**. Môi trường: Ubuntu, Python 3.13.13, PyTorch 2.14.0 với CUDA 13.0 và cuDNN 9.2.4. Online encoder và momentum encoder đều chạy trên GPU này.

```bash
conda create -n pcmea-reproduce python=3.13 -y
conda activate pcmea-reproduce
python -m pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Dòng cuối cần in ra `True`. Nếu là `False`, cần cài bản PyTorch hỗ trợ CUDA phù hợp với máy trước khi train. Phiên bản các thư viện nhóm đã dùng được ghi trong `requirements.txt`.

## 2. Tải và đặt dữ liệu

Nhóm dùng [HUMEA `data.zip`](https://drive.google.com/file/d/1a3aou1qe7Yzq6y_khyTq1La_kf3UI7sC/view?usp=drive_link) làm nguồn tải bộ dữ liệu đã chuyển đổi và đặc trưng ảnh VGG16. [Nguồn do tác giả PCMEA cung cấp](https://pan.baidu.com/s/1MLGBNyFjb9LLa4urCk4hCA) cũng có thể sử dụng; mã truy cập là `stdt`. Dữ liệu không nằm trong GitHub repo này.

Nếu dùng HUMEA, tải `data.zip` về máy rồi giải nén **từ thư mục `PCMEA`**:

```bash
unzip /duong/dan/den/data.zip -d data
```

Sau khi giải nén, kiểm tra các đường dẫn:

```text
PCMEA/data/
├── mmkb-datasets/
│   ├── FB15K_DB15K/
│   │   ├── ent_ids_1, ent_ids_2, ill_ent_ids
│   │   ├── triples_1, triples_2
│   │   ├── training_attrs_1, training_attrs_2
│   │   ├── FB15K_DB15K_id_img_feature_dict.pkl
│   │   ├── attribute_feature_dict.pkl
│   │   └── triples_feature_dict.pkl
│   └── FB15K_YAGO15K/          # cùng cấu trúc; file ảnh mang tên FB15K_YAGO15K_...
└── mmkb-master/
    ├── FB15K/FB15K_EntityTriples.txt
    ├── DB15K/DB15K_EntityTriples.txt
    └── YAGO15K/YAGO15K_EntityTriples.txt
```

File nén cung cấp dữ liệu graph, đặc trưng ảnh và hai bộ đặc trưng BOW (`attribute_feature_dict.pkl`, `triples_feature_dict.pkl`). Hai file đặc trưng PLM cần cho PCMEA được tạo ở bước 3. Nếu dùng nguồn Baidu và thư mục sau giải nén có tên khác, hãy sắp xếp lại theo cấu trúc trên.

## 3. Tạo đặc trưng PLM

Chạy lần lượt cho hai bộ dữ liệu:

```bash
bash scripts/prepare_text_features.sh FB15K_DB15K data/mmkb-master
bash scripts/prepare_text_features.sh FB15K_YAGO15K data/mmkb-master
```

Script dùng `data_process/relation_text_expand.py` và `data_process/PLM_embedding.py` để tạo `attribute_roberta_base_feature_all.pkl` bằng `roberta-base` và `relation_t5_feature_all.pkl` bằng `t5-base`. Lần đầu chạy cần Internet để tải hai mô hình này; sau đó chúng được lưu trong cache trên máy. Bước này có thể mất nhiều thời gian.

## 4. Kiểm tra dữ liệu

```bash
python tools/verify_data.py FB15K_DB15K
python tools/verify_data.py FB15K_YAGO15K
```

Mỗi lệnh cần kết thúc bằng `PASS`. Script kiểm tra số entity, số cặp alignment, số triple, số feature và kích thước vector. `scripts/run_experiment.sh` cũng tự chạy bước kiểm tra này trước khi train.

## 5. Chạy thí nghiệm

Một cấu hình được xác định bằng tên dataset, tỷ lệ alignment và seed. Ví dụ:

```bash
GPU_ID=0 bash scripts/run_experiment.sh FB15K_DB15K 0.2 42
```

Để chạy cả sáu cấu hình theo thứ tự, trên cùng một GPU:

```bash
GPU_ID=0 bash scripts/run_all.sh 42
```

Tên dataset hợp lệ: `FB15K_DB15K` hoặc `FB15K_YAGO15K`. Tỷ lệ hợp lệ: `0.2`, `0.5`, `0.8`. Seed nhóm sử dụng: `42`. Script đặt 4.500 epochs, batch size 512, learning rate `5e-4`, `il_start=500`, `semi_learn_step=5`, `stage2_start=500`, `momentum_update_span=20` và CSLS k=3.

Nếu muốn tiến trình tiếp tục chạy sau khi đóng terminal, dùng `tmux`:

```bash
tmux new -s pcmea
GPU_ID=0 bash scripts/run_experiment.sh FB15K_DB15K 0.2 42
```

Nhấn `Ctrl-b` rồi `d` để rời phiên; quay lại bằng `tmux attach -t pcmea`.

## 6. Xem output hoặc chạy tiếp sau khi gián đoạn

Với ví dụ FB15K–DB15K 20%, seed 42, file được lưu tại:

```text
logs/full_fb15k_db15k_02_seed42.log
checkpoints/full_fb15k_db15k_02_seed42/latest.pt
artifacts/full_fb15k_db15k_02_seed42/pred.txt
```

Log chứa tiến độ từng epoch, đánh giá định kỳ, tổng thời gian và đỉnh bộ nhớ GPU. Checkpoint được lưu mỗi 50 epochs. Nếu lần chạy bị gián đoạn, chạy lại **đúng lệnh của cấu hình đó**; script tự nhận checkpoint và tiếp tục từ epoch đã lưu.