# PCMEA reproduction code

Thư mục này chứa mã nguồn và script đã dùng để chạy lại các cấu hình PCMEA trong báo cáo. README chỉ mô tả cách thiết lập và chạy; kết quả thực nghiệm được trình bày trong báo cáo, không lưu trong repository này.

Mã nguồn được phát triển từ repository PCMEA chính thức tại commit `77b5ec2ddc3e35f26e5b5f32e312977efd563afa`:

- Paper: <https://arxiv.org/abs/2403.01203>
- Official code: <https://github.com/Bubble-bubble77/PCMEA>
- Original MMKB data: <https://github.com/mniepert/mmkb>

## Phần cứng đã kiểm thử

- 1 NVIDIA GeForce RTX 3090, 24 GB VRAM
- Cả online encoder và momentum encoder chạy trên cùng GPU `cuda:0`.
- Mỗi cấu hình chạy tuần tự; không cần multi-GPU.

## Môi trường đã dùng

- Ubuntu Linux
- Python 3.13.13
- PyTorch 2.14.0 + CUDA 13.0
- cuDNN 9.2.4

Tạo môi trường và cài dependency:

```bash
conda create -n pcmea-reproduce python=3.13 -y
conda activate pcmea-reproduce
python -m pip install -r requirements.txt
```

Có thể dùng phiên bản PyTorch/CUDA tương thích với GPU nếu bản wheel nêu trên không còn được phân phối. Các phiên bản chính xác đã kiểm thử được khóa trong `requirements.txt`.

## Chuẩn bị dữ liệu

### 1. Converted MMKB và VGG16 image embeddings

Dữ liệu không được commit vào repository. Tải một trong hai nguồn sau:

- Bản do tác giả PCMEA cung cấp: <https://pan.baidu.com/s/1MLGBNyFjb9LLa4urCk4hCA>, key `stdt`.
- Mirror `data.zip` từ HUMEA: <https://drive.google.com/file/d/1a3aou1qe7Yzq6y_khyTq1La_kf3UI7sC/view?usp=drive_link>.

Mirror HUMEA chỉ được dùng cho converted graph, BOW features và VGG16 image embeddings. Không dùng text/PLM embeddings có sẵn trong mirror; hai file PLM sẽ được tạo lại ở bước tiếp theo.

Sau khi giải nén, cấu trúc cần có dạng:

```text
data/mmkb-datasets/
├── FB15K_DB15K/
│   ├── ent_ids_1
│   ├── ent_ids_2
│   ├── ill_ent_ids
│   ├── triples_1
│   ├── triples_2
│   ├── training_attrs_1
│   ├── training_attrs_2
│   ├── FB15K_DB15K_id_img_feature_dict.pkl
│   ├── attribute_feature_dict.pkl
│   └── triples_feature_dict.pkl
└── FB15K_YAGO15K/
    └── ...
```

### 2. MMKB gốc và PLM embeddings

```bash
git clone https://github.com/mniepert/mmkb.git data/original/mmkb

bash scripts/prepare_text_features.sh \
  FB15K_DB15K data/original/mmkb

bash scripts/prepare_text_features.sh \
  FB15K_YAGO15K data/original/mmkb
```

Script trên tạo lại:

- `attribute_roberta_base_feature_all.pkl` bằng `roberta-base`;
- `relation_text.json` từ entity triples gốc;
- `relation_t5_feature_all.pkl` bằng `t5-base`.

Hugging Face models được tải tự động trong lần chạy đầu. Có thể đặt `HF_HOME` để chọn thư mục cache.

### 3. Kiểm tra input

```bash
python tools/verify_data.py FB15K_DB15K
python tools/verify_data.py FB15K_YAGO15K
```

Kiểm tra phải kết thúc bằng `PASS` trước khi train.

## Chạy một cấu hình

Cú pháp:

```bash
bash scripts/run_experiment.sh DATASET ALIGNMENT_RATE SEED
```

Ví dụ:

```bash
GPU_ID=0 bash scripts/run_experiment.sh FB15K_DB15K 0.2 42
GPU_ID=0 bash scripts/run_experiment.sh FB15K_YAGO15K 0.8 42
```

Script dùng đúng cấu hình chính: 4,500 epochs, batch size 512, learning rate `5e-4`, `il_start=500`, `semi_learn_step=5`, `stage2_start=500`, `momentum_update_span=20`, và CSLS k=3.

Để chạy tách khỏi terminal:

```bash
tmux new -s pcmea
GPU_ID=0 bash scripts/run_experiment.sh FB15K_DB15K 0.2 42
# Ctrl-b, sau đó d để detach
```

Checkpoint được lưu mỗi 50 epochs. Nếu tiến trình bị gián đoạn, chạy lại đúng câu lệnh; script tự phát hiện `latest.pt` và resume đầy đủ model, optimizer, RNG, pseudo-label và momentum state.

## Chạy toàn bộ ma trận cấu hình

```bash
GPU_ID=0 bash scripts/run_all.sh 42
```

Lệnh trên chạy tuần tự hai dataset và ba tỷ lệ alignment. Chỉ nên chạy khi GPU có thể hoạt động liên tục trong thời gian dài.

## Output cục bộ

Các output dưới đây bị loại khỏi Git bởi `.gitignore`:

```text
logs/         # stdout, runtime và peak VRAM
checkpoints/  # latest.pt để resume
artifacts/    # prediction output
```

## Cấu trúc code

```text
src/                         # training entry point và model runtime
data_process/                # tạo relation/attribute PLM embeddings
tools/verify_data.py         # kiểm tra dữ liệu trước khi train
scripts/prepare_text_features.sh
scripts/run_experiment.sh
scripts/run_all.sh
```
