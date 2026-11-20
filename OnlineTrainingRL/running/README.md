# OmniGen2 + Qwen3 reward: chạy từ thư mục riêng

## Cấu trúc

```text
OnlineTrainingRL/
  EditScore/                 # Repo gốc VectorSpaceLab/EditScore, không sửa source
  running/                   # Config và code train riêng
    settings.json
    train.yml
    run_train.slurm
    run_job.py
    prepare_run.py
    train.py
    reward_server.py
    reward_runtime.py
    compact_checkpoint.py
```

Repo gốc được tải tại commit `4609c5d2ebb62fdebf665d3c924686d896ef1f74`.
train.py trong running kế thừa bản tích hợp Qwen3 một GPU đã dùng trước đây,
giữ nguyên công thức Flow-GRPO, PPO clipping, CFG, KL và timestep reweighting.

running không cần nằm trong repo EditScore. Code model/dataset/scheduler/loss
được import từ repo chỉ định bằng editscore_repo trong settings.json.
Mặc định `../EditScore`, được resolve theo vị trí settings.json, không theo
working directory của terminal. Có thể đổi thành đường dẫn tuyệt đối để đặt
running ở bất kỳ thư mục nào có quyền truy cập source.

## Cấu hình cũ được giữ

Đây là cấu hình H200 200 step của job 64127, không phải recipe 8/32 GPU:

| Setting | Giá trị |
|---|---:|
| GPU policy và reward | Cùng 1 H200 full 141GB |
| global_batch_size / batch_size | 12 / 12 |
| Prompt mỗi vòng sampling | 1 |
| Rollout mỗi prompt | 12 |
| batch_size_per_forward | 1 |
| gradient_accumulation_steps | 12 |
| num_update_steps_per_sampling | 1 |
| max_train_steps | 200 optimizer update |
| learning_rate | 5e-5 |
| warmup_t / warmup_lr_init | 20 / 1e-7 |
| LoRA rank / alpha / dropout | 32 / 64 / 0 |
| Sampling steps / train fraction | 20 / 0.6 |
| Text / image guidance | 4 / 2 |
| CFG range | [0.0, 0.6] |
| sigma_coef | 0.7 |
| PPO clip | [1e-4, 5e-4] |
| Advantage clip | [-5, 5] |
| KL weight | 0.04 |
| checkpointing_steps / retention | 20 / 3 |

BF16 backbone, FP32 LoRA trainable, gradient checkpointing, EMA 0.
Chỉ LoRA transformer được cập nhật; text encoder, VAE và reward model đóng băng.
AdamW betas (0.9,0.95), weight decay 0.01, epsilon 1e-8, max grad norm 1.
Giới hạn diện tích input/output 262144 pixel, max side 2048, giữ tỉ lệ ảnh.

12 ảnh × 12 transition = 144 lần backward tích lũy cho 1 optimizer update.
Effective image batch là 12, không phải 144 ảnh. 200 update tương ứng 2400
rollout và 200 lượt điều kiện; không phải một lượt đầy đủ dataset 8132 mẫu.

## Đường dẫn: chỉnh settings.json

Các đường dẫn model, checkpoint reward, metadata và ảnh shared được giữ như
bản chạy trước. Ba pretrained policy mặc định nằm dưới:

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/Models/
  OmniGen2/transformer/
  Qwen2.5-VL-3B-Instruct/
  FLUX.1-dev/vae/
```

Trainer đọc trực tiếp các shard Safetensors trong OmniGen2/transformer/.
Nếu dùng bản pretrained đã đóng gói thành pytorch_model.bin, có thể trỏ thẳng đến file đó.

Để tạo `Models/OmniGen2/transformer/pytorch_model.bin` (BF16), chạy từ `running/`:

```bash
mkdir -p logs
sbatch prepare_pretrained.slurm
```

Script lấy thư mục transformer và môi trường Conda từ `settings.json`.

Các trường chính:

- editscore_repo: đường dẫn repo EditScore mới, mặc định ../EditScore.
- metadata, data_root: mặc định dùng bản sao unique_metadata.jsonl cạnh settings.json và ảnh trong thư mục shared.
- omnigen_transformer_bin, text_encoder, vae_model: pretrained policy.
- reward_project: project Qwen3 có src/models/custom_qwen3_vl_reranker.py.
- reward_pretrained, reward_checkpoint: Qwen3-VL-Reranker-8B và checkpoint Stage2.
- conda_sh, train_env, reward_env: đường dẫn môi trường đang dùng, giữ như cũ.

Reward infer 6 cặp cho 12 rollout, chọn metric có relevance trung bình nhóm
>= 0.5, lấy trung bình raw score làm reward, không chia 10. Sau đó chuẩn hóa
reward trong nhóm thành advantage. Empty metric policy mặc định error.

## Upload và chạy trên server

Upload OnlineTrainingRL gồm hai thư mục EditScore và running, ví dụ:

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL/
  EditScore/
  running/
```

Nếu không upload source, có thể clone repo gốc trên server:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL
git clone https://github.com/VectorSpaceLab/EditScore.git EditScore
```

Sau khi kiểm tra settings.json và tài nguyên trong Slurm:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL/running
mkdir -p logs
sbatch run_train.slurm
```

Slurm mặc định: account uit, partition full, QoS mig7-141, 1 GPU
gpu:7g.141gb:1, 16 CPU, 128G RAM, time limit 72 giờ.
run_job.py tự chuẩn bị data/config, bật Qwen3, chờ đúng server của job sẵn sàng,
chạy train và tắt server khi kết thúc. Nếu reward chết thì dừng policy.
Không cần tự export Conda hoặc chạy server trước.

Các môi trường Conda/thư viện và pretrained phải có sẵn. Bộ này không tự
cài môi trường hoặc tải trọng số. running vẫn đọc reward architecture, model,
checkpoint, ảnh và môi trường từ các vị trí ngoài thư mục như cấu hình.

## Output

```text
running/logs/omnigen2_qwen3_JOBID.out
running/logs/omnigen2_qwen3_JOBID.err
running/logs/reward_job_JOBID.log
running/outputs/job_JOBID/
  train.yml
  settings.json
  data.yml
  reward_server.yml
  training_pairs.jsonl
  repo_rl_path.txt
  source.json
  train.py
  compact_checkpoint.py
  reward_server.py
  reward_runtime.py
  checkpoints/checkpoint-N/lora_adapter/
```

Mỗi job dùng thư mục mới. Chuẩn bị run ghi đường dẫn source tuyệt đối vào
repo_rl_path.txt nên việc train không dựa vào quan hệ cha/con với EditScore.
Các script runtime và config được chụp khi job bắt đầu; thay đổi lúc job
queued vẫn áp dụng trước khi snapshot được tạo. Giữ source EditScore tại vị
trí đã cấu hình trong suốt quá trình chạy.

Checkpoint lưu LoRA + optimizer/scheduler/RNG, không sao chép frozen model.
Ghi vào thư mục .checkpoint-N.incomplete, publish khi hoàn tất. Kiểm tra ghi
512 MiB trước khi nạp model giúp phát hiện quota sớm, không đảm bảo quota
sẽ còn đủ khi các job khác cùng ghi dữ liệu.

Đã kiểm tra trên CPU: resolve source ngoài running, data, reward HTTP,
giữ công thức loss, checkpoint export và quản lý vòng đời policy/reward.
Chưa chạy train GPU trên server từ bản thư mục mới này.
