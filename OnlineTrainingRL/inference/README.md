# Infer OmniGen2 gốc và checkpoint trên GEdit

inference nằm ngang hàng với EditScore và running:

```text
OnlineTrainingRL/
  EditScore/
  running/
  inference/
    settings.json
    generate_omnigen2_gedit.py
    generate_omnigen2_gedit_fireward.py
    inference_config.py
    run_inference.sh
    run_omnigen2_base.slurm
    run_omnigen2_checkpoint.slurm
```

## Chỉnh settings.json

- editscore_repo: đường dẫn repo, mặc định ../EditScore. Không phụ thuộc vị trí
  thư mục inference trong repo; có thể dùng đường dẫn tuyệt đối.
- omnigen_transformer_bin, text_encoder, vae_model: trỏ đến các thư mục pretrained
  trong Models/ của TrainingEditingModel, cùng cấu hình với running.
- checkpoint: chọn chính xác checkpoint-N hoặc thư mục lora_adapter/.
- metadata/data_root: GEdit-Bench-EN metadata và thư mục gốc ảnh source.
- output_base/output_checkpoint: nơi lưu edited image của mỗi phiên bản.
- conda_sh/train_env: môi trường OmniGen2 đang dùng.
- seed, steps, text_guidance, image_guidance, max_long_edge, limit, cpu_offload:
  setting inference.

Trường checkpoint mặc định là null. Đặt đường dẫn checkpoint muốn đánh giá
trong settings.json hoặc truyền --checkpoint, ví dụ:

```text
/datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL/running/outputs/job_JOBID/checkpoints/checkpoint-200
```

Không tự tìm checkpoint latest hoặc fallback về pretrained khi checkpoint lỗi.
Bản base không đọc/nạp checkpoint, dù trường checkpoint trong settings có giá trị.

Mặc định: seed 20, Euler 50 bước, text guidance 5.0, image guidance 1.5,
CFG range [0,1], time_shift_base_res 168, max long edge 1024, CPU offload bật.
Đây là preset infer GEdit đã dùng trước đây; không lấy sampling 20 bước của train.
limit 0 = toàn bộ metadata; limit N = N sample từ start_index, gồm sample đã
có ảnh hợp lệ. CLI có thể ghi đè settings.

## Chạy trên server

Upload inference nằm cùng cấp với EditScore. Ví dụ:

```bash
cd /datastore/anhndt/Truongtq/TrainingEditingModel/OnlineTrainingRL/inference
mkdir -p logs

# OmniGen2 pretrained, không LoRA post-training.
sbatch run_omnigen2_base.slurm

# OmniGen2 pretrained + checkpoint đã train.
sbatch run_omnigen2_checkpoint.slurm
```

Hai job có thể chạy đồng thời vì output mặc định tách biệt.
Slurm mặc định normal/mig2-35, một MIG 35GB, 8 CPU, 64G RAM, 72 giờ.
Thay header nếu muốn full hoặc loại tài nguyên khác; giữ cpu_offload phù hợp.

Thử một sample trước:

```bash
sbatch run_omnigen2_base.slurm --limit 1
sbatch run_omnigen2_checkpoint.slurm --limit 1
```

Kiểm tra metadata, source và cấu trúc đường dẫn checkpoint trước khi nạp GPU:

```bash
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate /datastore/truongtq/miniconda3/envs/omnigen2-rl
python generate_omnigen2_gedit.py --dry-run
python generate_omnigen2_gedit_fireward.py --dry-run
```

Dry run chưa đọc toàn bộ tensor checkpoint. Việc kiểm tra key/tensor được thực
hiện khi load model để infer. Inference không cần reward server/Qwen3.

## Ảnh và cơ chế resume

```text
inference/results/OmniGen2-Base/GEdit-Bench-EN/<task_type>/<tên ảnh nguồn>.png
inference/results/OmniGen2-Fireward/GEdit-Bench-EN/<task_type>/<tên ảnh nguồn>.png
```

Chỉ edited PNG nằm trong results, không lưu source/prompt ở đó. Ảnh giữ tỉ lệ,
không upscale, resize nếu vượt cạnh dài tối đa, pad theo bội số 16 rồi crop
padding khỏi ảnh edited. Tên theo ảnh source trong metadata, không theo key
nếu hai tên khác nhau.

Chạy lại cùng cấu hình sẽ bỏ qua PNG hợp lệ có đúng kích thước. Ảnh hỏng hoặc
ghi dở được sinh lại; output PNG được ghi qua file tạm rồi rename.

state/ chứa lock và chữ ký cấu hình theo output root. Nếu đổi checkpoint,
seed hoặc cấu hình ảnh thì dùng output root mới; bộ chạy từ chối trộn vào
thư mục đã có signature khác. Chọn thư mục kết quả mới khi dùng ảnh được tạo
bởi công cụ khác mà chưa có signature của bộ chạy này.

## Checkpoint được hỗ trợ

- Compact checkpoint của running: ưu tiên checkpoint-N/lora_adapter/.
- PEFT adapter_model.safetensors hoặc adapter_model.bin cùng adapter_config.json.
- Legacy Accelerate full model.safetensors/pytorch_model.bin: cần YAML train
  hoặc adapter config để khôi phục đúng rank/alpha; có thể chỉ định --lora-config.
- Không hỗ trợ tự hợp nhất checkpoint FSDP sharded của recipe nhiều GPU.

LoRA được kiểm tra key/tensor sau khi nạp, rồi trả transformer có forward
signature gốc để pipeline vẫn truyền ảnh tham chiếu. Không có fallback im lặng
về model base.

Log tại logs/omnigen2_base_JOBID.{out,err} và
logs/omnigen2_checkpoint_JOBID.{out,err}; có tiến độ sample, thời gian infer
và số ảnh sinh/bỏ qua. Checkpoint hoặc setting mới không tự tạo run_1/run_2.

Config được đọc khi job bắt đầu; sửa settings khi job đang queued sẽ ảnh
hưởng job đó. Các file test chạy trên CPU/mocked pipeline và PEFT; chưa chạy
inference GPU từ thư mục mới này.
