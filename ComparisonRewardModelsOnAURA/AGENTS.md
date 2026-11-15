# AGENTS.md — ComparisonRewardModels

Hướng dẫn cho agent khi làm thí nghiệm so sánh reward model. Quy ước chung của repo xem `../AGENTS.md`;
file này chỉ ghi phần riêng và phần ghi đè.

## Mô tả dự án

So sánh reward model chỉnh sửa ảnh trên **AURA-Test** (`../AURA/AURA-TEST/AURA-Bench.jsonl`):
1561 cặp ảnh, mỗi cặp có nhãn người (A / B / Tie, A = `edited_images[0]`) cho **một** metric:
Aesthetic, Realism, Fidelity, Background Consistency (BC), Foreground Consistency (FC), Structure Consistency (SC).
Định nghĩa metric: `../AURA/AURA-TEST/metrics_definition.txt`; câu hỏi người chấm đã đọc:
`../UserStudy/UserStudyHTML/index.html`.

- Bảng 15 dòng: 12 model so sánh (EditScore-72B ghi "not_run") và 3 dòng của mình, backbone Qwen3-VL-Reranker-8B:
  - **Student-Stage1** (`student_warmstart.pt` = checkpoint-126, 1008 cặp); đại diện cả Teacher-Stage1 (bản sao giống hệt).
  - **Student-Stage2**, **Teacher-Stage2**: Student / Teacher Model của cùng checkpoint cuối Stage 2, version `Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5`:
    `output_training/Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5/checkpoint-927.pt` (`student_adapter_state_dict` / `teacher_adapter_state_dict`).
  - Không có dòng "Our Model". **Không đánh giá Inspector Baseline** (quyết định 13/11/2026): Inspector chỉ train
    và có system prompt cho overall, còn mỗi cặp AURA-Test chỉ có nhãn cho một metric.
  - Student / Teacher: mỗi cột dùng đầu ra của đúng metric đó (bench đã cho metric, Relevance Head không dùng để chọn);
    xác suất relevance được lưu và phân tích riêng (`scripts/relevance_analysis.py`).
- Mỗi model chỉ được đánh giá trên các metric mà chiều chấm điểm của nó nhắm tới, ô còn lại là "-".
  Cách ghép đã chốt: `EvaluationConfig/metric_mapping.json`.
- VLM gốc (Qwen2.5-VL-7B, Qwen3-VL-8B, Qwen3.5-9B) chấm bằng prompt theo metric:
  `EvaluationConfig/base_vlm_metric_prompts.json`.
- Input cho **mọi** model, cả 6 metric: ảnh gốc, ảnh đã sửa và chỉ dẫn. (Người chấm chỉ thấy ảnh đã sửa
  ở Aesthetic và Realism; điều đó không đổi input của model.) Reward model giữ nguyên cách dựng prompt gốc.
- Mỗi ô chỉ tính trên các cặp có nhãn của đúng metric đó; margin = điểm A − điểm B.
- Chỉ số (đã chốt, chi tiết trong `metric_mapping.json` → `evaluation_metrics`):
  - Bảng chính: Accuracy và Margin AUC trên cặp không hoà; Accuracy ba lớp (A / B / Tie) trên mọi cặp —
    Student-Stage1 / Student-Stage2 / Teacher-Stage2 dùng argmax P(A), P(B), P(Tie); model chỉ có điểm: Tie chỉ khi
    hai điểm bằng hệt nhau (không tune ngưỡng trên bench). Đây là điểm nổi bật của model của mình.
  - Bảng phụ: PLCC và SRCC giữa margin và độ chênh phiếu `(votes A − votes B) / số người chấm` trên mọi cặp,
    Tie AUC (AUC của −|margin| tách cặp hoà), Accuracy trên cặp người chấm đồng ý hoàn toàn;
    thêm Tie Head AUC (AUC của P(Tie)) cho các dòng có Tie Head, model khác để "-".
  - Khoảng tin cậy 95% bootstrap (2000 lần, seed cố định) cho mọi ô; bootstrap ghép cặp so với
    Student-Stage2 (dòng có `paired_reference` trong mapping).
- Làm **từng bước một**, chờ người dùng duyệt giữa các bước.

## Môi trường

```bash
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl      # torch 2.4, transformers 4.57.6, peft 0.12, pytest; không có vLLM
```
- Env khác có sẵn: `edit_reward` (torch 2.5.1, peft 0.21.2), `omnigen2-rl` (torch 2.7.1, peft 0.17.1).
- SpatialReward gốc cần vLLM 0.11 + torch 2.8 (chưa có env nào). Trước khi chạy một model:
  kiểm tra env chạy được 1 mẫu, rồi mới viết job.
- Không tạo hay sửa env khi chưa được duyệt.
- Không tải model từ Internet trong lúc đánh giá: luôn truyền đường dẫn trên máy (`--model_path`,
  `model_name_or_path`, `--lora_path`); code gốc của LMM4Edit, EditHF, EditReward sẽ tự tải nếu thiếu.

## Lệnh chính

Mọi việc nặng (inference, chạy cả bench, tải model lớn) **chỉ chạy qua Slurm**, không chạy trên login node.

```bash
# Tải backbone / checkpoint lớn (danh sách: /datastore/tuanld/Truongtq/RewardModels/models.txt)
cd /datastore/tuanld/Truongtq/RewardModels
python -B download_pretrained_models.py --dry-run [--only NAME ...]
umask 002 && [ALLOW_DIFFERENT_OWNER=1] DOWNLOAD_ONLY="NAME ..." sbatch download_pretrained_models.slurm
# Chuyển sở hữu (quota) sang tuanld — chạy trên tuanld
nice -n 19 ionice -c3 python -B take_model_ownership.py --dry-run [--only NAME ...]

# Đánh giá (chưa viết): một launcher / model, đặt trong script_slurm/
sbatch script_slurm/run_evaluate_<model>.slurm

# Test
pytest -q            # chạy trong ComparisonRewardModels/
```

**Slurm**
- Partition: `small` 18 GB (QOS `mig1-18`, tối đa 4 CPU, 56G RAM mỗi job) · `normal` 35 GB (`mig2-35`, 8 CPU, 64G)
  · `large` 71 GB (`mig3-71`) · `full` 141 GB (`mig7-141`). Ước lượng VRAM và chọn tier nhỏ nhất đủ dùng.
- Mỗi user chạy tối đa **3 job** cùng lúc trên mọi partition (job thừa chờ với `AssocGrpJobsLimit`).
- `sbatch --test-only` **không** kiểm tra giới hạn QOS: so `#SBATCH` với `sacctmgr show qos format=name,maxtres`.
- Slurm ẩn job của user khác: muốn biết trạng thái job của tài khoản khác thì hỏi người dùng hoặc đọc log.

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `EditHF/`, `EditReward/`, `EditScore/`, `LMM4Edit/`, `SpatialReward/` | Repo gốc (git clone). **Không sửa code**. LoRA / head tải về nằm trong đây, kèm `lora_download_manifest.json` (repo, commit) |
| `EditHF/lora_checkpoints_{visual,editing,preservation}/` | LoRA EditHF (`editing` dùng cho `--mode alignment`) |
| `LMM4Edit/weights/{visual,alignment,preservation}/` | LoRA + score head LMM4Edit |
| `EditScore/weights/<repo>/` | LoRA EditScore-7B, -Qwen3-VL-4B-Instruct, -Qwen3-VL-8B-Instruct |
| `EvaluationConfig/` | `metric_mapping.json` (model → metric, chỉ số), `base_vlm_metric_prompts.json` |
| `RewardModelsMetrics.txt` | Định nghĩa chiều chấm điểm của từng model (theo paper) |
| `scripts/` | Code của mình: adapter chấm điểm từng model, tính chỉ số |
| `script_slurm/` | Launcher `.slurm` cho đánh giá |
| `results/<Model>_<Pretrained>/` | Điểm từng cặp + chỉ số; mỗi bước một thư mục con |
| `tests/`, `pytest.ini` | Test pytest cho code trong `scripts/` và cho `EvaluationConfig/` |
| `submissions.log` | Nhật ký mọi lần submit job |
| `/datastore/tuanld/Truongtq/RewardModels/` | **Chỉ** backbone / checkpoint đầy đủ lớn (xem `models.txt`), mỗi thư mục có `download_manifest.json` |
| `/datastore/nghiatg/Truongtq/` | Benchmark |

Thư mục này hiện bị `.gitignore` gốc loại; muốn đưa code lên git phải thêm dòng `!/ComparisonRewardModels/<đường dẫn>/`
(chỉ code và config, không đưa repo clone, weights, kết quả).

## Quy ước

**Đặt tên**
- Tên model và pretrained đúng như bảng: `EditScore` + `Qwen3-VL-4B`, `EditReward` + `MiMo-VL-7B`…
  Thư mục kết quả: `<Model>_<Pretrained>` (vd `EditScore_Qwen3-VL-4B`). Không dùng mã tắt.
- Tên chiều chấm điểm giữ nguyên như paper (Visual Quality, Perceptual Quality, Semantic Consistency…).
- Viết đầy đủ thuật ngữ trong code, log, tài liệu.

**Weights và dung lượng**
- LoRA / head nhỏ: trong repo của model đó; backbone / checkpoint đầy đủ: `RewardModels/` (thêm dòng vào
  `models.txt`, chạy `--dry-run` trước). Luôn cố định commit Hugging Face và kiểm tra kích thước file.
- Quota 300 GB / tài khoản. Chưa xác nhận quota tính theo người ghi file hay theo thư mục: ghi rõ tài khoản
  sở hữu và dung lượng mỗi lần tải. `RewardModels/` hiện thuộc anhndt (146,9 GB).
- Job từ tài khoản khác: kiểm tra quyền hai chiều; `umask 002` trước `sbatch`, `chmod -R g+rwX` sau job.

**Output**
- Mỗi lần chạy lưu: điểm từng cặp (`sample_id`, metric, điểm A, điểm B, margin), output thô của model
  (để kiểm tra lỗi parse), cấu hình đầy đủ (đường dẫn model, commit, prompt, env, md5 code).
- Các bước của cùng một việc: một thư mục, mỗi bước một thư mục con.
- Output job thử (`*_test_*`) được xoá sau khi kiểm tra xong.

**File điểm** (`results/<Model>_<Pretrained>/scores.jsonl`, adapter của mỗi model ghi ra): mỗi dòng
`{"sample_id", "dimension", "score_a", "score_b"[, "probabilities": [P(A), P(B), P(Tie)]]}`;
`dimension` đúng tên trong `metric_mapping.json`; `score_a` / `score_b` = điểm của `edited_images[0]` / `[1]`.
Bảng: `python scripts/comparison_table.py --output results/comparison_table` (ghi `.json` + `.md`).

**Logging khi submit**
- Mỗi job ghi một dòng vào `submissions.log`: ngày, tài khoản, job ID, partition / QOS / GPU, model,
  metric, đường dẫn output, md5 của launcher và script.
- Đọc log và số liệu thật trước khi kết luận; báo nguyên văn lỗi.

## Testing

- Mỗi khi thêm feature mới hoặc sửa logic, **phải** viết test tương ứng trong `tests/` (pytest),
  file `test_<module>.py`, tên mỗi test mô tả rõ hành vi (vd `test_margin_auc_ignores_tie_pairs`).
- Viết test **trước hoặc cùng lúc** với code, gồm: trường hợp bình thường, trường hợp biên
  (input rỗng, 1 cặp, toàn cặp hoà, NaN, điểm bằng nhau), và trường hợp lỗi (input sai phải raise đúng exception).
- Bắt buộc có test cho: cách ghép metric (mỗi cột chỉ lấy đúng cặp của metric đó; "-" không được tính),
  parse output văn bản của model (JSON hỏng, thiếu điểm, ngoài thang), chiều của margin (A − B),
  và mọi chỉ số (Accuracy, Margin AUC, PLCC, SRCC, Tie AUC, bootstrap) so với giá trị tính tay.
- Code liên quan model: dùng model giả / model nhỏ, data giả tạo trong test, chạy trên CPU, mỗi test < 10 giây.
  Không tải checkpoint thật, không đọc ảnh thật của bench, không submit Slurm. `tests/conftest.py` ẩn GPU,
  bật chế độ offline của Hugging Face và chặn `sbatch` / `scontrol` / `scancel` / `srun`.
- Chạy `pytest -q` trong `ComparisonRewardModels/` sau khi sửa; chỉ báo hoàn thành khi tất cả test pass
  (báo nguyên văn lỗi nếu fail).
- Không được xoá hoặc sửa test cũ để cho pass, trừ khi người dùng đồng ý.

## Không được làm

- Không sửa code trong các repo clone; cần thay đổi thì viết wrapper trong `scripts/`.
- Không xoá checkpoint, LoRA, weights đã tải; chỉ người dùng xoá.
- Không sửa, ghi đè hay di chuyển AURA-Test, ảnh bench, nhãn user study, `metrics_definition.txt`.
- Không quét hay đọc toàn bộ thư mục ảnh trong session; chạy cả bench phải là job Slurm đã duyệt.
- Không dùng AURA-Test để tune prompt, ngưỡng hay chọn checkpoint của Student / Teacher.
- Không tải hay chạy EditScore-72B.
- Không submit, huỷ hay sửa job Slurm khi chưa được duyệt; không chạy việc nặng trên login node.
- Không commit hay push khi chưa được yêu cầu; không ghi token, mật khẩu, thông tin cá nhân.
