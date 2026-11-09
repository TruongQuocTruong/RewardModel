# Reward Model pipeline (Inspector → Stage 1 → Stage 2)

Mục tiêu: Student sau Stage 2 đạt hoặc vượt Inspector Baseline, trong khi chỉ dùng **1008 cặp có nhãn**. Backbone là Qwen3-VL-Reranker-8B, fine-tune bằng LoRA.
Công thức, head và loss xem tại `Scriptions/pipeline_core.md`. Kết quả các thí nghiệm Stage 2 xem tại `Scriptions/stage2_tuning.md`.

```
EditReward-Data (HumanGT full) ──► Inspector Baseline ──► checkpoint-63 (= 1008 cặp đầu) ──► nhãn Inspector cho 10k Unlabeled
GemmaGT 3k (1008 cặp đầu) ──────► Student-Stage1 (khởi tạo từ Inspector checkpoint-63) ──► Teacher = bản sao Student-Stage1
                                   chọn 1008 Unlabeled (Teacher–Inspector agreement) + kịch bản augmentation
Stage 2: Student–Teacher loop (HumanGT + GemmaGT 1008 cặp, Unlabeled 1008, Teacher EMA, 5 iteration) ──► đánh giá EditReward-Bench / VLM-Bench
```

## Cấu trúc thư mục

| Thư mục | Nội dung |
|---|---|
| `Qwen3-VL-Embedding/scripts/` | Script train và suy luận (bảng bên dưới) |
| `Qwen3-VL-Embedding/src/models/custom_qwen3_vl_reranker.py` | Model, các head (Score, Tie `margin_coupled`, Relevance) và loss |
| `Qwen3-VL-Embedding/output_training/` | Output của mọi run (trên repo để trống; checkpoint chia sẻ qua Drive) |
| `script_slurm/` | Launcher Slurm của từng module (định nghĩa các biến cấu hình mặc định) |
| `config_training.sh` | Submit Stage 2 qua Slurm; mỗi version = một tên thư mục output; mỗi lần submit được ghi vào `experiment_submissions.log` (config + md5 code) |
| `config_stage2.sh` | Lệnh Stage 2 của 3 run đã train, chạy được không cần Slurm |
| `TrainingDataCreation/` | Dữ liệu train đã chuẩn hoá (JSONL) và script tạo dữ liệu |
| `Scriptions/` | Tài liệu: pipeline, công thức, kết quả các ablation |
| `../TrainingDataSampling/` | Script và CSV lấy mẫu dữ liệu từ EditReward / EditHF |
| `../RewardModelEvaluation/` | Bộ đánh giá (`OurModel/`), baseline khác (`OtherModels/`), tập đánh giá (`EvalDatasets/`) |

## Dữ liệu

| File | Dùng ở |
|---|---|
| EditReward-Data `metadata.jsonl` (HumanGT full, bên ngoài repo) | Train Inspector Baseline |
| `TrainingDataCreation/editreward_overall_humanGT_3k.jsonl` | HumanGT của Stage 2 (chỉ 1008 cặp đầu theo `train_order.jsonl`) |
| `TrainingDataCreation/editreward_overall_finegrained_gemmaGT_3k.jsonl` | GemmaGT (nhãn 6 metric từ 3 lần chạy Gemma): Stage 1 và Stage 2, 1008 cặp đầu |
| `TrainingDataCreation/editreward_unlabeled_10k.jsonl` | Unlabeled: Inspector gán nhãn, Stage 2 chọn 1008 mẫu |
| `../RewardModelEvaluation/EvalDatasets/EditReward-Bench_pairwise.json` | EditReward-Bench (133 cặp, 115 không hoà) |
| `../RewardModelEvaluation/EvalDatasets/EditReward-Bench_VLM_judge/pairs.csv` | VLM-Bench: nhãn VLM của Bench, đa số 15 vote |

Đường dẫn ảnh nằm bên trong các file JSONL và phải tồn tại trên máy chạy. Ràng buộc: không dùng nhãn HumanGT/GemmaGT nào ngoài 1008 cặp, kể cả để validation. Không dùng tập Test khi tuning Stage 2.

## Các module, script, launcher và output

| # | Module | Script (`Qwen3-VL-Embedding/scripts/`) | Launcher (`script_slurm/`) | Output (`output_training/`) |
|---|---|---|---|---|
| 1 | **Inspector Baseline**: three-way P(A), P(B), P(Tie), LoRA r32, lr 5e-5 không đổi, 1 lượt, batch 16 | `train_overall_inspector.py` | `run_overall_inspector_training.slurm` | `inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/`: `train_order.jsonl`, `checkpoint-N` (checkpoint-63 = 1008 cặp đầu = Inspector pipeline) |
| 2 | **Nhãn Inspector** cho 10k Unlabeled (eval mode) | `infer_overall_inspector.py` | `run_overall_inspector_inference.slurm` | `.../checkpoint-63/inspector_pseudo_labels_10k.jsonl` |
| 3 | **Student-Stage1**: 6 metric × (Score, Tie, Relevance), hard label GemmaGT, 1008 cặp, LoRA r64 trên LLM+ViT, lr 4e-5, khởi tạo LoRA từ Inspector checkpoint-63 | `train_student_hardlabel.py` (gọi `train_student_warmstart.py`) | `run_student_hardlabel_training.slurm` | `student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/student_warmstart.pt` |
| 4 | **Khởi tạo Teacher** = bản sao chính xác Student-Stage1 | `initialize_teacher_model.py` | `run_teacher_model_initialization.slurm` | cùng thư mục ở bước 3: `teacher_initialized.pt` |
| 5 | **Chọn Unlabeled**: 1008 mẫu Teacher và Inspector cùng nhãn, xác suất lớn nhất của Inspector trong [0.5, 0.7) | `score_teacher_inspector_agreement.py` | `run_score_teacher_inspector_agreement.slurm` | `teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/selected_unlabeled_1008.json` |
| 6 | **Kịch bản augmentation** cho Student (đầu vào của Stage 2) | `build_augmentation_scenarios.py` (thao tác ảnh/chỉ dẫn trong `student_augmentation.py`) | `run_build_augmentation_scenarios.slurm` | cùng thư mục ở bước 5: `augmentation_combined_scenarios_1008.json` |
| 7 | **Stage 2**: Student–Teacher loop | `train_loop.py` | `run_train_loop.slurm` (submit qua `config_training.sh`) hoặc `config_stage2.sh` | `stage2_tuning_..._<run>/` (bên dưới) |

Bước 1–6 đã chốt. Chỉ còn Stage 2 đang được cải tiến.

## Stage 2

Mỗi optimizer step gồm 8 cặp có nhãn (HumanGT + GemmaGT gộp) và 8 mẫu Unlabeled:
- **Loss có nhãn:** `three_way`, gộp overall bằng `selected_equal`.
- **Loss Unlabeled:** cross-entropy với nhãn mềm của Inspector, cộng KL(Teacher‖Student) trên score (`inspector_soft_ema_consistency`).
- **Teacher:** cập nhật bằng EMA 0.99.
- **Augmentation:** Student nhận đầu vào augment (`composite+mask_content`), còn Teacher nhận ảnh gốc. Cặp có nhãn cũng được augment.
- **Iteration:** 5 iteration, mỗi iteration chọn lại Unlabeled theo agreement Teacher–Inspector của iteration trước. Mỗi iteration lưu 2 checkpoint, learning rate 1e-5 cosine.

Ba run đã train xong (tiền tố `stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_`):

| Run | `--teacher-update` | `--teacher-kl-relevance-weight` |
|---|---|---|
| `ema_step_0.99` | `optimizer_step` (EMA sau mỗi step) | 0 |
| `ema_step_0.99_teacher_kl_relevance_0.5` | `optimizer_step` | 0.5 (thêm KL trên Relevance) |
| `ema_iteration_0.99` | `after_round` (EMA sau mỗi iteration) | 0 |

Chạy lại không cần Slurm:
```bash
bash config_stage2.sh ema_iteration_0.99 --dry-run   # in lệnh
bash config_stage2.sh ema_iteration_0.99             # train (sửa các biến đường dẫn ở đầu file trước)
```

### R0 fixes for iteration EMA (`ema_iteration_0.99_r0`)

`ema_iteration_0.99_r0` in `config_stage2.sh` is `ema_iteration_0.99` with three changes in `scripts/train_loop.py`:

- **Learning-rate schedule (`--restart-lr-each-round`).** The original cosine spans all five iterations (630 planned steps, one 32-step warmup), so iterations 4 and 5 receive about 12% of the summed learning rate and iteration 5 trains at ≤1.4e-6. The flag restarts the cosine, with 5% warmup, over each iteration's actual step count.
- **Teacher export (always on with `--teacher-update after_round`).** The iteration-end `checkpoint-N.pt` and the `best.pt` snapshot were written before that iteration's EMA update, so every exported Teacher lagged one update (`checkpoint-126` held the Stage 1 Teacher; `best.pt` held four of five updates). Both are now re-saved after the update. The Student and the in-memory Teacher used for training are unchanged.
- **Per-iteration Unlabeled selection (`--replay-round-selections`).** Instead of re-running the Teacher–Inspector agreement filter, each iteration trains a fixed list of Unlabeled IDs (new shuffle and augmentation draws). `scripts/build_round_schedule.py` builds the list from a finished run's `round_XX_summary.json`; from `ema_iteration_0.99` it gives 1008, 993, 992, 990, 990 IDs, i.e. 126 + 125 + 124 + 124 + 124 = 623 steps. This holds data and budget fixed when comparing recipes; the loss-plateau stop is disabled while replaying.

Output của một run gồm:
- `checkpoint-N.pt`: Student và Teacher, 2 checkpoint mỗi iteration.
- `best.pt`.
- `round_XX_summary.json`.
- `iteration_selection_round_XX.jsonl`: các mẫu Unlabeled được chọn ở mỗi iteration.
- `training_log_<job>.jsonl`: loss thành phần theo từng step.
- `per_sample_log_<job>.jsonl`.

## Đánh giá (`../RewardModelEvaluation/OurModel/`)

| Việc | Script | Output (`evaluation_results/`) |
|---|---|---|
| Student/Teacher của từng checkpoint trên EditReward-Bench | `evaluate_student_checkpoints.py` + `run_evaluate_student_checkpoints.slurm` (`STUDENT_EVAL_ROLE=student hoặc teacher`) | `train_loop/<run>/checkpoint-N/{bench_evaluation.json, teacher_bench_evaluation.json}` |
| Bench và VLM-Bench theo iteration, bootstrap ghép cặp | `analyze_stage2_bench_iterations.py --run <run>` | `train_loop/<run>/bench_iteration_analysis/` |
| Loss thành phần theo step và theo iteration | `plot_iteration_losses.py [--per-run]` | `output_training/<run>/analysis/losses/` |
| Agreement Student–Teacher–Inspector theo iteration | `analyze_iteration_agreement.py` | `output_training/<run>/analysis/iteration_agreement/` |
| Tóm tắt log từng mẫu | `summarize_per_sample_log.py --version <run>` | `train_loop/<run>/per_sample/` |
| Đánh giá trên dữ liệu train Stage 2 (run `all_once` cũ) | `evaluate_stage2_tuning_checkpoints.py`, `plot_stage2_tuning.py` | `train_loop/<run>/` |

Chỉ số chính:
- **Cặp không hoà (115 cặp):** Accuracy và Margin AUC. Margin AUC là AUC đối xứng của margin trong từng cặp.
- **Có cặp hoà (133 cặp):** AUC theo −|margin| và Tie Head AUC theo P(Tie).
- Trên VLM-Bench chỉ có 2 cặp hoà, nên chỉ số Tie trên tập này không dùng để kết luận được.

## Checkpoint trên Drive

Các thư mục output đã chia sẻ, mỗi thư mục là một file `.zip` không nén:
- `inspector_baseline_...`
- `student_pipeline_...`
- `teacher_inspector_moderate_agreement_...`
- `teacher_inspector_agreement_...`
- 3 run Stage 2

Giải nén vào `Qwen3-VL-Embedding/output_training/`, rồi sửa `BASE_DIR` / các đường dẫn `/datastore/truongtq/...` trong launcher, hoặc đặt các biến ở đầu `config_stage2.sh`.
