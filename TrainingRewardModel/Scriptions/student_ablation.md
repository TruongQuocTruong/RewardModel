# Thí nghiệm cấu hình warm-start Student Model

Cập nhật: 2026-10-18

Phần I (mục 1–8) là ba job soft label ở `--max_pixels 200704`. Phần II (mục 9–17) là các job hard label với cách resize mặc định của processor.

Về độ phân giải: "resize mặc định của processor" là không truyền `max_pixels`. Ảnh vẫn được làm tròn hai cạnh về bội số của 32 và bị giới hạn tối đa 1,310,720 pixel (khoảng 1280 token mỗi ảnh) theo `preprocessor_config.json` của Qwen3-VL-Reranker-8B. Ví dụ ảnh 1080×1920 được đưa vào model ở 832×1504. Inspector cũng dùng đúng cách resize này.

**Kết luận phần I:** ba cấu hình warm-start dùng soft label không khác nhau có ý nghĩa thống kê. Cấu hình rank 64, learning rate 4e-5 có loss đánh giá thấp nhất và độ chính xác cao nhất, nên được dùng làm cơ sở dự đoán cấu hình Student cho phần hoàn thiện pipeline. Ba job này **chỉ dùng để tham khảo**, vì chúng được huấn luyện với soft label và `--max_pixels 200704`. Cả hai điều đó đều khác setting cố định đã thống nhất: hard label (majority vote từ GemmaGT Data) và không giới hạn max_pixels.

## 1. Bối cảnh

- Model gốc: Qwen3-VL-Reranker-8B, huấn luyện bằng MS-Swift (`scripts/train_student_warmstart.py`, `task_type generative_reranker`).
- Dữ liệu: GemmaGT Data `TrainingDataCreation/editreward_overall_finegrained_gemmaGT_3k.jsonl`. File có 3000 cặp, trùng đúng các cặp của HumanGT Data, gồm 2358 ảnh nguồn, mỗi cặp có 3 lần chấm của Gemma trên 6 metric: aesthetic, fidelity, realism, background_consistency, foreground_consistency, structure_consistency.
- Head của Student (`MultiMetricRewardHeads`, khoảng 5.6M tham số) gồm ba phần:
  - Score Head dùng chung, khởi tạo theo hướng yes/no và có thêm MLP phần dư.
  - Tie Head đối xứng.
  - Relevance Head chỉ nhìn ảnh nguồn.
- Loss `FactorizedWarmStartLoss` gồm ba thành phần binary cross-entropy:
  - cho tie;
  - cho người thắng có điều kiện, có trọng số theo relevance và phần khối lượng không hòa;
  - cho relevance.
- Mỗi cặp tạo 3 chuỗi đầu vào: ảnh nguồn + A, ảnh nguồn + B, và chỉ ảnh nguồn. Tổng cộng là 5 ảnh mỗi cặp, so với 4 ảnh của Inspector.
- Tách tập kiểm định theo ảnh nguồn, tỉ lệ 0.15, seed 42. Tập huấn luyện có 2558 cặp (2004 ảnh nguồn), tập kiểm định 442 cặp (354 ảnh nguồn).

## 2. Thiết lập chung

Giữ nguyên ở cả ba job:
- LoRA trên LLM + ViT + aligner, alpha = 2 × rank, `lora_dropout 0.05`, `weight_decay 0.01`.
- Batch 1 cặp × tích lũy gradient 8, warmup 0.05, cosine, 2 epoch (640 bước).
- Soft label (tỉ lệ của 3 lần chấm Gemma), `--max_pixels 200704`.
- Đánh giá mỗi 50 bước, `save_strategy best`, `save_total_limit 3`. Checkpoint được chọn theo loss đánh giá thấp nhất.

## 3. Các job

Ngân sách cập nhật = 2 × learning rate × rank, giống cách tính trong `inspector_ablation.md`.

| Ký hiệu | Thư mục | Rank | Learning rate | Tham số huấn luyện | Ngân sách cập nhật | Job huấn luyện | Job Metrics Weights Aggregator | Job đánh giá |
|---|---|---|---|---|---|---|---|---|
| S1 | `student_warmstart_r32_lr4e-5` | 32 | 4e-5 | 110.55M (1.24%) | 2.56e-3 | 4263 | 4298 | 4303 |
| S2 | `student_warmstart_r64_lr4e-5` | 64 | 4e-5 | 215.54M (2.40%) | 5.12e-3 | 4265 | 4299 | 4304 |
| S3 | `student_warmstart_r32_lr2e-5` | 32 | 2e-5 | 110.55M (1.24%) | 1.28e-3 | 4266 | 4300 | 4305 |

Chi phí tài nguyên:
- Huấn luyện: cả ba chạy khoảng 4 giờ trên partition normal (`gpu:2g.35gb`). Bộ nhớ GPU đỉnh là 20.68 GiB với rank 32 và 22.46 GiB với rank 64. Tốc độ khoảng 22.5 giây mỗi bước, đã tính cả thời gian đánh giá.
- Metrics Weights Aggregator: khoảng 30–33 phút mỗi job.
- Đánh giá: khoảng 15 phút mỗi job.

Lưu ý: tính đến 2026-10-18 00:46, các thư mục huấn luyện `output_training/student_warmstart_*` không còn trên đĩa. Số liệu trong file này lấy từ log (`logs/student_warmstart_*.out`, `logs/preference_aggregator_*.out`) và từ các file kết quả đánh giá còn lại.

## 4. Loss đánh giá trên tập kiểm định (442 cặp, soft target)

| Bước | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 640 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| S1 | 1.0547 | 0.9649 | 0.8192 | 0.7782 | 0.7599 | 0.7585 | 0.7294 | 0.7301 | 0.7081 | 0.7116 | 0.6945 | **0.6938** | 0.6967 |
| S2 | 1.0860 | 0.8996 | 0.7991 | 0.7607 | 0.7617 | 0.7300 | 0.7368 | 0.7196 | 0.6952 | 0.6989 | 0.6802 | **0.6792** | 0.6812 |
| S3 | 1.1808 | 1.0330 | 0.9404 | 0.8233 | 0.8189 | 0.7769 | 0.7729 | 0.7599 | 0.7500 | 0.7469 | 0.7375 | **0.7348** | 0.7355 |

Nhận xét:
- **Không có dấu hiệu học quá khớp.** Khác với Inspector, loss giảm đều đến bước 600 rồi gần như đứng yên. Checkpoint tốt nhất của cả ba đều là checkpoint-600.
- **S2 có loss thấp hơn S1 ở 10 trên 13 lần đánh giá.** Từ bước 450 trở đi, S2 thấp hơn ổn định khoảng 0.013–0.015.
- **S3 hội tụ chậm hơn và kết thúc với loss cao nhất.** Ở bước 600, loss của S3 cao hơn S1 0.041.
- Hai cấu hình có ngân sách cập nhật lớn hơn (S2 lớn nhất) cho loss thấp hơn. Với Student, chưa đạt tới ngưỡng ngân sách gây quá khớp như đã thấy ở Inspector.

## 5. Metrics Weights Aggregator

Aggregator được fit trên các cặp không hòa của HumanGT 3k, với tập kiểm định riêng tách theo ảnh nguồn, tỉ lệ 20%: 1863 cặp huấn luyện, 493 cặp kiểm định.

| | Trọng số fit (aesthetic, fidelity, realism, background, foreground, structure) | β | Độ chính xác kiểm định | Brier kiểm định |
|---|---|---|---|---|
| S1 | 0.258, 0.248, 0.162, 0.095, 0.125, 0.112 | 3.05 | 79.51% | 0.1399 |
| S2 | 0.261, 0.230, 0.178, 0.103, 0.123, 0.105 | 3.17 | 80.73% | 0.1263 |
| S3 | 0.232, 0.238, 0.176, 0.080, 0.120, 0.154 | 2.78 | 77.08% | 0.1575 |

Ở cả ba job, trọng số tập trung vào aesthetic, fidelity và realism (khoảng 63–67% tổng trọng số). Độ chính xác kiểm định ở đây **bị lạc quan**, vì Student đã được huấn luyện trên chính các ảnh này. Số liệu đáng tin là kết quả trên EditReward-Data-Test và EditReward-Bench ở phần 6.

## 6. Đánh giá trên EditReward-Data-Test và EditReward-Bench

Phạm vi và cách đánh giá:
- Chỉ dùng cặp không hòa: Test có 1000 cặp, Bench có 115 cặp.
- Script `Qwen3-VL-Embedding/scripts/evaluate_student_checkpoints.py`. Kết quả nằm ở `RewardModelEvaluation/OurModel/evaluation_results/<thư mục>/evaluation.json`.
- Độ chính xác, Brier và log loss dùng margin = β · Σ aₖ · signed_evidenceₖ.
- AUC gộp dùng điểm từng ảnh Σ aₖ · scoreₖ(ảnh) từ Score Head.
- Khoảng tin cậy 95% lấy bằng bootstrap 1000 lần trên các cặp.
- "Bằng nhau" là trọng số 1/6 cho mỗi metric. "Đã fit" là trọng số của Metrics Weights Aggregator.

| Cấu hình | Cách gộp | Test: độ chính xác | Test: AUC | Test: Brier | Test: log loss | Bench: độ chính xác | Bench: AUC | Bench: Brier | Bench: log loss |
|---|---|---|---|---|---|---|---|---|---|
| S1 | bằng nhau | 73.7 [71.0–76.3] | 68.30 [66.5–70.2] | 0.1797 | 0.5442 | 74.78 [67.0–81.7] | 70.71 [64.9–76.9] | 0.1967 | 0.6037 |
| S1 | đã fit | 73.9 [71.2–76.6] | 68.36 [66.6–70.3] | 0.1792 | 0.5398 | 73.91 [66.1–81.7] | **71.60** [65.7–77.8] | 0.1888 | 0.5783 |
| S2 | bằng nhau | 74.0 [71.2–76.6] | 68.19 [66.5–69.9] | 0.1777 | 0.5400 | 75.65 [67.8–83.5] | 70.12 [63.6–76.4] | 0.1971 | 0.6314 |
| S2 | đã fit | **74.8** [72.1–77.4] | 68.29 [66.6–70.0] | 0.1763 | 0.5346 | **76.52** [68.7–84.4] | 70.89 [64.6–76.9] | 0.1901 | 0.6028 |
| S3 | bằng nhau | 73.9 [71.2–76.6] | 67.99 [66.3–69.7] | 0.1767 | 0.5312 | 75.65 [67.8–82.6] | 70.61 [64.8–76.7] | 0.1827 | 0.5565 |
| S3 | đã fit | 74.2 [71.6–76.8] | 68.28 [66.6–70.0] | 0.1766 | **0.5297** | 74.78 [66.9–81.8] | 71.06 [65.0–76.9] | **0.1766** | **0.5415** |
| Inspector phiên bản 10, checkpoint-200 | — | 74.5 | 65.94 | **0.1713** | 0.5156 | 76.52 | **72.01** | 0.1894 | 0.5889 |

### Kiểm định McNemar

Phép so sánh dùng cách gộp đã fit. Mỗi ô ghi: số cặp chỉ bên thứ nhất đúng / số cặp chỉ bên thứ hai đúng, kèm giá trị p.

| So sánh | Test | Bench |
|---|---|---|
| S2 với S1 | 57 / 48, p = 0.44 | 4 / 1, p = 0.38 |
| S1 với S3 | 57 / 60, p = 0.85 | 6 / 7, p = 1.0 |
| S2 với S3 | 67 / 61, p = 0.66 | 6 / 4, p = 0.75 |
| S1 với Inspector phiên bản 10 | 80 / 86, p = 0.70 | 2 / 5, p = 0.45 |
| S2 với Inspector phiên bản 10 | 74 / 71, p = 0.87 | 1 / 1, p = 1.0 |
| S3 với Inspector phiên bản 10 | 82 / 85, p = 0.88 | 4 / 6, p = 0.75 |
| Đã fit với bằng nhau, S1 | 29 / 27, p = 0.89 | 2 / 3, p = 1.0 |
| Đã fit với bằng nhau, S2 | 33 / 25, p = 0.36 | 4 / 3, p = 1.0 |
| Đã fit với bằng nhau, S3 | 25 / 22, p = 0.77 | 2 / 3, p = 1.0 |

### Độ chính xác của từng metric riêng lẻ trên Test

Mỗi metric được tính theo dấu của signed_evidence.

| Cấu hình | aesthetic | fidelity | realism | background | foreground | structure |
|---|---|---|---|---|---|---|
| S1 | 72.4 | 71.6 | 72.0 | 71.2 | 70.8 | 70.8 |
| S2 | 72.4 | 72.5 | **74.1** | 71.0 | 70.3 | 70.1 |
| S3 | 71.5 | 71.7 | 72.1 | 71.7 | 71.2 | 71.4 |

## 7. Nhận xét

1. **Ba cấu hình không khác nhau có ý nghĩa thống kê.** Mọi giá trị p đều lớn hơn 0.35. Độ chính xác trên Test chỉ chênh tối đa 0.9 điểm, trong khi khoảng tin cậy rộng khoảng ±2.7 điểm. Bench chỉ có 115 cặp, nên khoảng tin cậy rộng khoảng ±8 điểm và không đủ để phân biệt các cấu hình.
2. **S2 (rank 64) tốt nhất về loss đánh giá và độ chính xác.** Đây là ba tín hiệu cùng chiều nhưng đều nhỏ: loss đánh giá thấp nhất (0.6792), độ chính xác Test và Bench cao nhất, và độ chính xác kiểm định của Metrics Weights Aggregator cao nhất (80.73%).
3. **S3 (learning rate 2e-5) hiệu chỉnh xác suất tốt nhất trên Bench**, với Brier 0.1766 và log loss 0.5415, tốt hơn cả Inspector. Tuy nhiên loss đánh giá của S3 cao nhất. Điều này cho thấy S3 học chưa đủ, nên margin của nó nhỏ hơn và ít khi tự tin quá mức.
4. **Student ngang Inspector phiên bản 10 về độ chính xác.** Trên Test là 74.8 so với 74.5 (p = 0.87). Trên Bench là 76.52 so với 76.52.
5. **AUC gộp trên Test của Student cao hơn Inspector khoảng 2.3–2.4 điểm** (khoảng 68.3 so với 65.94). *Ghi chú 21/10/2026: AUC ở phần I là AUC gộp cũ. Ở phần II, margin AUC cho thấy ưu thế này đến từ độ lệch điểm giữa các ngữ cảnh (mục 16); nhận xét này không còn đúng.* Score Head xếp hạng từng ảnh tốt hơn điểm yes/no của Inspector. Trên Bench thì Inspector cao hơn (72.0), nhưng khoảng tin cậy chồng lấn hoàn toàn.
6. **Về hiệu chỉnh xác suất trên Test, Inspector vẫn tốt hơn**: Brier 0.1713, so với tốt nhất của Student là 0.1763.
7. **Trọng số đã fit gần như không giúp gì so với trọng số bằng nhau.** Mức tăng chỉ 0.2–0.8 điểm, với p ≥ 0.36. Độ chính xác của sáu metric riêng lẻ đều quanh 70–74%. Như vậy các metric rất tương quan với nhau, và Student đang học một "chất lượng chung" nhiều hơn là sáu tiêu chí tách biệt.
8. **Chưa thấy giới hạn trên của ngân sách cập nhật.** Ngân sách lớn hơn cho kết quả tốt hơn hoặc bằng, và không có cấu hình nào quá khớp trong 2 epoch.

## 8. Hạn chế

- Ba job dùng soft label và `--max_pixels 200704`. Các lần chạy tiếp theo dùng hard label và resize mặc định của processor, nên thứ tự giữa các cấu hình có thể thay đổi.
- Phép so sánh với Inspector phiên bản 10 chưa hoàn toàn công bằng, vì Inspector được huấn luyện và đánh giá với resize mặc định của processor.
- Tập kiểm định của Metrics Weights Aggregator dùng chung ảnh với tập huấn luyện của Student.
- Mỗi cấu hình chỉ chạy với một seed.

# Phần II. Student hard label, resize mặc định của processor

Cập nhật: 2026-10-21. Gồm 8 cấu hình, gọi theo setting (LoRA rank, learning rate, số epoch).

## 9. Setting cố định mới

Đây là setting cố định cho Student ở cả giai đoạn 1 và giai đoạn 2, áp dụng từ phần này:

- **Hard label:** majority vote trên 3 lần chấm của Gemma, dùng script `scripts/train_student_hardlabel.py`.
  - Relevance bằng 1 khi có ít nhất 2 trên 3 lần chấm áp dụng metric.
  - Lựa chọn A, B hay Tie phải có đa số tuyệt đối (trên 50%) trong các lần chấm áp dụng; nếu không có thì gán Tie.
  - Tập kiểm định vẫn dùng soft target, nên loss đánh giá vẫn so sánh được với phần I.
  - Thống kê trên tập huấn luyện: 11503 ô metric có liên quan, 3797 ô không liên quan. Có 1699 ô đổi nhãn so với soft label, và 67 ô không có đa số nên bị chuyển thành Tie.
- **Không giới hạn max_pixels:** dùng cách resize mặc định của processor, giống Inspector.
- **Lưu checkpoint giống Inspector:**
  - `--save_strategy steps`, `--save_steps 50`, `--eval_steps 50`, `--save_total_limit 5`;
  - `--metric_for_best_model loss`;
  - không còn `save_strategy best` và không còn `load_best_model_at_end`.

  Transformers 4.57.6 vẫn ghi `best_model_checkpoint` và không xoá checkpoint tốt nhất khi xoay vòng. `student_warmstart.pt` được tạo từ checkpoint đó.
- **Chia train/validation ngẫu nhiên theo cặp**, giống `--split_dataset_ratio` của MS-Swift, với tỉ lệ 0.15 và seed 42. Một ảnh nguồn có thể xuất hiện ở cả hai tập với edit instruction và ảnh chỉnh sửa khác nhau.
  - Kết quả: 2550 cặp huấn luyện (2060 ảnh nguồn), 450 cặp kiểm định (432 ảnh nguồn), 134 ảnh nguồn chung giữa hai tập.
  - Mọi job trong phần II dùng **cùng tập kiểm định 450 cặp**, nên loss đánh giá so sánh trực tiếp được với nhau. Tập này khác tập 442 cặp của phần I.
- Các phần giữ nguyên như phần I: LoRA trên LLM + ViT + aligner, alpha = 2 × rank, batch 1 cặp × tích lũy gradient 8, warmup 0.05, cosine, `lora_dropout 0.05`, `weight_decay 0.01`.

## 10. Thay đổi code

Bản sao các file trước khi sửa nằm trong thư mục scratchpad `backup_before_pipeline` của phiên làm việc.

| File | Thay đổi |
|---|---|
| `run_student_hardlabel_training.slurm` | <ul><li>Thêm biến môi trường `WARMSTART_LORA_RANK` (mặc định 64), `WARMSTART_LEARNING_RATE` (4e-5), `WARMSTART_NUM_TRAIN_EPOCHS` (2), `WARMSTART_VALIDATION_RATIO` (0.15). `lora_alpha` bằng 2 × rank.</li><li>Bắt buộc truyền `WARMSTART_OUTPUT_DIR`.</li><li>Bỏ `--max_pixels 200704` và các `export MAX_PIXELS` / `MIN_PIXELS`.</li><li>Đổi sang lưu checkpoint theo bước, giữ 5 checkpoint. Bỏ `--greater_is_better false`, `--load_best_model_at_end true` và `--save_only_model false`.</li><li>Bỏ mặc định dùng `shared_source_split.json`.</li><li>Đường dẫn conda ghi cố định là `/datastore/truongtq/miniconda3`, để account khác chạy được.</li><li>Partition mặc định là large `gpu:3g.71gb`.</li></ul> |
| `Qwen3-VL-Embedding/scripts/train_student_warmstart.py` | Thêm hàm `random_pair_split`, thay cho `source_grouped_split` khi không có shared split. Checkpoint ghi `split_method = random_pair_v1`. Script hard label kế thừa thay đổi này. |
| `Qwen3-VL-Embedding/scripts/evaluate_student_checkpoints.py`, `fit_preference_aggregator.py`, `train_loop.py`, `RewardModelEvaluation/OurModel/evaluate_reward_model.py` và các slurm tương ứng | `--min-pixels` / `--max-pixels` mặc định không đặt, tức dùng cách resize mặc định của processor. |

## 11. Các job đã chạy

Thư mục huấn luyện nằm trong `Qwen3-VL-Embedding/output_training/`. Kết quả đánh giá nằm trong `RewardModelEvaluation/OurModel/evaluation_results/<cùng tên thư mục>/evaluation.json`.

| LoRA rank | Learning rate | Epoch | Số bước | Thư mục | Account | Partition | Job huấn luyện | Job Metrics Weights Aggregator | Job đánh giá |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 4e-5 | 2 | 638 | `student_hardlabel_r64_lr4e-5` (cấu hình gốc, pipeline đang dùng) | truongtq | full `7g.141gb` | 4310 | 4311 | 4349 |
| 32 | 4e-5 | 2 | 638 | `student_hardlabel_r32_lr4e-5` | anhndt | normal `2g.35gb` | 4314 | 4350 | 4351 |
| 64 | 2e-5 | 2 | 638 | `student_hardlabel_r64_lr2e-5` | anhndt | normal | 4316 | 4352 | 4353 |
| 64 | 8e-5 | 2 | 638 | `student_hardlabel_r64_lr8e-5` | anhndt | normal | 4317 | 4354 | 4355 |
| 64 | 4e-5 | 3 | 957 | `student_hardlabel_r64_lr4e-5_epoch3` | anhndt | normal | 4318 | 4469 | 4470 |
| 64 | 2e-5 | 3 | 957 | `student_hardlabel_r64_lr2e-5_epoch3` | anhndt | normal | 4319 | 4471 | 4472 |
| 32 | 2e-5 | 2 | 638 | `student_hardlabel_r32_lr2e-5` | anhndt | normal | 4611 | 4612 | 4613 |
| 32 | 8e-5 | 2 | 638 | `student_hardlabel_r32_lr8e-5` | anhndt | normal | 4614 | 4615 | 4616 |

Các cặp so sánh, mỗi cặp chỉ khác đúng một yếu tố:

| Yếu tố | Các cặp |
|---|---|
| LoRA rank (32 so với 64) | ở learning rate 2e-5, 4e-5 và 8e-5, cùng 2 epoch |
| Learning rate (2e-5, 4e-5, 8e-5) | ở rank 32 và ở rank 64, cùng 2 epoch |
| Số epoch (2 so với 3) | ở rank 64 với learning rate 4e-5 và 2e-5 |

Ghi chú:

- **Cấu hình gốc** (rank 64, 4e-5, 2 epoch) submit lần đầu là job 4308 lên partition large, nhưng GPU large đều bận nên đã huỷ và submit lại thành job 4310 trên partition full với `--qos=mig7-141`.
- **Job rank 128** nằm trong đề xuất ban đầu đã bị huỷ trước khi chạy. Sau đó thêm rank 32 với learning rate 2e-5 và 8e-5 để có lưới đầy đủ rank {32, 64} × learning rate {2e-5, 4e-5, 8e-5}.
- **Bộ nhớ GPU đỉnh:** 29.76–29.9 GiB với rank 64 và 27.89 GiB với rank 32, đều vừa với partition normal (35 GB).
- **Thời gian huấn luyện:** khoảng 8 giờ 30 phút cho 2 epoch và khoảng 13 giờ cho 3 epoch trên MIG `2g.35gb`; 2 giờ 46 phút cho 2 epoch trên GPU full. Mỗi job Metrics Weights Aggregator mất khoảng 1 giờ, mỗi job đánh giá khoảng 30 phút.
- **Chuỗi job tự động:** từ hai cấu hình rank 32 mới, ba bước huấn luyện → aggregator → đánh giá được submit cùng lúc, nối với nhau bằng `--dependency=afterok`:

```bash
cd /datastore/truongtq/TrainingRewardModel/TrainingRewardModel
OUT=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training
for LR in 2e-5 8e-5; do
    RUN=student_hardlabel_r32_lr$LR
    TRAIN_JOB=$(WARMSTART_LORA_RANK=32 WARMSTART_LEARNING_RATE=$LR WARMSTART_NUM_TRAIN_EPOCHS=2 \
                WARMSTART_OUTPUT_DIR=$OUT/$RUN \
                sbatch --parsable --partition=normal --gres=gpu:2g.35gb:1 run_student_hardlabel_training.slurm)
    AGG_JOB=$(AGGREGATOR_STUDENT_CHECKPOINT=$OUT/$RUN/student_warmstart.pt \
              AGGREGATOR_OUTPUT=$OUT/$RUN/preference_aggregator.json \
              sbatch --parsable --dependency=afterok:$TRAIN_JOB run_preference_aggregator.slurm)
    STUDENT_EVAL_RUN_DIR=$OUT/$RUN \
    sbatch --dependency=afterok:$AGG_JOB run_evaluate_student_checkpoints.slurm
done
```

## 12. Loss đánh giá trên tập kiểm định (450 cặp, soft target)

Ô in đậm là checkpoint được chọn (loss thấp nhất), ghi trong `student_warmstart.pt`.

**2 epoch (638 bước):**

| LoRA rank, learning rate | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 638 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 32, 2e-5 | 1.2147 | 1.0960 | 0.9953 | 0.8510 | 0.8100 | 0.7539 | 0.7403 | **0.7341** | 0.7477 | 0.7514 | 0.7387 | 0.7405 | 0.7377 |
| 32, 4e-5 | 1.0608 | 1.0362 | 0.8064 | 0.7514 | 0.8051 | 0.7003 | 0.7002 | 0.6865 | 0.6900 | **0.6616** | 0.6862 | 0.6756 | 0.6745 |
| 32, 8e-5 | 1.1005 | 0.8763 | 0.7955 | 0.7086 | 0.7197 | 0.6703 | 0.7174 | 0.6703 | 0.6522 | **0.6221** | 0.6397 | 0.6270 | 0.6262 |
| 64, 2e-5 | 1.2054 | 1.0816 | 0.8646 | 0.7751 | 0.7675 | 0.7255 | 0.7121 | 0.7197 | 0.7228 | **0.7073** | 0.7189 | 0.7132 | 0.7132 |
| 64, 4e-5 | 1.0479 | 0.9473 | 0.8217 | 0.7417 | 0.7339 | 0.7005 | 0.7100 | 0.7305 | 0.6838 | 0.6856 | 0.6732 | **0.6687** | 0.6701 |
| 64, 8e-5 | 1.1522 | 0.8359 | 0.7424 | 0.7180 | 0.7048 | 0.6481 | 0.6710 | 0.6571 | 0.6686 | 0.6317 | 0.6367 | 0.6285 | **0.6275** |

**3 epoch (957 bước), rank 64:**

| Learning rate | 50 | 100 | 150 | 200 | 250 | 300 | 350 | 400 | 450 | 500 | 550 | 600 | 650 | 700 | 750 | 800 | 850 | 900 | 950 | 957 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 4e-5 | 1.1013 | 1.1240 | 0.8198 | 0.7434 | 0.7401 | 0.7358 | 0.6968 | 0.7390 | 0.7265 | 0.6780 | **0.6425** | 0.6498 | 0.6808 | 0.7393 | 0.7236 | 0.7175 | 0.7082 | 0.7096 | 0.7102 | 0.7139 |
| 2e-5 | 1.2460 | 1.0882 | 0.8477 | 0.7910 | 0.7794 | 0.7347 | 0.7104 | 0.7234 | 0.7220 | 0.6914 | 0.6742 | **0.6703** | 0.6983 | 0.7028 | 0.6998 | 0.6968 | 0.6998 | 0.6963 | 0.6925 | 0.6930 |

Nhận xét về loss kiểm định:

1. **Learning rate càng cao thì loss càng thấp, ở cả hai rank:** 0.7341 → 0.6616 → 0.6221 với rank 32, và 0.7073 → 0.6687 → 0.6275 với rank 64. Rank 32, learning rate 8e-5 có loss thấp nhất trong toàn bộ ablation.
2. **Learning rate 2e-5 học chưa đủ trong 2 epoch:** loss gần như đứng yên từ bước 350–400.
3. **Epoch thứ 3 gây học quá khớp.** Với learning rate 4e-5, loss xuống thấp nhất ở bước 550 (ngay cuối epoch 2) rồi tăng lên 0.7393 ở bước 700, khi dữ liệu bắt đầu lặp lại lần thứ ba. Learning rate 2e-5 tăng nhẹ hơn. Checkpoint được chọn của cả hai nằm ở giữa lịch cosine 3 epoch, khi learning rate còn khoảng 30–40% giá trị ban đầu.
4. **Đường loss của hard label dao động mạnh giữa hai lần đánh giá liền nhau** (ví dụ rank 32, 4e-5: 0.7514 → 0.8051 → 0.7003 ở bước 200–300), vì target hard sắc hơn còn tập kiểm định dùng soft target. Chênh lệch loss dưới khoảng 0.01 giữa hai cấu hình không đáng tin.

## 13. Metrics Weights Aggregator

Cài đặt giống phần I: đặc trưng `signed_evidence`, 100 lần bootstrap, phạt L2 0.01, tập kiểm định 20% tách theo ảnh nguồn, seed 42, resize mặc định của processor.

| LoRA rank, learning rate, epoch | Trọng số fit (aesthetic, fidelity, realism, background, foreground, structure) | β | Độ chính xác kiểm định | Brier kiểm định |
|---|---|---|---|---|
| 32, 2e-5, 2 | 0.267, 0.205, 0.142, 0.096, 0.161, 0.128 | 2.55 | 74.65% | 0.1745 |
| 32, 4e-5, 2 | 0.203, 0.246, 0.169, 0.098, 0.130, 0.154 | 2.85 | 79.11% | 0.1526 |
| 32, 8e-5, 2 | 0.223, 0.266, 0.184, 0.075, 0.113, 0.139 | 3.11 | 81.34% | 0.1374 |
| 64, 2e-5, 2 | 0.200, 0.272, 0.190, 0.094, 0.127, 0.117 | 2.77 | 77.89% | 0.1573 |
| 64, 4e-5, 2 | 0.224, 0.215, 0.238, 0.071, 0.112, 0.140 | 2.85 | 80.53% | 0.1353 |
| 64, 8e-5, 2 | 0.212, 0.235, 0.187, 0.075, 0.115, 0.175 | 3.19 | 84.58% | 0.1158 |
| 64, 4e-5, 3 | 0.198, 0.187, 0.171, 0.093, 0.155, 0.195 | 3.18 | 84.99% | 0.1218 |
| 64, 2e-5, 3 | 0.216, 0.359, 0.084, 0.131, 0.108, 0.102 | 3.01 | 80.53% | 0.1428 |

- Độ chính xác kiểm định ở đây bị lạc quan vì Student đã được huấn luyện trên chính các ảnh này.
- Trọng số không ổn định giữa các lần fit. Ví dụ cấu hình rank 64, 2e-5, 3 epoch cho fidelity 0.359 nhưng realism chỉ 0.084, trong khi ở các cấu hình khác hai metric này gần nhau. Nguyên nhân là sáu metric tương quan mạnh, nên nhiều bộ trọng số khác nhau cho cùng một kết quả.

## 14. Kết quả trên EditReward-Data-Test và EditReward-Bench

- Test: EditReward-Data-Test, 1000 cặp không hòa. Bench: EditReward-Bench, 115 cặp không hòa.
- **Cách gộp chính là `selected_equal`** (quyết định ngày 2026-10-20): mỗi cặp chỉ dùng các metric mà Relevance Head cho xác suất ≥ 0.5 (nếu không metric nào đạt thì lấy metric có xác suất cao nhất), rồi lấy trung bình bằng nhau của `signed_evidence`. Cách gộp này không có β nên không có Brier và log loss. Kết quả được tính lại từ `evaluation.json` bằng `RewardModelEvaluation/OurModel/recompute_overall_selected_metrics.py`.
- Cách gộp phụ là `all_metrics_fitted`: cả sáu metric với trọng số và β của Metrics Weights Aggregator. Đây là cách gộp dùng trong các mục 6 và các phân tích trước ngày 2026-10-20.
- Khoảng tin cậy 95% lấy bằng bootstrap 1000 lần trên các cặp. **AUC là margin AUC đối xứng trong cặp** (từ 21/10/2026; mục 16 và `pipeline_process.md` mục 3.6), tính trên đúng margin dùng để quyết định của mỗi cách gộp. Kết quả ở `RewardModelEvaluation/OurModel/evaluation_results/margin_auc_summary.json`, sinh bằng `recompute_margin_auc.py`. Cột AUC gộp cũ chỉ để tham khảo. In đậm là giá trị tốt nhất của mỗi cột.

### Cách gộp chính: `selected_equal`

| LoRA rank | Learning rate | Epoch | Checkpoint | Loss kiểm định | Test-Acc | Test margin AUC | Bench-Acc | Bench margin AUC | AUC gộp cũ (Test / Bench) |
|---|---|---|---|---|---|---|---|---|---|
| 32 | 2e-5 | 2 | 400 | 0.7341 | 73.8 [71.0–76.5] | 80.66 [77.95–83.18] | 73.91 [65.2–81.7] | 78.66 [69.62–87.45] | 67.66 / 71.12 |
| 32 | 4e-5 | 2 | 500 | 0.6616 | 75.3 [72.6–77.9] | 81.97 [79.40–84.61] | 75.65 [67.8–83.5] | **81.16** [72.84–88.83] | 69.01 / 72.71 |
| 32 | 8e-5 | 2 | 500 | **0.6221** | 75.7 [73.1–78.2] | 82.09 [79.53–84.39] | 75.65 [68.7–82.6] | 80.57 [72.01–88.67] | 68.94 / 71.47 |
| 64 | 2e-5 | 2 | 500 | 0.7073 | 74.4 [71.7–77.1] | 81.76 [79.03–84.31] | 75.65 [67.8–82.6] | 78.12 [68.92–87.15] | 67.86 / 70.21 |
| 64 | 4e-5 | 2 | 600 | 0.6687 | 75.1 [72.4–77.7] | 81.88 [79.40–84.47] | **77.39** [69.6–85.2] | 80.72 [72.10–88.56] | 67.72 / 71.93 |
| 64 | 8e-5 | 2 | 638 | 0.6275 | **75.9** [73.4–78.5] | 82.38 [79.75–84.81] | 74.78 [67.0–82.6] | 80.76 [72.00–88.81] | 68.37 / 72.61 |
| 64 | 4e-5 | 3 | 550 | 0.6425 | 74.3 [71.6–76.8] | 82.05 [79.51–84.46] | 74.78 [67.0–82.6] | 80.50 [71.37–88.57] | 67.35 / 71.14 |
| 64 | 2e-5 | 3 | 600 | 0.6703 | 73.7 [70.8–76.3] | 81.57 [79.05–84.16] | 74.78 [67.0–82.6] | 78.91 [70.00–87.20] | 68.35 / 70.17 |
| *Inspector phiên bản 10* | | | | | 74.5 | **82.41** [79.73–84.81] | 76.52 | 79.81 [71.16–87.71] | 65.94 / 72.01 |

### Cách gộp phụ: `all_metrics_fitted`

| LoRA rank | Learning rate | Epoch | Test-Acc | Test margin AUC | Test Brier | Test log loss | Bench-Acc | Bench margin AUC | Bench Brier | Bench log loss | AUC gộp cũ (Test / Bench) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 32 | 2e-5 | 2 | 73.5 | 80.65 | 0.1802 | 0.5412 | 75.65 | 79.22 | 0.1882 | 0.5709 | 67.99 / 71.67 |
| 32 | 4e-5 | 2 | 75.5 | 81.88 | **0.1748** | **0.5302** | 73.91 | **80.76** | 0.1866 | **0.5675** | 68.56 / 72.52 |
| 32 | 8e-5 | 2 | **75.9** | 82.01 | 0.1757 | 0.5368 | 73.91 | 80.49 | 0.1915 | 0.5882 | 68.54 / 71.87 |
| 64 | 2e-5 | 2 | 74.2 | 81.67 | 0.1770 | 0.5329 | 73.91 | 78.48 | 0.1910 | 0.5870 | 67.54 / 69.73 |
| 64 | 4e-5 | 2 | 74.5 | 81.70 | 0.1776 | 0.5385 | 75.65 | 80.75 | **0.1859** | 0.5711 | 67.26 / 71.31 |
| 64 | 8e-5 | 2 | 75.8 | 82.26 | 0.1756 | 0.5395 | 75.65 | 79.93 | 0.1888 | 0.5983 | 67.60 / 71.95 |
| 64 | 4e-5 | 3 | 75.3 | 82.08 | 0.1761 | 0.5365 | **77.39** | 79.83 | 0.1873 | 0.5916 | 66.68 / 70.17 |
| 64 | 2e-5 | 3 | 74.3 | 81.63 | 0.1783 | 0.5379 | 75.65 | 79.40 | 0.1919 | 0.5936 | 68.49 / 70.31 |
| *Inspector phiên bản 10* | | | 74.5 | **82.41** | **0.1713** | **0.5156** | **76.52** | 79.81 | 0.1894 | 0.5889 | 65.94 / 72.01 |

## 15. So sánh ghép cặp (cách gộp chính `selected_equal`)

- Độ chính xác: kiểm định McNemar. Cột ghi số cặp chỉ bên trái đúng / chỉ bên phải đúng.
- Chênh lệch margin AUC: bên trái trừ bên phải, khoảng tin cậy 95% từ 1000 lần lấy mẫu lặp **ghép theo cùng một tập cặp** cho cả hai model. Cách ghép này nhạy hơn nhiều so với việc so khoảng tin cậy riêng của từng model ở mục 14.
- In đậm là các trường hợp p < 0.05 hoặc khoảng tin cậy không chứa 0.

| So sánh | Yếu tố | Test: McNemar | Test: p | Test: chênh lệch margin AUC | Bench: McNemar | Bench: chênh lệch margin AUC |
|---|---|---|---|---|---|---|
| rank 32 với rank 64, ở 2e-5 | rank | 50 / 56 | 0.63 | −1.11 [−2.26, +0.09] | 1 / 3, p = 0.63 | +0.54 [−2.28, +3.49] |
| rank 32 với rank 64, ở 4e-5 | rank | 45 / 43 | 0.92 | +0.09 [−0.89, +1.12] | 1 / 3, p = 0.63 | +0.44 [−1.26, +2.55] |
| rank 32 với rank 64, ở 8e-5 | rank | 48 / 50 | 0.92 | −0.28 [−1.49, +0.94] | 7 / 6, p = 1.0 | −0.19 [−3.24, +2.99] |
| rank 32: 4e-5 với 2e-5 | learning rate | 64 / 49 | 0.19 | **+1.31 [+0.06, +2.52]** | 5 / 3, p = 0.73 | +2.50 [−0.24, +5.39] |
| rank 32: 8e-5 với 4e-5 | learning rate | 47 / 43 | 0.75 | +0.12 [−0.89, +1.16] | 3 / 3, p = 1.0 | −0.59 [−2.72, +1.55] |
| rank 32: 8e-5 với 2e-5 | learning rate | 78 / 59 | 0.12 | **+1.44 [+0.01, +2.82]** | 4 / 2, p = 0.69 | +1.91 [−1.42, +5.31] |
| rank 64: 4e-5 với 2e-5 | learning rate | 53 / 46 | 0.55 | +0.12 [−1.04, +1.30] | 5 / 3, p = 0.73 | +2.60 [−0.12, +5.09] |
| rank 64: 8e-5 với 4e-5 | learning rate | 62 / 54 | 0.52 | +0.50 [−0.97, +1.83] | 3 / 6, p = 0.51 | +0.04 [−3.65, +3.37] |
| rank 64, 4e-5: 3 epoch với 2 epoch | số epoch | 50 / 58 | 0.50 | +0.17 [−1.10, +1.45] | 2 / 5, p = 0.45 | −0.22 [−2.71, +2.22] |
| rank 64, 2e-5: 3 epoch với 2 epoch | số epoch | 41 / 48 | 0.53 | −0.19 [−1.26, +0.96] | 3 / 4, p = 1.0 | +0.79 [−2.09, +3.67] |
| rank 32, 8e-5 với rank 64, 4e-5 (cấu hình pipeline) | rank và learning rate | 57 / 51 | 0.63 | +0.21 [−0.94, +1.38] | 3 / 5, p = 0.73 | −0.15 [−2.39, +2.27] |
| rank 32, 8e-5 với Inspector phiên bản 10 | — | 73 / 61 | 0.34 | −0.32 [−2.03, +1.29] | 4 / 5, p = 1.0 | +0.76 [−2.79, +4.39] |
| rank 32, 4e-5 với Inspector phiên bản 10 | — | 66 / 58 | 0.53 | −0.44 [−2.04, +1.21] | 2 / 3, p = 1.0 | +1.35 [−2.37, +4.66] |
| rank 64, 4e-5 với Inspector phiên bản 10 | — | 74 / 68 | 0.68 | −0.53 [−2.26, +1.14] | 3 / 2, p = 1.0 | +0.91 [−2.87, +4.42] |
| rank 64, 8e-5 với Inspector phiên bản 10 | — | 79 / 65 | 0.28 | −0.03 [−1.74, +1.54] | 2 / 4, p = 0.69 | +0.95 [−2.96, +4.82] |

Với cách gộp phụ `all_metrics_fitted`, chênh lệch margin AUC trên Test giữa rank 32 và rank 64 là −1.02 [−2.15, +0.16] ở 2e-5, +0.18 [−0.78, +1.19] ở 4e-5 và −0.25 [−1.48, +1.00] ở 8e-5. Ở rank 32, 4e-5 và 8e-5 hơn 2e-5 khoảng +1.2 đến +1.4 điểm, nhưng khoảng tin cậy chạm 0. Mọi Student so với Inspector phiên bản 10 nằm trong khoảng −0.71 đến −0.15, đều chứa 0. Chênh lệch Brier và log loss giữa các cấu hình Student đều có khoảng tin cậy chứa 0. So với Inspector phiên bản 10, Brier của Student kém hơn nhẹ; chỉ rank 32, 2e-5 kém hơn có ý nghĩa thống kê (+0.0089 [+0.0008, +0.0172]).

**So với kết luận cũ dựa trên AUC gộp:** ưu thế của rank 32 so với rank 64 ở 4e-5 (+1.30) và 8e-5 (+0.57; +0.94 với cách gộp phụ), và ưu thế +1.8 đến +3.1 điểm của Student so với Inspector, đều biến mất với margin AUC. Các chênh lệch đó đến từ độ lệch điểm giữa các ngữ cảnh (mục 16), không phải từ khả năng xếp hạng trong cặp.

## 16. Margin AUC và AUC gộp: cách đọc khi độ chính xác bằng nhau

**Độ chính xác** chỉ xét dấu của margin trong từng cặp. Mỗi cặp đóng góp 0 hoặc 1. Trên Bench, mỗi cặp bằng 0.87 điểm phần trăm, nên khoảng tin cậy rộng khoảng ±8 điểm.

**Margin AUC (AUC chính từ 21/10/2026).** Với mỗi cặp đặt d = nhãn × margin (dương nếu model đúng), rồi tính ROC AUC của {d} so với {−d}. Kết quả bằng xác suất d_i + d_j > 0 trên hai cặp ngẫu nhiên và không phụ thuộc thứ tự A/B. Margin là đúng đại lượng dùng để quyết định: với `selected_equal` là trung bình `signed_evidence` trên các metric được chọn. Margin AUC xấp xỉ p² + 2p(1 − p)·q, với p là độ chính xác và q là xác suất độ tự tin ở một cặp đúng lớn hơn độ tự tin ở một cặp sai. Trên Test, q khoảng 68–71%, nên margin AUC cao hơn độ chính xác khoảng 6–8 điểm. Hai cấu hình cùng độ chính xác mà khác margin AUC nghĩa là một model đặt margin lớn vào đúng các cặp dễ tốt hơn model kia.

**AUC gộp (cũ, chỉ tham khảo)** so mọi ảnh thắng với mọi ảnh thua, **kể cả của các cặp khác nhau**. Phần lớn phép so sánh là giữa hai ảnh khác ảnh nguồn và khác instruction. Kết quả bị chi phối bởi mức điểm riêng của từng ngữ cảnh. Trên Test, độ lệch chuẩn của điểm trung bình mỗi cặp lớn gấp 1.5–2 lần nửa khoảng cách trong cặp (ví dụ rank 32, 4e-5: 1.82 so với 1.21; rank 64, 4e-5: 2.55 so với 1.54). Margin AUC chính là AUC gộp sau khi trừ điểm trung bình của từng cặp, nên khoảng 13–16 điểm chênh giữa hai cách là do độ lệch giữa các ngữ cảnh. AUC gộp chỉ có ý nghĩa nếu cần dùng điểm từng ảnh như reward tuyệt đối để so giữa các prompt. Điều này chưa được kiểm tra đúng cách vì Test và Bench chỉ có nhãn theo cặp.

**Giới hạn của margin AUC:**
- Bench chỉ có 115 cặp, khoảng tin cậy rộng khoảng ±8 điểm. Kết luận nên dựa trên Test và trên chênh lệch ghép cặp.
- Margin AUC không đo calibration; Brier và log loss vẫn cần khi dùng xác suất làm nhãn mềm.
- AUC theo margin thô (không qua `signed_evidence`) lệch tối đa 0.6 điểm và không đổi kết luận.

## 17. Nhận xét phần II

1. **Độ chính xác của 8 cấu hình không phân biệt được trên tập ngoài.** Với `selected_equal`, độ chính xác Test nằm trong khoảng 73.7–75.9, mọi so sánh một yếu tố có p ≥ 0.12. Bench quá nhỏ để phân biệt.
2. **Rank 32 và rank 64 ngang nhau (cập nhật 21/10/2026).** Chênh lệch margin AUC trên Test là −1.11 (2e-5), +0.09 (4e-5) và −0.28 (8e-5), đều có khoảng tin cậy chứa 0, với cả hai cách gộp. Ưu thế của rank 32 trước đây chỉ có ở AUC gộp, tức là đến từ độ lệch điểm giữa các ngữ cảnh. Rank 32 vẫn có lợi về số tham số (khoảng một nửa).
3. **Ở rank 32, learning rate 2e-5 học chưa đủ; 4e-5 và 8e-5 ngang nhau.** 4e-5 và 8e-5 hơn 2e-5 về margin AUC trên Test (+1.31 [+0.06, +2.52] và +1.44 [+0.01, +2.82]); khoảng tin cậy chỉ vừa không chứa 0, nên đây là tín hiệu yếu. Hai mức 4e-5 và 8e-5 không khác nhau (+0.12). Ở rank 64, learning rate không tạo khác biệt có ý nghĩa.
4. **Thêm epoch thứ 3 không giúp gì:** không cải thiện độ chính xác (mọi p ≥ 0.50) hay margin AUC (chênh lệch −0.19 và +0.17), tốn thêm khoảng 50% thời gian huấn luyện, và gây học quá khớp ở epoch 3.
5. **So với Inspector phiên bản 10, Student ngang về cả độ chính xác lẫn margin AUC** (cập nhật 21/10/2026). Chênh lệch margin AUC trên Test là −0.53 đến −0.03, trên Bench +0.76 đến +1.35, đều có khoảng tin cậy chứa 0. Kết luận cũ "AUC gộp trên Test của Student cao hơn rõ rệt" đến từ độ lệch điểm giữa các ngữ cảnh, không phải từ khả năng xếp hạng trong cặp. Xác suất theo cặp (Brier, log loss) của Student kém hơn nhẹ.
6. **Sáu metric tương quan mạnh**, nên trọng số do Metrics Weights Aggregator fit không ổn định và không hơn trung bình bằng nhau.
7. **Lựa chọn Student cho pipeline.** Theo tiêu chí đã thống nhất (loss kiểm định thấp nhất), cấu hình tốt nhất là **rank 32, learning rate 8e-5, 2 epoch** (0.6221). Cấu hình này cũng thuộc nhóm cao nhất về độ chính xác Test (75.7) và margin AUC Test (82.09), và chỉ dùng khoảng một nửa số tham số huấn luyện của rank 64. So với cấu hình pipeline đang dùng (rank 64, 4e-5), độ chính xác không khác (p = 0.63) và margin AUC cũng không khác: +0.21 [−0.94, +1.38] trên Test, −0.15 [−2.39, +2.27] trên Bench. Lý do đổi Student trước đây (ưu thế AUC gộp +1.2 đến +1.3) không còn. Lợi ích còn lại chỉ là loss kiểm định thấp hơn và ít tham số hơn. Đổi Student nghĩa là chạy lại pipeline; quyết định thuộc về người làm thí nghiệm.

Hạn chế và việc còn lại:
- Mỗi cấu hình chỉ chạy một seed. Ở Inspector, chạy lại cùng seed cho độ dao động khoảng ±0.4 điểm độ chính xác. Với margin AUC, rank 32 không còn ưu thế nên không cần chạy lại seed để xác nhận điểm này. Nếu cần đo độ dao động theo seed, phải thêm biến `WARMSTART_SEED` vào `run_student_hardlabel_training.slurm`, vì hiện đang ghi cố định `--seed 42`.
- Có nhiều phép so sánh cùng lúc, nên các giá trị p gần 0.05 chỉ là tín hiệu yếu.
- Quyết định có đổi Student của pipeline từ rank 64, 4e-5 sang rank 32 hay không.
