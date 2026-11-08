# Ablation Tie Head

Cập nhật: 2026-10-26; mục 16 thêm 2026-11-06

**Trạng thái:** cả 6 job training và 7 job đánh giá đã xong (job 5983–5989). Kết quả ở mục 12 và 13. Lựa chọn Tie Head cho pipeline ở mục 14: dùng `margin_coupled` cho Inspector Baseline, Inspector pipeline và Student pipeline. Chưa đo Tie Head AUC và margin AUC (−|d|) trên 450 cặp validation của Inspector. Các nhận xét ở mục 8 chỉ dựa trên loss đánh giá của MS-Swift.

## 1. Ký hiệu

- **d** = điểm ảnh A − điểm ảnh B (chênh lệch điểm). **|d|** là độ lớn của chênh lệch, bỏ dấu.
- **w** = σ(d), xác suất A thắng khi cặp không hoà (σ là hàm sigmoid).
- **t** = P(Tie), xác suất model cho rằng cặp hoà.
- **Head ba lớp:** P(A) = (1 − t)·w, P(B) = (1 − t)·(1 − w), P(Tie) = t. Loss = −log P(nhãn đúng).
- **h_A, h_B:** hidden state của ảnh A và ảnh B lấy từ backbone.
- **θ₀:** mức hoà nền, khởi tạo bằng logit(0.2), tức tỉ lệ Tie khoảng 20%.
- **κ:** độ nhạy của t theo |d|, luôn dương (κ = softplus(tham số), khởi tạo bằng 1.0). |d| càng lớn thì t càng nhỏ.
- **r(h_A, h_B):** phần residual, chính là Tie Head hiện tại (`_SymmetricTieHead`), nhưng lớp cuối khởi tạo bằng 0.
- **AUC trên tập có Tie samples** (quy chuẩn chung, `AUC-Margin.md` mục 4): cặp hoà là lớp dương, mọi cặp còn lại là lớp âm. 50 là đoán ngẫu nhiên, 100 là tách hoàn hảo. Có hai chỉ số:
  - **Tie Head AUC:** cho điểm bằng t = P(Tie) của Tie Head. Model không có Tie Head ghi "—".
  - **Margin AUC (−|d|):** cho điểm bằng −|d|, tức chỉ dùng chênh lệch điểm: |d| càng nhỏ thì càng có khả năng hoà.
  - Trên tập không có Tie, margin AUC là AUC đối xứng {d} so với {−d} (`AUC-Margin.md` mục 3).

## 2. Vấn đề của Tie Head hiện tại

### 2.1. Phân tích trên Inspector Baseline (job 5620)

Số liệu lấy trên khoảng 90k cặp đầu của `progressive_loss.jsonl` (loss trên các cặp model chưa thấy):

| Số cặp đã train | t trung bình (cặp Tie) | t trung bình (cặp A/B) | Tie Head AUC | \|d\| trung bình (cặp Tie) | \|d\| trung bình (cặp A/B) | Margin AUC (−\|d\|) |
|---|---|---|---|---|---|---|
| 0–16k | 0.250 | 0.232 | 0.576 | 0.90 | 1.30 | 0.590 |
| 32k–48k | 0.249 | 0.196 | 0.657 | 1.08 | 1.80 | 0.653 |
| 80k–90k | 0.273 | 0.180 | 0.714 | 1.22 | 2.48 | 0.704 |

- Tie Head có học (AUC tăng từ 0.58 lên 0.71), nhưng chỉ ngang mức dùng riêng −|d| (0.704). Nghĩa là Tie Head gần như chỉ tự học lại thông tin đã có sẵn trong d.
- |d| của cặp Tie tăng dần từ 0.90 lên 1.22, vì không có lực nào kéo cặp hoà về d ≈ 0.

### 2.2. Quan hệ giữa t và d

- **Chiều t → P(A), P(B): có liên hệ.** Ba xác suất có tổng bằng 1; t tăng thì P(A) và P(B) cùng giảm.
- **Chiều d → t: không có liên hệ.** t được tính từ một nhánh riêng, độc lập với d. Cặp có d = 0 vẫn có thể nhận t rất thấp, và cặp có |d| rất lớn vẫn có thể nhận t cao.
- Các mô hình hoà kinh điển đều gắn t với d:
  - **Rao-Kupper:** P(A) = σ(d − θ), P(B) = σ(−d − θ), P(Tie) là phần còn lại, với θ > 0 là ngưỡng hoà.
  - **Davidson:** P(Tie) tỉ lệ với ν·√(π_A·π_B), với π_A, π_B là "sức mạnh" của hai ảnh và ν là tham số mức hoà.

### 2.3. Vì sao Tie Head khởi tạo ngẫu nhiên

- Score Head có sẵn hướng khởi tạo W_yes − W_no (W là trọng số của LM head), vì reranker đã được pretrain với token yes/no.
- Không có token hay hướng nào được pretrain cho "hoà". Vì vậy phép chiếu và MLP của Tie Head dùng khởi tạo mặc định của PyTorch.
- Đặc trưng đối xứng của Tie Head là [½(u_A + u_B), |u_A − u_B|], với u_A, u_B là hidden state sau một phép chiếu ngẫu nhiên. Phép chiếu này không thẳng hàng với hướng điểm, nên Tie Head phải tự học lại "hai ảnh khác nhau bao nhiêu theo hướng chất lượng".

### 2.4. Các nguyên nhân có khả năng (xếp theo mức độ)

1. **Không có ràng buộc giữa d và t.** Đây là thiếu sót quan trọng nhất về cấu trúc.
2. **Cặp hoà không gửi gradient vào Score Head.** Loss của cặp hoà chỉ là −log t, phần loss của w bị triệt tiêu. Score Head chỉ bị đẩy về |d| lớn trên các cặp A/B.
3. **Learning rate nhỏ cho một head khởi tạo ngẫu nhiên.** Head được train qua `modules_to_save` với cùng learning rate như LoRA (4e-5 đến 8e-5), cộng lịch cosine giảm dần.
4. **Tie là lớp thiểu số (20.7%) và không đồng nhất.** Có cặp "cả hai đều tốt", có cặp "cả hai đều tệ", có cặp do người chấm phân vân. Nhãn hoà của người vốn nhiễu.
5. **Cách đo cũ dễ gây hiểu lầm.** Chỉ nhìn loss trên cặp hoà thì tưởng Tie Head không học, nhưng Tie Head AUC cho thấy có tiến bộ.

### 2.5. So sánh Tie Head của Inspector và Student

| | Overall Inspector | Student |
|---|---|---|
| Cấu trúc | `_SymmetricTieHead`, 1 đầu ra | `_SymmetricTieHead`, 6 đầu ra (1 cho mỗi metric) |
| Khởi tạo | Ngẫu nhiên; t không phụ thuộc d | Giống hệt |
| Nhãn hoà | Người, cho cả cặp | Gemma, cho từng metric (majority vote 3 lượt), chỉ tính các metric có liên quan |
| Nhãn hoà của người trong loss | Có | Không có, ở cả Stage 1 và Stage 2 |

- Hai Tie Head có chung hai điểm yếu: t không ràng buộc với d, và cặp hoà không gửi gradient vào Score Head.
- Ở Stage 2, Tie Head của Student gần như đứng yên (Brier trên ô Tie từ 0.308 xuống 0.289). Lý do: ở nhánh HumanGT và Unlabeled Data, `tie_probs` bị detach (tách khỏi đồ thị tính gradient), nên Tie Head chỉ nhận gradient từ phần GemmaGT trộn lại trong batch.

### 2.6. Bằng chứng từ setting 300 (checkpoint-100)

AP và AUC trên EditReward-Bench (133) w Tie samples, cặp hoà là lớp dương (18 cặp hoà, AP ngẫu nhiên 13.53):

| Model | Margin (−\|d\|): AP / AUC | Tie Head: AP / AUC |
|---|---|---|
| Overall Inspector-Stage1 | 25.34 / 68.41 | 22.06 / 69.23 |
| Student-Stage1 | 30.86 / 70.00 | 17.82 / 63.24 |

- Trên 60 cặp GemmaGT validation của Student (gộp 6 metric), Tie Head AUC là 68.6, bằng −|d| là 76.1. Ở 4/6 metric, −|d| tốt hơn Tie Head; chỉ background_consistency và structure_consistency là Tie Head tốt hơn.
- Kết luận: Tie Head hiện tại không tốt hơn −|d| của chính model đó. Các tập này có ít cặp hoà nên số liệu rất nhiễu. Đây là lý do chạy ablation ở setting 3k.

## 3. Thiết kế ablation

### 3.1. Bốn dạng Tie Head

Chọn qua biến môi trường `TIE_HEAD_MODE`; mặc định là `independent`, nên các run cũ không đổi.

| Dạng | Công thức | Gradient qua d từ loss Tie | Xử lý điểm yếu |
|---|---|---|---|
| `independent` (hiện tại) | logit(t) = MLP(h_A, h_B) | Không | Không xử lý điểm yếu nào |
| `margin_coupled_detached` | logit(t) = θ₀ − κ·\|d\| + r(h_A, h_B) | Không (d bị detach) | t phụ thuộc d (nguyên nhân 1) |
| `margin_coupled` | logit(t) = θ₀ − κ·\|d\| + r(h_A, h_B) | Có | Thêm: cặp hoà kéo \|d\| về 0 (nguyên nhân 2) |
| `margin_only` | logit(t) = θ₀ − κ·\|d\| (bỏ residual) | Có | Đúng dạng Rao-Kupper kinh điển |

Mỗi dạng chỉ khác một dạng khác đúng một điểm:

| So sánh | Câu hỏi được trả lời |
|---|---|
| `margin_coupled_detached` với `independent` | Ràng buộc t với d có giúp không? |
| `margin_coupled` với `margin_coupled_detached` | Cho cặp hoà gửi gradient vào Score Head có giúp không? |
| `margin_only` với `margin_coupled` | Residual học từ h_A, h_B có thêm thông tin ngoài \|d\| không (ví dụ "cả hai đều tốt" với "cả hai đều tệ")? |

Student không có job `independent` và `margin_only`. Mốc so sánh `independent` của Student là run có sẵn `student_hardlabel_r64_lr4e-5` (cùng dữ liệu, cùng cách chia, cùng cấu hình).

### 3.2. Vì sao chọn setting 3k thay vì 300

- Với 300 cặp, validation chỉ có 60 cặp (khoảng 20 cặp hoà), nên Tie Head AUC và margin AUC (−|d|) quá nhiễu.
- Với 3,000 cặp, MS-Swift chia 0.15 (seed 42) được 450 cặp validation:
  - Inspector: 198 cặp A thắng, 154 cặp B thắng, **98 cặp hoà**. Cả 4 job Inspector dùng cùng một tập validation (đã đối chiếu `val_dataset.jsonl`).
  - Student: 450 cặp GemmaGT, khoảng 311 ô metric mang nhãn Tie. Hai job Student và mốc `student_hardlabel_r64_lr4e-5` dùng cùng một tập validation.

## 4. Thay đổi code

Bản gốc của các file đã sửa nằm trong `Qwen3-VL-Embedding/scripts/backup_tie_head/`.

- **`Qwen3-VL-Embedding/src/models/custom_qwen3_vl_reranker.py`:**
  - Thêm `TIE_HEAD_MODES`, `resolve_tie_head_mode`, `tie_head_mode_from_state_keys`, `_SymmetricTieHead.zero_init_output`, `_MarginCoupledTie` và `_build_tie_coupling`.
  - `OverallInspectorHead` và `MultiMetricRewardHeads` nhận `tie_head_mode`. θ₀ và κ là tham số của head, được lưu cùng checkpoint. Với Student, mỗi metric có θ₀ và κ riêng.
  - `margin_only` truyền residual bằng 0.
- **Các script nạp checkpoint** tự nhận dạng Tie Head từ tên khoá trọng số (khoá chứa `tie_coupling`): `evaluate_inspector_checkpoints.py`, `infer_overall_inspector.py`, `evaluate_reward_model.py`, `train_loop.py`, `fit_preference_aggregator.py`. Các dạng có ràng buộc đều nạp và suy luận theo cấu trúc `margin_coupled`; `margin_only` nạp vào cấu trúc này cho kết quả trùng khớp.
- **`train_overall_inspector.py`, `train_student_warmstart.py`:** checkpoint tổng hợp ghi lại `tie_head_mode`.
- **`evaluate_inspector_checkpoints.py`:** Bench giữ đủ 133 cặp, thêm margin AUC và các chỉ số ba lớp.
- **`evaluate_student_checkpoints.py`:**
  - Bench giữ đủ 133 cặp. Chỉ số A/B vẫn tính trên 115 cặp không hoà.
  - Thêm phần `selected_equal_three_way`: P(A), P(B), P(Tie) là trung bình xác suất ba lớp trên các metric được chọn; tính accuracy ba lớp, log loss ba lớp, Tie Head AUC và bằng −|trung bình d_k| (d_k là chênh lệch điểm của metric k).
  - Thêm báo cáo Tie trên GemmaGT validation theo từng metric: AUC bằng t_k và bằng −|d_k|, Brier trên ô Tie và ô không Tie, P(Tie) trung bình.
  - File Metrics Weights Aggregator thành tuỳ chọn; không có file thì chỉ tính `selected_equal`.
- **Launcher:** `run_overall_inspector_training.slurm` (thêm `INSPECTOR_SAVE_TOTAL_LIMIT`, in dòng `Tie Head:`), `run_student_hardlabel_training.slurm` (thêm `WARMSTART_EVAL_STEPS`, `WARMSTART_SAVE_STEPS`, in dòng `Tie Head:`).
- **Kiểm tra trên CPU trước khi chạy:**
  - t đối xứng khi đổi chỗ A và B; xác suất ba lớp có tổng bằng 1;
  - lúc khởi tạo, t đúng bằng σ(θ₀ − κ·|d|);
  - với loss chỉ từ cặp hoà: `margin_coupled_detached` có gradient vào Score Head bằng 0, `margin_coupled` có gradient khác 0;
  - `margin_only`: residual không nhận gradient, còn Score Head, θ₀ và κ đều có gradient.

## 5. Thiết lập chung

- **Inspector (4 job):** dữ liệu `TrainingDataCreation/editreward_overall_humanGT_3k.jsonl`, giữ cặp hoà, head ba lớp. Cấu hình tốt nhất của ablation Inspector cũ (phiên bản 10): rank 32 chỉ LLM, `lora_alpha` 64, learning rate 8e-5, 2 lượt dữ liệu, batch 1 × tích lũy gradient 8, validation 0.15 (seed 42), gradient checkpointing bật, đánh giá và lưu mỗi 50 step, giữ 5 checkpoint. 638 step mỗi job.
- **Student (2 job):** giữ nguyên setting của `student_hardlabel_r64_lr4e-5`: rank 64, learning rate 4e-5, 2 epoch, `random_pair_split` 0.15 (seed 42), nhãn Gemma cứng. 638 step mỗi job.
- Tất cả chạy trên partition normal (`2g.35gb`), account anhndt.

## 6. Các job đã xong và thư mục output

Thư mục output nằm trong `Qwen3-VL-Embedding/output_training/`. Log nằm trong `logs/`. Dòng `Tie Head:` trong log đã xác nhận đúng dạng head ở cả 6 job.

| Job | Thư mục | Dạng Tie Head | Thời gian | Bộ nhớ GPU | Checkpoint còn lưu | Checkpoint có loss đánh giá thấp nhất |
|---|---|---|---|---|---|---|
| 5838 | `inspector_tie_ablation_independent_3k` | independent | 4 giờ 18 phút | 24.0 GiB | 300, 500, 550, 600, 638 | checkpoint-300 |
| 5839 | `inspector_tie_ablation_margin_coupled_detached_3k` | margin_coupled_detached | 4 giờ 17 phút | 24.0 GiB | 300, 500, 550, 600, 638 | checkpoint-300 |
| 5840 | `inspector_tie_ablation_margin_coupled_3k` | margin_coupled | 4 giờ 18 phút | 24.0 GiB | 350, 500, 550, 600, 638 | checkpoint-350 |
| 5843 | `inspector_tie_ablation_margin_only_3k` | margin_only | 4 giờ 12 phút | 24.0 GiB | 250, 500, 550, 600, 638 | checkpoint-250 |
| 5841 | `student_tie_ablation_margin_coupled_detached_3k` | margin_coupled_detached | 8 giờ 45 phút | 29.8 GiB | 300, 500, 550, 600, 638 | checkpoint-300 |
| 5842 | `student_tie_ablation_margin_coupled_3k` | margin_coupled | 8 giờ 29 phút | 29.8 GiB | 450, 500, 550, 600, 638 | checkpoint-638 |

- **Đã dọn ngày 2026-10-26** (sau khi đánh giá xong): xoá 24 checkpoint không phải checkpoint có loss thấp nhất, và xoá `optimizer.pt`, `scheduler.pt`, `rng_state.pth` trong 6 checkpoint có loss thấp nhất. Dung lượng giảm từ khoảng 47 GiB xuống khoảng 6 GiB. Mỗi thư mục còn: checkpoint có loss thấp nhất (adapter, `trainer_state.json`), file `.pt` tổng hợp, `logging.jsonl`, `val_dataset.jsonl` và các log khác. Vẫn đánh giá lại được (ví dụ Tie Head AUC và margin AUC (−|d|) trên validation), nhưng không train tiếp được từ các checkpoint này. Cột "Checkpoint còn lưu" ở bảng trên là trạng thái trước khi dọn.
- Log: `logs/overall_inspector_<job>.out` cho Inspector, `logs/student_hardlabel_gemma_<job>.out` cho Student.
- Checkpoint tổng hợp: `overall_inspector_checkpoint.pt` (Inspector) và `student_warmstart.pt` (Student) trong từng thư mục.
- Mốc Student `student_hardlabel_r64_lr4e-5` (independent) có checkpoint loss thấp nhất là checkpoint-600.
- Tên thư mục chưa đổi theo quy ước `_overall_humanGT_3k` / `_overall_finegrained_gemmaGT_3k`. Bảng đổi tên đã đề xuất nhưng chưa được xác nhận.

## 7. Loss đánh giá trên validation

### 7.1. Inspector

Loss ba lớp trên 450 cặp validation (gồm 98 cặp hoà). In đậm là bước có loss thấp nhất. Để tham khảo: model đoán hằng số theo tỉ lệ ba lớp (44.0% / 34.2% / 21.8%) có loss khoảng 1.060.

| Bước | independent | margin_coupled_detached | margin_coupled | margin_only |
|---|---|---|---|---|
| 50 | 0.9771 | 0.9536 | 0.9794 | 1.0039 |
| 100 | 0.9576 | 0.9800 | 0.9475 | 0.9784 |
| 150 | 0.9367 | 0.9260 | 0.9212 | 0.9651 |
| 200 | 0.9504 | 0.9944 | 0.9336 | 0.9755 |
| 250 | 0.9140 | 0.9819 | 0.9161 | **0.9590** |
| 300 | **0.9125** | **0.9099** | 0.9222 | 0.9997 |
| 350 | 0.9231 | 1.0439 | **0.9105** | 0.9663 |
| 400 | 0.9869 | 1.4723 | 0.9164 | 0.9591 |
| 450 | 0.9812 | 0.9373 | 0.9634 | 1.0027 |
| 500 | 0.9606 | 0.9968 | 0.9367 | 0.9865 |
| 550 | 1.0286 | 1.0732 | 0.9571 | 1.0265 |
| 600 | 1.0057 | 1.0493 | 0.9527 | 1.0127 |
| 638 | 1.0055 | 1.0490 | 0.9511 | 1.0125 |

Loss này tính cả cặp hoà, nên không so trực tiếp được với loss hai lớp của ablation Inspector cũ.

### 7.2. Student

Loss trên 450 cặp GemmaGT validation. `independent` là mốc `student_hardlabel_r64_lr4e-5`.

| Bước | independent (mốc) | margin_coupled_detached | margin_coupled |
|---|---|---|---|
| 50 | 1.0479 | 1.1823 | 1.0677 |
| 100 | 0.9473 | 0.9273 | 0.9695 |
| 150 | 0.8217 | 0.8284 | 0.7404 |
| 200 | 0.7417 | 0.7920 | 0.7075 |
| 250 | 0.7339 | 0.7699 | 0.7253 |
| 300 | 0.7005 | **0.6895** | 0.6744 |
| 350 | 0.7100 | 0.7049 | 0.6702 |
| 400 | 0.7305 | 0.7861 | 0.6751 |
| 450 | 0.6838 | 0.7175 | 0.6511 |
| 500 | 0.6856 | 0.7160 | 0.6468 |
| 550 | 0.6732 | 0.7204 | 0.6433 |
| 600 | **0.6687** | 0.7119 | 0.6404 |
| 638 | 0.6701 | 0.7075 | **0.6385** |

## 8. Nhận xét ban đầu (chỉ dựa trên loss đánh giá)

- **Inspector:**
  - Loss thấp nhất của `independent`, `margin_coupled_detached` và `margin_coupled` gần như bằng nhau (0.9099–0.9125). Chênh lệch này nhỏ hơn mức dao động giữa hai lần đánh giá liền nhau, nên chưa phân biệt được.
  - `margin_coupled` ổn định nhất: sau bước 350 loss tăng ít (0.9511 ở cuối, so với 1.0055 của `independent` và 1.0490 của `margin_coupled_detached`).
  - `margin_coupled_detached` dao động mạnh (vọt lên 1.4723 ở bước 400).
  - `margin_only` kém nhất (thấp nhất 0.9590). Gợi ý residual có đóng góp thêm thông tin ngoài |d|.
- **Student:**
  - `margin_coupled` thấp hơn mốc ở mọi bước từ bước 150 trở đi; loss thấp nhất 0.6385, so với 0.6687 của mốc. Loss vẫn giảm đến bước cuối.
  - `margin_coupled_detached` kém hơn mốc từ bước 400 trở đi (thấp nhất 0.6895).
  - Ở cả Inspector và Student, cho cặp hoà gửi gradient vào Score Head (`margin_coupled`) tốt hơn dạng detached.
- **Giới hạn:** loss đánh giá gộp cả phần A/B và phần Tie. Chưa biết dạng nào tách Tie tốt hơn −|d|, và chưa biết accuracy A/B trên Test và Bench có bị ảnh hưởng không. Cần chạy đánh giá ở mục 9.

## 9. Việc còn lại

Trạng thái ngày 2026-10-26: ý 1 đã xong, trừ Tie Head AUC và margin AUC (−|d|) trên 450 cặp validation của Inspector. Ý 2 đã xong. Ý 3 và ý 4 chưa làm. Kết quả ở mục 12 và 13.

1. **Đánh giá 4 checkpoint Inspector** (checkpoint có loss thấp nhất của mỗi job) theo bảng chuẩn:
   - Acc và AUC trên EditReward-Test(1k), EditReward-Bench(115), EditReward(VLM-Reannotate)-Bench(115);
   - AP và AUC (cặp hoà là lớp dương) trên EditReward-Bench(133) w Tie samples và EditReward(VLM-Reannotate)-Bench(133) w Tie samples, tách hai cột margin (−|d|) và Tie Head;
   - thêm Tie Head AUC và bằng −|d| trên 450 cặp validation (98 cặp hoà), vì đây là tập có nhiều cặp hoà nhất.
2. **Đánh giá 2 checkpoint Student** và mốc `student_hardlabel_r64_lr4e-5` bằng `evaluate_student_checkpoints.py` đã sửa: các chỉ số như trên, cộng báo cáo Tie theo từng metric trên GemmaGT validation.
3. **Sửa quy tắc chọn checkpoint** trong `evaluate_inspector_checkpoints.py`: hiện chọn theo accuracy Test cao nhất, cần đổi sang loss đánh giá thấp nhất cho đúng quy ước.
4. **Để sau, khi đã chọn được dạng Tie Head:**
   - learning rate riêng cho các head (Score, Tie, θ₀, κ), ví dụ gấp 10 lần LoRA;
   - Stage 2: không detach `tie_probs` và đưa cặp hoà HumanGT vào loss.

## 10. Lệnh submit job đánh giá (account anhndt)

### 10.1. Phạm vi và tài nguyên

- **7 job:** 4 Inspector (mỗi job chỉ chấm checkpoint có loss đánh giá thấp nhất), 2 Student, và mốc Student `student_hardlabel_r64_lr4e-5` (chấm lại bằng `evaluate_student_checkpoints.py` đã sửa, vì kết quả cũ chưa có phần ba lớp và phần Tie).
- **GPU dự kiến:** khoảng 18–22 GB mỗi job (trọng số bf16 khoảng 16.5 GB, batch 1, độ phân giải ảnh mặc định của processor như lúc train). Dưới 35 GB, nên chạy trên partition normal `2g.35gb` bằng account anhndt.
- **Dữ liệu được chấm:** EditReward-Test(1k) và đủ 133 cặp EditReward-Bench. Student chấm thêm 450 cặp GemmaGT validation (báo cáo Tie theo từng metric).
- **Không chấm lại model gốc** (`INSPECTOR_EVAL_INCLUDE_BASE=0`), vì đã có hàng Original Qwen3-VL-Reranker.
- **Các chỉ số trên bộ nhãn VLM** (EditReward(VLM-Reannotate)-Bench(115) và (133) w Tie samples), cùng AP tách Tie, được tính sau từ file dự đoán của các job này. Phần tính này chạy trên CPU, không cần GPU, và không đọc file Reannotation của người.
- **Chưa có trong các lệnh:** Tie Head AUC và margin AUC (−|d|) trên 450 cặp validation của Inspector (mục 9, ý 1). `evaluate_inspector_checkpoints.py` chưa hỗ trợ tập validation, nên phần này cần sửa code và được đề xuất riêng.

### 10.2. Lệnh

```bash
cd /datastore/truongtq/TrainingRewardModel/TrainingRewardModel
OUT=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training
RES=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OurModel/evaluation_results
SB="sbatch -p normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=8 --mem=64G"

# Inspector: checkpoint có loss đánh giá thấp nhất của mỗi job, có Test 1k, không chấm lại model gốc
env INSPECTOR_EVAL_RUN_DIR=$OUT/inspector_tie_ablation_independent_3k INSPECTOR_EVAL_CHECKPOINTS=checkpoint-300 \
    INSPECTOR_EVAL_INCLUDE_BASE=0 INSPECTOR_EVAL_INCLUDE_TEST=1 \
    $SB --job-name=eval_insp_tie_independent run_evaluate_inspector_checkpoints.slurm
env INSPECTOR_EVAL_RUN_DIR=$OUT/inspector_tie_ablation_margin_coupled_detached_3k INSPECTOR_EVAL_CHECKPOINTS=checkpoint-300 \
    INSPECTOR_EVAL_INCLUDE_BASE=0 INSPECTOR_EVAL_INCLUDE_TEST=1 \
    $SB --job-name=eval_insp_tie_detached run_evaluate_inspector_checkpoints.slurm
env INSPECTOR_EVAL_RUN_DIR=$OUT/inspector_tie_ablation_margin_coupled_3k INSPECTOR_EVAL_CHECKPOINTS=checkpoint-350 \
    INSPECTOR_EVAL_INCLUDE_BASE=0 INSPECTOR_EVAL_INCLUDE_TEST=1 \
    $SB --job-name=eval_insp_tie_coupled run_evaluate_inspector_checkpoints.slurm
env INSPECTOR_EVAL_RUN_DIR=$OUT/inspector_tie_ablation_margin_only_3k INSPECTOR_EVAL_CHECKPOINTS=checkpoint-250 \
    INSPECTOR_EVAL_INCLUDE_BASE=0 INSPECTOR_EVAL_INCLUDE_TEST=1 \
    $SB --job-name=eval_insp_tie_margin_only run_evaluate_inspector_checkpoints.slurm

# Student: student_warmstart.pt là checkpoint có loss đánh giá thấp nhất của mỗi job
env STUDENT_EVAL_RUN_DIR=$OUT/student_tie_ablation_margin_coupled_detached_3k \
    $SB --job-name=eval_student_tie_detached run_evaluate_student_checkpoints.slurm
env STUDENT_EVAL_RUN_DIR=$OUT/student_tie_ablation_margin_coupled_3k \
    $SB --job-name=eval_student_tie_coupled run_evaluate_student_checkpoints.slurm

# Mốc Student (independent): ghi vào thư mục con mới, vì launcher không ghi đè evaluation.json cũ
env STUDENT_EVAL_RUN_DIR=$OUT/student_hardlabel_r64_lr4e-5 \
    STUDENT_EVAL_OUTPUT_DIR=$RES/student_hardlabel_r64_lr4e-5/tie_evaluation \
    $SB --job-name=eval_student_tie_independent run_evaluate_student_checkpoints.slurm
```

### 10.3. Kết quả và log

| Job | Thư mục kết quả (trong `RewardModelEvaluation/OurModel/evaluation_results/`) |
|---|---|
| Inspector independent | `inspector_checkpoints/inspector_tie_ablation_independent_3k_maxpixelsdefault/` |
| Inspector margin_coupled_detached | `inspector_checkpoints/inspector_tie_ablation_margin_coupled_detached_3k_maxpixelsdefault/` |
| Inspector margin_coupled | `inspector_checkpoints/inspector_tie_ablation_margin_coupled_3k_maxpixelsdefault/` |
| Inspector margin_only | `inspector_checkpoints/inspector_tie_ablation_margin_only_3k_maxpixelsdefault/` |
| Student margin_coupled_detached | `student_tie_ablation_margin_coupled_detached_3k/evaluation.json` |
| Student margin_coupled | `student_tie_ablation_margin_coupled_3k/evaluation.json` |
| Mốc Student independent | `student_hardlabel_r64_lr4e-5/tie_evaluation/evaluation.json` (tên `tie_evaluation` đã xác nhận) |

Job đã chạy: 5983 (Inspector independent), 5984 (Inspector margin_coupled_detached), 5985 (Inspector margin_coupled), 5986 (Inspector margin_only), 5987 (Student margin_coupled_detached), 5988 (Student margin_coupled), 5989 (mốc Student independent). Cả 7 job xong, không lỗi. Mốc Student có chấm 450 cặp GemmaGT validation.

- Log nằm ở `TrainingRewardModel/logs/eval_inspector_checkpoints_<job>.out` và `TrainingRewardModel/logs/eval_student_checkpoints_<job>.out`.
- Mốc Student có sẵn `preference_aggregator.json`, nên launcher tự tính thêm phần trọng số đã fit. Bảng so sánh chỉ dùng `selected_equal`, giống hai job Student còn lại.
- Báo cáo Tie trên GemmaGT validation của Student lấy danh sách cặp validation lưu trong checkpoint. Nếu checkpoint của mốc (tạo ngày 2026-10-18) không lưu danh sách này, script sẽ bỏ qua phần GemmaGT validation và không báo lỗi. Cần kiểm tra trong log sau khi job xong.

## 11. Quyền đọc, ghi, thực thi của account anhndt

anhndt (uid 659439956) và truongtq cùng nhóm `Domain Users`. Đã kiểm tra ngày 2026-10-26:

| Đường dẫn | Quyền hiện tại | anhndt cần | Kết quả |
|---|---|---|---|
| `/datastore/truongtq`, `/datastore/truongtq/miniconda3` (môi trường `qwen_vl`) | `drwxr-xr-x` truongtq | Đọc, thực thi | Đủ |
| `/datastore/truongtq/TrainingRewardModel` | `drwxrwxrwx` truongtq | Ghi (tạo `hf_cache` khi chạy) | Đủ |
| `TrainingRewardModel/logs/` | `drwxrwxrwx` truongtq | Ghi log | Đủ |
| `run_evaluate_inspector_checkpoints.slurm` | `-rwxrwxrwx` | Đọc | Đủ |
| `run_evaluate_student_checkpoints.slurm`, `scripts/evaluate_student_checkpoints.py` | `-rw-r--r--` | Đọc (sbatch và python không cần quyền thực thi) | Đủ |
| `Qwen3-VL-Embedding/scripts/`, `scripts/evaluate_inspector_checkpoints.py` | `drwxrwxrwx`, `-rwxrwxrwx` | Đọc | Đủ |
| `Models/Qwen3-VL-Reranker-8B/`, `RewardModelEvaluation/EvalDatasets/` | `drwxrwxrwx` | Đọc | Đủ |
| 6 thư mục `*_tie_ablation_*_3k` | `drwxrwxrwx` anhndt (file `-rw-rw-rw-`) | Đọc | Đủ |
| `output_training/student_hardlabel_r64_lr4e-5/` (mốc) | `drwxr-xr-x` truongtq, `student_warmstart.pt` là `-rw-r--r--` | Đọc | Đủ |
| `evaluation_results/`, `evaluation_results/inspector_checkpoints/` | `drwxrwxrwx` truongtq | Ghi (tạo thư mục kết quả mới) | Đủ |
| `evaluation_results/student_hardlabel_r64_lr4e-5/` | `drwxrwxrwx` anhndt, đã có `evaluation.json` | Ghi thư mục con mới | Đủ; không ghi đè file cũ |

Các điểm cần xử lý trên account anhndt (truongtq không đổi quyền được file của anhndt):

1. **30 file `adapter_model.safetensors`** trong các thư mục `*_tie_ablation_*_3k` có quyền `-rw-------`. Chỉ anhndt đọc được, nên job đánh giá trên anhndt vẫn chạy được, nhưng truongtq không nạp được các checkpoint này (ví dụ khi tự đánh giá hoặc dùng cho Stage 2). Lệnh mở quyền đọc:

   ```bash
   find /datastore/truongtq/TrainingRewardModel/TrainingRewardModel/Qwen3-VL-Embedding/output_training/*_tie_ablation_*_3k \
       -name adapter_model.safetensors -exec chmod a+r {} +
   ```

2. **Thư mục kết quả do các job đánh giá tạo ra** sẽ thuộc anhndt. Sau khi job xong, mở quyền đọc cho truongtq:

   ```bash
   RES=/datastore/truongtq/TrainingRewardModel/RewardModelEvaluation/OurModel/evaluation_results
   chmod -R a+rX $RES/inspector_checkpoints/inspector_tie_ablation_*_3k_maxpixelsdefault \
       $RES/student_tie_ablation_*_3k $RES/student_hardlabel_r64_lr4e-5/tie_evaluation
   ```

3. **Tuỳ chọn:** các thư mục ablation đang là `drwxrwxrwx` với file `-rw-rw-rw-`, nên mọi user đều sửa hoặc xoá được checkpoint. Training đã xong, có thể thu quyền ghi bằng `chmod -R go-w` trên 6 thư mục này. Chỉ làm sau khi chốt việc đổi tên thư mục, vì đổi tên cần quyền ghi trên thư mục cha `output_training/` (vẫn là `drwxrwxrwx`, không bị ảnh hưởng).

## 12. Kết quả đánh giá ngoại tuyến (2026-10-26)

### 12.1. Cách tính

- Mọi số liệu được tính lại trên CPU từ file dự đoán của job 5983–5989, chỉ đọc file JSON. Các chỉ số trên nhãn người khớp với số script đánh giá đã ghi.
- **Nhãn VLM (EditReward(VLM-Reannotate)):** 3 VLM (`gemini-3.5-flash`, `claude-opus-5-5`, `gpt-5.6-sol`), mỗi model trả lời 5 lần mỗi cặp, tổng 15 phiếu có trọng số bằng nhau. Mỗi phiếu chọn 0 (ảnh thứ nhất, tức A), 1 (ảnh thứ hai, tức B) hoặc 2 (Tie). Nhãn là lựa chọn nhiều phiếu nhất và duy nhất đứng đầu, lấy từ `EvalDatasets/EditReward-Bench_VLM_judge/pairs.csv` (nhóm `2pair`).
  - VLM-Bench(115): 115 cặp mà người không gán Tie (`EditReward-Bench_pairwise_VLM_judge.json`), dùng nhãn VLM.
  - VLM-Bench(133) w Tie samples: cả 133 cặp, dùng nhãn VLM: 61 A, 70 B, **chỉ 2 Tie**. Chưa có file JSON riêng cho bản này; nhãn được đọc thẳng từ `pairs.csv`.
  - Toàn bộ 29 phiếu Tie của 3 VLM trên 133 cặp đều rơi vào 18 cặp mà người gán Tie. 16/18 cặp Tie của người được VLM gán A hoặc B.
- **Margin của Student:** `selected_equal` là trung bình `signed_evidence` trên các metric được chọn. `evaluate_student_checkpoints.py` hiện ghi margin bằng **tổng** (`evaluate_selected_aggregation`, trọng số 1 trên mỗi metric được chọn, không chia số metric). Accuracy không đổi, nhưng margin AUC ghi trong `evaluation.json` của Student khác số ở đây. Cần sửa script.
- −|d| của Student là −|trung bình d_k trên các metric được chọn|, với d_k = điểm A − điểm B của metric k, như trong script.
- Script tính: `scratchpad/tie_ablation_summary.py` (chưa đưa vào repo).

### 12.2. A/B: Acc / margin AUC (%)

| Model | Test(1k) | Bench(115) | VLM-Bench(115) |
|---|---|---|---|
| Inspector independent (checkpoint-300) | 74.40 / 82.64 | 73.04 / 77.50 | 91.30 / 97.20 |
| Inspector margin_coupled_detached (checkpoint-300) | 74.50 / 82.31 | 73.04 / 76.48 | 91.30 / 97.23 |
| Inspector margin_coupled (checkpoint-350) | 73.80 / 81.54 | 74.78 / 78.24 | 93.91 / 98.04 |
| Inspector margin_only (checkpoint-250) | 73.90 / 81.32 | 74.78 / 79.79 | 93.04 / 97.86 |
| Student independent (mốc, checkpoint-600) | 75.10 / 81.88 | 77.39 / 80.72 | 95.65 / 98.31 |
| Student margin_coupled_detached (checkpoint-300) | 75.80 / 82.33 | 73.04 / 78.93 | 93.04 / 97.82 |
| Student margin_coupled (checkpoint-638) | 74.90 / 82.14 | 77.39 / 81.17 | 92.17 / 97.92 |

### 12.3. EditReward-Bench(133) w Tie samples, nhãn người (18 Tie, AP ngẫu nhiên 13.53)

| Model | Margin (−\|d\|): AP / AUC | Tie Head: AP / AUC | Acc ba lớp | P(Tie) trung bình: cặp Tie / cặp A/B |
|---|---|---|---|---|
| Inspector independent | 35.61 / 71.16 | 30.61 / 71.81 | 63.16 | 0.204 / 0.149 |
| Inspector margin_coupled_detached | 31.42 / 65.72 | 27.85 / 65.39 | 63.91 | 0.235 / 0.169 |
| Inspector margin_coupled | 36.96 / 70.29 | 36.30 / 70.56 | 64.66 | 0.228 / 0.147 |
| Inspector margin_only | 36.75 / 71.23 | 36.75 / 71.23 | 64.66 | 0.127 / 0.085 |
| Student independent (mốc) | 36.88 / 68.84 | 14.95 / 53.14 | 64.66 | 0.174 / 0.173 |
| Student margin_coupled_detached | 32.40 / 71.35 | 32.08 / 67.44 | 63.91 | 0.206 / 0.142 |
| Student margin_coupled | 37.40 / 70.72 | 32.79 / 66.38 | 66.92 | 0.189 / 0.126 |

Với `margin_only`, t = σ(θ₀ − κ·|d|) nên thứ hạng của t trùng thứ hạng của −|d|, hai cột giống hệt nhau.

### 12.4. EditReward(VLM-Reannotate)-Bench(133) w Tie samples (2 Tie, AP ngẫu nhiên 1.50)

| Model | Margin (−\|d\|): AP / AUC | Tie Head: AP / AUC | Acc ba lớp | A/B trên 131 cặp không hoà (tham khảo): Acc / margin AUC đối xứng |
|---|---|---|---|---|
| Inspector independent | 70.00 / 98.85 | 26.79 / 96.95 | 87.22 | 88.55 / 95.99 |
| Inspector margin_coupled_detached | 70.00 / 98.85 | 26.67 / 96.37 | 85.71 | 88.55 / 95.65 |
| Inspector margin_coupled | 45.00 / 98.47 | 36.67 / 98.09 | 88.72 | 90.08 / 96.66 |
| Inspector margin_only | 58.33 / 99.43 | 58.33 / 99.43 | 88.72 | 90.08 / 96.39 |
| Student independent (mốc) | 29.76 / 92.37 | 1.55 / 28.63 | 87.22 | 92.37 / 97.13 |
| Student margin_coupled_detached | 26.67 / 96.95 | 9.00 / 87.79 | 86.47 | 90.08 / 97.01 |
| Student margin_coupled | 21.11 / 95.80 | 26.79 / 96.95 | 88.72 | 90.08 / 96.77 |

Với 2 cặp Tie, AP, margin AUC (−|d|) và Tie Head AUC chỉ để tham khảo: một cặp đổi thứ hạng làm số thay đổi hàng chục điểm. Phần A/B trên 131 cặp vẫn dùng được.

### 12.5. Student: tách Tie trên 450 cặp GemmaGT validation (319 ô Tie / 2,020 ô, gộp 6 metric)

| Model | t: AP / AUC | −\|d_k\|: AP / AUC |
|---|---|---|
| independent (mốc) | 60.82 / 85.29 | 23.66 / 63.66 |
| margin_coupled_detached | 55.43 / 82.09 | 18.80 / 55.11 |
| margin_coupled | 58.59 / 85.51 | 30.96 / 74.24 |

AUC theo từng metric (bằng t / bằng −|d_k|):

| Model | aesthetic | fidelity | realism | background | foreground | structure |
|---|---|---|---|---|---|---|
| independent (mốc) | 75.68 / 75.79 | 70.99 / 74.65 | 63.72 / 70.86 | 77.21 / 65.19 | 81.53 / 73.47 | 81.38 / 71.50 |
| margin_coupled_detached | 62.21 / 71.02 | 70.97 / 68.95 | 62.67 / 62.30 | 72.26 / 56.96 | 73.90 / 59.52 | 73.54 / 71.73 |
| margin_coupled | 76.83 / 78.54 | 69.33 / 69.03 | 71.13 / 71.47 | 80.42 / 78.90 | 78.29 / 76.48 | 74.91 / 74.14 |

### 12.6. So sánh từng cặp (trái − phải)

Test Acc dùng kiểm định McNemar chính xác. Khoảng tin cậy 95% của chênh lệch AUC tính bằng 1000 lần bootstrap trên cùng các cặp (seed 42).

| So sánh | Test Acc: chỉ trái đúng / chỉ phải đúng, p | Test margin AUC [CI 95%] | Bench(133) Tie Head AUC [CI 95%] | Bench(133) Margin AUC (−\|d\|) [CI 95%] |
|---|---|---|---|---|
| Inspector: detached − independent | 30 / 29, p = 1.000 | −0.33 [−0.89, +0.28] | **−6.43 [−13.95, −0.65]** | **−5.43 [−10.80, −1.04]** |
| Inspector: coupled − detached | 56 / 63, p = 0.582 | −0.77 [−2.08, +0.61] | +5.17 [−0.24, +10.53] | +4.57 [−1.04, +10.20] |
| Inspector: margin_only − coupled | 43 / 42, p = 1.000 | −0.22 [−1.23, +0.88] | +0.68 [−3.50, +4.81] | +0.94 [−3.24, +5.00] |
| Inspector: coupled − independent | 54 / 60, p = 0.640 | −1.10 [−2.32, +0.14] | −1.26 [−7.64, +4.41] | −0.87 [−6.12, +3.71] |
| Student: detached − mốc | 59 / 52, p = 0.569 | +0.45 [−0.85, +1.79] | **+14.30 [+1.79, +27.14]** | +2.51 [−2.26, +7.96] |
| Student: coupled − detached | 48 / 57, p = 0.435 | −0.18 [−1.36, +1.02] | −1.06 [−7.85, +5.68] | −0.63 [−7.11, +5.31] |
| Student: coupled − mốc | 51 / 53, p = 0.922 | +0.26 [−0.92, +1.49] | +13.24 [−0.23, +27.96] | +1.88 [−3.27, +6.85] |

### 12.7. Nhận xét

- **A/B:** không có khác biệt có ý nghĩa giữa các dạng, ở cả Inspector và Student. Đổi Tie Head không làm hỏng khả năng so sánh A/B. Riêng Inspector `margin_coupled` thấp hơn `independent` 1.10 điểm margin AUC trên Test, khoảng tin cậy vừa chạm 0.
- **Inspector, tách Tie:**
  - `margin_coupled_detached` kém `independent` có ý nghĩa ở cả hai cách đo, khớp với loss đánh giá dao động mạnh (mục 7.1).
  - `margin_coupled` và `margin_only` ngang `independent`. Không dạng nào vượt −|d| trên Bench.
  - Residual không thêm thông tin so với `margin_only` trên Bench (+0.68, khoảng tin cậy cắt 0). Nhưng trên validation, loss của `margin_only` kém hơn rõ (0.9590 so với 0.9105).
- **Student, tách Tie:**
  - Tie Head của mốc gần như ngẫu nhiên trên Bench (AUC 53.14). P(Tie) ở cặp Tie và cặp A/B gần bằng nhau (0.174 so với 0.173).
  - Ràng buộc với |d| kéo AUC lên khoảng 66–67, nhưng vẫn chưa vượt −|d| (khoảng 71).
  - Trên GemmaGT validation, `margin_coupled` giữ AUC bằng t cao nhất (85.51) và làm −|d_k| tốt lên rõ (63.66 lên 74.24). Đây là dấu hiệu cặp hoà kéo |d| về 0 đúng như thiết kế. `margin_coupled_detached` kém mốc ở cả hai cột.
- **Giới hạn:** Bench chỉ có 18 cặp Tie nên khoảng tin cậy rất rộng. Bản VLM chỉ có 2 cặp Tie. Inspector chưa có Tie Head AUC và margin AUC (−|d|) trên 450 cặp validation (98 cặp Tie).

## 13. Student: Tie Head ở setting 300 và setting 3k

Tie Head AUC trên Bench(133) của Student là 63.24 ở setting 300 (`student_pipeline_overall_finegrained_gemmaGT_300`, checkpoint-100) nhưng chỉ 53.14 ở mốc 3k. Cả hai đều là `independent`. Script: `scratchpad/student_tie_300_vs_3k.py`.

- **Về thống kê, chưa có ý nghĩa:** 300 là 63.24 [51.06, 74.00]; mốc 3k là 53.14 [39.60, 65.73]; 3k `margin_coupled` là 66.38 [50.92, 80.91]. Chênh lệch mốc 3k − 300 là −10.10 [−24.47, +5.31].
- **Về cơ chế, có khác biệt thật.** P(Tie) trung bình theo metric, cặp Tie / cặp A/B:

| Metric | 300 | 3k mốc | 3k margin_coupled |
|---|---|---|---|
| aesthetic | 0.289 / 0.259 | 0.041 / 0.040 | 0.037 / 0.014 |
| fidelity | 0.358 / 0.333 | 0.110 / 0.109 | 0.121 / 0.053 |
| realism | 0.336 / 0.315 | 0.062 / 0.063 | 0.081 / 0.032 |
| background_consistency | 0.472 / 0.467 | 0.397 / 0.402 | 0.464 / 0.395 |
| foreground_consistency | 0.340 / 0.320 | 0.294 / 0.296 | 0.295 / 0.215 |
| structure_consistency | 0.353 / 0.342 | 0.280 / 0.285 | 0.293 / 0.209 |

Tie Head AUC / margin AUC (−|d_k|) trên Bench khi dùng riêng từng metric (t_k / −|d_k|):

| Run | aesthetic | fidelity | realism | background | foreground | structure |
|---|---|---|---|---|---|---|
| 300 | 66.1 / 70.0 | 63.2 / 70.7 | 61.2 / 69.8 | 56.1 / 67.8 | 62.2 / 69.0 | 61.8 / 69.0 |
| 3k mốc | 53.6 / 74.8 | 50.5 / 73.6 | 51.2 / 69.2 | 50.9 / 62.3 | 51.3 / 61.4 | 48.2 / 65.2 |
| 3k margin_coupled | 72.3 / 71.6 | 72.3 / 72.5 | 70.1 / 69.9 | 57.1 / 62.2 | 62.5 / 64.2 | 66.2 / 68.4 |

- Ở mốc 3k, t_k gần như bằng nhau giữa cặp Tie và cặp A/B ở cả 6 metric. Nguyên nhân không phải cách gộp metric: gộp bằng trung bình trên 6 metric, max hay min đều cho AUC 50–54.
- Ở setting 300, Tie Head mới train 100 step. t_k vẫn gần mức khởi tạo (khoảng 0.3) và chênh nhẹ nhưng đều giữa cặp Tie và cặp A/B, tức mang thông tin chung "hai ảnh khác nhau bao nhiêu".
- Ở mốc 3k, sau 600 step, Tie Head độc lập chuyên biệt hoá theo Tie của Gemma cho từng metric: AUC trên GemmaGT validation tăng từ 68.59 (300, 60 cặp) lên 85.29 (3k, 450 cặp). t của aesthetic, fidelity và realism bị kéo về khoảng 0.04–0.11. Tie theo từng metric của Gemma không chuyển sang Tie tổng thể của người. Đây là điểm yếu "t không ràng buộc với d" (mục 2.4, nguyên nhân 1): train càng lâu, Tie Head càng đi xa khỏi thông tin chênh lệch điểm.
- `margin_coupled` khôi phục khoảng chênh: t_k của cặp Tie cao hơn rõ so với cặp A/B, và AUC theo metric ở aesthetic, fidelity, realism (70–72) ngang −|d_k|.
- Đây không phải so sánh chỉ khác một điểm: hai run khác số cặp train, số step, checkpoint, và mức chọn metric (structure_consistency được chọn trên 100% cặp Tie ở 300, 22% ở 3k).

## 14. Lựa chọn Tie Head cho pipeline (2026-10-26)

**Chọn `margin_coupled`** (logit(t) = θ₀ − κ·|d| + r(h_A, h_B), có gradient qua d) cho cả ba nơi dùng Tie Head trong `pipeline_process.md`:

| Nơi dùng | Tie Head | Cơ sở |
|---|---|---|
| Inspector Baseline (Overall Inspector trên toàn bộ EditReward-Data, mục 6.5 của `pipeline_process.md`) | `margin_coupled` | Xem lý do Inspector bên dưới |
| Inspector pipeline (Overall Inspector, setting 300) | `margin_coupled` | Như trên |
| Student pipeline (Student-Stage1 và Student Giai đoạn 2, setting 300; Teacher clone từ Student nên dùng cùng head) | `margin_coupled` | Xem lý do Student bên dưới |

Theo nguyên tắc ở mục 13.1 của `pipeline_process.md`, lựa chọn dựa trên validation. Test và Bench chỉ dùng để kiểm tra lựa chọn không làm kết quả ngoài tệ đi.

**Student: bằng chứng rõ.**

- Loss trên 450 cặp GemmaGT validation thấp hơn mốc ở mọi bước từ bước 150 (thấp nhất 0.6385 so với 0.6687), và vẫn giảm đến bước cuối.
- Tie trên GemmaGT validation: giữ AUC bằng t (85.51 so với 85.29) và làm −|d_k| tốt lên rõ (74.24 so với 63.66).
- Bench(133): Tie Head AUC 66.38 so với 53.14 (+13.24, khoảng tin cậy vừa chạm 0). P(Tie) phân biệt được cặp Tie và cặp A/B ở mọi metric (mục 13), trong khi mốc thì không.
- A/B trên Test và Bench không đổi (Test p = 0.92; Bench(115) 77.39 bằng mốc).
- `margin_coupled_detached` bị loại: loss validation kém mốc từ bước 400, Tie trên GemmaGT validation kém mốc ở cả hai cột.

**Inspector: chọn `margin_coupled`, nhưng bằng chứng yếu hơn.**

- Loss thấp nhất trên 450 cặp validation ngang `independent` và `margin_coupled_detached` (0.9105 so với 0.9125 và 0.9099). Nhưng `margin_coupled` ổn định nhất sau điểm tốt nhất: cuối train 0.9511, so với 1.0055 (`independent`) và 1.0490 (`margin_coupled_detached`).
- Độ ổn định quan trọng cho Inspector Baseline, vì Inspector Baseline train trên khoảng 170k cặp và đã có hiện tượng quá tự tin khi train lâu: |d| tăng dần, kể cả trên cặp Tie (vấn đề mở 7 và 8 trong `pipeline_process.md`). `margin_coupled` là dạng duy nhất có lực kéo |d| của cặp Tie về 0.
- Không làm kết quả ngoài tệ đi có ý nghĩa: Test Acc p = 0.64; Test margin AUC −1.10 [−2.32, +0.14]; Bench(115) 74.78 so với 73.04; Bench(133) Tie Head AUC ngang `independent`.
- Dùng cùng một dạng head cho Inspector và Student giúp Giai đoạn 2 nhất quán: P(Tie) của Inspector (nguồn Overall Pseudo-label Cache) và của Student được tính theo cùng một cấu trúc.
- Bị loại: `margin_coupled_detached` (Tie trên Bench kém `independent` có ý nghĩa, loss dao động mạnh); `margin_only` (loss validation kém nhất, 0.9590).
- **Điều kiện kiểm lại:** tính Tie Head AUC và margin AUC (−|d|) trên 450 cặp validation (98 cặp Tie) cho 4 checkpoint Inspector. Nếu `margin_coupled` kém `independent` rõ trên tập này, xem lại lựa chọn cho Inspector.

**Cách áp dụng:**

- Đặt `TIE_HEAD_MODE=margin_coupled` khi train Inspector Baseline, Inspector pipeline và Student-Stage1 của Student pipeline. Các script nạp checkpoint tự nhận dạng head từ tên khoá, nên Giai đoạn 2 (`train_loop.py`) và Teacher clone không cần thêm cấu hình.
- Đây là một thay đổi so với các run 300 hiện có (`inspector_pipeline_overall_humanGT_300`, `student_pipeline_overall_finegrained_gemmaGT_300`, cả hai là `independent`). Phải train lại cả hai run, sau đó chạy lại Giai đoạn 2, rồi so sánh với bản `independent` theo đúng nguyên tắc chỉ khác một điểm.
- Chưa gộp các thay đổi ở mục 9, ý 4 (learning rate riêng cho head; không detach `tie_probs` ở Giai đoạn 2; đưa cặp hoà HumanGT vào loss). Mỗi thay đổi là một thí nghiệm riêng.

## 15. Việc tiếp theo (đề xuất, chưa xác nhận)

1. Thêm Tie Head AUC và margin AUC (−|d|) trên validation vào `evaluate_inspector_checkpoints.py` và chấm 4 checkpoint Inspector (điều kiện kiểm lại ở mục 14).
2. Sửa margin `selected_equal` trong `evaluate_student_checkpoints.py` từ tổng sang trung bình (mục 12.1).
3. Tạo file `EditReward-Bench_pairwise_VLM_judge_with_tie.json` (133 cặp, nhãn VLM majority vote) để các lần đánh giá sau tính trực tiếp bản VLM có Tie.
5. Train lại Inspector pipeline và Student pipeline (setting 300) với `margin_coupled`.

## 16. Ghi chú Stage 2 (06/11/2026)

- Trên Unlabeled, Tie Head nhận gradient từ CE Inspector (qua P(Tie) overall) và KL Teacher, chỉ ở các metric trong S của Student. Relevance Head không học từ Unlabeled; chỉ học lại L_M trên 1008 cặp Stage 1.
- Augmentation làm S của Student đổi (composite: khác Teacher ở 22% mẫu, chủ yếu realism, background, structure) và đẩy A/B thành Tie ở metric consistency (background: baseline 12.3%, composite 32.1% mẫu Teacher chọn A/B nhưng Student chọn Tie).
- Khi đo Tie Head AUC sau Stage 2 có augmentation, tách theo metric và theo việc metric có trong S của cả hai model.
