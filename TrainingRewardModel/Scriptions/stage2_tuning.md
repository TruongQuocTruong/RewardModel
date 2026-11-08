# STAGE 2 TUNING: KIỂM CHỨNG GIAI ĐOẠN 2 TRÊN DỮ LIỆU NHỎ

Trạng thái tài liệu: 07/11/2026 (mục 8.5: kết quả các tổ hợp augmentation và Inspector one-hot; mục 8.6: augmentation và tập metric được chọn). Liên quan: `pipeline_process.md` (mục 0: proposal gốc; mục 3.11: các phiên bản Giai đoạn 2 đã chạy), `pipeline_core.md` (công thức). Nội dung của bản trước batch 16 (kế hoạch ban đầu, các run sai batch) đã gộp vào mục 9; bản gốc lưu ở `Qwen3-VL-Embedding/scripts/backups/stage2_tuning.before_batch16_unlabeled1008_settings.md`.

## 1. Mục tiêu

- **Câu hỏi chính:** Stage 2 có học thêm được từ Unlabeled và cải thiện Student không. Dữ liệu có nhãn chỉ có 1008 cặp; Unlabeled chỉ có nhãn mềm của Inspector, bản chất không phải nhãn sạch. Vì vậy Student phải dựa vào cặp có nhãn và học chậm từ Unlabeled.
- Thí nghiệm nhỏ: 1008 Unlabeled, 1 loop, khoảng 3 giờ trên partition normal. Với ít dữ liệu, phải đọc loss và norm gradient của từng thành phần để biết Student học được gì.
- Experiment chỉ chạy theo claim của người dùng, mỗi run đổi một thay đổi, không lặp seed:
  - **Claim 1:** khởi tạo Student từ Inspector có giúp không. 1a: so sánh ở Stage 1 (mục 7.1). 1b: Stage 2 từ mỗi Student-Stage1 (mục 4.1, kết quả mục 7.2). Chẩn đoán gradient của run khởi tạo từ Inspector: mục 7.3.
  - **Claim 2:** trọng số các loss (mục 4.2). Kết quả mục 7.4.
  - **Đóng băng ViT/aligner ở Stage 2** (từ chẩn đoán mục 7.3): mục 4.3, kết quả mục 7.5.
  - Thống kê mẫu khó giữa Teacher, Student và Inspector trên mọi run trên: mục 7.6.

## 2. Quy tắc dữ liệu

- Dữ liệu có nhãn **chỉ có 1008 cặp** (cặp của Inspector checkpoint-63 / Student-Stage1). Không dùng bất kỳ cặp HumanGT / GemmaGT nào ngoài 1008 cặp này, kể cả để làm validation.
- **Unlabeled:** 1008 mẫu rút ngẫu nhiên từ 10,000 mẫu của Overall Pseudo-label Cache (`inspector_pseudo_labels_10k.jsonl`, Inspector checkpoint-63) với seed 50042 (selection seed 42 + 50,000). Mỗi mẫu đi đúng một lần. ID lưu trong `unlabeled_subset.json` của thư mục output.
  - 1008 mẫu này **không phải** phần đầu của tập 2,000 mẫu cũ, vì Python rút mẫu theo cách khác khi kích thước khác. Mô phỏng trên chỉ số cho thấy 972 / 1008 mẫu nằm trong tập 2,000 cũ. Vì vậy các tập đánh giá Unlabeled cũ không dùng lại.
- **Không dùng Test** cho Stage 2 tuning. EditReward-Bench chỉ chạy sau khi log train và đánh giá trên dữ liệu train Stage 2 cho thấy Stage 2 có học được (mục 5).

## 3. Config đã chốt (02/11/2026)

| Thành phần | Giá trị | Biến trong `config_training.sh` |
|---|---|---|
| Batch | 16 cặp mỗi optimizer step = **8 cặp có nhãn + 8 Unlabeled** | `REFINEMENT_LABELED_PAIRS_PER_STEP=8` |
| Microbatch | 1 cặp; gradient accumulation 8 | `REFINEMENT_UNLABELED_BATCH_SIZE=1`, `REFINEMENT_GRADIENT_ACCUMULATION_STEPS=8` |
| Unlabeled | 1008 mẫu, một lần | `REFINEMENT_UNLABELED_SUBSET_SIZE=1008`, `REFINEMENT_UNLABELED_PER_ROUND=1008`, `REFINEMENT_SELECTION_SEED=42` |
| Số step | 1008 / 8 = **126 step**, 1 loop | `REFINEMENT_MAX_ROUNDS=1` |
| Cặp có nhãn | 126 × 8 = 1008: **mỗi cặp đúng một lần**. HumanGT và GemmaGT cùng một forward | `REFINEMENT_LABELED_SOURCE=merged` |
| Teacher | Một lượt forward, **dropout bật** (dropout của LoRA 0.05 và của metric_heads) | `REFINEMENT_MC_PASSES=1`; cần sửa code (mục 6) |
| Cập nhật Teacher | **Không cập nhật trong loop**: Teacher = Student-Stage1 suốt run (mục 3.2) | cần sửa code (mục 6) |
| Loss Unlabeled | CE với nhãn mềm ba lớp của Inspector + **KL(Teacher ‖ Student)** trên các metric được chọn (S) | `REFINEMENT_UNLABELED_OBJECTIVE=inspector_soft_ema_consistency`, `REFINEMENT_CONSISTENCY_METRICS=selected` |
| λ_H, λ_G, λ_M, λ_U, λ_C | 1, 1, 1, 1, 1 | |
| Ramp λ_U | **Không ramp**: λ_U = 1 từ step đầu | `REFINEMENT_SET_RAMP_STEPS=1` (giá trị 0 có nghĩa là ramp 20% số step) |
| Trọng số theo cặp | Mỗi cặp có nhãn 1/8, mỗi Unlabeled λ_U / 8: tỉ lệ 1 : 1 khi λ_U = 1 | `REFINEMENT_LOSS_WEIGHTING=per_pair` |
| Learning rate | 1e-5, metric_heads dùng chung | `REFINEMENT_LEARNING_RATE=1e-5` |
| Lịch learning rate | cosine, warmup 5% (khoảng 6 step) | |
| Optimizer | AdamW, β 0.9 / 0.95, ε 1e-8, weight decay 0.01, clip gradient 1.0 | |
| max_pixels | Không đặt (mặc định của processor), đồng nhất với Inspector và Student-Stage1 | |
| Checkpoint | Mỗi 21 step: 21, 42, 63, 84, 105, 126; save limit 10; giữ tất cả cho đến khi đánh giá xong | `REFINEMENT_EVAL_SAVE_STEPS=21` |
| Log | Mỗi step, có diagnostics từng step | `REFINEMENT_LOGGING_STEPS=1`, `REFINEMENT_STEP_DIAGNOSTICS=true` |
| Theo dõi trong lúc train | 32 cặp Gemma cố định trong 1008 cặp (chỉ là báo động sớm) | `REFINEMENT_GEMMA_MONITOR_PAIRS=32` |
| Tập theo dõi cố định 32 Unlabeled (probe) | **Bỏ**: chỉ số của nó (độ rộng hộp, tỉ lệ nằm trong hộp, projection KL, phương sai giữa các lượt Teacher) thuộc mục tiêu projection cũ, không dùng cho mục tiêu hiện tại | cần sửa code (mục 6) |
| Tài nguyên | partition normal, 2g.35gb, giới hạn 5 giờ | |

Loss mỗi step, với \(n_L = 8\) cặp có nhãn:

\[
\mathcal L=\frac{1}{8}\Big[\sum_{i\in L}\big(\ell^{H}_i+\ell^{G}_i+\ell^{M}_i\big)+\sum_{i\in U}\big(\ell^{CE}_i+\ell^{KL}_i\big)\Big].
\]

**Setting mặc định cho mọi experiment mới (chốt 03/11/2026).** Config bảng trên, cộng với:
- Student-Stage1 xuất phát: batch 16 **khởi tạo từ Inspector checkpoint-63** (`CLAIM2_STUDENT`). Teacher = Student-Stage1 này.
- Mọi λ = 1. **Train cả LoRA của LLM và ViT** (không đóng băng ViT/aligner).
- **Luôn bật log từng mẫu** (mục 6.1 việc 10 và 11): ở mỗi step, mỗi Unlabeled và mỗi cặp có nhãn một dòng, có nhãn argmax của Teacher, Student, Inspector và cờ giống / khác.
- Trong `config_training.sh`: mảng `DEFAULT_STAGE2`; một phiên bản mới là `tuning "${DEFAULT_STAGE2[@]}" <một thay đổi>`. `COMMON_CONFIG` vẫn trỏ Student-Stage1 khởi tạo mới để giữ nguyên các phiên bản cũ.
- Run mốc: `<gốc>_student_initialized_from_inspector_checkpoint63` (mục 4.1, cùng seed, cùng config, chưa có log từng mẫu). Checkpoint đã xoá; kết quả đánh giá checkpoint 126 (dữ liệu train và Bench) dùng lại được.

### 3.1. Vì sao giữ KL(Teacher ‖ Student)

- Với nhãn mềm của Teacher, KL(T ‖ S) = CE(T, S) − H(T). H(T) không phụ thuộc Student, nên đổi KL thành CE hay BCE với cùng nhãn mềm **không đổi gì khi train**; gradient theo logit đều là S − T. KL không "chặt" hơn CE.
- Teacher = Student-Stage1, nên ở đầu run S ≈ T và số hạng này gần 0 (chỉ còn nhiễu do dropout của Teacher). Khi loss Inspector và loss có nhãn kéo Student đi xa, số hạng này kéo Student về dự đoán của Student-Stage1, mạnh tỉ lệ với độ lệch. Trong setting 1 loop nó chỉ có vai trò **giữ Student gần Student-Stage1**, không mang thông tin mới; thông tin mới từ Unlabeled chỉ đến từ nhãn Inspector.
- Các dạng thực sự khác (nhãn cứng argmax của Teacher, MSE trên xác suất) chưa dùng. Tác dụng của số hạng này đo bằng λ_C (claim 2).

### 3.2. Teacher trong setting 1 loop

- Theo hướng tiếp cận gốc, Teacher là EMA của Student **cập nhật sau mỗi loop**, không phải sau mỗi optimizer step. Code hiện tại (`update_teacher_ema`, lịch cosine 0.99 → 0.9999) cập nhật sau mỗi optimizer step; các run Stage 2 trước đây đều dùng cách này, **không đúng thiết kế**.
- Với 1 loop, Teacher không cập nhật trong run: Teacher = Student-Stage1 (`teacher_initialized.pt`).
- Teacher chạy một lượt có dropout. Hiện code tắt dropout khi J = 1 (`model.eval()`); chỉ J ≥ 2 mới bật dropout.

### 3.3. Thời gian và bộ nhớ

Số đo trên 2g.35gb từ các run trước, microbatch 1 cặp:

| Phần | Số đo |
|---|---|
| Student forward + backward | khoảng 4.4 giây mỗi cặp (có nhãn và Unlabeled như nhau) |
| Teacher forward | khoảng 1.66 giây mỗi Unlabeled |
| Optimizer | khoảng 0.5 giây mỗi step |
| Một step 8 + 8 | 16 × 4.4 + 8 × 1.66 + 0.5 ≈ 84 giây |
| 126 step | khoảng 2.9 giờ; cả nạp model và đánh giá ban đầu khoảng 3.1 giờ |
| Bộ nhớ đỉnh | khoảng 29 GiB (phụ thuộc microbatch, không phụ thuộc số cặp mỗi step) |

- Association của truongtq (account `uit`) có `GrpJobs=3`: tối đa **3 job chạy cùng lúc** trên mọi partition.
- Mỗi run khoảng 6 checkpoint × 1.7 GB.

## 4. Các run (đã submit 02/11/2026)

Mọi run dùng config mục 3, cùng `REFINEMENT_SEED=42` và selection seed 42: cùng 1008 Unlabeled, cùng thứ tự cặp có nhãn và Unlabeled. Tên đầy đủ bắt đầu bằng `stage2_tuning_unlabeled1008_teacher_dropout_without_ema` (dưới đây viết là `<gốc>`).

### 4.1. Claim 1b: khởi tạo Student từ Inspector có giúp không

| Phiên bản | Student-Stage1 xuất phát (checkpoint-126) | Account |
|---|---|---|
| `<gốc>` | batch 16, khởi tạo mới | truongtq |
| `<gốc>_student_initialized_from_inspector_checkpoint63` | batch 16, khởi tạo từ Inspector checkpoint-63 | truongtq |

- Hai run chỉ khác Student-Stage1 xuất phát.
- So sánh hai run **ở cùng step**: mặc định checkpoint 126, thêm checkpoint giữa nếu một trong hai run cần (mục 5.1).
- Thước đo cuối là **EditReward-Bench**, không dùng Test. Với 115 cặp không Tie, chênh lệch vài cặp có thể không có ý nghĩa thống kê (như claim 1a, mục 7.1).

### 4.2. Claim 2: trọng số các loss thành phần

Điểm xuất phát: `<gốc>_student_initialized_from_inspector_checkpoint63` (mọi λ = 1). Mỗi run đổi đúng một trọng số, nhân hoặc chia 2.

| Nhóm | Phiên bản (`<gốc>_student_initialized_from_inspector_checkpoint63_…`) | Account | Câu hỏi |
|---|---|---|---|
| λ_U | `lambda_unlabeled0`, `lambda_unlabeled0.5`, `lambda_unlabeled2` | anhndt (chạy trước) | Học từ Unlabeled giúp hay hại. λ_U nhân cả CE Inspector và KL Teacher, nên λ_U = 0 là đối chứng không Unlabeled (Unlabeled vẫn forward, trọng số 0, cùng luồng dữ liệu) |
| λ_C | `lambda_consistency0`, `lambda_consistency2` | truongtq (λ_C = 0 chạy trước) | KL Teacher (giữ Student gần Student-Stage1) có cản việc học từ Unlabeled không |
| λ_M | `lambda_mask0.5`, `lambda_mask2` | anhndt (hàng đợi) | Relevance là phần lệch nặng nhất ở các run cũ |
| λ_G | `lambda_gemma0.5` (anhndt), `lambda_gemma2` (truongtq) | hàng đợi | Mức giữ nhãn preference Gemma |

- **λ_H = 1 là mốc.** Mọi step đều bị clip (mục 7.3), nên nhân mọi λ với cùng hằng số không đổi gì; chỉ tỉ lệ giữa các λ quyết định hướng cập nhật.
- **Loss con bên trong** (chưa thử, tầng 2): L_H là một CE ba lớp; L_G = r·BCE(P(Tie), g_T) + r·(g_A + g_B)·BCE(P(A thắng | không Tie)), tie : winner = 1 : 1 cố định trong code (cần thêm tuỳ chọn nếu muốn thử); L_M là một BCE; trong L_U tỉ lệ CE : KL là λ_C. Hệ số replay 1.5 của L_G / L_M chỉ đổi sau một loop, nên luôn là 1 ở đây.
- **Lưu ý khi đọc kết quả:** ở step 2 của run `<gốc>`, norm gradient trước clip là 41: phần có nhãn 40, phần Unlabeled 8.3 (tỉ lệ 0.21); theo nhóm ViT 39.9, LLM 4.8, heads 8.4. Clip toàn cục chia mọi gradient cho khoảng 41, nên hướng cập nhật chủ yếu do gradient ViT của phần có nhãn. Nếu các run λ_G, λ_M gần như trùng nhau, có thể do clip toàn cục chứ không phải trọng số không quan trọng; phải xem tỉ lệ gradient theo nguồn và nhóm trong diagnostics trước khi kết luận.

Job và md5 code lúc submit: mục 6.1.

### 4.3. Đóng băng ViT/aligner ở Stage 2 (03/11/2026)

| Phiên bản | Thay đổi so với `<gốc>_student_initialized_from_inspector_checkpoint63` | Account |
|---|---|---|
| `<gốc>_student_initialized_from_inspector_checkpoint63_vit_aligner_frozen` | `REFINEMENT_FREEZE_VISUAL=true`: chỉ train LoRA của LLM và metric_heads (như Inspector); LoRA ViT và aligner giữ giá trị của Student-Stage1, không nằm trong optimizer | truongtq |

- Câu hỏi: gradient ViT là nhiễu chi phối hệ số clip (mục 7.3); bỏ nó có giúp Stage 2 không.
- Cấu trúc adapter giữ nguyên (Student-Stage1 có LoRA ViT), nên backward vẫn đi qua ViT: thời gian mỗi step không đổi.
- Phiên bản `…_vit_aligner_frozen_test_2_steps` (16 Unlabeled, lưu mỗi step, 30 phút) có trong `config_training.sh` để kiểm tra code; không dùng. Job 9653 (bản này cộng log từng mẫu) bị huỷ khi còn chờ, không chạy (03/11).

## 5. Đánh giá

Thứ tự: (1) log train → (2) dữ liệu train Stage 2 trên các checkpoint được chọn → (3) EditReward-Bench. Không dùng Test. Mọi phân tích đều có hình.

### 5.1. Log train (sau khi job train xong, không cần GPU)

Job train không tự vẽ hình. Ngay sau khi một run xong, chạy `RewardModelEvaluation/OurModel/plot_stage2_tuning.py --versions <phiên bản>`: script chỉ đọc log train (không đọc dữ liệu), bỏ qua job chưa xong, ghi hình vào `evaluation_results/train_loop/<phiên bản>/`. Đã kiểm tra trên log dở dang của run `<gốc>`: đọc đúng định dạng log mới (`ema_decay=1.00000`, không probe) và vẽ được hình loss, norm gradient, learning rate, diagnostics từng step.

Hình theo step, có đường trung bình trượt (loss từng step của 8 cặp rất nhiễu):

| Chỉ số | Cho biết điều gì |
|---|---|
| Loss tổng và từng thành phần chưa nhân λ: HumanGT, Gemma preference, Gemma relevance mask, CE Unlabeled với Inspector, KL(Teacher ‖ Student) | Thành phần nào giảm, thành phần nào tăng |
| Norm gradient trước clip: tổng, phần có nhãn, phần Unlabeled; theo nhóm vit / aligner / llm / heads | Gradient có bùng nổ không; trọng số thực tế của Unlabeled so với phần có nhãn |
| Cosine giữa gradient có nhãn và gradient Unlabeled | Hai nguồn kéo cùng hướng hay ngược hướng |
| Tỉ lệ cập nhật ‖ΔW‖ / ‖W‖ theo nhóm; learning rate | Tốc độ học thật |

- Mỗi Unlabeled chỉ đi một lần, nên CE Unlabeled ở mỗi step được tính **trước khi** Student học mẫu đó: đường này đã là phép đo trên mẫu Student chưa thấy.
- Norm gradient trong log là giá trị trước clip; ở các run trước gần như mọi step đều bị clip về 1.0.
- Dựa vào hình loss và norm gradient để **chọn checkpoint cần đánh giá**: mặc định checkpoint 126; thêm tối đa một checkpoint giữa nếu hình có biến động (loss Unlabeled đi ngang hoặc tăng, norm gradient bất thường). Người dùng xác nhận trước khi chạy.
- **Log từng mẫu** (mọi run mới): chạy `RewardModelEvaluation/OurModel/summarize_per_sample_log.py --version <phiên bản>` (chỉ đọc file log, không cần GPU; job lấy từ `experiment_submissions.log`). Ghi vào `evaluation_results/train_loop/<phiên bản>/per_sample/`:
  - `per_sample_summary.json`: kiểm tra số dòng mỗi step; tỉ lệ Student = Teacher, Student = Inspector, Teacher = Inspector trên Unlabeled và Student = nhãn người trên cặp có nhãn, theo từng step và toàn run; số nhãn A / B / Tie của mỗi bên; các kiểu khác (Teacher X → Student Y); tỉ lệ theo mức tự tin của Inspector (P lớn nhất < 0.45, 0.45–0.55, ≥ 0.55).
  - `student_teacher_disagreements.jsonl`: mọi Unlabeled mà Student khác Teacher, đủ các trường của log.
  - `per_sample_agreement.png`: tỉ lệ cùng nhãn theo step (có trung bình trượt 10 step), kiểu khác, theo mức tự tin của Inspector.
  - Teacher ở đây là lượt có dropout dùng làm target của loss; Student là forward lúc train (train mode, dropout bật, trước khi cập nhật ở step đó). Khác với mục 7.6 (sau training, eval mode, 256 mẫu).

### 5.2. Dữ liệu train Stage 2 (chỉ trên checkpoint được chọn)

| Tập | Số mẫu | Ghi chú |
|---|---|---|
| Unlabeled đã train | 256: dùng lại tập `trained_unlabeled_seed50042_size256` của file tập đánh giá cũ | Cả 256 mẫu nằm trong 1008 Unlabeled của mọi run (cùng seed rút mẫu, kiểm tra 02/11 trên `unlabeled_subset.json` thật), là một tập con ngẫu nhiên đều của 1008 mẫu. Step 0 của cả hai Student-Stage1 batch 16 đã có |
| Cặp có nhãn | 256, rút từ 1008 cặp | Dùng lại tập 256 cặp cũ; step 0 đã có kết quả |

File tập đánh giá: `evaluation_results/train_loop/stage2_tuning_evaluation_sets.json` (đường dẫn mặc định của launcher). File gốc ở `stage2_tuning_comparison/` đã bị xoá cùng thư mục đó (03/11); file hiện tại được dựng lại 03/11 từ đúng các `sample_id` lưu trong kết quả step 0 của hai Student-Stage1 batch 16 (hai file trùng nhau), chỉ gồm 2 tập đang dùng, md5 Overall Pseudo-label Cache a876c5172ff46b22aec7a612c897bd11. Đã kiểm tra: 256 Unlabeled nằm trong `unlabeled_subset.json` của run, 256 cặp nằm trong 1008 cặp của cả hai Student-Stage1 batch 16. Biến của job đánh giá: `STAGE2_EVAL_RUN_DIR=<thư mục run>`, `STAGE2_EVAL_TRAINED_SET_KEY=trained_unlabeled_seed50042_size256` (script kiểm tra mọi ID của tập nằm trong `unlabeled_subset.json` của run), `STAGE2_EVAL_SKIP_UNTRAINED=true` (không đánh giá Unlabeled chưa train).

- Mỗi checkpoint một job, khoảng 12 phút (khoảng 1.4 giây mỗi cặp).
- Trên Unlabeled: CE / KL với Inspector, tỉ lệ cùng argmax (tách Inspector tự tin ≥ 0.7 và < 0.7), Spearman margin, trung bình |d|, P(Tie), entropy. Các chỉ số này đo **mức giống Inspector**, không đo đúng hay sai.
- Trên cặp có nhãn: CE ba lớp và accuracy HumanGT, loss Gemma, relevance BCE, Tie Brier, số cặp đổi dự đoán và |Δmargin| so với step 0. Đây là dữ liệu đã train, nên chỉ đo **giữ được hay bị lệch**.
- So với step 0 bằng khoảng tin cậy bootstrap ghép cặp; ưu tiên chỉ số liên tục vì accuracy ít nhạy.

### 5.3. EditReward-Bench

- Chỉ chạy sau 5.1–5.2, trên checkpoint được chọn. 133 cặp (115 không Tie, 18 Tie), mỗi checkpoint một job khoảng 5 phút.
- Đây là tập duy nhất đo đúng / sai theo nhãn người trên dữ liệu chưa thấy. Với 115 cặp, chênh lệch vài cặp không có ý nghĩa thống kê (McNemar, bootstrap ghép cặp).

Kết quả và hình: `RewardModelEvaluation/OurModel/evaluation_results/train_loop/<phiên bản>/`.

## 6. Thay đổi code

### 6.1. Thay đổi cho config mục 3 (xong 02/11/2026)

Sao lưu: `Qwen3-VL-Embedding/scripts/backups/train_loop.before_teacher_update_after_round.py`, `run_train_loop.before_teacher_update_after_round.slurm`, `config_training.before_teacher_update_after_round.sh`, `plot_stage2_tuning.before_probe_optional.py`. Mọi tuỳ chọn mới có mặc định giữ hành vi cũ.

| # | File | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | `train_loop.py`, `run_train_loop.slurm` | `--teacher-update {optimizer_step, after_round}` (`REFINEMENT_TEACHER_UPDATE`, mặc định `optimizer_step`). `after_round`: không cập nhật Teacher sau mỗi optimizer step; cập nhật EMA một lần cuối mỗi loop với `--ema-decay`. Với 1 loop, lần cập nhật này xảy ra sau step cuối, không ảnh hưởng training. Mỗi checkpoint và cuối loop in `Teacher max absolute change` so với đầu loop (phải bằng 0). Log từng step ghi `ema_decay=1.00000` (Teacher giữ nguyên) | xong |
| 2 | `train_loop.py`, `run_train_loop.slurm` | `--teacher-dropout {auto, on}` (`REFINEMENT_TEACHER_DROPOUT`, mặc định `auto` = J = 1 không dropout). `on`: lượt Teacher duy nhất bật dropout của LoRA (0.05) và metric_heads (0.05); chỉ áp dụng khi tạo target lúc train, đánh giá vẫn ở eval mode | xong |
| 3 | `train_loop.py`, `plot_stage2_tuning.py` | `--diagnostic-probe-size 0`: không chọn, không chạy, không lưu probe (JSON ghi `null`, log ghi `probe=none`); hình probe bỏ qua run không có probe. Mặc định 128 giữ nguyên | xong |
| 4 | `config_training.sh` | `TUNING_CONFIG` theo mục 3 (thêm `REFINEMENT_SEED=42`); 2 phiên bản claim 1b, giới hạn 5 giờ trên normal | xong |
| 5 | `train_loop.py` | Khi λ_U = 0: Unlabeled vẫn forward (Teacher và Student), loss nhân 0; run λ_U = 0 tốn thời gian như các run khác | đã kiểm tra, không sửa |
| 6 | `config_training.sh` | 9 phiên bản claim 2 (mục 4.2) dùng mảng `CLAIM2_STUDENT`; sao lưu `config_training.before_claim2_loss_weights.sh` | xong |
| 7 | `evaluate_stage2_tuning_checkpoints.py`, `run_evaluate_stage2_tuning_checkpoints.slurm` | `--trained-set-key` (`STAGE2_EVAL_TRAINED_SET_KEY`): dùng một tập đã train có sẵn trong file tập đánh giá, báo lỗi nếu có ID nằm ngoài `unlabeled_subset.json` của run; `--skip-untrained-set` (`STAGE2_EVAL_SKIP_UNTRAINED`). Sao lưu `*.before_trained_set_key.*` | xong |
| 8 | `plot_stage2_tuning.py` | Sửa 2 lỗi ở hình đánh giá trên dữ liệu train: (1) step 0 (file Student-Stage1 chứa mọi tập đã train) đọc **cùng tập** với các checkpoint, trước đây đọc tập đầu tiên (`size2000`, sai); (2) file 1008 cặp ở step 0 cũng cho điểm 256 cặp. Ô của tập không được đánh giá ghi "not evaluated". Đã kiểm tra với file step 0 thật và một file checkpoint giả: step 0 KL đúng 0.0544 (tập `size256`) | xong |
| 9 | `train_loop.py`, `run_train_loop.slurm`, `config_training.sh` | `--freeze-visual {false, true}` (`REFINEMENT_FREEZE_VISUAL`, mặc định `false`). `true`: đặt `requires_grad=False` cho mọi tham số ViT và aligner trước khi tạo optimizer (không vào optimizer, không weight decay); báo lỗi nếu không tìm thấy tham số ViT/aligner hoặc optimizer còn tham số `visual`; mỗi checkpoint in `visual LoRA max absolute change since Student-Stage1` (phải bằng 0). Mọi run in `Trainable parameters per group`. Resume: checkpoint cũ không có tuỳ chọn này được coi là `false`. Thêm 2 phiên bản mục 4.3. Sao lưu `*.before_freeze_visual.*` | xong (03/11) |
| 10 | `train_loop.py`, `run_train_loop.slurm` | `--per-sample-log {false, true}` + `--per-sample-log-path` (`REFINEMENT_PER_SAMPLE_LOG`, mặc định `false`; launcher ghi vào `TrainingRewardModel/logs/train_loop_<job>_per_sample.jsonl`, không ghi vào `output_training/`). Mỗi cặp đã train và step một dòng JSON, lấy từ forward lúc train (Student trước khi cập nhật, train mode), không forward thêm. Unlabeled: xác suất ba lớp của Inspector, Teacher (gộp theo metric Teacher chọn), Student; trạng thái Teacher; CE Inspector, KL(Inspector ‖ Student), KL(Teacher ‖ Student); số metric được chọn. Cặp có nhãn: nhãn người, xác suất ba lớp của Student, CE HumanGT, loss Gemma. Chỉ với `merged` và loss ba lớp; từ chối ghi vào file đã có; resume không so đường dẫn file. Kiểm tra trên CPU với dữ liệu giả: tổng CE, KL Teacher và loss từng mẫu khớp đúng hàm loss. Khi `false`, lệnh python sinh ra giống hệt trước khi sửa. Sao lưu `*.before_per_sample_log.*`; md5 sau khi sửa: `train_loop.py` 77def13eb07e776490ce5139cba6dcd5, `run_train_loop.slurm` 235317ec4397f5632fe2ceafca955534 | xong (03/11); bật mặc định cho mọi run mới (việc 11) |
| 11 | `train_loop.py`, `config_training.sh`, `RewardModelEvaluation/OurModel/summarize_per_sample_log.py` (file mới) | (a) Mỗi dòng Unlabeled thêm `teacher_label`, `student_label`, `inspector_label` (argmax A / B / Tie) và `student_same_as_teacher`, `student_same_as_inspector`, `teacher_same_as_inspector` (true / false); mỗi dòng cặp có nhãn thêm `student_label`, `student_same_as_human`. Chỉ thêm trường vào log, không đổi loss. Sao lưu `train_loop.before_per_sample_labels.py`; md5 sau khi sửa 4f518d81aeabcda5acc1233a3ae03a1a. Kiểm tra trên CPU với dữ liệu giả: nhãn đúng argmax, cờ đúng. (b) `config_training.sh`: mảng `DEFAULT_STAGE2` (mục 3) và phiên bản thử `<gốc>_student_initialized_from_inspector_checkpoint63_test_2_steps` (16 Unlabeled = 2 step, lưu mỗi step, 30 phút; khác run mốc chỉ ở số Unlabeled, checkpoint và log từng mẫu). Sao lưu `config_training.before_default_stage2_per_sample_log.sh`; md5 56b96042bda431eaa05db99f4615b945. (c) Script tổng hợp (mục 5.1), đã chạy thử trên log giả 30 step | xong (03/11); job thử **9658** COMPLETED 04/11 (9 phút, exit 0): 32 dòng = 2 step × (8 Unlabeled + 8 cặp có nhãn), mỗi Unlabeled một lần, nhãn đúng argmax, cờ đúng, `teacher_kl` / `inspector_kl` / CE trung bình mỗi step khớp đúng log train; script tổng hợp chạy được trên log thật (env qwen_vl) |
| 12 | `Qwen3-VL-Embedding/scripts/score_teacher_inspector_agreement.py`, `run_score_teacher_inspector_agreement.slurm` (file mới) | Chọn Unlabeled đồng thuận Teacher – Inspector (mục 8.1, bước 1): duyệt theo entropy Inspector tăng dần, Teacher eval mode (tắt dropout), tính CE(Inspector, Teacher) như L_U với Student thay bằng Teacher; nhận mẫu khi nhãn argmax trùng, dừng khi đủ 1008; ghi từng dòng (chạy tiếp được nếu job dừng) | xong; job **9667** (04/11, 28 phút) |
| 13 | `Qwen3-VL-Embedding/scripts/student_augmentation.py`, `build_augmentation_scenarios.py` (file mới) | Online augmentation cho Student (mục 8.1, bước 2): quy tắc masking / injection / ghép ảnh / lật ngang / đổi chỗ A ↔ B, hàm ghép và lật ảnh, kiểm tra tự động; biến thể instruction tạo lúc train theo seed (mẫu, iteration), không đạt kiểm tra thì dùng instruction gốc. Script build chỉ xác định kịch bản được phép cho từng mẫu + file xem trước. Sao lưu `*.before_online_variants.py` | xong (04/11); chưa nối vào `train_loop.py` |

Nhãn log chưa sửa: `run_train_loop.slurm` dòng `MC-Dropout passes:   4` in cứng, giá trị thật là dòng `Teacher MC passes: 1`.

Kiểm tra trên CPU (model nhỏ giả lập, không nạp model thật): Teacher `on` cho hai lượt khác nhau, `auto` cho hai lượt giống nhau, model trở về eval mode sau lượt Teacher; `--diagnostic-probe-size -1` bị chặn, 0 được chấp nhận; hàm đo độ lệch Teacher đúng.

Submit 02/11/2026 qua `config_training.sh`, ghi trong `experiment_submissions.log`; md5 lúc submit: `train_loop.py` b00b8bb5e3469df0c0124376c1bfe2a6, `run_train_loop.slurm` c4a9557c2d141625de1b237334feb67c.

| Job | Phiên bản | Student-Stage1 xuất phát |
|---|---|---|
| 9099 | `stage2_tuning_unlabeled1008_teacher_dropout_without_ema` | batch 16, khởi tạo mới |
| 9100 | `stage2_tuning_unlabeled1008_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint63` | batch 16, khởi tạo từ Inspector checkpoint-63 |

Claim 2 (mục 4.2), cùng md5 `train_loop.py` và `run_train_loop.slurm`:

| Job | Account | Thay đổi so với điểm xuất phát |
|---|---|---|
| 9117 | truongtq | λ_C = 0 |
| 9118 | truongtq | λ_C = 2 |
| 9119 | truongtq | λ_G = 2 |
| 9126 | anhndt | λ_U = 0 |
| 9127 | anhndt | λ_U = 0.5 |
| 9128 | anhndt | λ_U = 2 |
| 9129 | anhndt | λ_M = 0.5 |
| 9130 | anhndt | λ_M = 2 |
| 9131 | anhndt | λ_G = 0.5 |

Đánh giá (03/11/2026), checkpoint 126, mỗi run một job dữ liệu train Stage 2 (`run_evaluate_stage2_tuning_checkpoints.slurm`, tập `trained_unlabeled_seed50042_size256` + 256 cặp có nhãn) và một job EditReward-Bench (`run_evaluate_student_checkpoints.slurm`, `STUDENT_EVAL_DATASETS=bench`):

| Claim | Phiên bản | Job dữ liệu train | Job Bench |
|---|---|---|---|
| 1b | `<gốc>` | 9225 | 9246 |
| 1b | `<gốc>_student_initialized_from_inspector_checkpoint63` | 9226 | 9247 |
| 2 | λ_U = 0 / 0.5 / 2 | 9389 / 9391 / 9393 | 9390 / 9392 / 9394 |
| 2 | λ_C = 0 / 2 | 9395 / 9397 | 9396 / 9398 |
| 2 | λ_M = 0.5 / 2 (anhndt) | 9408 / 9410 | 9409 / 9411 |
| 2 | λ_G = 0.5 / 2 (anhndt) | 9412 / 9414 | 9413 / 9415 |

Đóng băng ViT/aligner (mục 4.3): train job **9484** (truongtq, 03/11, xong sau 3 giờ 8 phút); md5 lúc submit `train_loop.py` 71649ab5d2745f4dd8d8864b7f30b22c, `run_train_loop.slurm` 9c07b2dc660e08f7d131fc85cce1eee7. Lệnh sbatch chỉ khác `<gốc>_student_initialized_from_inspector_checkpoint63` ở `REFINEMENT_FREEZE_VISUAL=true` (đã so danh sách biến). Đánh giá checkpoint 126: dữ liệu train job 9598 (truongtq), Bench job 9599 (anhndt).

Rà soát sau khi submit (03/11/2026): cả hai Student-Stage1 lấy từ checkpoint-126, cùng 1008 ID cặp và cùng thứ tự; Teacher ban đầu trùng Student (độ lệch 0); head fp32, có theta0 / kappa. Log 2 step đầu của run `<gốc>`: Teacher `on` / `after_round`, `ema_decay=1.00000`, 1008 Unlabeled, 1008 cặp có nhãn (validation 0), probe 0, mọi λ = 1 từ step 1, không đặt max_pixels, Student được đặt lại `model.train()` sau mỗi lượt Teacher; khoảng 74–94 giây mỗi step, bộ nhớ đỉnh 27.9 GiB.

### 6.2. Đã làm

| # | File | Nội dung | Trạng thái |
|---|---|---|---|
| 1 | `config_training.sh` | Partition `normal` (4 CPU / 36 GB), giới hạn thời gian theo phiên bản, mục job thử | xong (02/11) |
| 2 | `train_loop.py` | `--unlabeled-subset-size` (tập con cố định, seed + 50,000, lưu `unlabeled_subset.json`; phần còn lại là Unlabeled chưa train) và `--unlabeled-passes` (chỉ với `all_once`, mỗi lượt xáo lại, seed + 60,000) | xong (02/11), sao lưu `train_loop.before_unlabeled_subset.py` |
| 2b | `train_loop.py`, `run_train_loop.slurm`, `config_training.sh` | `--gemma-monitor-pairs N` (theo dõi Gemma trên N cặp train cố định, seed + 70,000; 0 = cả 1008 cặp); launcher thêm `REFINEMENT_UNLABELED_SUBSET_SIZE`, `REFINEMENT_UNLABELED_PASSES`, `REFINEMENT_GEMMA_MONITOR_PAIRS` | xong (02/11), sao lưu `train_loop.before_gemma_monitor_pairs.py`, `run_train_loop.before_stage2_tuning.slurm` |
| 3 | `train_loop.py`, `run_train_loop.slurm` | `--lambda-human` (`REFINEMENT_LAMBDA_HUMAN`); `--loss-weighting per_pair` (`REFINEMENT_LOSS_WEIGHTING`, chỉ với `merged`): mỗi cặp chia cho số cặp có nhãn của step (không có cặp có nhãn thì chia cho số cặp Unlabeled) | xong (02/11) |
| 4 | `train_loop.py`, `run_train_loop.slurm` | `--labeled-source merged` (`REFINEMENT_LABELED_SOURCE`): một forward cho HumanGT + GemmaGT + mask, báo lỗi nếu hai tập khác ID, ảnh hoặc thứ tự A/B; `--labeled-pairs-per-step` (`REFINEMENT_LABELED_PAIRS_PER_STEP`); `separate` giữ hành vi cũ | xong (02/11), sao lưu `train_loop.before_merged_labeled_source.py`, `run_train_loop.before_merged_labeled_source.slurm` |
| 5 | `train_loop.py`, `run_train_loop.slurm` | `--mc-passes J` (`REFINEMENT_MC_PASSES`, mặc định 4); hiện J = 1 là một lượt Teacher không dropout, J ≥ 2 là MC-Dropout | xong (02/11); J = 1 có dropout cần sửa (6.1 #2) |
| 6 | `train_loop.py`, `run_train_loop.slurm`, `config_training.sh` | `--step-diagnostics` (`REFINEMENT_STEP_DIAGNOSTICS`, chỉ với `merged`): mỗi step một dòng `step=N diagnostics {json}` với giá trị của riêng step đó: loss HumanGT / Gemma / mask / tie / winner; loss Unlabeled tách thành CE với Inspector, entropy nhãn Inspector, KL(Inspector ‖ Student), KL(Teacher ‖ Student); norm gradient trước clip của phần có nhãn, phần Unlabeled và tổng, theo nhóm vit / aligner / llm / heads; cosine giữa hai gradient; tỉ lệ cập nhật theo nhóm; learning rate; λ_U(t); hệ số clip | xong (02/11), sao lưu `train_loop.before_step_diagnostics.py`, `run_train_loop.before_step_diagnostics.slurm`, `config_training.before_step_diagnostics.sh` |
| 7 | `RewardModelEvaluation/OurModel/evaluate_stage2_tuning_checkpoints.py`, `run_evaluate_stage2_tuning_checkpoints.slurm` | Đánh giá offline trên dữ liệu train Stage 2. Lệnh `sets` chọn các tập cố định (`evaluation_results/train_loop/stage2_tuning_comparison/stage2_tuning_evaluation_sets.json`); lệnh `evaluate` nạp model một lần, đổi adapter theo từng checkpoint; `--labeled-all-pairs` cho cả 1008 cặp. Đường forward offline (`processor`) gần nhưng không trùng đường forward lúc train (template MS-Swift), nên số offline chỉ so với nhau và với Bench | xong (02/11). Tập 1008 Unlabeled mới cần thêm vào file tập đánh giá |
| 8 | `train_loop.py`, `run_train_loop.slurm` | `--head-learning-rate` (`REFINEMENT_HEAD_LEARNING_RATE`); `--lr-scheduler-type cosine / constant_with_warmup` (`REFINEMENT_LR_SCHEDULER_TYPE`); `REFINEMENT_WARMUP_RATIO`; `--selection-seed` (`REFINEMENT_SELECTION_SEED`) | xong (02/11), sao lưu `train_loop.before_head_learning_rate.py`, `run_train_loop.before_head_learning_rate.slurm` |
| 9 | `RewardModelEvaluation/OurModel/plot_stage2_tuning.py` | Hình từ log train và kết quả đánh giá; hình so sánh các phiên bản | xong (02/11) |
| 10 | `config_training.sh` | `TUNING_CONFIG`, hàm `tuning`; `experiment_submissions.log` mở quyền ghi cho anhndt. `TUNING_CONFIG` đã cập nhật theo mục 3 (02/11); `OVERFIT_CONFIG`, `LABELED_ONE_PASS_CONFIG` và các phiên bản trước 02/11 vẫn là config sai batch, không dùng nữa | xong (02/11), sao lưu `config_training.before_stage2_tuning_versions.sh` |
| 11 | `Qwen3-VL-Embedding/scripts/diagnose_gradient_components.py`, `run_diagnose_gradient_components.slurm` (file mới) | Chẩn đoán gradient từng loss thành phần (mục 7.3): dựng model đúng như train (`run_ms_swift_online`), chạy lại N step đầu trên đúng dữ liệu và thứ tự của run, backward riêng từng thành phần (L_H, phần tie và phần winner của L_G, L_M, CE Inspector, KL Teacher = loss đầy đủ − loss với λ_C = 0); tính norm, cosine, ‖tổng‖² / Σ‖thành phần‖², tỉ lệ ‖gradient trung bình qua step‖ / trung bình ‖gradient từng step‖, norm theo từng mẫu. Step đầu tự kiểm tra: tổng các thành phần so với backward của tổng, sai số phải dưới 3 lần mức làm tròn bf16 (backward(3L)/3 so với backward(L)); so loss và gradient norm với log train. Trọng số Student: Student-Stage1 hoặc `checkpoint-N.pt` | xong (03/11), md5 script 8a9cb6f0f16c2542e93886f90d6fef92 |
| 12 | `RewardModelEvaluation/OurModel/plot_gradient_components.py` (file mới) | Hình từ JSON của việc 11: norm từng thành phần theo step và nhóm, ma trận cosine, tỉ lệ tín hiệu / nhiễu, gradient ViT theo số patch ảnh | xong (03/11) |

## 7. Kết quả

Output training của 12 run mục 7.2, 7.4, 7.5 (claim 1b, claim 2, đóng băng ViT/aligner): mọi file `.pt` (checkpoint-21…126, `best.pt`, `round_01.pt`, `screening_round_01.pt`) **đã xoá 03/11/2026** theo yêu cầu người dùng; còn `args.json`, `round_01_summary.json`, `unlabeled_subset.json`. Kết quả đánh giá và hình trong `evaluation_results/train_loop/` giữ nguyên; các run này không đánh giá lại được.

### 7.1. Claim 1a: Student-Stage1 batch 16, khởi tạo mới so với khởi tạo từ Inspector checkpoint-63

Hai run giống nhau ngoài cách khởi tạo: 1008 cặp nhãn Gemma, batch 16, 126 step (2 epoch), learning rate 4e-5 cosine.

- **Log train:** loss của bản khởi tạo từ Inspector thấp hơn khoảng 0.03–0.1 suốt run (train loss trung bình 0.951 so với 1.026); norm gradient trước clip có nhiều đỉnh hơn (67.6, 86.8, 47.0 so với tối đa 35.4).
- **1008 cặp có nhãn (dữ liệu train, checkpoint cuối):** bản khởi tạo từ Inspector khớp tốt hơn (CE ba lớp 0.785 so với 0.841; accuracy không Tie 84.4% so với 81.3%).
- **256 Unlabeled chưa train:** bản khởi tạo từ Inspector lại xa Inspector hơn (KL 0.114 so với 0.065; cùng argmax 82.4% so với 87.1%) và tự tin hơn (|d| 0.51 so với 0.47).
- **EditReward-Bench (115 cặp không Tie):** bản khởi tạo từ Inspector cao hơn ở 5 / 6 checkpoint về accuracy (checkpoint 126: 75.7% so với 73.9%) và 6 / 6 về margin AUC (78.6 so với 78.1). Chênh lệch chỉ 1–5 cặp; McNemar p ≥ 0.22, khoảng tin cậy 95% đều chứa 0. Từ checkpoint 63, log loss ba lớp và Tie AUC của bản khởi tạo từ Inspector xấu hơn (18 cặp Tie, rất nhiễu). Tham chiếu: Inspector checkpoint-63 accuracy 73.0%, margin AUC 74.6 (head overall, khác cách gộp selected_equal).
- **Kết luận:** chưa đủ bằng chứng để nói khởi tạo từ Inspector giúp Student tốt hơn. Chỉ một seed mỗi bên.

### 7.2. Claim 1b: Stage 2 từ Student-Stage1 khởi tạo mới và từ Student-Stage1 khởi tạo từ Inspector checkpoint-63

EditReward-Bench, checkpoint 126 (133 cặp: 115 không Tie, 18 Tie). Hình: `evaluation_results/train_loop/claim1b_student_initialization/` (`claim1b_bench_metrics.png`, `claim1b_bench_paired_differences.png`, `claim1b_gradient_norm_by_group.png`; số liệu `claim1b_bench_paired_comparison.json`).

| Checkpoint 126 | Accuracy | Margin AUC | Log loss ba lớp | Tie AUC (Tie Head) |
|---|---|---|---|---|
| Student-Stage1 khởi tạo mới | 73.91 | 78.07 | 0.907 | 64.3 |
| Student-Stage1 khởi tạo từ Inspector checkpoint-63 | 75.65 | 78.59 | 0.954 | 60.4 |
| Stage 2, Student khởi tạo mới | 73.91 | 79.07 | 0.871 | 70.2 |
| Stage 2, Student khởi tạo từ Inspector checkpoint-63 | 74.78 | 78.46 | 0.932 | 59.7 |

- **Xếp hạng tương đương** trong giới hạn của Bench: chỉ 4–5 cặp khác dự đoán; khoảng tin cậy của chênh lệch accuracy khoảng [−2.7, +4.8].
- Ở Stage 2, bản khởi tạo từ Inspector **xấu hơn** về log loss ba lớp (+0.062, p = 0.02) và Tie AUC (−10 điểm, p = 0.01–0.03).
- **Kết luận claim 1:** khởi tạo Student từ Inspector không giúp. Chỉ một seed và 18 cặp Tie, nên chênh lệch 1–4 điểm không phân định được.

### 7.3. Chẩn đoán gradient của run Stage 2 khởi tạo từ Inspector checkpoint-63 (03/11/2026)

**Hiện tượng trong log train (job 9100):** mọi step bị clip (norm trước clip trung vị 58, có step vượt 200, so với max_grad_norm 1.0). Theo nhóm, trung vị: ViT 57.6, LLM 5.4, heads 5.5, aligner 1.3; **ViT chiếm khoảng 97.5% bình phương norm**. Gradient norm gần như không tương quan với giá trị loss (r = 0.05–0.22). Với AdamW, hệ số clip dao động 8–13 lần nhưng tỉ lệ cập nhật / learning rate chỉ dao động 1.7–2.3 lần: clip chủ yếu đổi trọng số tương đối giữa các step (theo 1 / ‖g‖), không thu nhỏ bước cập nhật.

**Cách đo:** script việc 11 mục 6.1 chạy lại 16 step đầu trên đúng dữ liệu train, với hai trạng thái trọng số: Student-Stage1 khởi tạo từ Inspector checkpoint-63 (job 9341) và checkpoint 126 (job 9342). Cả hai qua bước tự kiểm tra; loss và gradient norm ở step 1 khớp log train (65.57 so với 65.53). Độ chính xác do bf16: norm khoảng 2–3%, cosine khoảng ±0.03. Chỉ 16 step mỗi trạng thái, nên con số là ước lượng. Kết quả và hình: `evaluation_results/train_loop/<gốc>_student_initialized_from_inspector_checkpoint63/gradient_components/`.

**1. Gradient norm lớn không phải do loss có vấn đề, mà do ViT.**
- Các step có gradient norm tăng vọt do **một mẫu duy nhất** trong batch: ở các step đó, mẫu lớn nhất chiếm 90–100% bình phương norm của thành phần tăng vọt.
- Ví dụ checkpoint 126, step 9 (norm tổng 483): mẫu có nhãn `1c8d00fa…_18` chiếm 98% gradient của L_H và phần winner của L_G. Trong mẫu này, gradient ViT là 255, nhưng LLM chỉ 2.3 và heads 1.2. Student-Stage1, step 13: L_M từ mức 3–4 lên 76.9, 100% do một mẫu.
- Cùng một mẫu, gradient ở heads và LLM bình thường nhưng ở ViT lớn gấp khoảng 100 lần: **gradient bị khuếch đại khi lan ngược vào ViT**, không do giá trị loss. Chưa rõ vì sao ViT nhạy với các ảnh này. Số patch ảnh giải thích rất ít (tương quan log–log r ≤ 0.29).
- Gradient ViT ở mỗi step chủ yếu là nhiễu riêng của mẫu: cosine giữa gradient tổng của hai step liên tiếp ≈ 0; tỉ lệ ‖gradient trung bình qua 16 step‖ / trung bình ‖gradient từng step‖ của ViT 0.25–0.4, gần mức nhiễu thuần tuý 1/√16 = 0.25.
- **Tác hại:** clip theo norm của toàn bộ tham số, nên nhiễu ViT quyết định hệ số clip của cả LLM và heads. Ở step 9 của checkpoint 126, mọi gradient bị chia cho 483 (step thường khoảng 40): gradient LLM và heads của step đó gần như bị bỏ vì một ảnh.
- Mức 1.0 chỉ là ngưỡng clip mặc định, không phải giá trị gradient norm "nên có".

**2. Các loss có giằng co không.**
- **Student-Stage1: không.** Các thành phần cùng hướng: ‖tổng‖² / Σ‖thành phần‖² = 1.51 (1 là trực giao, < 1 là triệt tiêu, > 1 là cùng hướng). L_H và phần winner của L_G gần cùng hướng (cosine +0.68). Gradient có nhãn và gradient Unlabeled gần trực giao (≈ 0): không kéo ngược, chỉ không giúp nhau.
- **Checkpoint 126: có một chỗ giằng co thật, trong L_U.** CE Inspector và KL Teacher kéo ngược nhau: cosine −0.43 trên toàn bộ tham số, −0.49 trên LLM (ở Student-Stage1 là +0.12). Ở heads, KL Teacher đẩy cùng một hướng ở hầu hết các step (tỉ lệ tín hiệu 0.94, mức nhiễu 0.25) và ngược L_H (cosine −0.24). Teacher = Student-Stage1 đóng băng và nhãn Inspector không đồng ý nhau; khi Student đi về phía Inspector, KL Teacher kéo lại. Tỉ lệ ‖tổng‖² / Σ‖thành phần‖² giảm còn 1.12, một số step 0.4–0.67 (các thành phần triệt tiêu một phần).

**3. Ghi nhận khác.**
- `score_head.residual_output.bias` không có gradient: bias này cộng như nhau vào điểm ảnh A và B (`custom_qwen3_vl_reranker.py`, dòng 442–443) nên triệt tiêu trong hiệu điểm. Đây là lý do log train ghi `zero_gradient: 1`; không phải lỗi. Tham số của Relevance Head không nhận gradient từ L_U.
- Stage 1 Student cũng bị clip ở mọi step (norm khoảng 14–16).

### 7.4. Claim 2: trọng số các loss (03/11/2026)

So mỗi run với `<gốc>_student_initialized_from_inspector_checkpoint63` (mọi λ = 1), checkpoint 126. Khoảng tin cậy 95% và p từ bootstrap ghép cặp (5,000 lần trên dữ liệu train, 3,000 lần trên Bench). Hình và số liệu: `evaluation_results/train_loop/claim2_loss_weights/` (`claim2_training_log_by_group.png`, `claim2_training_data_differences.png`, `claim2_bench_differences.png`, `claim2_paired_comparison.json`, script `claim2_analysis.py`).

**1. Log train.**
- KL(Inspector ‖ Student) trên Unlabeled (tính trước khi Student học mẫu đó, tức mẫu chưa thấy), trung bình 30 step cuối: λ_U = 0 / 0.5 / 1 / 2 cho 0.134 / 0.087 / 0.076 / 0.058. Khi λ_U = 0, chỉ train cặp có nhãn cũng làm Student trôi xa Inspector (0.089 → 0.134).
- λ_M chỉ làm L_M tách khỏi run gốc, các loss khác gần như trùng: clip toàn cục **không** làm mất tác dụng của trọng số (lo ngại ở mục 4.2 không xảy ra).
- Gradient norm trung vị 50–80 ở mọi run; các bậc nhảy do vài step tăng vọt như mục 7.3.

**2. Dữ liệu train Stage 2** (256 Unlabeled đã train, 256 cặp có nhãn). Run gốc so với Student-Stage1: KL với Inspector 0.099 → 0.059, CE ba lớp HumanGT 0.778 → 0.715, accuracy ba lớp +3.9 điểm, loss preference Gemma −0.019, BCE relevance −0.037 (đều p ≤ 0.01). So với run gốc, mỗi trọng số tác động đúng loss nó điều khiển và đổi lại phần khác:

| Thay đổi | KL với Inspector | CE HumanGT | Loss Gemma |
|---|---|---|---|
| λ_U = 0 | +0.054 (xa hơn) | −0.020 | preference −0.012 |
| λ_U = 0.5 | +0.010 | không đổi | không đổi |
| λ_U = 2 | −0.015 (gần hơn) | +0.023 | preference +0.012, relevance +0.003 |
| λ_C = 0 | −0.008 | không đổi | preference +0.009 |
| λ_C = 2 | không đổi | +0.016 (accuracy −2.3 điểm) | không đổi |
| λ_M = 0.5 / 2 | không đổi | 2: +0.014 | relevance +0.014 / −0.013 |
| λ_G = 0.5 / 2 | 0.5: −0.004 / 2: +0.011 | 2: +0.013 | preference +0.012 / −0.007 |

Cặp có nhãn là dữ liệu đã train, nên bảng chỉ đo giữ được hay bị lệch.

**3. EditReward-Bench** (133 cặp: 115 không Tie, 18 Tie).

| Checkpoint 126 | Accuracy | Margin AUC | Log loss ba lớp | Tie AUC |
|---|---|---|---|---|
| Student-Stage1 khởi tạo từ Inspector checkpoint-63 | 75.65 | 78.59 | 0.954 | 60.4 |
| Run gốc (mọi λ = 1) | 74.78 | 78.46 | 0.932 | 59.7 |
| λ_U = 0 / 0.5 / 2 | 73.91 / 74.78 / 73.91 | 78.04 / 78.41 / 78.04 | 0.996 / 0.942 / 0.910 | 63.6 / 62.6 / 61.6 |
| λ_C = 0 / 2 | 74.78 / 74.78 | 77.81 / 78.62 | 0.927 / 0.926 | 62.1 / 59.7 |
| λ_M = 0.5 / 2 | 73.91 / 73.91 | 77.77 / 77.88 | 0.933 / 0.942 | 62.2 / 57.9 |
| λ_G = 0.5 / 2 | 73.91 / 73.91 | 78.33 / 78.42 | 0.922 / 0.938 | 62.1 / 59.8 |

- **Accuracy và margin AUC:** mọi run khác run gốc 0–1 cặp đúng (McNemar p = 1); khoảng tin cậy của chênh lệch accuracy khoảng ±3–4 điểm, margin AUC khoảng ±1.2. Bench không phân biệt được các trọng số.
- **Log loss giảm đều theo λ_U** (0.996 → 0.942 → 0.932 → 0.910); hai đầu khác run gốc có ý nghĩa (λ_U = 0: +0.06, λ_U = 2: −0.02, cả hai p = 0.004). Nguyên nhân: nhãn Inspector rất mềm, học nhiều Unlabeled làm Student bớt tự tin (P lớn nhất trung bình 0.749 khi λ_U = 0, 0.698 khi λ_U = 2; log loss trên cặp đoán sai 2.26 → 1.90). Đây là thay đổi độ tự tin, không phải xếp hạng tốt hơn.
- λ_U = 0 có log loss xấu hơn cả Student-Stage1 (0.996 so với 0.954): chỉ train cặp có nhãn ở Stage 2 làm Student quá tự tin.
- Tie AUC cao hơn khi λ_U nhỏ (λ_U = 0: +3.9, p = 0.005; λ_U = 0.5: +2.9, p = 0.01), nhưng chỉ 18 cặp Tie.
- Có 36 phép so sánh trên Bench, nên với ngưỡng 0.05 sẽ có khoảng 2 kết quả "có ý nghĩa" do ngẫu nhiên. Kết quả log loss theo λ_U đáng tin hơn vì đơn điệu qua 4 giá trị và có nguyên nhân rõ; Tie AUC chưa đủ để kết luận.

**Kết luận claim 2:**
1. Học từ Unlabeled có tác dụng nhưng chỉ ở độ tự tin: Student gần Inspector hơn, log loss Bench tốt hơn; accuracy và margin AUC không đổi; độ khớp nhãn người trên dữ liệu train giảm nhẹ.
2. λ_C, λ_M, λ_G chỉ đổi loss mà chúng điều khiển trên dữ liệu train, không tạo khác biệt trên Bench.
3. Không có cơ sở chọn trọng số khác 1 (một seed, 133 cặp Bench). Ưu tiên log loss thì λ_U = 2 có lợi nhỏ, đổi lại lệch nhãn người trên dữ liệu train.

### 7.5. Đóng băng ViT/aligner ở Stage 2 (03/11/2026)

So `<gốc>_student_initialized_from_inspector_checkpoint63_vit_aligner_frozen` (mục 4.3) với `<gốc>_student_initialized_from_inspector_checkpoint63` (train cả ViT), checkpoint 126. Hình và số liệu: `evaluation_results/train_loop/<gốc>_student_initialized_from_inspector_checkpoint63_vit_aligner_frozen/comparison_with_base/` (`vit_aligner_frozen_training_log.png`, `vit_aligner_frozen_training_data_differences.png`, `vit_aligner_frozen_bench_differences.png`, `vit_aligner_frozen_paired_comparison.json`, script `vit_aligner_frozen_analysis.py`); hình log train ở thư mục cha.

**Kiểm tra code trong log:** đóng băng 30,799,872 tham số ViT và 4,587,520 tham số aligner; tham số được train chỉ còn LLM (174,587,904) và heads (5,565,214); gradient ViT/aligner bằng 0 ở mọi step; LoRA ViT/aligner không đổi so với Student-Stage1 ở cả 6 checkpoint (độ lệch lớn nhất 0). Dòng "215.5M Trainable" của MS-Swift được in trước khi đóng băng. Ở step 2, gradient heads trùng đúng run gốc (learning rate step 1 bằng 0 nên trọng số hai run vẫn giống nhau), tức đóng băng không đổi forward và gradient của phần còn train.

**1. Log train.**

| Trung vị theo step | Run gốc | Đóng băng ViT/aligner |
|---|---|---|
| Gradient norm trước clip (toàn bộ) | 58.1 | **7.9** |
| Theo nhóm: ViT / LLM / heads | 57.6 / 5.4 / 5.5 | 0 / 5.4 / 5.4 |
| Tỉ lệ cập nhật mỗi step: LLM / heads | 1.84e-4 / 0.96e-4 | 2.14e-4 / 1.11e-4 |
| Thời gian Student mỗi step / bộ nhớ đỉnh | 70.5 giây / 29.4 GiB | 70.7 giây / 28.6 GiB |

- Gradient LLM và heads không đổi; phần lớn gradient norm trước đây là ViT. Vẫn bị clip ở mọi step (norm 5–20 > 1.0), nhưng hệ số clip giờ do LLM và heads quyết định. Tỉ lệ cập nhật chỉ tăng khoảng 16% vì AdamW tự chuẩn hoá độ lớn bước.
- Các đường loss L_H, L_G, L_M, KL với Inspector trên Unlabeled chưa thấy và KL Teacher gần như trùng run gốc.

**2. Dữ liệu train Stage 2.**

| | Student-Stage1 | Run gốc | Đóng băng | Đóng băng − run gốc [khoảng tin cậy 95%] |
|---|---|---|---|---|
| KL(Inspector ‖ Student) | 0.099 | 0.059 | 0.059 | −0.000 [−0.003, +0.003] |
| CE ba lớp HumanGT | 0.778 | 0.715 | 0.736 | **+0.021** [+0.011, +0.032], p < 0.001 |
| Accuracy ba lớp HumanGT | 67.2 | 71.1 | 67.6 | **−3.5 điểm** [−5.9, −1.6], p = 0.001 |
| Loss preference Gemma | 0.462 | 0.443 | 0.449 | +0.006, p = 0.07 |
| BCE relevance Gemma | 0.218 | 0.181 | 0.180 | không đổi |

- Khớp Inspector như run gốc: học từ Unlabeled không cần ViT.
- Nhãn người trên cặp đã train khớp kém hơn, accuracy gần như về mức Student-Stage1. So với Teacher, cặp sai thành đúng 1.2% (run gốc 4.3%), cặp đổi dự đoán 3.1% (run gốc 7.4%): phần lớn mức cải thiện nhãn người trên dữ liệu đã train ở run gốc đến từ LoRA ViT. Chưa biết đó là học thật hay khớp ảnh đã thấy.

**3. EditReward-Bench.**

| Checkpoint 126 | Accuracy | Margin AUC | Log loss ba lớp | Tie AUC |
|---|---|---|---|---|
| Student-Stage1 khởi tạo từ Inspector checkpoint-63 | 75.65 | 78.59 | 0.954 | 60.4 |
| Run gốc | 74.78 | 78.46 | 0.932 | 59.7 |
| Đóng băng ViT/aligner | 75.65 | 78.22 | 0.932 | 60.6 |

So với run gốc: đúng thêm 1 cặp (McNemar 1 so với 0), margin AUC −0.23 (p = 0.65), log loss không đổi, Tie AUC +1.0 (p = 0.44). Không có khác biệt có ý nghĩa.

**Kết luận:**
1. Gradient norm lớn do ViT **không phải** nguyên nhân Stage 2 không cải thiện Bench: bỏ nhiễu ViT làm tối ưu sạch hơn (58 → 8) nhưng Bench và mức khớp Inspector không đổi.
2. LoRA ViT có đóng góp vào việc khớp nhãn người trên cặp đã train (−3.5 điểm accuracy khi bỏ), nhưng không thấy trên Bench.
3. Kết hợp với claim 2: chỉnh trọng số loss hay nhóm tham số được train đều chưa giúp Stage 2 vượt Student-Stage1 trên Bench. Một seed, 115 cặp không Tie.

### 7.6. Mẫu khó: so sánh từng mẫu giữa Teacher, Student và Inspector, sau training (03/11/2026)

Từ các file đánh giá đã có, không cần GPU, trên 12 run Stage 2 ở checkpoint 126: 2 run claim 1b, 9 run claim 2, run đóng băng ViT/aligner. Teacher của mọi run là Student-Stage1 đóng băng, nên dự đoán Teacher lấy từ file đánh giá step 0 (eval mode, không dropout). Hình, danh sách mẫu khó (`hard_samples.json`) và script: `evaluation_results/train_loop/claim1_claim2_hard_samples/`.

**Lưu ý:** đây là phân tích **sau training**, chỉ so trạng thái cuối (checkpoint 126) với Teacher ở eval mode, trên 256 mẫu đánh giá. Đây **chưa phải** thống kê trong lúc training (nhãn Teacher có dropout thật sự dùng cho loss, Student thay đổi theo từng step, trên đủ 1008 Unlabeled). Thống kê đó cần log từng mẫu (mục 6.1 việc 10).

Tỉ lệ argmax ba lớp của Student ở checkpoint 126 giống Teacher (eval mode):

| Run | 256 Unlabeled đã train | 256 cặp có nhãn |
|---|---|---|
| Claim 1b: khởi tạo mới | 96.1% | 94.1% |
| Khởi tạo từ Inspector, mọi λ = 1 | 95.3% | 92.6% |
| λ_U = 0 | 91.0% | 90.6% |
| λ_U = 0.5 | 93.0% | 94.5% |
| λ_U = 2 | 96.1% | 93.8% |
| λ_C = 0 | 90.6% | 93.8% |
| λ_C = 2 | 95.3% | 94.5% |
| λ_M = 0.5 | 94.9% | 93.8% |
| λ_M = 2 | 93.4% | 94.1% |
| λ_G = 0.5 | 92.2% | 96.1% |
| λ_G = 2 | 94.1% | 93.8% |
| Đóng băng ViT/aligner | 95.3% | 96.9% |

**256 Unlabeled đã train** (Inspector: P lớn nhất trung bình 0.56, không bao giờ chọn Tie):
- Teacher khác argmax Inspector ở 18.4% mẫu (khởi tạo từ Inspector), 14.1% (khởi tạo mới).
- Student đổi argmax so với Teacher ở 4–9% mẫu; mỗi run sửa được 2–5% mẫu Teacher khác Inspector và làm sai thêm 1–4%, nên Student vẫn khác Inspector 14–20%.
- Với mọi λ = 1, khoảng một phần ba mẫu ở checkpoint 126 lại xa Inspector hơn Teacher (λ_U = 0: 61%, λ_U = 2: 24%).
- **Unlabeled khó** (Student khác argmax Inspector ở cả 12 run): 19 mẫu (**7.4%**); 71.9% mẫu không bao giờ khác. Mẫu khó gần như cố định giữa các run, và chỉ nằm ở nơi Inspector không chắc: tỉ lệ khó 17.9% khi P lớn nhất của Inspector < 0.45, 4.7% khi 0.45–0.55, 0% khi ≥ 0.55. "Khác Inspector" ở đây chủ yếu là khác một nhãn gần như ngẫu nhiên.

**256 cặp có nhãn** (106 A, 100 B, 50 Tie):
- So với Teacher, Student sửa đúng 1–5.5% cặp và làm sai thêm 0–1.2% cặp.
- **Cặp khó** (sai ở cả 14 model: 2 Teacher + 12 Student): 65 cặp (**25.4%**), gồm 45 / 50 cặp Tie (90%) và 20 / 206 cặp không Tie (9.7%).
- Hai Teacher sai 100% cặp Tie: P(Tie) không bao giờ là xác suất lớn nhất. Đây là giới hạn của dự đoán ba lớp, không phải do từng mẫu khó.
- 20 cặp không Tie khó đều bị đảo chiều (11 người A → model B, 9 người B → model A); ở các cặp này loss preference Gemma của Teacher cũng cao (0.625 so với 0.407).

Đóng băng ViT/aligner không đổi tỉ lệ hay danh sách mẫu khó. Thống kê trong lúc train, trên đủ 1008 Unlabeled và theo từng step, lấy từ log từng mẫu (mục 6.1 việc 10, 11; bật mặc định từ 03/11) cho các run mới.

### 7.7. Các run Stage 2 tuning sai batch (đã xoá output training)

Mọi run Stage 2 tuning trước 02/11 lệch batch 16 (24, 8, 12 hoặc 36 cặp mỗi step, 2,000 Unlabeled, khoảng 100 step), dùng Student-Stage1 batch 8 và EMA Teacher cập nhật sau mỗi optimizer step. Kết quả đánh giá (`evaluation_results/train_loop/stage2_tuning_*`) được giữ để tham khảo. Các nhận xét dưới đây chỉ là **bằng chứng định tính về cơ chế**, không dùng để chọn giá trị:

1. **Stage 2 khớp được nhãn Inspector trên Unlabeled và phần khớp tổng quát hoá.** Chỉ train Unlabeled (4 lượt × 256 mẫu): KL(Inspector ‖ Student) trên mẫu đã train 0.093 → 0.007, trên Unlabeled chưa train 0.100 → 0.017. Nhãn Inspector rất mềm (P lớn nhất trung bình 0.56, entropy 0.90).
2. **Khớp Inspector làm lệch nhãn Gemma, và mức lệch đi theo mức khớp, không theo learning rate.** Relevance Head lệch nặng nhất. Giảm learning rate chỉ làm chậm cả hai.
3. **λ_U tạo đánh đổi đơn điệu:** λ_U càng lớn, Student càng khớp Inspector và càng lệch nhãn Gemma.
4. **KL(Teacher ‖ Student) giữ Student gần Student-Stage1.** Teacher EMA gần như không đổi trong khoảng 100 step. Bỏ KL làm Student khớp Inspector hơn nhưng lệch Gemma gấp đôi.
5. **Giảm learning rate backbone (2e-6, head giữ 1e-5) không giảm lệch Gemma.**
6. **Trong khoảng 100 step, Stage 2 chủ yếu đổi độ tự tin, ít đổi argmax.** Chỉ số chính phải là đại lượng liên tục theo từng cặp với khoảng tin cậy ghép cặp.
7. **Mọi step đều bị clip** (norm trước clip 30–130 so với 1.0, chủ yếu do ViT). Độ lớn mỗi bước cập nhật vì vậy chủ yếu do learning rate quyết định.
8. **Cosine giữa gradient có nhãn và Unlabeled khoảng 0:** hai nguồn gần như vuông góc, không kéo ngược nhau.

### 7.8. Bài học về cách đánh giá

| Vấn đề | Khắc phục |
|---|---|
| 32 cặp Gemma theo dõi trong lúc train quá ít; đường forward lúc train khác offline | Chỉ dùng làm báo động sớm; so sánh dựa trên đánh giá offline |
| Đánh giá mọi checkpoint trên mọi tập quá tốn thời gian | Chọn checkpoint từ hình loss và norm gradient; chỉ đánh giá dữ liệu train Stage 2 (mục 5) |
| Accuracy không nhạy | Chỉ số liên tục với khoảng tin cậy bootstrap ghép cặp |
| Loss từng step bị chi phối bởi dữ liệu của step | Trung bình trượt |
| Phân tích chưa có hình | Script vẽ hình chạy trong job đánh giá |
| Bench 115 cặp quá nhỏ | Chỉ dùng ở bước cuối; kiểm định ghép cặp |
| Gradient norm tổng không cho biết loss nào gây ra, có giằng co hay không | Script chẩn đoán gradient từng loss thành phần (việc 11 mục 6.1) |

## 8. Hướng tiếp theo (cập nhật 04/11/2026)

Đã làm từ bản trước: phiên bản đóng băng ViT/aligner (mục 7.5) và phân tích claim 2 (mục 7.4). Cả hai cho cùng kết luận: Stage 2 trong setting này chưa vượt Student-Stage1 trên Bench, và Bench (115 cặp không Tie) không phân biệt được chênh lệch 1–2 điểm.

- **Log từng mẫu trong lúc train** (mục 6.1 việc 10, 11): đã bật mặc định cho mọi run mới (mục 3). Job thử 2 step 9658 kiểm tra trên GPU: đủ 16 dòng mỗi step (8 Unlabeled + 8 cặp có nhãn), `teacher_kl` khớp KL Teacher trong log train, có đủ trường nhãn và cờ giống / khác.
  - Kết quả 9658 (04/11): đạt cả ba kiểm tra. Số liệu 2 step chỉ để kiểm tra code, chưa dùng kết luận: Student = Teacher 16/16, Student = Inspector 15/16; KL Teacher trung bình 0.007 / 0.015, KL Inspector 0.11 / 0.16 trong khi CE 0.96 / 1.10 (phần lớn CE là entropy nhãn mềm Inspector 0.85 / 0.95); gradient Unlabeled chỉ bằng 19% / 10% gradient cặp có nhãn; clip 1.0 với tổng gradient norm 64 / 175 (ViT chiếm phần lớn). Kết quả: `evaluation_results/train_loop/<phiên bản test_2_steps>/per_sample/`.
- **Tie:** Teacher và Student gần như không bao giờ chọn Tie làm argmax; 45 / 50 cặp Tie trên dữ liệu train sai ở mọi model (mục 7.6). Cần xem cách quyết định Tie (ngưỡng P(Tie)) trước khi so sánh accuracy ba lớp.
- **Unlabeled mà Inspector không chắc** (P lớn nhất < 0.45, khoảng 37% mẫu) là nơi Student khác Inspector ở mọi run (mục 7.6). Có thể thử giảm trọng số hoặc bỏ các mẫu này trong L_U (mỗi run một thay đổi).
- **Giằng co CE Inspector và KL Teacher** trong L_U (mục 7.3): λ_C = 0 làm Student gần Inspector hơn nhưng không đổi Bench (mục 7.4); thiết kế L_U cần xem lại cùng hướng trên.
- Ưu tiên thấp hơn sau mục 7.5: clip riêng theo nhóm tham số, giảm learning rate ViT, tìm hiểu vì sao ViT nhạy với một số ảnh. Bỏ nhiễu ViT không đổi kết quả Bench.

### 8.1. Hướng mới (04/11/2026): mẫu đồng thuận Teacher – Inspector + online augmentation cho Student

**Nhận định (anh, 04/11):** Student Stage 2 gần như không học thêm: loss không giảm, Bench / accuracy / AUC không khác Student-Stage1. Khi Teacher và Inspector ra cùng nhãn (và gần cùng phân phối) với Student, gần như không có gradient để học. **Mục tiêu:** bắt Student học input khó; loss ở iteration đầu có thể rất cao và phải giảm dần qua các iteration, tránh loss đi ngang.

Lưu ý khi đọc loss (job 9658): CE theo Inspector có mức sàn bằng entropy nhãn mềm Inspector; theo dõi KL(Inspector ‖ Student) = CE − entropy thay vì CE thô. Mỗi Unlabeled thấy một lần nên loss mỗi step tính trên mẫu mới, giống loss kiểm tra; muốn thấy loss giảm cần đi qua cùng tập nhiều lần.

**Bước 1. Chọn 1008 Unlabeled đồng thuận Teacher – Inspector (xong, job 9667).**
- Độ đồng thuận = CE(Inspector, Teacher) (L_U với Student thay bằng Teacher); Teacher = `teacher_initialized.pt` của Student-Stage1 khởi tạo từ Inspector checkpoint-63, eval mode, tắt dropout, gộp `selected_equal`.
- Duyệt theo Inspector chắc chắn trước (entropy tăng dần, là mức sàn của CE); nhận khi nhãn argmax Teacher = Inspector; dừng khi đủ 1008 (không chấm hết 10k).
- Kết quả: duyệt 1009, nhận 1008 (1 mẫu loại: Inspector A, Teacher B). P lớn nhất Inspector ≥ 0.853 (trung bình 0.914). CE trung bình 0.386, KL trung bình 0.050. Nhãn A 490, B 518, **không có Tie**: Unlabeled không còn dạy lớp Tie.
- 1008 cặp chỉ có 640 instruction / 644 ảnh nguồn (432 ảnh nguồn góp 1 cặp, tối đa 9 cặp): tập 10k có 3,074 ảnh nguồn, mỗi ảnh nguồn nhiều cặp A/B từ các model sửa ảnh khác nhau, bước 1 không giới hạn số cặp mỗi ảnh nguồn.
- File: `output_training/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint63/` (`agreement_scored.jsonl`, `selected_unlabeled_1008.json`, `selection_summary.json`); hình `evaluation_results/train_loop/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint63/teacher_inspector_agreement.png`.

**Bước 2. Online augmentation cho Student (code xong, chưa nối vào train_loop).** Trong cùng step, Teacher và Inspector nhận mẫu gốc; Student nhận mẫu đã biến đổi. Chỉ Unlabeled; cặp có nhãn giữ nguyên.
- Model đọc 3 chuỗi cho mỗi cặp: (ảnh nguồn + instruction + A) → h_A, (ảnh nguồn + instruction + B) → h_B, (ảnh nguồn + instruction) → h_src. Score / Tie head dùng h_A, h_B; Relevance head dùng h_src (chọn metric chỉ theo ảnh nguồn + instruction, chung cho A và B). Vì reranker chỉ đọc một cặp (truy vấn, tài liệu) mỗi lần nên input 4 thành phần được mã hoá thành 3 chuỗi.
- Kịch bản (mỗi mẫu, mỗi iteration bốc 1 kịch bản chính đều nhau trong các kịch bản được phép, cộng đổi chỗ A ↔ B xác suất 0.5):

| Kịch bản | Cách làm | Mẫu dùng được |
|---|---|---|
| masking | xoá "the / a / an", câu mở đầu "please, help me, I need you to, can you", cụm "in / from / to the / this image / picture / photo"; không xoá phần trong ngoặc, màu, số, vị trí, danh từ, động từ; còn ≥ 3 từ | 960 |
| injection | 10 câu mở đầu, 8 câu kết thúc trung tính (không thêm yêu cầu), dạng đầu / cuối / cả hai | 1008 |
| ghép ảnh | ảnh lạ (ảnh nguồn của 2,430 Unlabeled khác ảnh nguồn của mẫu đã chọn) ghép trái / phải / trên / dưới ảnh nguồn; giữ tỉ lệ, thu về **cùng số pixel ảnh nguồn** (bộ nhớ không tăng); thay ảnh nguồn ở cả 3 chuỗi, A và B giữ nguyên; instruction thêm vị trí ("In the left image, …", "… (edit only the top image)", "Only edit the right image: …") | 1008 |
| lật ngang | lật ảnh nguồn, A, B; đổi left ↔ right trong instruction; loại instruction có chữ / số / ngoặc / ký tự ngoài ASCII / biển hiệu (285) và chiều xoay / mirror (7) | 716 |
| đổi chỗ A ↔ B | Student nhận (B, A), mục tiêu (P_B, P_A, P_Tie) | 1008 |

- Vì sao ghép ảnh chỉ vào ảnh nguồn là hợp lý: A và B chịu cùng ảnh nguồn ghép nên so sánh A hơn B vẫn đúng; instruction chỉ rõ nửa ảnh cần sửa, Student phải học bỏ qua nửa ảnh lạ.
- Đọc toàn bộ 640 instruction, sửa 5 lỗi quy tắc: (1) "Make this **a** market", "Make it **a** …": không xoá mạo từ sau tân ngữ; (2) "the skies found in this image" → không xoá cụm ảnh sau found / visible / shown / seen / present / depicted; (3) "Can you …?" bỏ "can you" thì "?" thành "."; (4) viết hoa đầu câu sau khi xoá "The"; (5) gộp ". ." và thêm dấu chấm sau dấu nháy đóng ở injection. Các lỗi này đã thành kiểm tra tự động.
- **Online:** biến thể instruction tạo lúc train theo seed (seed, iteration, mẫu); qua 5 iteration mô phỏng, 461 / 1008 mẫu có 5 instruction khác nhau, 380 có 4; chỉ 2 / 5,040 lần dùng instruction gốc do không đạt kiểm tra. Ảnh ghép / lật tạo lúc train (vài ms), ghi thư mục tạm.
- Hạn chế: ảnh có chữ ngẫu nhiên (biển hiệu) mà instruction không nói tới vẫn bị lật; A và B bị lật như nhau nên so sánh vẫn đúng. Quy tắc chỉ cho tiếng Anh; lỗi nghĩa chỉ tìm được khi đọc, nên mỗi run đọc lại khoảng 50 dòng log instruction thật.
- File: `scripts/student_augmentation.py`, `scripts/build_augmentation_scenarios.py`; `augmentation_scenarios_1008.json`, `augmentation_scenarios_preview.md` (cùng thư mục bước 1).

**Bước 3. Training Stage 2 với online augmentation (chuẩn bị).**
- Trước khi cải tiến tiếp: kiểm chứng augmentation có làm input khó cho Student như mục tiêu (loss / KL Student so với Teacher và Inspector ở các step đầu cao hơn run không augmentation, theo từng kịch bản).
- **Luôn bật monitor** khi train; kiểm tra log từng mẫu ngay ở các step đầu. Log từng mẫu ghi thêm: kịch bản, đổi chỗ A ↔ B, instruction thật đưa vào Student, vị trí ghép, ảnh lạ, lý do dùng instruction gốc.
- Thiết kế nhiều iteration (chưa làm): Teacher cập nhật EMA **sau mỗi iteration** (không theo step), cả head và backbone, hệ số cosine 0.99 → 0.9999 qua các iteration (với 3 iteration: 0.99, 0.99495, 0.9999 → Teacher chỉ dịch 1% rồi 0.5% về phía Student); sau mỗi iteration đo độ bất định Teacher – Student, giữ mẫu chắc chắn cho iteration sau; augmentation mới mỗi iteration. Số iteration không cố định, dừng theo K như bài Co-rewarding, tối đa 4–5 iteration với 1008 mẫu (đọc kỹ bài trước khi thiết kế). Setting này **không** cập nhật Inspector.

**Job thử 9895 (04/11, 2 step, 16 mẫu đầu của tập con = CE thấp nhất): đạt.** Log từng mẫu đủ trường augmentation, `teacher_kl` khớp log train, ảnh tạm được xoá, đổi ngược A ↔ B đúng (Student sau đổi ngược khớp Teacher). Bộ nhớ đỉnh 26.1 GiB, khoảng 83 giây / step. Dấu hiệu sớm (chỉ 16 mẫu dễ nhất, chưa kết luận): Student = Teacher = Inspector 16 / 16 ở mọi kịch bản; KL Teacher trung bình 0.00055 / 0.00024; gradient Unlabeled chỉ 1.0 / 0.47 so với 60 / 94 của cặp có nhãn; CE 0.07 / 0.06 gần bằng entropy Inspector 0.064 / 0.059. Augmentation hiện tại gần như không làm đổi dự đoán của Student trên các mẫu này.

**Run đầy đủ: job 9915** (`…_teacher_inspector_agreement_subset_student_online_augmentation`, 1008 mẫu, 126 step, normal, 5 giờ), submit 04/11 22:26.

**Run đối chứng (ghi nhận 04/11, chạy sau):** tập con đồng thuận, **không** augmentation, để tách tác động của augmentation khỏi tác động của tập con lên loss và kết quả cuối. Hiện tập trung vào phiên bản có augmentation; ở các step đầu Teacher gần như trùng Student-Stage1, nên tỉ lệ Student khác nhãn Teacher phần lớn phản ánh trực tiếp augmentation.

**Phân tích bắt buộc cho mỗi run (theo từng step):** loss từng thành phần (HumanGT, GemmaGT, mask, CE Inspector, entropy Inspector, KL Inspector = CE − entropy, KL Teacher), gradient norm trước clip theo nhóm (ViT, aligner, LLM, heads) và theo nguồn (có nhãn / Unlabeled), hệ số clip, learning rate; tỉ lệ nhãn Student = Teacher, Student = Inspector, Teacher = Inspector theo step và theo kịch bản augmentation, theo đổi chỗ A ↔ B, theo mức tự tin Inspector; kiểu khác nhãn (Teacher X → Student Y); đọc mẫu instruction thật Student nhận.

**Cho các setting sau:**
- Giới hạn 1 (hoặc 2) cặp mỗi ảnh nguồn để đa dạng instruction; đổi lại phải nhận mẫu Inspector kém chắc chắn hơn; chạy lại bước 1 (khoảng 30 phút).
- EMA cho Inspector (chỉ backbone): Inspector LoRA hạng 32 chỉ trên LLM, Student hạng 64 trên LLM + ViT; Student-Stage1 đã chép LoRA Inspector vào 32 hạng đầu (alpha / hạng đều bằng 2), nên biểu diễn Inspector thành hạng 64 (32 hạng sau = 0) là chính xác và lấy EMA được; cần chấm lại nhãn Inspector sau mỗi lần cập nhật; quyết định có lấy EMA phần ViT không (ở thử nghiệm Inspector, ViT không có lợi, p = 0.711).
- Paraphrase instruction bằng Gemma (tạo sẵn bằng job), instruction ngôn ngữ khác; nhiễu ở mức đặc trưng; 2 kịch bản / mẫu (kiểu RandAugment) để tăng độ khó.

### 8.2. Tổng kết các đợt augmentation đã chạy (04–05/11/2026)

Số liệu là tỉ lệ Student ≠ Teacher (nhãn one-hot overall) trên Unlabeled trong lúc train, loop 1. Teacher và Inspector luôn nhận mẫu gốc. Một số job chỉ đo đến một phần quá trình train, nên các số chỉ dùng để so sánh tương đối.

| Nhóm | Strategy | Job (setting) | Student ≠ Teacher |
|---|---|---|---|
| Giữ nhãn | Online (một phép mỗi mẫu: ghép ảnh vào source, lật, thêm câu, bỏ từ; đổi chỗ A ↔ B) | 9915 (1008, đồng thuận cao, huỷ) | 0% |
| | Combined strong (phép văn bản + bố cục + giảm chất lượng ảnh + đổi chỗ A ↔ B) | 9968 (1008, đồng thuận cao, huỷ) / 9992 (512) | ~2% / ~3% |
| | Combined strong + instruction viết tay | 9980 (1008, đồng thuận cao) | ~2.6% |
| | Combined strong, tập đồng thuận vừa phải | 9997 (1008, Inspector 0.6–0.8) / 10120 (512) / 10116 (1008, viết tay) | ~11% / ~7% / ~7% |
| | Chỉ instruction / chỉ ảnh source | 10248 / 10249 (512) | ~2% / ~2.5% |
| | Không augmentation | 9991 (512 mẫu ngẫu nhiên) | ~5% |
| Phá thông tin | Extreme (bỏ 90% từ, đen 80–100% source) | 10250 (512) / 10351 (1008) / 10349, 10350 (nhiều iteration) | ~29–33% |
| | Blank (instruction rỗng, source đen hết) | 10295 (512) | ~36% |

Theo từng phép trong 9980: làm mờ 4.5%, nhiễu 4.0%, viết tay tối giản 4.4%, thu nhỏ 3.2%, lỗi chính tả 2.8%, lật 2.7%, ghép ảnh 2.5%, JPEG 2.5%, chèn instruction lạ 2.3%. Đo trên Teacher ngoài lúc train (job 9965), ở mức nặng nhất: làm mờ và thu nhỏ 2.8%, JPEG và nhiễu 0.9%.

**Nhận xét:**
- Augmentation giữ nhãn gần như không làm Student khác Teacher (dưới 3% trên tập đồng thuận cao), và tỉ lệ này còn giảm dần trong lúc train.
- Độ khó của tập mẫu ảnh hưởng mạnh hơn độ mạnh của augmentation: tập đồng thuận vừa phải cho khoảng 7–11%.
- Khi bỏ hết instruction và source, Student vẫn trùng Teacher khoảng 64%. Model chủ yếu so trực tiếp A với B.
- Phép `drop_action` cũ từng bỏ cả động từ chỉnh sửa (đã sửa). Số liệu "bỏ từ" trước khi sửa có phần do lỗi này.

**Các phép bị loại (anh, 05/11):**
- Instruction viết tay: do tôi viết offline, không làm online được và không mở rộng được.
- Extreme và Blank: mất quá nhiều thông tin của source và instruction, Student chỉ học thuộc A/B.
- JPEG và thu nhỏ: yếu; thu nhỏ có tác dụng gần giống làm mờ.
- Đổi chỗ A ↔ B: gần như không có tác dụng.

### 8.3. Hướng mới (05/11/2026): mỗi experiment một phép augmentation

**Mục tiêu:** tách riêng tác động của từng phép, rồi chọn 4–5 phép mạnh nhất để kết hợp. Bộ phép được chọn phải đạt hai điều kiện:
- **Theo nhãn one-hot overall:** Student = Teacher khoảng **75–80%** (anh chỉnh 06/11; bản 05/11 ghi 70–75%).
- **Theo soft label:** Teacher tự tin vào nhãn của nó **cao hơn nhiều** so với Student (đã augmentation) trên cùng nhãn đó. Kể cả khi hai model cùng nhãn, Student kém tự tin hơn nhiều thì loss vẫn tăng, và như vậy vẫn đúng mục tiêu.

Hai điều kiện được đo ở mức overall và ở từng metric trong 6 metric.

**Setting chung:**
- Setting 1008: Student-Stage1 khởi tạo từ Inspector checkpoint-63; Inspector checkpoint-63 cùng cache của nó.
- Tập Unlabeled mới gồm 1008 mẫu:
  - Teacher (eval mode) cùng nhãn với Inspector;
  - xác suất lớn nhất của Inspector trong **[0.5, 0.7)**;
  - duyệt theo **thứ tự ngẫu nhiên** (seed 42) để phủ đều cả khoảng;
  - mỗi instruction chỉ xuất hiện một lần.

  Trong cache Inspector, khoảng này có 2750 mẫu với 1476 instruction khác nhau. Thư mục: `output_training/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/` (job 10450). File kịch bản augmentation dựng bằng job 10451, dùng lại dạng combined.
- all_once một loop, 126 step, batch 8 cặp có nhãn + 8 Unlabeled; mọi λ bằng 1; partition normal; giới hạn 5 giờ (job 10351 cùng cấu hình chạy 3 giờ 03 phút).
- Không đổi chỗ A ↔ B ở mọi job. Teacher và Inspector luôn nhận mẫu gốc.

**12 phiên bản** (`config_training.sh`, mỗi phiên bản kèm một bản `_test_2_steps`):

| Phiên bản (`stage2_tuning_unlabeled1008_…`) | Phép | Áp lên | Tham số |
|---|---|---|---|
| `baseline` | không augmentation | — | — |
| `augmentation_composite` | ghép cùng một ảnh lạ, cùng một phía (trái / phải / trên / dưới, chọn đều) vào source, A và B, để ba ảnh giữ cùng bố cục; mỗi ảnh ghép thu về số pixel của ảnh gốc tương ứng; instruction do code tạo từ 4 template nói rõ ảnh chính và ảnh phải bỏ qua | source, A, B, instruction | ảnh lạ là source của mẫu Unlabeled ngoài tập |
| `augmentation_blur` | làm mờ Gaussian, cùng tham số trên ba ảnh | source, A, B | bán kính 2–3 (mức nhẹ 1–1.5 cho instruction về chữ); không dùng cho instruction về chất lượng hoặc độ nét |
| `augmentation_noise` | nhiễu Gaussian, cùng tham số trên ba ảnh | source, A, B | sigma 15–25 (mức nhẹ 6–12); không dùng cho instruction về chất lượng |
| `augmentation_mask_content` | thay từ nội dung bằng `___`, giữ từ đầu và phần trong ngoặc kép | instruction | **20–40%** số từ |
| `augmentation_black_rectangles` | 1–3 hình chữ nhật đen | source | tổng diện tích **10–20%** |
| `augmentation_typo` | viết sai chính tả từ dài từ 4 chữ cái, giữ từ chỉ vị trí và phần trong ngoặc kép | instruction | **20–40%** số từ |
| `augmentation_flip` | lật ngang ba ảnh; đổi left ↔ right trong instruction | source, A, B, instruction | không dùng cho instruction về chữ, số, xoay, lật |
| `augmentation_drop_action` | bỏ cụm mở đầu ("Make this look like…"), không bỏ động từ chỉnh sửa | instruction | chỉ khoảng 14% instruction đủ điều kiện |
| `augmentation_distractor_instruction` | chèn instruction của mẫu khác, ghi rõ là không thực hiện | instruction | — |
| `augmentation_rotation` | xoay, góc trống tô đen | source | 5–15°, dấu ngẫu nhiên |
| `augmentation_injection` | thêm câu mở đầu hoặc kết thúc trung tính, không thêm yêu cầu | instruction | — |

**Các nguyên tắc:**
- Mỗi biến đổi do `student_augmentation.py` tạo online theo seed (seed, iteration, mẫu); không viết hay sửa tay.
- Mẫu không đủ điều kiện cho phép thì giữ input gốc và được ghi `applied = False` kèm lý do.
- Mức độ thực tế của từng mẫu được ghi lại: tỉ lệ từ, bán kính, sigma, góc, diện tích, vị trí ghép.
- Mỗi job lưu 25 mẫu ví dụ trong `augmentation_tmp/`.

**Log chi tiết** (mọi job, kể cả baseline; mỗi mẫu Unlabeled ở mỗi step):
- **Overall:** xác suất 3 lớp và nhãn của Teacher, Student, Inspector; Student = Teacher; CE với Inspector; `teacher_kl` (thành phần loss); độ tự tin của Teacher vào nhãn của nó; độ tự tin của Student vào cùng nhãn đó; khoảng chênh; KL(Teacher‖Student) trên phân phối overall.
- **Từng metric trong 6 metric:** xác suất và nhãn của Teacher và Student, có trùng không, hai độ tự tin và khoảng chênh, KL, CE, relevance của hai model, metric có được chọn không.
- **Loss theo step** (dòng `diagnostics`): HumanGT, Gemma preference, Gemma relevance, CE Inspector, entropy Inspector, KL Inspector, KL Teacher, gradient theo nhóm, hệ số clip, learning rate.

**Phân tích (anh xác nhận 05/11).** Kết quả nằm **ngay trong thư mục output training của từng phiên bản**, ở `output_training/<phiên bản>/analysis/`, để dễ kiểm soát. Có thể xoá checkpoint mà vẫn giữ phần phân tích. Bench và đánh giá checkpoint vẫn ở `evaluation_results/`.

- **Script:** `RewardModelEvaluation/OurModel/analyze_single_operation_augmentation.py`. Script chạy một lần cho cả nhóm, chỉ đọc `training_log_<job>.jsonl` và `per_sample_log_<job>.jsonl` trong thư mục run (không cần model, không đọc ảnh). Phiên bản nào thiếu log thì bỏ qua. Tham số `--suffix _test_2_steps` dùng cho các job test.
- **Mỗi phiên bản augmentation**, trong `analysis/`:
  - `step_losses.csv`: mỗi step một dòng: HumanGT CE, Gemma preference, Gemma relevance, Unlabeled loss và các thành phần của nó (CE Inspector, entropy Inspector, KL Inspector, KL Teacher), loss tổng, learning rate, hệ số clip. Thêm 06/11: `unlabeled_floor_inspector_entropy` (mức sàn = entropy nhãn mềm Inspector, không tạo gradient), `unlabeled_learnable` (= Unlabeled − mức sàn = KL Inspector + λ_C·KL Teacher) và `total_learnable`. HumanGT, Gemma preference và Gemma relevance dùng nhãn one-hot / 0-1 (Majority-vote) nên mức sàn bằng 0, cả giá trị là phần học được.
  - `samples_overall.csv`: mỗi mẫu Unlabeled ở mỗi step một dòng: nhãn overall và xác suất của Teacher, Student, Inspector; có trùng nhãn không; độ tự tin, khoảng chênh, KL, CE; phép có được áp dụng không, lý do nếu không, mức độ thực tế; instruction gốc và instruction Student nhận.
  - `samples_metrics.csv`: mỗi mẫu × mỗi step × 6 metric một dòng: nhãn và xác suất của Teacher và Student, có trùng không, độ tự tin, khoảng chênh, KL, CE, relevance, metric có được chọn không.
  - `loss_vs_baseline.png`: 6 ô loss theo step (HumanGT, Gemma preference, Gemma relevance, CE Inspector, KL Teacher, loss tổng). Mỗi ô có hai đường, phép (xanh) và baseline (cam), gồm giá trị gốc và trung bình trượt 8 step.
  - `metrics_vs_baseline.png`: overall và 6 metric, mỗi nhóm so phép với baseline về: Student = Teacher (có vùng mục tiêu 75–80%), độ tự tin của Teacher vào nhãn của nó, độ tự tin của Student vào nhãn Teacher, khoảng chênh, KL(Teacher‖Student).
  - `label_transitions.png`: ma trận Teacher → Student (A/B/Tie, chuẩn hoá theo hàng nhãn Teacher) cho overall và 6 metric. Ba hàng: phép, baseline, và hiệu (phép − baseline).
  - `summary.json`: toàn bộ số liệu trên, cộng thêm:
    - hiệu so với baseline, ở mức overall và theo từng metric;
    - chia theo mức độ của phép: tỉ lệ từ dưới / trên 30%; ba khoảng đều nhau theo bán kính làm mờ, sigma nhiễu, góc xoay, diện tích che; vị trí ghép;
    - tỉ lệ mẫu được áp dụng và lý do không áp dụng.
- **Thư mục baseline**, trong `analysis/`:
  - cùng các file CSV và `summary.json`; hình riêng `loss_per_step.png`, `metrics.png`, `label_transitions.png`;
  - **bảng xếp hạng chung** `ranking.csv` và `ranking.png`. Mỗi phép một dòng: tỉ lệ được áp dụng; Student = Teacher và hiệu so với baseline; khoảng chênh độ tự tin và hiệu; KL và hiệu; có nằm trong vùng 75–80% không; tỉ lệ trùng nhãn và khoảng chênh của từng metric. Bảng này dùng để chọn top 4–5 phép.
  - **bảng so sánh các phép** (thêm 06/11) trong `analysis/comparison/`:
    - `comparison_tables.md`: mọi bảng ở dạng đọc được. Hàng là overall và 6 metric; với các dạng loss có thêm hàng "loss (selected metrics)" (trung bình trên các metric Student chọn, đúng như lúc train). Cột là baseline và 11 phép. Ô ghi giá trị kèm hiệu so với baseline, hoặc số lần so với baseline với các dạng loss.
    - Mỗi chỉ số một file `<chỉ số>.csv`, gồm cột giá trị, hiệu và số lần so với baseline.
    - `difference_from_baseline_bootstrap.csv`: hiệu so với baseline trên cùng mẫu, khoảng tin cậy 95% (bootstrap ghép cặp, 1000 lần).
    - 9 file `heatmap_<loss>.png`.
  - **Các dạng loss / độ lệch** giữa Teacher và Student, tính trên 3 lớp A/B/Tie của từng metric:
    - KL(T‖S): loss đang dùng;
    - KL ngược (S‖T); Jensen–Shannon; CE mềm (= KL + entropy Teacher); CE với nhãn cứng của Teacher;
    - BCE và KL nhị phân, tách thành hai phần: phần A-vs-B (tỉ phần T_A/(T_A+T_B)) và phần Tie (P(Tie)).
  - **Các chỉ số khác:**
    - "Teacher chắc chắn, Student kém chắc chắn": bản 3 lớp, có Tie (Teacher ≥ 0.75, Student ≤ 0.50 trên nhãn Teacher) và bản A-vs-B (tỉ phần Teacher ≥ 0.75, Student ≤ 0.60). Các ngưỡng chỉ để minh hoạ, chưa thống nhất.
    - Tỉ lệ khoảng chênh ≥ 0.2.
    - Các chỉ số Tie: tỉ lệ Teacher / Student chọn Tie; Teacher chọn A/B nhưng Student chọn Tie và ngược lại; P(Tie) trung bình.
- Mọi tỉ lệ của một phép được tính trên **các mẫu thực sự được áp dụng phép**. Baseline dùng toàn bộ mẫu.
- **Về Tie:** KL, CE, tỉ lệ cùng nhãn, độ tự tin và khoảng chênh đều tính trên đủ 3 lớp, và trên cả các mẫu Teacher chọn Tie. Đã kiểm tra lại độc lập từ CSV và khớp log train. Bản đầu (06/11) có hai chỉ số bỏ qua Tie (BCE A-vs-B, và "Student gần ngẫu nhiên giữa A và B"). Hai chỉ số này đã được bổ sung phần Tie như trên.
- Đã chạy thử trên dữ liệu giả trước, rồi chạy trên 12 run thật (06/11, khoảng 1 phút).

**Triển khai (đã xong 06/11):**
- **Đổi tên:** các phiên bản ban đầu mang thêm `probability_0.5_0.7_`. Anh xác nhận bỏ phần này vì cả nhóm dùng chung một tập mẫu. Các job test tên cũ (10452–10463, chưa chạy) đã huỷ.
- **Job test:** chỉ giữ 5 job test 2 step, đủ phủ mọi đường code mới: baseline 10483, composite 10484, blur 10485, black_rectangles 10488, drop_action 10491. Cả 5 đều đạt; output test đã xoá.
- **12 job thật**, chờ `afterok` của 10451 và 5 job test:
  - truongtq: baseline 10496, blur 10497, mask_content 10498, typo 10499, drop_action 10500, rotation 10501;
  - danhnt (anh submit, `umask 0002`): composite 10503, noise 10504, black_rectangles 10505, flip 10506, distractor_instruction 10507, injection 10508.

  Cả 12 job chạy đủ 126 step, mỗi job khoảng 3 giờ 05 phút đến 3 giờ 16 phút, không lỗi. Mỗi job có 1008 dòng log, mỗi dòng đủ 6 metric, và 6 checkpoint.
- Không dùng tài khoản anhndt cho các experiment này. QoS `mig2-35` cho mỗi tài khoản chạy tối đa 5 job cùng lúc.
- **Tập mẫu (job 10450):**
  - duyệt 1610 mẫu, chấm điểm 1087, nhận 1008; Teacher cùng nhãn Inspector 92.7%;
  - nhãn A 516, B 492;
  - xác suất lớn nhất của Inspector: trung vị 0.585, phủ [0.50, 0.70);
  - xác suất lớn nhất của Teacher: trung bình 0.73.

**Thay đổi code** (bản gốc trong `scripts/backups/*.before_single_operation_augmentation.*`):
- `score_teacher_inspector_agreement.py`: thêm `--scan-order random`.
- `student_augmentation.py`: thêm `choose_single_augmentation` và `single_augmented_images`.
- `train_loop.py`: thêm `--student-augmentation single_operation`, `--augmentation-operation`, và các trường soft label và theo metric trong per-sample log.
- `run_train_loop.slurm`: thêm `REFINEMENT_AUGMENTATION_OPERATION`.
- `config_training.sh`: thêm 12 phiên bản.

Đã kiểm tra trên CPU: 11 phép lặp lại được theo seed; ảnh ghép đúng bố cục; `teacher_kl` bằng trung bình KL của các metric Student chọn.

### 8.4. Kết quả: mỗi experiment một phép (06/11/2026)

Số liệu đầy đủ nằm trong `output_training/stage2_tuning_unlabeled1008_baseline/analysis/comparison/comparison_tables.md`. Các số dưới đây tính trên các mẫu được áp dụng phép.

**Số mẫu được áp dụng** trên 1008:

| Phép | Số mẫu | Lý do không áp dụng |
|---|---|---|
| composite, black_rectangles, distractor, injection, rotation | 1008 | — |
| noise | 1006 | instruction về chất lượng |
| typo | 1002 | không có từ dài từ 4 chữ cái, hoặc kiểm tra tự động thấy từ chỉ vị trí bị đổi |
| blur | 1000 | instruction về độ nét hoặc chất lượng |
| mask_content | 963 | ít hơn 2 từ nội dung có thể che |
| flip | 843 | 165 mẫu bị loại vì instruction có chữ, số, ngoặc hoặc nói về xoay / lật |
| drop_action | **128** | chỉ áp dụng được khi câu mở đầu bằng cụm yêu cầu |

Quy tắc loại của `flip` hơi chặt:
- 29 mẫu bị loại sai vì dấu nháy trong sở hữu cách (person's);
- một phần trong 9 mẫu có chữ số bị loại sai, ví dụ "2D".

`blur` cũng loại cả câu chỉ tình cờ có từ "clear" hoặc "detail". Chưa sửa các quy tắc này.

**Teacher** chắc chắn như nhau ở cả 12 run (overall 0.733–0.735), vì luôn nhận mẫu gốc. Mọi khác biệt giữa các run đều đến từ Student.

**Student = Teacher (one-hot)**

| | baseline | composite | blur | noise | mask_content | flip | black_rect. | typo | drop_action | distractor | injection | rotation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| overall | 0.988 | **0.888** | **0.900** | 0.921 | 0.920 | 0.954 | 0.966 | 0.965 | 0.961 | 0.971 | 0.977 | 0.978 |
| aesthetic | 0.976 | 0.878 | 0.885 | 0.913 | 0.902 | 0.940 | 0.952 | 0.957 | 0.945 | 0.956 | 0.957 | 0.966 |
| fidelity | 0.971 | 0.869 | 0.885 | 0.918 | 0.906 | 0.947 | 0.948 | 0.954 | 0.961 | 0.956 | 0.957 | 0.967 |
| realism | 0.988 | 0.891 | 0.898 | 0.925 | 0.925 | 0.963 | 0.967 | 0.963 | 0.945 | 0.977 | 0.978 | 0.980 |
| background | 0.849 | **0.616** | 0.653 | 0.695 | 0.792 | 0.792 | 0.807 | 0.825 | 0.781 | 0.827 | 0.842 | 0.820 |
| foreground | 0.903 | 0.717 | 0.718 | 0.762 | 0.837 | 0.829 | 0.869 | 0.883 | 0.875 | 0.888 | 0.897 | 0.879 |
| structure | 0.937 | 0.843 | 0.846 | 0.868 | 0.892 | 0.911 | 0.917 | 0.920 | 0.898 | 0.926 | 0.928 | 0.918 |

**Độ tự tin của Student vào nhãn Teacher / khoảng chênh (Teacher − Student)**

Tính trên 3 lớp, có Tie. Ba phép yếu nhất (distractor, injection, rotation) có khoảng chênh overall 0.052–0.059, gần mức baseline.

| | baseline | composite | blur | noise | mask_content | flip | black_rect. | typo | Teacher |
|---|---|---|---|---|---|---|---|---|---|
| overall | 0.685 / 0.049 | 0.599 / **0.134** | 0.612 / 0.122 | 0.634 / 0.100 | 0.636 / 0.097 | 0.668 / 0.067 | 0.664 / 0.070 | 0.673 / 0.060 | 0.734 |
| aesthetic | 0.829 / 0.039 | 0.738 / 0.130 | 0.750 / 0.118 | 0.775 / 0.094 | 0.764 / 0.105 | 0.808 / 0.063 | 0.806 / 0.063 | 0.811 / 0.056 | 0.868 |
| fidelity | 0.785 / 0.053 | 0.690 / 0.147 | 0.704 / 0.134 | 0.730 / 0.108 | 0.720 / 0.118 | 0.765 / 0.077 | 0.759 / 0.079 | 0.767 / 0.069 | 0.839 |
| realism | 0.826 / 0.041 | 0.725 / 0.142 | 0.745 / 0.123 | 0.767 / 0.100 | 0.764 / 0.104 | 0.804 / 0.066 | 0.802 / 0.066 | 0.813 / 0.055 | 0.868 |
| background | 0.529 / 0.069 | 0.435 / **0.161** | 0.458 / 0.140 | 0.471 / 0.127 | 0.501 / 0.095 | 0.510 / 0.086 | 0.511 / 0.087 | 0.522 / 0.075 | 0.598 |
| foreground | 0.540 / 0.067 | 0.457 / 0.149 | 0.467 / 0.140 | 0.484 / 0.123 | 0.507 / 0.099 | 0.522 / 0.084 | 0.521 / 0.086 | 0.534 / 0.073 | 0.607 |
| structure | 0.527 / 0.046 | 0.477 / 0.096 | 0.467 / 0.107 | 0.483 / 0.091 | 0.503 / 0.070 | 0.516 / 0.056 | 0.514 / 0.059 | 0.519 / 0.056 | 0.573 |

**KL(Teacher‖Student) theo metric, giá trị (số lần so với baseline)**

Tính trên 3 lớp, có Tie. Hàng đầu là loss đúng như lúc train: trung bình trên các metric Student chọn.

Cách `train_loop.py` tính thành phần Unlabeled (`inspector_supervision_loss`, objective `inspector_soft_ema_consistency`, `--consistency-metrics selected`):
- **Với Inspector:** gộp trước rồi mới tính loss. Phân phối overall của Student là trung bình đều (`selected_equal`) phân phối 3 lớp của các metric mà relevance của Student chọn (relevance ≥ 0.5, nếu không có metric nào thì lấy metric relevance cao nhất). Sau đó tính CE(Inspector, overall Student). Gradient đi vào các metric được chọn qua bước trung bình.
- **Với Teacher:** tính loss **riêng từng metric** rồi mới lấy trung bình. Với mỗi metric m, tính KL(T_m‖S_m) trên phân phối 3 lớp của chính metric đó. Sau đó lấy trung bình đều trên các metric mà relevance của Student chọn. Không gộp thành phân phối overall trước khi tính KL.
  - Metric không được chọn có trọng số 0, nên không nhận gradient từ thành phần này.
  - Phép chọn metric dùng relevance đã tách khỏi đồ thị gradient (detach), nên KL Teacher không cập nhật relevance head.
  - Nhãn phân bố theo metric của Teacher lấy từ lượt Teacher chạy trên mẫu gốc. Việc chọn metric dựa trên relevance của Student, không dùng metric mà Teacher chọn.
- Loss Unlabeled của mỗi mẫu = CE với Inspector + λ_C × KL Teacher, với λ_C = 1.

Trong `comparison_tables.md`, hàng "overall" của KL là KL trên phân phối overall đã gộp (gộp trước rồi mới tính KL). Hàng này chỉ để tham khảo, không phải đại lượng được dùng để train.

| | baseline | composite | blur | noise | mask_content | flip | black_rect. | typo | drop_action | distractor | injection | rotation |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| loss (metric Student chọn) | 0.027 | 0.138 (×5.1) | 0.134 (×5.0) | 0.100 (×3.7) | 0.089 (×3.3) | 0.058 (×2.1) | 0.054 (×2.0) | 0.042 (×1.5) | 0.038 (×1.4) | 0.040 (×1.5) | 0.032 (×1.2) | 0.042 (×1.5) |
| aesthetic | 0.026 | 0.150 (×5.9) | 0.147 (×5.8) | 0.104 (×4.1) | 0.116 (×4.6) | 0.065 (×2.5) | 0.061 (×2.4) | 0.047 (×1.9) | 0.041 (×1.6) | 0.041 (×1.6) | 0.032 (×1.2) | 0.044 (×1.7) |
| fidelity | 0.031 | 0.161 (×5.1) | 0.157 (×5.0) | 0.113 (×3.6) | 0.122 (×3.9) | 0.073 (×2.3) | 0.068 (×2.2) | 0.053 (×1.7) | 0.045 (×1.4) | 0.049 (×1.6) | 0.038 (×1.2) | 0.053 (×1.7) |
| realism | 0.023 | 0.153 (×6.8) | 0.145 (×6.4) | 0.104 (×4.6) | 0.104 (×4.6) | 0.059 (×2.6) | 0.056 (×2.5) | 0.040 (×1.8) | 0.036 (×1.6) | 0.038 (×1.7) | 0.029 (×1.3) | 0.039 (×1.7) |
| background | 0.030 | 0.122 (×4.0) | 0.119 (×3.9) | 0.098 (×3.2) | 0.055 (×1.8) | 0.050 (×1.7) | 0.045 (×1.5) | 0.036 (×1.2) | 0.037 (×1.2) | 0.037 (×1.2) | 0.033 (×1.1) | 0.039 (×1.3) |
| foreground | 0.027 | 0.108 (×4.0) | 0.110 (×4.1) | 0.088 (×3.3) | 0.055 (×2.0) | 0.047 (×1.7) | 0.041 (×1.5) | 0.033 (×1.2) | 0.033 (×1.2) | 0.034 (×1.3) | 0.029 (×1.1) | 0.035 (×1.3) |
| structure | 0.020 | 0.070 (×3.5) | 0.076 (×3.8) | 0.059 (×2.9) | 0.042 (×2.1) | 0.032 (×1.6) | 0.030 (×1.5) | 0.027 (×1.4) | 0.027 (×1.3) | 0.027 (×1.3) | 0.024 (×1.2) | 0.027 (×1.3) |

**Các dạng loss khác**, hàng "loss (metric Student chọn)", số lần so với baseline:

| Dạng | composite | blur | noise | mask_content | flip | black_rect. | typo | các phép còn lại |
|---|---|---|---|---|---|---|---|---|
| KL ngược (S‖T) | ×5.7 | ×5.7 | ×4.2 | ×3.5 | ×2.2 | ×2.1 | ×1.5 | ×1.2–1.7 |
| Jensen–Shannon | ×5.0 | ×4.8 | ×3.6 | ×3.2 | ×2.1 | ×1.9 | ×1.5 | ×1.2–1.6 |
| KL phần A-vs-B | ×5.7 | ×5.6 | ×4.0 | ×3.8 | ×2.3 | ×2.2 | ×1.7 | ×1.2–1.6 |
| CE mềm | ×1.2 | ×1.2 | ×1.1 | ×1.1 | ×1.0 | ×1.0 | ×1.0 | ×1.0 |
| BCE phần A-vs-B | ×1.2 | ×1.2 | ×1.1 | ×1.1 | ×1.1 | ×1.1 | ×1.0 | ×1.0 |
| CE nhãn cứng | ×1.4 | ×1.4 | ×1.3 | ×1.2 | ×1.1 | ×1.1 | ×1.1 | ×1.0–1.1 |

CE mềm và BCE chỉ tăng ×1.1–1.2. Lý do: hai đại lượng này chứa entropy của Teacher, giống nhau ở mọi run và khá lớn trên tập [0.5, 0.7) (khoảng 0.6 với CE mềm, 0.39 với BCE A-vs-B). Mức tăng tuyệt đối thì bằng mức tăng của KL. Vì vậy, để so độ mạnh giữa các phép nên dùng KL, KL A-vs-B hoặc Jensen–Shannon. Nếu đổi loss train từ KL sang CE hay BCE mềm, gradient theo Student không đổi, vì entropy Teacher không phụ thuộc Student. Riêng BCE A-vs-B còn bỏ hẳn lớp Tie.

**Tie ở các metric consistency**

- Ở overall, aesthetic, fidelity và realism không có nhãn Tie nào.
- Teacher chọn Tie ở 308 / 1008 mẫu (background), 167 / 1008 (foreground) và 16 / 1008 (structure).
- Augmentation đẩy Student sang Tie. Số mẫu Student chọn Tie ở background: baseline 404, composite 576. Tỉ lệ mẫu Teacher chọn A/B nhưng Student chọn Tie:
  - background: baseline 12.3%, composite 32.1%, blur 27.0%, noise 23.6%;
  - foreground: baseline 6.2%, composite 20.1%, blur 19.3%.
- Ở các metric này, một phần lớn tác động của augmentation là biến A/B thành Tie, không phải làm Student dao động quanh 50/50 giữa A và B.

**Kỳ vọng "cùng nhãn nhưng Student kém chắc chắn"** (ngưỡng minh hoạ)
- Tỉ lệ mẫu có khoảng chênh ≥ 0.2 ở mức overall: baseline 2.2%, composite 31.3%, blur 27.7%, noise 21.2%, mask_content 16.1%.
- Bản 3 lớp, mức overall: Teacher ≥ 0.75, Student ≤ 0.50 và vẫn cùng nhãn: baseline 0.1%, composite 4.6%, blur 5.3%. Số này thấp vì trên tập này Teacher hiếm khi đạt ≥ 0.75.

**Hiệu so với baseline trên cùng mẫu** (overall, khoảng tin cậy 95%, bootstrap ghép cặp):

| Phép | Δ Student = Teacher | Δ khoảng chênh |
|---|---|---|
| composite | −0.100 [−0.121, −0.080] | +0.086 [+0.077, +0.095] |
| blur | −0.088 [−0.109, −0.071] | +0.073 [+0.063, +0.082] |
| noise | −0.067 [−0.083, −0.052] | +0.051 [+0.043, +0.059] |
| mask_content | −0.067 [−0.086, −0.049] | +0.048 [+0.040, +0.056] |
| flip | −0.036 [−0.050, −0.023] | +0.018 [+0.012, +0.025] |
| black_rectangles | −0.022 [−0.032, −0.014] | +0.021 [+0.016, +0.027] |
| typo | −0.023 [−0.035, −0.010] | +0.011 [+0.006, +0.017] |
| distractor / rotation / injection | −0.010 đến −0.017 | +0.005 đến +0.011 |
| drop_action (128 mẫu) | −0.023 [−0.055, 0.000] | +0.007 [−0.007, +0.023], không khác baseline |

**Kết luận:**
- **Nhóm mạnh:** composite, blur, noise, mask_content. Khoảng tin cậy không chồng lên nhóm còn lại, và thứ tự này giống nhau ở mọi metric và mọi dạng loss.
- **Nhóm giữa:** flip, black_rectangles, typo.
- **Gần như không tác dụng:** distractor_instruction, rotation, injection, drop_action.
- **Không phép đơn nào đạt mục tiêu 75–80%.** Thấp nhất là composite với 88.8%. Phải kết hợp các phép và chỉnh lại mức độ.
- Blur và noise cùng làm giảm chất lượng A và B như nhau, nên có thể trùng cơ chế: kết hợp hai phép này có thể không cộng dồn tác động.

**Dọn checkpoint (06/11, anh yêu cầu):** đã xoá `checkpoint-*.pt`, `best.pt` và `round_01.pt` của 11 run augmentation (212 GB). Log, per-sample log, `analysis/`, `augmentation_tmp/` và các file json được giữ lại. Checkpoint của baseline (job 10496) tạm giữ.

### 8.5. Kết hợp top 4 phép và Inspector one-hot (06–07/11/2026, đã có kết quả)

**Kết hợp phép:** gồm composite, blur, noise và mask_content, đủ mọi tổ hợp 2, 3 và 4 phép, tổng cộng 11 phiên bản (`stage2_tuning_unlabeled1008_augmentation_<các phép nối bằng _>`, thứ tự cố định composite → blur → noise → mask_content).
- **Cách áp:** mọi phép trong tổ hợp được áp cùng lúc lên mỗi mẫu. Mỗi phép **dùng lại đúng lượt bốc ngẫu nhiên của chính nó khi chạy riêng**, cùng seed, iteration và mẫu, nên tổ hợp chỉ khác các phép đơn ở việc kết hợp.
- **Mẫu không đủ điều kiện cho một phép:** chỉ bỏ phép đó; các phép còn lại vẫn áp.
- **Thứ tự:**
  - instruction: `mask_content` áp lên instruction gốc, sau đó template composite bọc ra ngoài;
  - ảnh: composite (source, A, B), rồi blur, rồi noise, cùng tham số trên cả ba ảnh.
- Không đổi chỗ A ↔ B.
- **Số mẫu được áp với tổ hợp 4 phép:** 953 mẫu nhận đủ 4 phép, 55 mẫu nhận 3 phép.

**Inspector one-hot:** tham số mới `--inspector-target one_hot` (biến `REFINEMENT_INSPECTOR_TARGET`).
- Thay đổi: CE(one-hot của nhãn argmax Inspector, Student overall) thay cho nhãn mềm. Mức sàn từ khoảng 0.94 xuống 0. KL Teacher giữ nguyên.
- Gradient bằng (Student − one-hot), nên phần Unlabeled kéo Student mạnh hơn so với nhãn mềm.
- Trên tập này, Teacher luôn cùng nhãn với Inspector.
- Có hai phiên bản, mỗi phiên bản so với đúng một phiên bản đối chứng:
  - `stage2_tuning_unlabeled1008_baseline_inspector_one_hot`, so với baseline;
  - `stage2_tuning_unlabeled1008_augmentation_composite_blur_noise_mask_content_inspector_one_hot`, so với tổ hợp 4 phép dùng nhãn mềm.
- Per-sample log ghi nhãn đích thực tế (`inspector`, `inspector_target`) và nhãn mềm (`inspector_soft`).

**Job:**
- 4 job test 2 step: 10687 (tổ hợp 4 phép + one-hot), 10688 (blur + noise), 10689 (noise + mask_content), 10690 (baseline + one-hot).
- 13 job chính, chờ `afterok` của 4 job test:
  - truongtq: 10691–10697 (tổ hợp 4 phép; composite + blur + noise; composite + noise + mask_content; composite + blur; blur + noise; noise + mask_content; baseline + one-hot);
  - danhnt: tổ hợp 4 phép + one-hot; composite + blur + mask_content; blur + noise + mask_content; composite + noise; composite + mask_content; blur + mask_content.

**Code** (bản gốc trong `scripts/backups/*.before_combined_operations_inspector_one_hot.*`):
- `student_augmentation.py`: thêm `choose_combined_operations` và `combined_operations_images`.
- `train_loop.py`: thêm `--inspector-target` và tổ hợp phép nối bằng "+" trong `--augmentation-operation`.
- `run_train_loop.slurm`, `config_training.sh`.

**Bước tiếp theo:** phân tích các tổ hợp so với baseline và so với các phép đơn: tỉ lệ Student = Teacher (mục tiêu 75–80%), khoảng chênh độ tự tin, KL theo metric, Tie ở các metric consistency. Chọn tổ hợp đạt mục tiêu rồi tuning mức độ. Trước đây, mục này ghi: chọn các phép mạnh nhất (đề xuất composite, blur, noise, mask_content) rồi kết hợp. Mỗi mẫu bốc 2–3 phép, hoặc áp cả 4. Sau đó tuning tỉ lệ và mức độ đến khi Student = Teacher đạt 75–80% và khoảng chênh lớn. Cần chú ý các metric consistency, nơi augmentation chủ yếu biến A/B thành Tie. Cách kết hợp cụ thể sẽ do anh xác nhận trước khi sửa code. Sau đó, nếu cần, quay lại thiết kế nhiều iteration (Teacher cập nhật EMA, chỉ loại các mẫu Teacher và Inspector bất đồng).

**Kết quả (07/11/2026).** 13 job xong đủ 126 step, không lỗi (truongtq 10691–10697, danhnt 10699–10704). Số liệu do `analyze_single_operation_augmentation.py --group combined` xuất (`stage2_tuning_unlabeled1008_baseline/analysis/combined_operations/`: `summary_tables.md`, `comparison_tables.md`, các CSV). Mọi số là trung bình trên các mẫu Unlabeled được áp dụng (tổ hợp: cả 1008 mẫu), 126 step. Ngày 07/11 anh đã xoá folder của 6 tổ hợp 2 phép; số liệu của chúng dưới đây lấy từ output script lúc 04:12 (07/11), trước khi xoá — lần chạy script sau đó chỉ còn 7 run. Số liệu phép đơn lấy từ `stage2_tuning_unlabeled1008_baseline/analysis/comparison/` (mục 8.4); số theo step của phép đơn (*) lấy từ output script ngày 06/11.

*Student = Teacher (one-hot, overall), theo step; "cùng nhãn, Student sát ngẫu nhiên" = Teacher chọn A/B với tỉ phần A-so-với-B ≥ 0.75, Student giữ nhãn nhưng tỉ phần ≤ 0.60 (tính theo xác suất).*

| Phiên bản | Student = Teacher | Hiệu so với baseline | Step 1–15 / 16–63 / 64–126 | Cùng nhãn, Student sát ngẫu nhiên |
|---|---|---|---|---|
| baseline | 0.988 | — | 1.000 / 0.987 / 0.986 | 0.1% |
| composite+blur+noise+mask_content | 0.705 | −0.283 | 0.683 / 0.690 / 0.722 | 12.6% |
| ↳ Inspector one-hot | 0.729 | −0.259 | 0.700 / 0.714 / 0.748 | 9.7% |
| composite+blur+noise | 0.739 | −0.249 | 0.683 / 0.727 / 0.762 | 12.4% |
| **composite+blur+mask_content** | **0.776** | −0.212 | 0.800 / 0.779 / 0.768 | 11.6% |
| composite+noise+mask_content | 0.803 | −0.186 | 0.775 / 0.794 / 0.815 | 10.0% |
| blur+noise+mask_content | 0.805 | −0.184 | 0.792 / 0.784 / 0.823 | 9.3% |
| composite+blur | 0.814 | −0.174 | 0.808 / 0.802 / 0.825 | 10.5% |
| composite+mask_content | 0.839 | −0.149 | 0.842 / 0.852 / 0.829 | 9.2% |
| composite+noise | 0.845 | −0.143 | 0.825 / 0.815 / 0.873 | 8.4% |
| blur+mask_content | 0.859 | −0.129 | 0.867 / 0.865 / 0.853 | 7.2% |
| blur+noise | 0.867 | −0.121 | 0.850 / 0.857 / 0.879 | 10.5% |
| noise+mask_content | 0.869 | −0.119 | 0.883 / 0.872 / 0.863 | 5.5% |
| composite (phép đơn) | 0.888 | −0.100 | 0.867 / 0.878 / 0.901 * | 7.5% |
| blur (phép đơn) | 0.900 | −0.088 | 0.915 / 0.885 / 0.908 * | 5.5% |
| noise (phép đơn) | 0.921 | −0.067 | 0.933 / 0.911 / 0.926 * | 3.5% |
| mask_content (phép đơn) | 0.920 | −0.068 | 0.941 / 0.920 / 0.915 * | 3.1% |
| baseline_inspector_one_hot | 0.987 | −0.001 | 1.000 / 0.984 / 0.986 | 0.0% |

*Theo metric, các phiên bản chính (BG / FG / ST = background / foreground / structure consistency). Ô: Student = Teacher; độ tự tin Student vào nhãn Teacher / khoảng chênh (Teacher − Student); KL(Teacher‖Student) (số lần so với baseline).*

| Phiên bản | | overall | aesthetic | fidelity | realism | BG | FG | ST |
|---|---|---|---|---|---|---|---|---|
| Teacher | độ tự tin | 0.734 | 0.868 | 0.839 | 0.868 | 0.598 | 0.607 | 0.573 |
| baseline | S = T | 0.988 | 0.976 | 0.971 | 0.988 | 0.849 | 0.903 | 0.937 |
| | Student / chênh | 0.685 / 0.049 | 0.829 / 0.039 | 0.785 / 0.053 | 0.826 / 0.041 | 0.529 / 0.069 | 0.540 / 0.067 | 0.527 / 0.046 |
| | KL | 0.021 | 0.026 | 0.031 | 0.023 | 0.030 | 0.027 | 0.020 |
| composite+blur+mask_content | S = T | 0.776 | 0.781 | 0.769 | 0.777 | 0.488 | 0.584 | 0.757 |
| | Student / chênh | 0.527 / 0.206 | 0.644 / 0.224 | 0.599 / 0.240 | 0.629 / 0.239 | 0.389 / 0.209 | 0.410 / 0.197 | 0.442 / 0.131 |
| | KL | 0.193 (×9.3) | 0.282 (×11.0) | 0.288 (×9.2) | 0.275 (×12.1) | 0.176 (×5.8) | 0.160 (×5.9) | 0.101 (×5.0) |
| composite+blur+noise+mask_content | S = T | 0.705 | 0.718 | 0.718 | 0.702 | 0.441 | 0.519 | 0.705 |
| | Student / chênh | 0.493 / 0.240 | 0.596 / 0.272 | 0.553 / 0.285 | 0.581 / 0.287 | 0.373 / 0.225 | 0.392 / 0.214 | 0.428 / 0.145 |
| | KL | 0.237 (×11.4) | 0.357 (×14.0) | 0.358 (×11.4) | 0.345 (×15.2) | 0.198 (×6.5) | 0.184 (×6.8) | 0.118 (×5.8) |
| composite (phép đơn) | S = T | 0.888 | 0.878 | 0.869 | 0.891 | 0.616 | 0.717 | 0.843 |
| | Student / chênh | 0.599 / 0.134 | 0.738 / 0.130 | 0.690 / 0.147 | 0.725 / 0.142 | 0.435 / 0.161 | 0.457 / 0.149 | 0.477 / 0.096 |
| | KL | 0.111 (×5.3) | 0.150 (×5.9) | 0.161 (×5.1) | 0.153 (×6.8) | 0.122 (×4.0) | 0.108 (×4.0) | 0.070 (×3.5) |

*Loss KL Teacher đúng như lúc train (trung bình trên metric Student chọn) và khoảng chênh overall, mọi phiên bản:* baseline 0.027 / 0.049; 4 phép 0.287 (×10.6) / 0.240; composite+blur+noise 0.266 (×9.8) / 0.224; composite+blur+mask_content 0.235 (×8.7) / 0.206; composite+noise+mask_content 0.220 (×8.2) / 0.194; blur+noise+mask_content 0.215 (×8.0) / 0.180; composite+blur 0.206 (×7.6) / 0.184; composite+mask_content 0.185 (×6.9) / 0.173; composite+noise 0.178 (×6.6) / 0.164; blur+noise 0.175 (×6.5) / 0.147; blur+mask_content 0.173 (×6.4) / 0.154; noise+mask_content 0.146 (×5.4) / 0.134; composite đơn 0.138 (×5.1) / 0.134.

*Inspector one-hot so với Inspector mềm (trung bình 126 step; CE one-hot = −log P_Student(nhãn Inspector), mức sàn 0).*

| | baseline | baseline_inspector_one_hot | 4 phép | 4 phép, one-hot |
|---|---|---|---|---|
| CE(Inspector, Student) | 1.024 | 0.269 | 1.042 | 0.719 |
| mức sàn (entropy Inspector) | 0.941 | 0 | 0.941 | 0 |
| phần học được = KL(Inspector‖Student) | 0.082 | 0.269 (×3.3) | 0.101 | 0.719 (×7.1) |
| KL(Teacher‖Student) | 0.027 | 0.031 | 0.287 | 0.286 |
| L_human / L_preference / L_relevance | 0.775 / 0.455 / 0.210 | 0.786 / 0.457 / 0.213 | 0.790 / 0.456 / 0.213 | 0.800 / 0.463 / 0.215 |
| L_total học được (đã trừ sàn) | 1.548 | 1.756 | 1.847 | 2.483 |
| gradient Unlabeled / có nhãn | 0.32 | 0.62 | 0.98 | 1.66 |
| Δ Student = Teacher (bootstrap 95%) | — | −0.001 [−0.005, +0.003] | — | +0.024 [+0.009, +0.039] |
| Δ khoảng chênh | — | −0.094 (chênh = −0.046) | — | −0.053 |

**Nhận xét:**
- Số phép quyết định mức tách: 4 phép 0.70–0.73, 3 phép 0.74–0.81, 2 phép 0.81–0.87, phép đơn 0.89–0.98. Tổ hợp có composite thường mạnh hơn. Tác động không cộng đủ: ở 6 cặp 2 phép, mức giảm thực tế bằng khoảng 78–90% tổng mức giảm của 2 phép đơn.
- Ba cách đo (accuracy one-hot, khoảng chênh độ tự tin, KL) xếp hạng các phiên bản gần như giống nhau. Ở các tổ hợp, 5–13% mẫu có Teacher chắc chắn mà Student giữ nhãn ở mức gần 50/50 giữa A và B (baseline 0.1%).
- Hai nhóm metric phản ứng khác nhau: aesthetic / fidelity / realism (Teacher tự tin 0.84–0.87) có khoảng chênh và KL tăng mạnh nhất; các metric consistency (Teacher chỉ 0.57–0.61) có KL tăng ít hơn nhưng accuracy one-hot giảm sâu nhất (Teacher sát ngưỡng, lệch nhỏ đã đổi nhãn). mask_content đơn gần như chỉ tác động lên aesthetic / fidelity / realism.
- Đa số tổ hợp có Student = Teacher tăng dần theo step (Student thích nghi); composite+blur+mask_content giảm dần (0.800 → 0.768).
- Loss trên cặp có nhãn gần như không đổi ở mọi phiên bản (L_human 0.775–0.800, L_preference 0.453–0.463, L_relevance 0.210–0.215): augment Unlabeled không hại nhưng cũng không làm phần có nhãn học thêm.
- **Inspector one-hot:** loss học được lớn hơn 3–7 lần và gradient Unlabeled mạnh hơn, nhưng trên tập con này nhãn Inspector trùng nhãn Teacher ở 98.3–98.7% mẫu, nên CE one-hot đẩy Student tự tin vào đúng nhãn Teacher: ở baseline Student tự tin hơn cả Teacher (khoảng chênh âm ở mọi metric, P(nhãn) 0.780 so với Teacher 0.734); ở 4 phép, Student = Teacher tăng và khoảng chênh giảm. One-hot đi ngược mục tiêu tăng khoảng cách.

**So với Stage 1 (Student-Stage1 batch 16 khởi tạo từ Inspector checkpoint-63).** MS-Swift chỉ ghi loss tổng (= L_preference + L_relevance trên GemmaGT, không có L_human), grad_norm và learning rate, mỗi 5 step (`student_pipeline_…_initialized_from_inspector_checkpoint63/analysis/losses.png`). Stage 1: 2 epoch trên 1008 cặp, learning rate đỉnh 4e-5; loss 2.69 (step 1) → khoảng 1.0 (step 40) → 0.66–0.78 (step 100–125). Stage 2 bắt đầu đúng ở mức đó (L_preference + L_relevance 0.80 ở step 1, trung bình 0.665) và đi ngang: cùng 1008 cặp lần thứ ba, input không đổi, learning rate nhỏ hơn 4 lần. Đây là lý do augment cả cặp có nhãn (bên dưới).

**Plot của mỗi run** (`analysis/`, chỉ run đó, tiếng Anh, tên loss thống nhất L_human, L_preference = L_tie + L_winner, L_relevance, L_unlabeled = CE(Inspector, Student) + λ_C·KL(Teacher‖Student), L_total): `losses.png`, `learnable_loss.png`, `training_dynamics.png`, `metrics.png`, `label_transitions.png`. So sánh giữa các run nằm trong các bảng của `combined_operations/`.

**Đề xuất (chờ anh xác nhận):** chọn composite+blur+mask_content (nằm trong vùng 75–80%, ổn định theo step), dự phòng composite+noise+mask_content; giữ Inspector mềm.

**Dọn (07/11, anh yêu cầu):** anh xoá folder của 6 tổ hợp 2 phép; checkpoint (`checkpoint-*.pt`, `best.pt`, `round_01.pt`, khoảng 135 GB) của 7 run còn lại (10691–10693, 10697, 10699–10701) đã xoá, giữ log, `analysis/`, json.

**Augment cả cặp có nhãn (06/11/2026, tối).** Mục tiêu sau ablation: chọn tổ hợp tăng khoảng cách Student–Teacher, áp tổ hợp đó lên cả cặp có nhãn và Unlabeled để xử lý việc các loss trên cặp có nhãn đi ngang, giám sát loss từng step. Giữ chỗ slot trong lúc chờ ablation:

| Phiên bản (`stage2_tuning_unlabeled1008_augmentation_composite_blur_noise_mask_content_…`) | Thay đổi | Đối chứng |
|---|---|---|
| `labeled_augmented` (job 10842) | cặp có nhãn (HumanGT + GemmaGT, một lần forward) nhận cùng tổ hợp 4 phép | tổ hợp 4 phép chỉ augment Unlabeled; baseline |
| `inspector_one_hot_labeled_augmented` (job 10843) | như trên, Inspector one-hot | `…_inspector_one_hot`; baseline |

**Thêm 2 tổ hợp 3 phép, augment cả cặp có nhãn (07/11, anh xác nhận; không có bản one-hot; submit trên danhnt):**

| Phiên bản | Job | Lý do |
|---|---|---|
| `stage2_tuning_unlabeled1008_augmentation_composite_blur_mask_content_labeled_augmented` | 10926 | tổ hợp duy nhất trong vùng 75–80% (0.776), không tăng theo step |
| `stage2_tuning_unlabeled1008_augmentation_composite_noise_mask_content_labeled_augmented` | 10927 | sát mép trên (0.803), mức augment nhẹ hơn |

- Cùng code với 10842 (đã qua test 10841), Inspector mềm; `config_training.sh` mở hậu tố `_labeled_augmented` cho 2 tổ hợp này (bản gốc `scripts/backups/config_training.before_labeled_augmented_three_operations.sh`).
- Không thêm bản one-hot: trên tập con này nhãn Inspector trùng nhãn Teacher ở khoảng 98.5% mẫu, one-hot thu hẹp khoảng chênh; quyết định lại sau khi đánh giá EditReward-Bench cặp 10842 (mềm) / 10843 (one-hot), cặp duy nhất còn checkpoint.
- Sơ bộ 10842 / 10843 (step 1–38, so với run chỉ augment Unlabeled): L_human 0.996 so với 0.794, L_preference 0.685 so với 0.472, L_relevance 0.294 so với 0.223; loss có nhãn bắt đầu giảm lại (L_human 20 step cuối 0.952), không còn đi ngang ở mức Stage 1 kết thúc.

- Cặp có nhãn dùng input đã augment (không giữ bản gốc); nhãn giữ nguyên (HumanGT, Gemma preference và relevance majority-vote). Giả định augmentation không đổi nhãn; cần theo dõi riêng L_relevance (composite thêm ảnh lạ, mask_content che chữ).
- Cách bốc giống Unlabeled (theo sample_id, cùng seed); danh sách ảnh ghép giữ nguyên nên phần Unlabeled bốc giống hệt job đối chứng; đã kiểm tra ảnh ghép không trùng ảnh nguồn với cặp có nhãn.
- Điều kiện áp dụng của cặp có nhãn dựng ngay trong run bằng đúng quy tắc của `build_augmentation_scenarios.py` (combined), lưu `augmentation_labeled_scenarios.json` trong folder run; log từng mẫu của cặp có nhãn ghi phép đã áp và instruction mới.
- Code: `train_loop.py` thêm `--augment-labeled`; `run_train_loop.slurm` thêm `REFINEMENT_AUGMENT_LABELED`; `config_training.sh` thêm hậu tố `_labeled_augmented` (chỉ cho tổ hợp 4 phép). Bản gốc: `scripts/backups/*.before_labeled_augmentation.*`.
- Phân tích: `analyze_single_operation_augmentation.py --group combined` (thêm plot phần học được và gradient / learning rate; baseline và phép đơn chỉ đọc).
- Job: test 10841 (2 step) xong, không lỗi; 10842 và 10843 đang chạy (07/11). Log xác nhận: điều kiện áp dụng dựng cho 1008 cặp (mask_content 979, blur 1001, noise 1000); trong 280 cặp đầu, 269 cặp nhận đủ 4 phép, 11 cặp nhận 3 phép.

### 8.6. Relevance Head trên Unlabeled (06/11/2026)

**Phát hiện** (đã kiểm tra trên code và log; chi tiết và các vấn đề liên quan: `pipeline_process.md` mục 3.12). Chưa xử lý: hiện tập trung vào augmentation ablation; sau khi xong có thể phải triển khai lại các experiment cho các vấn đề này.
- Relevance Head ở Stage 2 chỉ học từ L_M trên 1008 cặp đã train ở Stage 1 (lượt thứ ba, input gốc). Không có loss relevance trên Unlabeled; relevance được detach khi chọn S ở mọi cặp.
- Augmentation đi vào chuỗi source, nên S của Student được chọn trên input đã biến đổi:

| Phép | S Student trùng S Teacher | Student = Teacher (overall): S trùng / S khác |
|---|---:|---|
| baseline | 0.92 | 0.990 / 0.965 |
| composite | 0.78 | 0.896 / 0.858 |
| mask_content | 0.83 | 0.929 / 0.877 |
| distractor_instruction | 0.85 | 0.973 / 0.961 |
| blur, noise, flip, rotation, black_rectangles | 0.90–0.92 | |

Metric hay đổi: realism, background, structure consistency; aesthetic và fidelity không đổi.

## 9. Phụ lục: kế hoạch ban đầu và các run trước batch 16 (gộp từ bản cũ)

Phần này giữ lại từ bản `stage2_tuning.before_batch16_unlabeled1008_settings.md` (02/11/2026) những nội dung không có ở các mục trên. Mọi run ở đây dùng batch sai (mục 7.7), chỉ có giá trị tham khảo về cơ chế.

### 9.1. Mục tiêu và giả định ban đầu

Các phiên bản Giai đoạn 2 trên 10k Unlabeled (setting 1008 cặp) mất 14–19 giờ mỗi run trên partition full, chưa cải thiện và chưa tách khỏi Student-Stage1. Stage 2 tuning thu nhỏ để chạy nhiều job song song trên normal (3–4 giờ mỗi job) và trước hết kiểm tra Stage 2 có overfit được dữ liệu train không. Các giả định cần kiểm chứng:

1. Trọng số các loss thành phần (λ_U = 1 hay 6; λ_G, λ_M, λ_C).
2. Tỉ lệ cặp có nhãn : Unlabeled mỗi step.
3. Ramp λ_U: total loss tăng trong các step ramp một phần chỉ do λ_U(t) tăng; phải xem từng loss thành phần chưa nhân λ.
4. Learning rate (backbone LoRA có thể giảm 5–10 lần; warmup, lịch learning rate).
5. Có cần KL(Teacher ‖ Student) không.
6. Teacher MC-Dropout J = 4 có tốt hơn J = 1–3 không (J = 4 tốn hơn cả phần train Student).
7. EMA Teacher cosine 0.99 → 0.9999 có hợp với 100–200 step không.

Kế hoạch ban đầu có 49 job chia giai đoạn: thử (0), overfit nhóm A + 3 seed + đối chứng λ_U = 0 (1), mỗi run đổi một biến: λ_U, ramp, learning rate, tỉ lệ dữ liệu, λ_C, EMA, λ_G = λ_M, J, chia ba round theo độ tự tin `inspector_teacher_rank` (2), tổ hợp và setting cuối (3). Chỉ đợt đầu 16 job được chạy, sau đó bị huỷ vì sai batch.

### 9.2. Job thử thời gian (giai đoạn 0)

- `stage2_tuning_timing_test_normal` (huỷ sau 22 phút): đánh giá ban đầu kiểu cũ (Teacher và Student trên 1008 cặp Gemma, probe 128 mẫu với J = 4, khoảng 2,500 lượt forward) quá nặng.
- `stage2_tuning_timing_test_normal_subset` (240 Unlabeled, 60 step, J = 4): Teacher **1.64 giây mỗi lượt forward**, Student **4.27 giây mỗi cặp forward + backward**, bộ nhớ đỉnh **29.0 GiB**.

### 9.3. Nhóm A: Stage 2 có overfit được dữ liệu train không

256 Unlabeled cố định, 4 lượt × 32 step (8 Unlabeled mỗi step), learning rate constant, đánh giá ở eval mode. A1: chỉ CE Unlabeled, lr 1e-5; A2: như A1, lr 2e-6; A3: thêm KL Teacher; A4: loss đầy đủ + 4 cặp có nhãn mỗi step.

KL(Inspector ‖ Student) = CE − entropy nhãn Inspector (mức sàn 0.915 trên 256 mẫu), trung bình từng lượt trong log:

| Run | Lượt 1 | Lượt 2 | Lượt 3 | Lượt 4 |
|---|---|---|---|---|
| A1 lr 1e-5 | 0.050 | 0.013 | 0.009 | 0.007 |
| A2 lr 2e-6 | 0.069 | 0.034 | 0.024 | 0.018 |

Nhánh Unlabeled khớp được nhãn Inspector, không underfit; learning rate lớn hơn khớp nhanh hơn. Chỉ train Unlabeled thì gradient norm khoảng 3 (so với 60–80 khi có cặp có nhãn).

### 9.4. Phân tích đợt đầu (6 job, log train, batch 24)

| Phiên bản | Loss Unlabeled, 25 step đầu → cuối (hoặc theo lượt) | 32 cặp Gemma theo dõi, preference loss Student (Teacher), đầu 0.313 | Norm gradient trước clip, trung vị |
|---|---|---|---|
| điểm xuất phát seed 42 | 0.992 → 0.981 | 0.330 (0.320) | 68 |
| điểm xuất phát seed 43 | 1.012 → 0.979 | 0.340 (0.328) | 65 |
| λ_U = 0 | 1.006 → 1.126 (không train, chỉ đo) | 0.299 (0.315) | 81 |
| A1 lr 1e-5 | 0.965 / 0.928 / 0.924 / 0.922 | 0.441 (0.349) | 3.0 |
| A2 lr 2e-6 | 0.984 / 0.949 / 0.939 / 0.933 | 0.393 (0.335) | 5.8 |
| A4 loss đầy đủ | 0.975 / 0.962 / 0.947 / 0.948 | 0.325 (0.318) | 64 |

- **Nhãn Inspector rất mềm:** P lớn nhất trung bình 0.56, chỉ khoảng 25% mẫu có P lớn nhất ≥ 0.7 (cả 10k: entropy 0.904). Giả thuyết: khớp nhãn này kéo Student về dè dặt (|d| nhỏ, P(Tie) cao). Hướng sửa nếu đúng: nhãn Inspector sắc hơn (nhiệt độ), chỉ dùng hoặc tăng trọng số mẫu Inspector tự tin.
- Loss Unlabeled làm Student lệch nhãn Gemma (A1: preference 0.313 → 0.441, mask BCE 0.18 → 0.29); ngược lại λ_U = 0 làm loss Unlabeled tăng dần. Hai mục tiêu mâu thuẫn.
- Ở step đầu, norm theo nhóm ViT 54.8, LLM 8.7, heads 5.7, aligner 1.9: ViT đã chiếm phần lớn norm từ các run này.
- Probe 32 mẫu với J = 1: hộp Teacher rất hẹp (0.039), chỉ số trong hộp ít thông tin; probe đã bỏ ở config mục 3.

### 9.5. Sai batch (phát hiện 02/11/2026)

Batch chuẩn của Giai đoạn 2 là 16 cặp mỗi optimizer step (như Student-Stage1 và Inspector). Các run cũ lệch: đợt đầu 4 có nhãn + 20 Unlabeled = 24 (100 step, 400 / 1008 cặp có nhãn); nhóm A 8 hoặc 12; các run `stage2_tuning_labeled_one_pass_*` 24 hoặc 36 (huỷ khi đang chạy). Nguyên nhân: số Unlabeled mỗi step được chọn để 2,000 Unlabeled chia thành khoảng 100 step, không đối chiếu với batch 16. Ngoài ra Student-Stage1 cũ (`student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_head_fp32`, dùng cho mọi Giai đoạn 2 trước đó, kể cả các run 10k) train với batch 8. Bộ nhớ phụ thuộc microbatch, không phụ thuộc số cặp mỗi step: microbatch 1 cặp khoảng 29 GiB.
