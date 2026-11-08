# AUGMENTATION ABLATION CHO STUDENT Ở STAGE 2

Trạng thái tài liệu: 07/11/2026. Liên quan: `stage2_tuning.md` (mục 8.1–8.5: diễn biến chi tiết, bảng đầy đủ), `pipeline_process.md` (mục 3.12: Relevance Head trên Unlabeled). Tài liệu này gom các thông tin của augmentation ablation: mục tiêu, setting, định nghĩa các phép, cách phân tích, kết quả, quyết định và các lỗi đã biết.

## 1. Mục tiêu

- **Vấn đề:** Student Stage 2 gần như không học thêm. Loss không giảm và Bench / accuracy / AUC không khác Student-Stage1. Khi Teacher và Inspector ra cùng nhãn và gần cùng phân phối với Student, gần như không có gradient để học.
- **Cách làm:** trong cùng một step, Teacher và Inspector nhận mẫu Unlabeled gốc; Student nhận chính mẫu đó sau khi augment online.
- **Mục tiêu của tổ hợp phép được chọn**, đo ở overall và ở từng metric trong 6 metric:
  - theo nhãn one-hot overall: Student = Teacher khoảng **75–80%** (anh chỉnh ngày 06/11; bản ngày 05/11 ghi 70–75%);
  - theo nhãn mềm: Teacher tự tin vào nhãn của nó **cao hơn nhiều** so với Student đã augment trên cùng nhãn đó (khoảng chênh độ tự tin lớn, KL(Teacher‖Student) lớn).
- **Sau ablation:** áp tổ hợp được chọn lên cả cặp có nhãn và Unlabeled, để xử lý việc các loss trên cặp có nhãn đi ngang. Giám sát loss theo từng step.

## 2. Setting chung

- Setting 1008: Student-Stage1 khởi tạo từ Inspector checkpoint-63; Inspector checkpoint-63 cùng Overall Pseudo-label Cache của nó.
- **Tập Unlabeled** (job 10450), 1008 mẫu:
  - Teacher (eval mode) cùng nhãn với Inspector;
  - xác suất lớn nhất của Inspector trong [0.5, 0.7);
  - duyệt theo thứ tự ngẫu nhiên (seed 42);
  - mỗi instruction chỉ xuất hiện một lần.
- Thư mục tập Unlabeled: `output_training/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63/`. File điều kiện áp dụng: `augmentation_combined_scenarios_1008.json` (job 10451).
- all_once, một loop, 126 step. Batch gồm 8 cặp có nhãn + 8 Unlabeled. Mọi λ bằng 1. Learning rate 1e-5. Partition normal.
- Không đổi chỗ A ↔ B. Teacher và Inspector luôn nhận mẫu gốc.
- Mỗi biến đổi do `scripts/student_augmentation.py` tạo online theo seed (seed, iteration, mẫu), không viết hay sửa tay.
  - Mẫu không đủ điều kiện cho một phép thì giữ input gốc và được ghi `applied = False` kèm lý do.
  - Mức độ thực tế của từng mẫu được ghi trong `choice["strength"]`.
  - Mỗi job lưu 25 mẫu ví dụ trong `augmentation_tmp/`.

## 3. Định nghĩa các phép (giữ nguyên như code hiện tại, anh xác nhận 07/11)

Code: `choose_single_augmentation`, `single_augmented_images` (phép đơn); `choose_combined_operations`, `combined_operations_images` (tổ hợp); trong `scripts/student_augmentation.py`.

| Phép | Cách làm | Áp lên | Tham số |
|---|---|---|---|
| `composite` | ghép **đúng 1 ảnh lạ** vào một phía của ảnh chính: trái / phải / trên / dưới, chọn đều. Cùng ảnh lạ và cùng phía cho source, A và B, để ba ảnh giữ cùng bố cục. Ảnh lạ được resize cho cùng chiều cao (ghép trái / phải) hoặc cùng chiều rộng (ghép trên / dưới) với ảnh chính, rồi cả khung thu về số pixel của ảnh gốc tương ứng. Instruction bọc trong 1 trong 4 template (`COMPOSITE_TEMPLATES`), luôn nêu vị trí ảnh cần sửa và vị trí ảnh lạ phải bỏ qua | source, A, B, instruction | ảnh lạ là source của mẫu Unlabeled ngoài tập, không trùng source của mẫu |
| `mask_content` | thay từ nội dung bằng `___`. Không che: từ đầu tiên, stopword, từ dưới 3 chữ cái, text trong ngoặc kép. Luôn giữ lại ít nhất 1 từ nội dung | instruction | **20–40% số từ nội dung** (`SINGLE_TEXT_FRACTION = (0.2, 0.4)`, anh chốt 05/11) |
| `blur` | làm mờ Gaussian, cùng tham số trên ba ảnh | source, A, B | bán kính 2–3 (mức nhẹ 1–1.5 cho instruction về chữ); không dùng cho instruction về chất lượng hoặc độ nét |
| `noise` | nhiễu Gaussian, cùng tham số trên ba ảnh | source, A, B | sigma 15–25 (mức nhẹ 6–12); không dùng cho instruction về chất lượng |
| `black_rectangles` | 1–3 hình chữ nhật đen | source | tổng diện tích 10–20% |
| `typo` | viết sai chính tả từ dài từ 4 chữ cái; giữ từ chỉ vị trí và text trong ngoặc kép | instruction | 20–40% số từ đủ điều kiện |
| `flip` | lật ngang ba ảnh; đổi left ↔ right trong instruction | source, A, B, instruction | không dùng cho instruction về chữ, số, xoay, lật |
| `drop_action` | bỏ cụm mở đầu ("Make this look like…"), không bỏ động từ chỉnh sửa | instruction | chỉ áp được khi câu mở đầu bằng cụm yêu cầu |
| `distractor_instruction` | chèn instruction của mẫu khác, ghi rõ là không thực hiện | instruction | — |
| `rotation` | xoay, góc trống tô đen | source | 5–15°, dấu ngẫu nhiên |
| `injection` | thêm câu mở đầu hoặc kết thúc trung tính, không thêm yêu cầu | instruction | — |

**Tổ hợp phép** (`choose_combined_operations`):
- Chỉ tổ hợp được trong `composite`, `blur`, `noise`, `mask_content`, theo thứ tự cố định đó.
- Mỗi phép dùng lại đúng lượt bốc ngẫu nhiên của chính nó khi chạy riêng (cùng seed, iteration và mẫu).
- Mẫu không đủ điều kiện cho một phép thì chỉ bỏ phép đó.
- Instruction: `mask_content` áp lên instruction gốc trước, sau đó template `composite` bọc ra ngoài, nên câu chỉ vị trí không bao giờ bị che.
- Ảnh: composite → blur → noise, cùng tham số trên source, A, B.

**Augment cả cặp có nhãn** (`--augment-labeled`, biến `REFINEMENT_AUGMENT_LABELED`):
- Cặp có nhãn (HumanGT + GemmaGT, một lần forward) nhận cùng phép, cùng quy tắc bốc theo sample_id.
- Nhãn giữ nguyên: HumanGT, và Gemma preference / relevance theo majority vote.
- Điều kiện áp dụng dựng ngay trong run bằng quy tắc của `build_augmentation_scenarios.py`, lưu `augmentation_labeled_scenarios.json` trong folder run.

**Inspector one-hot** (`--inspector-target one_hot`, biến `REFINEMENT_INSPECTOR_TARGET`): CE với one-hot của nhãn argmax Inspector thay cho nhãn mềm. KL Teacher giữ nguyên.

## 4. Các nhóm experiment

| Nhóm | Phiên bản (`stage2_tuning_unlabeled1008_…`) | Job | Ghi chú |
|---|---|---|---|
| Baseline | `baseline` | 10496 | không augmentation; còn checkpoint cuối |
| Phép đơn (05–06/11) | `augmentation_<phép>`, 11 phép | truongtq: blur 10497, mask_content 10498, typo 10499, drop_action 10500, rotation 10501; danhnt: composite 10503, noise 10504, black_rectangles 10505, flip 10506, distractor_instruction 10507, injection 10508 | test 2 step: 10483, 10484, 10485, 10488, 10491 |
| Tổ hợp và Inspector one-hot (06/11) | `augmentation_<các phép nối bằng _>` (11 tổ hợp), `baseline_inspector_one_hot`, `augmentation_composite_blur_noise_mask_content_inspector_one_hot` | truongtq: 10691–10697; danhnt: 10699–10704 | test 2 step: 10687–10690 |
| Augment cả cặp có nhãn (06–07/11) | `augmentation_composite_blur_noise_mask_content_labeled_augmented`, `augmentation_composite_blur_noise_mask_content_inspector_one_hot_labeled_augmented` | 10842, 10843 (truongtq) | test 2 step: 10841 |
| | `augmentation_composite_blur_mask_content_labeled_augmented`, `augmentation_composite_noise_mask_content_labeled_augmented` | 10926, 10927 (danhnt) | |

Mọi submit được ghi trong `experiment_submissions.log` (config và md5 code).

**Dọn dữ liệu:**
- 06/11, anh yêu cầu: đã xoá checkpoint của 11 run phép đơn.
- 07/11:
  - anh xoá folder của 6 tổ hợp 2 phép;
  - checkpoint của 7 run tổ hợp 3–4 phép / one-hot đã được xoá theo yêu cầu của anh; log và `analysis/` được giữ.
- Kiểm tra ngày 07/11: trong `output_training/` chỉ còn folder của `baseline`, `baseline_inspector_one_hot` và 2 run `_labeled_augmented` của truongtq (10842, 10843).

## 5. Phân tích

- **Script:** `RewardModelEvaluation/OurModel/analyze_single_operation_augmentation.py`.
  - Chỉ đọc `training_log_<job>.jsonl` và `per_sample_log_<job>.jsonl`, không cần model, không đọc ảnh.
  - `--group combined` cho nhóm tổ hợp; `--plots-only` và `--only` để chỉ vẽ một số run.
- **Mỗi run**, trong `output_training/<phiên bản>/analysis/`:
  - `step_losses.csv`, `samples_overall.csv`, `samples_metrics.csv`, `summary.json`;
  - plot của riêng run đó (tiếng Anh): `losses.png`, `learnable_loss.png`, `training_dynamics.png`, `metrics.png`, `label_transitions.png`.
- **Bảng so sánh**, trong `stage2_tuning_unlabeled1008_baseline/analysis/`:
  - phép đơn: `ranking.csv`, `comparison/comparison_tables.md`;
  - tổ hợp: `combined_operations/summary_tables.md`, `combined_operations/ranking.csv`.
- Mọi tỉ lệ của một phép được tính trên các mẫu thực sự được áp dụng phép. Baseline dùng toàn bộ mẫu. KL, CE, độ tự tin và khoảng chênh tính trên 3 lớp A/B/Tie.
- **Đọc loss:**
  - CE(Inspector, Student) với nhãn mềm có mức sàn bằng entropy của Inspector (khoảng 0.94 trên tập này), không tạo gradient.
  - Phần học được = loss − λ_U·entropy Inspector.
  - L_human, L_preference, L_relevance và CE one-hot có mức sàn bằng 0.

## 6. Kết quả

Số liệu lấy từ các file của script ở mục 5. Mỗi giá trị là trung bình 126 step trên các mẫu Unlabeled được áp dụng.

**Student = Teacher (one-hot, overall) và khoảng chênh độ tự tin overall (Teacher − Student).** Teacher tự tin vào nhãn của nó 0.734 ở mọi run.

| Phiên bản | Student = Teacher | Khoảng chênh | Nguồn |
|---|---|---|---|
| baseline | 0.988 | 0.049 | `combined_operations/summary_tables.md` |
| composite+blur+noise+mask_content | 0.705 | 0.240 | như trên |
| ↳ Inspector one-hot | 0.729 | 0.187 | như trên |
| composite+blur+noise | 0.739 | 0.224 | như trên |
| composite+blur+mask_content | 0.776 | 0.206 | như trên |
| composite+noise+mask_content | 0.803 | 0.194 | như trên |
| blur+noise+mask_content | 0.805 | 0.180 | như trên |
| composite+mask_content | 0.839 | 0.173 | output của script lúc 04:12 ngày 07/11 (*) |
| composite (phép đơn) | 0.888 | 0.134 | `ranking.csv` |
| blur (phép đơn) | 0.900 | 0.122 | như trên |
| noise (phép đơn) | 0.921 | 0.100 | như trên |
| mask_content (phép đơn) | 0.920 | 0.097 | như trên |
| baseline_inspector_one_hot | 0.987 | −0.046 | `combined_operations/summary_tables.md` |

(*) Các tổ hợp 2 phép: file của lần chạy script lúc 04:12 ngày 07/11 đã bị ghi đè khi chạy lại sau khi anh xoá folder. Số liệu chép trong `stage2_tuning.md` mục 8.5. Hiện không còn file nào chứa các số này.

Bảng theo từng metric, KL theo metric, các dạng loss khác và hiệu so với baseline (bootstrap) nằm trong `comparison/comparison_tables.md`, `combined_operations/summary_tables.md` và `stage2_tuning.md` mục 8.4–8.5.

**Nhận xét chính** (chi tiết: `stage2_tuning.md` mục 8.4–8.5):
- Nhóm phép đơn mạnh là composite, blur, noise, mask_content. Không phép đơn nào đạt 75–80%.
- Số phép quyết định mức tách. Tác động không cộng đủ khi kết hợp.
- Ở các metric consistency, augmentation chủ yếu biến A/B thành Tie.
- Loss trên cặp có nhãn đi ngang ở mọi run chỉ augment Unlabeled.
- Inspector one-hot thu hẹp khoảng chênh, vì trên tập này nhãn Inspector gần như luôn trùng nhãn Teacher.

**Augment cả cặp có nhãn:**
- 10842 và 10843 chạy xong ngày 07/11 (khoảng 3 giờ 22 phút mỗi job). Plot của từng run nằm trong `analysis/` của run đó.
- Chưa phân tích loss trên cặp có nhãn so với run chỉ augment Unlabeled và so với Stage 1.
- 10926 và 10927 chưa phân tích.

## 7. Quyết định

- **07/11 (anh):** thống nhất dùng tổ hợp **composite+mask_content**. Không dùng các tổ hợp có blur hoặc noise, vì hai phép này làm giảm chất lượng ảnh, ảnh hưởng đến các metric aesthetic và realism, và khiến model học thuộc nhãn thay vì phân tích source image.
- **Giữ nguyên định nghĩa các phép như code hiện tại** (mục 3):
  - `mask_content` che 20–40% số từ nội dung;
  - `composite` chỉ ghép thêm 1 ảnh ở trên / dưới / trái / phải, instruction luôn có phần chỉ vị trí source image.
- Đã kiểm tra trên code ngày 07/11: mọi lượt `composite` (đơn và trong tổ hợp) đều có câu nêu vị trí ảnh cần sửa và ảnh phải bỏ qua. Câu này không bị `mask_content` che.
- Inspector mềm được giữ (đề xuất ở `stage2_tuning.md` mục 8.5). Quyết định cuối về one-hot chờ đánh giá EditReward-Bench của cặp 10842 / 10843.

## 8. Lỗi và hạn chế đã biết (chưa sửa)

- **`mask_content` che từ chỉ vị trí trong instruction gốc** (ghi nhận 07/11, anh: chưa cần sửa). Danh sách từ được phép che không loại left / right / top / bottom.
  - Vì vậy một số mẫu có từ chỉ vị trí sẵn trong instruction gốc bị che, ví dụ "the man on the left" → "the man on the ___".
  - Câu chỉ vị trí source image của `composite` không bị ảnh hưởng. Chỉ phần vị trí bên trong nội dung chỉnh sửa bị mất.
  - Nếu sửa: dùng tên operation mới để các run cũ vẫn tái lập được.
  - Số mẫu bị ảnh hưởng chưa được đo bằng script.
- **Tỉ lệ che của `mask_content` tính trên từ nội dung, không phải trên tổng số từ.** Instruction ngắn và từ nội dung chỉ chiếm một phần số từ, nên tỉ lệ che trên tổng số từ thấp hơn nhiều so với 20–40%. Lúc xác nhận ngày 05/11, tôi ghi "20–40% số từ", không nói rõ điểm này. Anh giữ nguyên như hiện tại. Tỉ lệ trên tổng số từ chưa được đo bằng script.
- Quy tắc loại của `flip` hơi chặt: dấu nháy sở hữu cách (person's) và một số chữ số như "2D" bị loại sai.
- `blur` loại cả câu chỉ tình cờ có từ "clear" hoặc "detail".
- Relevance Head không học gì từ Unlabeled, và tập metric S của Student được chọn trên input đã augment. Chi tiết: `stage2_tuning.md` mục 8.6, `pipeline_process.md` mục 3.12.
