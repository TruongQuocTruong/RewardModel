# QUY TRÌNH PHÁT TRIỂN PIPELINE REWARD MODEL 6 METRICS

Trạng thái tài liệu: 02/11/2026. Bản trước khi sắp xếp lại: `backups/pipeline_process.before_restructure.md`.

Tài liệu gồm bốn mục:

0. **Proposal gốc của Giai đoạn 2**: thiết kế ban đầu, giữ nguyên để so sánh với pipeline đã triển khai.
1. **Thành phần pipeline**: tên các module đã thống nhất, kiến trúc, công thức loss, cách đánh giá, file và launcher.
2. **Các hướng triển khai trước đây (đã cũ)**: setting 3k và các phiên bản Giai đoạn 2 đã chạy, chỉ giữ setting và kết luận.
3. **Hướng hiện tại**: bám mục tiêu cốt lõi, gồm các bước đã làm và đang làm.

Tài liệu chi tiết riêng:

- `inspector_ablation.md`: cấu hình Overall Inspector Model (setting 3k).
- `student_ablation.md`: cấu hình Student Model warm-start (setting 3k).
- `tie_head_ablation.md`: chọn dạng Tie Head cho Overall Inspector và Student.
- `AUC-Margin.md`: quy chuẩn margin AUC cho tập có Tie và không có Tie.

Proposal là giả thuyết nghiên cứu cần được kiểm chứng bằng thực nghiệm, không phải bằng chứng rằng pipeline chắc chắn cải thiện mô hình. Tiêu chí thành công là **Student Model**; kết quả tốt hơn của Teacher Model không được xem là thành công nếu Student không cải thiện.

Quy tắc làm việc:

- Mỗi thực nghiệm chỉ khác phiên bản tham chiếu đúng một thay đổi.
- Mọi job đều chạy qua Slurm, không chạy job nặng trên login node.
- Mọi thay đổi file, tên thư mục hoặc job đều phải được đề xuất và xác nhận trước khi thực hiện.
- Dự đoán bộ nhớ GPU trước mỗi lần submit và chọn partition nhỏ nhất đủ dùng.

---

## 0. Proposal gốc của Giai đoạn 2 (trạng thái 10/10/2026, giữ nguyên để so sánh)

Mục này giữ nguyên thiết kế Giai đoạn 2 ban đầu, không cập nhật theo các thay đổi sau. Dùng làm mốc để so sánh pipeline đã triển khai khác proposal ở đâu (bảng so sánh ở mục 3.11).

**Đầu vào từ Giai đoạn 1**
- Overall Inspector nhị phân (scalar head) train trên HumanGT non-tie.
- Overall Pseudo-label Cache có ngưỡng chấp nhận từ calibration (0.7295). Mỗi mẫu được gán A, B hoặc Abstain; 62% số mẫu là Abstain.
- Student warm-start trên GemmaGT; Teacher là bản sao chính xác của Student.
- Metrics Weights Aggregator \((a_k,\beta)\) đã fit và đóng băng.

**Mỗi round**
1. **Teacher screening:** một lượt Teacher không dropout trên tập ứng viên (4 × ngân sách round). Mẫu Unlabeled được chia thành ba nhóm:
   - *high agreement:* Inspector confidence ≥ 0.80 và được chấp nhận, Teacher cùng chiều Inspector với margin ≥ κ;
   - *low agreement:* chiếm 25%;
   - *box-only:* Inspector Abstain, tối đa 25%.

   Cộng thêm 25% mẫu ôn lại khó từ round trước, mỗi mẫu tối đa 2 lần.
2. **Batch Mixing Controller:** 5 HumanGT + 5 GemmaGT + 6 Unlabeled mỗi optimizer step; HumanGT và GemmaGT mỗi nguồn ít nhất 15%.
3. **Teacher MC-Dropout:** J = 4 lần trên batch Unlabeled, dựng box \(\mathcal B_i\) (nới 0.02). Half-space \(y_i\sum_kw_k(P_{k,A}-P_{k,B})\ge\kappa\) với κ = 0.05, chỉ thêm khi Inspector chấp nhận nhãn.
4. **Metrics and Overall Projection:** \(P_i^*=\arg\min_{P\in\mathcal C_i}\sum_kD_{KL}(P_k\parallel p^{student}_{ik})\). Set loss bằng 0 khi Student nằm trong miền; ngoài miền, \(P^*_i\) được detach làm target. KL lấy trung bình trên 6 metric; box-only nhân 0.25.
5. **Composite Loss Objective:** \(\mathcal L_{\text{human}}+\mathcal L_{\text{Gemma}}+\mathcal L_{\text{mask}}+\lambda_{\text{set}}(t)\,\mathcal L_{\text{set}}\).
   - \(\mathcal L_{\text{human}}\) là BCE trên \(\sigma(\beta\sum_ka_ks_k)\), với `tie_probs` detach.
   - \(\lambda_{\text{set}}\) tăng tuyến tính trong 20% số step đầu, nhân thêm hệ số thích nghi `set_scale` trong khoảng [0.5, 4] (momentum 0.5).
6. **EMA Parameter Updater:** cập nhật Teacher sau mỗi optimizer step (giá trị decay gốc không còn lưu).
7. **Cuối round:**
   - Đánh giá **Teacher** trên HumanGT validation, chọn `best.pt` theo Teacher.
   - Semantic drift: Brier trên Gemma validation tăng quá 0.02 thì GemmaGT / mask replay nhân 1.5 ở round sau.
   - Dừng khi đủ 5 round, sau 2 round không tăng quá 0.002, hoặc khi số mẫu qua lọc dưới 50% ngân sách.

**Những gì proposal gốc không có:**
- loss trực tiếp từ nhãn Inspector;
- KL(Teacher ‖ Student);
- đánh giá Student trong loop;
- supervised control cùng số step.

Inspector chỉ dùng để xếp hạng, chia nhóm và xác định chiều half-space.

---

## 1. Thành phần pipeline và danh pháp đã thống nhất

### 1.1. Sáu metrics

Mọi tensor sáu chiều tuân theo thứ tự cố định:

1. `aesthetic`
2. `fidelity`
3. `realism`
4. `background_consistency`
5. `foreground_consistency`
6. `structure_consistency`

### 1.2. Nguồn dữ liệu và tập đánh giá

| Tên | Nội dung |
|---|---|
| **HumanGT Data** | Cặp ảnh có nhãn sở thích của người (A / B / Tie). |
| **GemmaGT Data** | Cặp ảnh có nhãn 6 metrics, tie và relevance do Gemma-31B sinh ba lượt, trên đúng các cặp HumanGT. |
| **Unlabeled Data** | Cặp ảnh không có nhãn, dùng trong training loop Giai đoạn 2. |

File dữ liệu (`TrainingDataCreation/`):

| File | Vai trò |
|---|---|
| `editreward_overall_humanGT_3k.jsonl` | HumanGT: 3,000 cặp, trong đó 644 tie. |
| `editreward_overall_finegrained_gemmaGT_3k.jsonl` | GemmaGT: đúng 3,000 cặp HumanGT, ba lượt mỗi cặp (gom bằng `gemmaGT_data_creation.py`). |
| `editreward_unlabeled_10k.jsonl` | Unlabeled: 10,000 cặp. |
| `editreward_overall_humanGT_test_1k.jsonl` | EditReward-Data-Test: 1,000 cặp không hoà. |

- **Toàn bộ EditReward-Data**: `/datastore/shared/datasets/EditReward-Data/extracted_data/metadata.jsonl`, 170,772 cặp (73,096 A / 62,353 B / 35,323 Tie), 8,132 ảnh nguồn (trung bình khoảng 21 cặp mỗi ảnh nguồn), 12 model chỉnh sửa. 1,000 cặp Test nằm trong file này.
- **EditReward-Bench**: 133 cặp, trong đó 115 cặp không hoà và 18 cặp hoà.
- **VLM Bench**: cùng 133 cặp Bench, nhãn do VLM gán lại (đa số 15 phiếu). Đây là bản gán lại duy nhất được dùng khi đánh giá.
- Prompt sinh GemmaGT cho Gemma xem lựa chọn tổng thể của người làm ngữ cảnh, nhưng yêu cầu không sao chép lựa chọn đó vào nhãn từng metric. Nhãn Gemma vì vậy chứa thông tin về phiếu của người, nhưng không rò rỉ vào đánh giá vì Bench nằm ngoài dữ liệu train.

### 1.3. Các module

| Module | Vai trò |
|---|---|
| **Overall Inspector Model** | Reward Model tổng thể ba lớp (Score Head + Tie Head), học từ nhãn người A / B / Tie, sinh nhãn tổng thể ba lớp cho Unlabeled Data. |
| **Overall Pseudo-label Cache** | Bảng nhãn tổng thể ba lớp đóng băng, do Overall Inspector sinh offline một lần cho Unlabeled Data: P(A), P(B), P(Tie) và quyết định A / B / Tie (mục 1.5.2). |
| **Student Model** | Mô hình 6 metrics nhận gradient. Đây là mô hình chính cần cải thiện. |
| **Teacher Model** | Bản sao EMA của Student, không nhận gradient. Làm target KL(Teacher ‖ Student); khi chia round còn xếp hạng mẫu Unlabeled và chọn mẫu ôn lại. |
| **Metrics Weights Aggregator** | Bộ trọng số đóng băng \((a_1,\ldots,a_6,\beta)\) fit trên Student-Stage1. Hiện chỉ dùng cho kết quả phụ và phân tích (mục 2.4). |
| **Batch Mixing Controller** | Trộn HumanGT, GemmaGT và Unlabeled trong mỗi optimizer step. |
| **Metrics and Overall Projection** | Dựng miền ràng buộc và thực hiện I-Projection. Chỉ là nhánh thực nghiệm, chưa dùng ở phiên bản nào. |
| **Composite Loss Objective** | Kết hợp các loss để cập nhật Student. |
| **EMA Parameter Updater** | Đồng bộ mềm trọng số Student sang Teacher sau mỗi optimizer step. |

Không dùng các tên `aspect`, `bridge`, `anchor`, `refinement` hoặc tên tự đặt khác cho module, file, log hay tài liệu mới. Ngoại lệ còn tồn tại: các biến môi trường `REFINEMENT_*` của `run_train_loop.slurm` là tên cũ.

### 1.4. Tên mô hình theo vai trò và thuật ngữ

| Tên | Ý nghĩa |
|---|---|
| **Inspector Baseline** | Overall Inspector train có giám sát trên toàn bộ EditReward-Data. Là đối chứng và mốc cần đạt của pipeline. |
| **Inspector pipeline** | Overall Inspector dùng trong pipeline đề xuất. Không train riêng: là checkpoint của Inspector Baseline tại một mốc ít cặp (mục 3.3). |
| **Student-Stage1** | Student sau warm-start Giai đoạn 1, dùng để khởi tạo Giai đoạn 2. |
| **Student pipeline / Teacher pipeline** | Student và Teacher của pipeline đề xuất. |
| **Soft Inspector label** | Phân phối ba lớp \((P(A),P(B),P(\text{Tie}))\) của Inspector trên mỗi mẫu Unlabeled. Setting 3k cũ dùng Inspector nhị phân nên chỉ có `probability_A` (mục 2.2). |
| **Hard Inspector label** | Quyết định ba lớp ∈ {A, B, Tie}, là lớp có xác suất lớn nhất. Chỉ dùng ở nhánh ablation. |
| **Soft Gemma target** | Tỉ lệ phiếu A/B/Tie và relevance trung bình qua ba lượt Gemma. Chỉ dùng để đo Gemma validation. |
| **Majority-vote Gemma target** | Nhãn one-hot theo đa số trên ba lượt Gemma: metric liên quan khi ít nhất 2/3 lượt đánh dấu; A/B/Tie cần hơn 50% số lượt áp dụng, không có đa số thì gán Tie. Setting cố định khi train Student ở cả hai giai đoạn. |
| **Round** | Một vòng của training loop: chọn mẫu, train, đánh giá, quyết định giữ hoặc dừng. |
| **Semantic guard** | Điều kiện chặn chọn checkpoint nếu Brier trên Gemma validation bị drift (mục 1.6.4). |
| **`selected_equal`** | Overall prediction chỉ gộp các metric được Relevance Head chọn, trung bình đều `signed_evidence` (mục 1.7). |

Tên phiên bản là tên thư mục output trong `output_training/`; thư mục kết quả đánh giá trong `RewardModelEvaluation/OurModel/evaluation_results/` có cùng tên. Không dùng ký hiệu viết tắt cho phiên bản.

### 1.5. Kiến trúc mô hình

#### 1.5.1. Backbone

- Backbone `Qwen3-VL-Reranker-8B` (`/datastore/truongtq/TrainingRewardModel/Models/Qwen3-VL-Reranker-8B`). Toàn bộ quá trình train dùng MS-Swift 4.5.3.
- LoRA dropout 0.05, `lora_alpha` luôn bằng 2 × rank.
  - Overall Inspector: rank 32, chỉ trên LLM.
  - Student: rank 64, trên LLM + ViT + aligner, không gắn vào các head.
- Không đặt `max_pixels` ở bất kỳ bước nào; processor tự resize theo mặc định.
- Biểu diễn của mỗi ảnh là hidden state của token hợp lệ cuối.
- Mỗi cặp của Student tạo ba dòng backbone: candidate A, candidate B và source. Mỗi cặp của Inspector tạo hai dòng (A và B).
- Prompt lấy từ `prompts/reward_model_template.json`, với hai profile `six_metric` và `overall_inspector`.

#### 1.5.2. Overall Inspector Model

- LoRA độc lập với Student. Head ba lớp (từ 24/10/2026): Score Head một đầu ra + Tie Head một đầu ra, không có Relevance Head. Không nhìn trực tiếp sáu metrics. Train trên mọi cặp, giữ cặp hoà.
- **Score Head:** điểm của mỗi ảnh lấy từ phép chiếu khởi tạo đúng bằng \(W_\text{yes}-W_\text{no}\) của LM head, phần residual khởi tạo bằng 0, nên ban đầu \(d\) trùng điểm gốc của reranker. \(d\) = điểm A − điểm B, \(w=\sigma(d)\) là xác suất A thắng nếu cặp không hoà.
- **Tie Head `margin_coupled`** (cùng cấu trúc với Student, mục 1.5.3, một đầu ra):

  \[
  \operatorname{logit}(t)=\theta_0-\kappa|d|+r(h_A,h_B).
  \]

- **Phân phối ba lớp:**

  \[
  P(A)=(1-t)\,w,\qquad P(B)=(1-t)(1-w),\qquad P(\text{Tie})=t,
  \]

  quyết định là lớp có xác suất lớn nhất; margin dùng để đánh giá là \(d\) hoặc `signed_evidence` = P(A) − P(B).
- Sau khi chọn checkpoint, toàn bộ mô hình được đóng băng để suy luận offline.
- **Overall Pseudo-label Cache** (sinh một lần cho Unlabeled Data, không calibration). Mỗi mẫu lưu:

  | Trường | Ý nghĩa |
  |---|---|
  | `probability_A`, `probability_B`, `probability_Tie` | \(P(A),P(B),P(\text{Tie})\) ba lớp |
  | `cond_win_probability_A` | \(w=\sigma(d)\), xác suất A thắng nếu không hoà |
  | `decision`, `pseudo_label` | Lớp có xác suất lớn nhất: A / B / Tie (tương ứng 1 / −1 / 0) |
  | `signed_evidence` | \(P(A)-P(B)\) |
  | `margin_A_minus_B`, `score_A`, `score_B` | \(d\) và điểm từng ảnh |

  Giai đoạn 2 dùng target \(q=P(A)+\tfrac12P(\text{Tie})\) (mục 3.6); cách dùng P(Tie) cho loss ba lớp chưa chốt (mục 3.7).
- Bản nhị phân cũ (listwise, bỏ cặp hoà, cache chỉ có `probability_A` theo ngưỡng 0.5) nằm trong `scripts/backup_binary_inspector/`; các kết quả Inspector và cache của setting 3k (mục 2.2) là của bản cũ.

#### 1.5.3. Student Model

Thông số head: hidden dim 512, tie projection dim 256, dropout 0.05, orthogonal init scale 0.35. `lm_head` gốc được giữ nguyên.

**Score Head**

\[
\operatorname{scores}_I=W_{\text{pre}}h_I+W_{\text{res}}\,\operatorname{Dropout}(\operatorname{act}(W_hh_I)),\quad I\in\{A,B\},\qquad
d=\operatorname{scores}_A-\operatorname{scores}_B,\quad w=\sigma(d).
\]

- \(W_\text{pre}\) khởi tạo từ \(W_\text{yes}-W_\text{no}\) cộng thành phần trực giao (scale 0.35) để sáu metrics không trùng hướng; \(W_\text{res}\) khởi tạo bằng 0.
- A và B dùng chung một dropout mask, nên margin phản đối xứng chính xác kể cả khi chạy MC-Dropout.
- Mọi loss chỉ dùng hiệu A − B, nên mức tuyệt đối của điểm không có ý nghĩa.

**Tie Head `margin_coupled` (chốt 26/10/2026)**

\[
u_I=W_ph_I,\quad
z=\left[\tfrac{u_A+u_B}{2};|u_A-u_B|\right],\quad
\operatorname{logit}(t_k)=\theta_{0,k}-\kappa_k\,|d_k|+r_k(z),\quad \kappa_k=\operatorname{softplus}(\cdot)>0.
\]

- \(\theta_0\) khởi tạo bằng logit(0.2), \(\kappa\) bằng 1.0, lớp cuối của residual \(r\) bằng 0, nên ban đầu \(t=\sigma(\theta_0-\kappa|d|)\), đúng dạng Rao-Kupper.
- Gradient đi qua \(d\): cặp hoà kéo \(|d|\) về 0, nên cặp hoà cũng gửi gradient vào Score Head.
- \(t\) không đổi khi hoán vị A/B. Overall Inspector dùng cùng cấu trúc với một đầu ra.
- **Lỗi đã phát hiện (28/10/2026, mục 3.5):** các head được train ở bf16, nên trong mọi run đến nay \(\theta_0\) và \(\kappa\) không được cập nhật (đứng yên ở logit(0.2) và 1.0), và khoảng 65% phần tử của phép chiếu Score Head và lớp đầu MLP của Tie Head cũng đứng yên. Dạng thực tế đã chạy là \(\theta_0,\kappa\) cố định. Các run mới phải giữ tham số head ở fp32.
- Chọn qua `TIE_HEAD_MODE`. **Mặc định trong code vẫn là `independent`** (\(t=\sigma(r(z))\), không phụ thuộc \(d\)), nên mọi run mới phải đặt `TIE_HEAD_MODE=margin_coupled`. Các script nạp checkpoint tự nhận dạng head.

**Relevance Head**

\[
\mu=\sigma(f_{\text{rel}}(h_{\text{src}}))\in[0,1]^6.
\]

Head chỉ nhìn source và edit instruction, nên hai ảnh của một cặp luôn có cùng tập metric được chọn. Multi-label, sigmoid độc lập, chỉ nhận gradient từ relevance loss trên GemmaGT.

**Đầu ra của Student và Teacher** (mỗi đại lượng `[B, 6]`):

\[
p_k=\big((1-t_k)w_k,\;(1-t_k)(1-w_k),\;t_k\big),\qquad
s_k=(1-t_k)(2w_k-1)\in[-1,1]\;\;(\texttt{signed\_evidence}).
\]

Metric được chọn: \(S=\{k:\mu_k\ge0.5\}\); nếu rỗng, lấy metric có relevance cao nhất. Kiểm định bắt buộc: margin phản đối xứng, tie bất biến khi đổi A/B, tổng ba xác suất bằng 1.

#### 1.5.4. Teacher Model

- Clone chính xác từng tensor của Student-Stage1 (`teacher_initialized.pt`). Không nhận gradient, không có optimizer.
- Trong training loop, Teacher không phải mô hình thứ hai trên GPU: trọng số EMA được hoán đổi tại chỗ khi cần một lượt Teacher.
- Sau mỗi optimizer step: \(\bar\theta\leftarrow\alpha\bar\theta+(1-\alpha)\theta\).

#### 1.5.5. Metrics Weights Aggregator

\[
F_\phi(s_i)=\sigma\Big(\sum_k\gamma_ks_{ik}\Big),\qquad \gamma_k=\beta a_k\ge0,\qquad \sum_ka_k=1.
\]

Fit một lần sau Giai đoạn 1 trên `signed_evidence` của Student-Stage1 (HumanGT non-tie, chia 80/20 theo ảnh nguồn): hồi quy logistic không âm, không intercept, L2 = 0.01, bootstrap 100 lần theo nhóm source, rồi đóng băng. Không cần khi Giai đoạn 2 dùng `selected_equal` với P(A) tuyến tính (mục 1.6.2).

### 1.6. Loss

#### 1.6.0. Vai trò của Score Head và Tie Head trong loss

Mọi loss có nhãn ba lớp (Inspector, GemmaGT, HumanGT/Unlabeled ở Giai đoạn 2) đều dựa trên phân phối \(P(A)=(1-t)w,\ P(B)=(1-t)(1-w),\ P(\text{Tie})=t\). Với một đầu ra, \(-\log P(\text{nhãn})\) tách thành hai thành phần:

\[
\underbrace{\operatorname{BCE}(y_T,t)}_{\text{thành phần Tie}}+\underbrace{(y_A+y_B)\operatorname{BCE}\Big(\tfrac{y_A}{y_A+y_B},\sigma(d)\Big)}_{\text{thành phần chọn bên thắng (Bradley–Terry trên }d)}
\]

- **Không có loss riêng cho điểm A hay điểm B.** Hai điểm chỉ vào loss qua \(d\) = điểm A − điểm B; cộng cùng một hằng số vào cả hai thì loss không đổi.
- **Thành phần chọn bên thắng** quyết định *hướng* của \(d\). Cặp hoà có trọng số \(y_A+y_B=0\) nên không đẩy hướng của \(d\). Thành phần này không có gradient vào \(\theta_0,\kappa\) và MLP Tie.
- **Thành phần Tie** train \(\theta_0,\kappa\), MLP residual của Tie Head, và qua `margin_coupled` cả *độ lớn* \(|d|\):

  \[
  \frac{\partial\,\operatorname{BCE}(y_T,t)}{\partial|d|}=-\kappa\,(t-y_T).
  \]

  Cặp hoà (\(y_T=1\)) kéo \(|d|\) nhỏ lại; cặp không hoà (\(y_T=0\)) đẩy \(|d|\) lớn ra. Thành phần này không quyết định A hay B thắng.

#### 1.6.1. Giai đoạn 1

**Overall Inspector**, với nhãn one-hot của người \(y=(y_A,y_B,y_T)\), giữ cặp hoà:

\[
\mathcal L_{\text{insp}}=-\log P(\text{nhãn đúng})=\operatorname{BCE}(y_T,t)+(y_A+y_B)\operatorname{BCE}\Big(\tfrac{y_A}{y_A+y_B},w\Big).
\]

Một đầu ra, BCE tính trên logit, trung bình theo batch.

**Student warm-start** (`train_student_hardlabel.py`), với Majority-vote Gemma target \(g=(g_A,g_B,g_T)\) và relevance target \(\bar m_k\) trên từng metric:
- \(\bar m_k=1\) khi ít nhất 2/3 lượt Gemma đánh dấu metric áp dụng được, ngược lại \(\bar m_k=0\) và \(g=0\).
- Khi \(\bar m_k=1\): \(g\) là one-hot của lựa chọn chiếm quá bán trong các lượt áp dụng được; không có quá bán thì là Tie.
- Validation giữ nhãn mềm (tỉ lệ phiếu trên các lượt áp dụng được, relevance = số lượt áp dụng / 3).

\[
\ell_{\text{pref},k}=\operatorname{BCE}(g_T,t_k)+(g_A+g_B)\operatorname{BCE}\Big(\tfrac{g_A}{g_A+g_B},w_k\Big),\qquad
\mathcal L_{\text{mask}}=\tfrac16\sum_k\operatorname{BCE}(\bar m_k,\mu_k),
\]
\[
\mathcal L_{\text{warm}}=\tfrac16\sum_k\bar m_k\,\ell_{\text{pref},k}+\lambda\mathcal L_{\text{mask}},\qquad \lambda=1\ (\texttt{WARMSTART\_RELEVANCE\_WEIGHT}),
\]

lấy trung bình theo cặp.
- Phần preference nhân với relevance target nên metric không liên quan chỉ train Relevance Head, không bị coi là hoà. Loss đã theo từng metric nên không cần cách gộp overall.
- So với Inspector: có sáu đầu ra, trọng số \(\bar m_k\) và thêm Relevance Head. BCE tính trên xác suất (`F.binary_cross_entropy`), không trên logit; khi xác suất bão hoà gần 0/1, log bị chặn ở −100.

#### 1.6.2. Giai đoạn 2

**Overall prediction của Student**, với \(m_i=\frac{1}{|S|}\sum_{k\in S}s_{ik}\) (relevance detach):

\[
P_i(A)=\begin{cases}
\sigma\big(\beta\sum_{k=1}^{6}a_ks_{ik}\big) & \text{6 metric, trọng số Aggregator}\\
\sigma(\beta\,m_i) & \text{trung bình đều trên } S,\ \text{giữ }\beta\\
(1+m_i)/2 & \text{trung bình đều trên } S,\ \text{tuyến tính, không có }\beta
\end{cases}
\]

Với cách tuyến tính, logit tính bằng \(2\operatorname{atanh}(m)\), \(m\) bị chặn trong \([-1+10^{-6},\,1-10^{-6}]\).

**Human Preference Loss nhị phân** (`--overall-loss binary`, setting 3k cũ): \(\mathcal L_{\text{human}}=\operatorname{BCE}(h_i,P_i(A))\). Gradient chảy qua `signed_evidence` về Score Head; `tie_probs` bị detach; HumanGT tie không tham gia.

**Loss ba lớp** (`--overall-loss three_way`, dùng từ setting 1008 cặp). Phân phối overall của Student là trung bình đều trên \(S\) của phân phối ba lớp từng metric (relevance detach, `tie_probs` không detach):

\[
\bar p_i=\tfrac{1}{|S_i|}\sum_{k\in S_i}p_{ik},\qquad
\mathcal L_{\text{human}}=-\log\bar p_i[\text{nhãn người}],
\]

nhãn người one-hot, giữ cặp hoà. Khác với Inspector:
- \(-\log\) của một trung bình không tách thành hai thành phần như mục 1.6.0. Mỗi metric nhận gradient theo tỉ lệ \(p_{ik,c}/(|S_i|\,\bar p_{i,c})\), nên metric đang đồng ý với nhãn nhận phần lớn hơn.
- \(\bar p_i(\text{Tie})\) là trung bình \(t_k\) trên \(S\). Một nhãn hoà overall (từ người hoặc P(Tie) của Inspector) đẩy \(t_k\) của mọi metric trong \(S\) lên, và qua `margin_coupled` kéo các \(|d_k|\) nhỏ lại.
- Tie Head metric nhận nhãn từ ba nguồn có ý nghĩa khác nhau: Tie theo metric của Gemma, Tie overall của người, và Tie overall mềm của Inspector.

**Nhánh Unlabeled** (`inspector_soft_ema_consistency`, dùng ở mọi phiên bản đã chạy):

\[
\mathcal L_{\text{unlabeled}}=\operatorname{BCE}\big(q_i,P_i(A)\big)+\lambda_C\,D_{KL}\big(\bar p^{\,teacher}\parallel p^{student}\big),
\]

với \(q_i\) là `probability_A` của Inspector nhị phân (setting 3k cũ), \(\bar p^{\,teacher}\) là trung bình `three_way_probs` của 4 lần MC-Dropout Teacher; KL lấy trung bình trên 6 metric hoặc trên \(S\). Trọng số toàn bộ \(\mathcal L_{\text{unlabeled}}\) tăng tuyến tính từ 0 lên 1 trong 20% số step đầu. Với Inspector ba lớp và loss nhị phân, target cho nhánh này là \(q=P(A)+\tfrac12P(\text{Tie})\) (mục 3.6); cấu hình Giai đoạn 2 đầu tiên của setting 1008 cặp dùng loss ba lớp (bên dưới, mục 3.7). Các nhánh khác chưa dùng: Hard Inspector label (có hoặc không có KL) và Projection.

**Metrics and Overall Projection** (chưa dùng):

\[
P_i^*=\arg\min_{P\in\mathcal C_i}\sum_kD_{KL}(P_k\parallel p_{ik}^{student}),\qquad
\mathcal C_i=\mathcal B_i\cap\Big\{y_i\sum_kw_k(P_{k,A}-P_{k,B})\ge\kappa\Big\},
\]

với \(\mathcal B_i\) là box Teacher từ 4 lần MC-Dropout (nới 0.02), half-space chỉ thêm khi Inspector chấp nhận nhãn, κ = 0.05. Student trong miền thì loss bằng 0; ngoài miền thì \(P^*_i\) được detach làm soft target.

Với loss ba lớp, nhánh Unlabeled thay BCE bằng cross-entropy với phân phối ba lớp của Inspector, KL giữ nguyên:

\[
\mathcal L_{\text{unlabeled}}=-\sum_{c\in\{A,B,\text{Tie}\}}P_I(c)\log\bar p_i(c)+\lambda_C\tfrac{1}{|S_i|}\sum_{k\in S_i}D_{KL}\big(\bar p^{\,teacher}_{k}\parallel p^{student}_{k}\big).
\]

**Composite Loss Objective**:

\[
\mathcal L_{\text{total}}=\mathcal L_{\text{human}}+\lambda_G\mathcal L_{\text{Gemma}}+\lambda_M\mathcal L_{\text{mask}}+\lambda_U(\text{step})\,\mathcal L_{\text{unlabeled}},\qquad \lambda_G=\lambda_M=\lambda_C=1,
\]

- \(\mathcal L_{\text{Gemma}},\mathcal L_{\text{mask}}\) là loss warm-start trên GemmaGT replay, cùng nhãn Majority-vote như Giai đoạn 1 (`--gemma-targets majority_vote`).
- \(\lambda_U\) tăng tuyến tính từ 0 lên 1 trong 20% số step đầu (khi `REFINEMENT_SET_RAMP_STEPS=0`).
- Mỗi thành phần được chia cho số microbatch của step rồi cộng dồn gradient.
- Mỗi thành phần là **trung bình trên batch của nguồn đó**, nên trọng số được chia theo nguồn chứ không theo số cặp. Mỗi cặp nặng \(1/\text{batch của nguồn}\). Với 5 + 5 + 6 (mặc định cũ), một cặp HumanGT và một cặp Unlabeled nặng gần ngang nhau (1/5 và 1/6). Với 1 + 1 + 6 (setting 1008 cặp), một cặp HumanGT nặng gấp 6 lần một cặp Unlabeled, và toàn bộ Unlabeled chỉ chiếm 1 trong 4 hệ số loss (HumanGT, Gemma, mask, Unlabeled), dù chiếm 75% số cặp.
- Phần entropy \(H(P_I)\) của nhãn mềm Inspector là hằng số trong \(\mathcal L_{\text{unlabeled}}\) và không tạo gradient. Chỉ phần \(D_{KL}(P_I\parallel\bar p)\) kéo Student.
- Hệ số 1.5 cho GemmaGT replay khi có semantic drift chỉ áp dụng từ round sau; với `all_once` (1 round), các hệ số giữ bằng 1 trong suốt run.
- Teacher không nhận gradient từ loss nào, chỉ cập nhật bằng EMA.

#### 1.6.3. Chọn mẫu Unlabeled

128 mẫu Unlabeled được giữ riêng làm probe chẩn đoán; các mẫu cùng ảnh nguồn với probe bị loại khỏi train.

| Cách chọn | Mô tả |
|---|---|
| `all_once` | 1 round, không có Teacher chọn mẫu: toàn bộ Unlabeled được xáo và dùng đúng một lần. |
| `inspector_teacher_rank` | Nhiều round theo Inspector confidence giảm dần. Round 1 chỉ giữ mẫu Teacher cùng chiều Inspector, mẫu ngược chiều đưa sang đầu round 2. Từ round 2 cộng thêm mẫu ôn lại của round trước có margin Teacher nhỏ nhất. |
| `confidence_order`, `agreement_strata` | Chế độ cũ (trước 19/10), không còn dùng. |

#### 1.6.4. Batch, chọn checkpoint và tham số chung Giai đoạn 2

- Batch Mixing Controller: mặc định 5 HumanGT + 5 GemmaGT + 6 Unlabeled = 16 cặp mỗi step; HumanGT và GemmaGT mỗi nguồn ít nhất 15%. Khi có semantic drift, GemmaGT replay được nhân trọng số 1.5 ở round sau.
- Chọn `best.pt` theo accuracy của **Student** trên HumanGT validation, cùng cách gộp với loss. Chỉ nhận best khi accuracy tăng hơn 0.002 và semantic guard đạt (Brier tie, winner, mask trên Gemma validation không vượt mức ban đầu quá 0.02). Teacher chỉ dùng để chẩn đoán.
- Đánh giá và lưu mỗi 200 step, giữ 5 checkpoint (không tính best), và đánh giá cuối mỗi round. Dừng khi hết số round, sau 2 round không cải thiện, hoặc khi số mẫu qua lọc quá ít.
- Learning rate 1e-5 cosine, warmup 5%; AdamW β = (0.9, 0.95); weight decay 0.01; grad clip 1.0; 4 lần MC-Dropout; seed 42. EMA decay 0.99 cố định, hoặc cosine theo step 0.99 → 0.999.

### 1.7. Đánh giá

**Overall prediction (`selected_equal`, chốt 20/10/2026).** Dấu của \(m=\frac{1}{|S|}\sum_{k\in S}s_k\): dương thì A tốt hơn. Trung bình lấy trên `signed_evidence` (bị chặn trong [−1, 1], có tính Tie Head), không trên margin thô. Kết quả phụ: `selected_fitted` (\(a_k\) giới hạn trên S) và `all_metrics_fitted` (\(\beta\sum_k a_ks_k\)). Inspector dùng \(d\) (hoặc `signed_evidence` = P(A) − P(B)).

**Margin AUC đối xứng là AUC chính (chốt 21/10/2026).** Với nhãn \(y_i\in\{+1,-1\}\) và margin \(m_i\), đặt \(d_i=y_im_i\). Margin AUC là ROC AUC của \(\{d_i\}\) so với \(\{-d_i\}\), bằng \(P(d_i+d_j>0)\), không phụ thuộc thứ tự A/B.

| Tập | Chỉ số |
|---|---|
| Không có Tie: Test (1000), Bench(115), VLM Bench(115) | Accuracy / margin AUC đối xứng |
| Có Tie: Bench(133), VLM Bench(133) | Cặp hoà là lớp dương. **Margin AUC** cho điểm −\|d\|; **Tie Head AUC** cho điểm P(Tie). Model không có Tie Head ghi "—". |

- Khoảng tin cậy 95%: 1,000 lần lấy mẫu lặp theo cặp, seed 42. So sánh hai model dùng McNemar và chênh lệch margin AUC theo cặp.
- Bench chỉ có 115 cặp, khoảng tin cậy rộng khoảng ±8 điểm; một cặp là 0.87 điểm accuracy.
- Margin AUC không đo calibration; khi cần thì báo thêm Brier và log loss.
- Tập so sánh với Inspector Baseline chỉ là Bench, vì Test nằm trong dữ liệu train của Inspector Baseline.

### 1.8. File và launcher

Source code ở `Qwen3-VL-Embedding/`, launcher ở `TrainingRewardModel/`, đánh giá ở `RewardModelEvaluation/OurModel/`.

| Chức năng | File | Launcher, partition |
|---|---|---|
| Kiến trúc Inspector, Student, Teacher và loss | `src/models/custom_qwen3_vl_reranker.py` | — |
| Train Overall Inspector | `scripts/train_overall_inspector.py` | `run_overall_inspector_training.slurm`, full |
| Sinh Overall Pseudo-label Cache (head ba lớp từ 25/10: P(A), P(B), P(Tie), d, `signed_evidence`) | `scripts/infer_overall_inspector.py` | `run_overall_inspector_inference.slurm`, normal |
| Student warm-start (Majority-vote Gemma target) | `scripts/train_student_hardlabel.py`, kế thừa `scripts/train_student_warmstart.py` | `run_student_hardlabel_training.slurm`, normal |
| Clone Teacher | `scripts/initialize_teacher_model.py` | `run_teacher_model_initialization.slurm`, small |
| Fit Metrics Weights Aggregator | `scripts/fit_preference_aggregator.py` | `run_preference_aggregator.slurm`, normal |
| Training loop Giai đoạn 2 và supervised control | `scripts/train_loop.py` | `run_train_loop.slurm`, full (thiếu dòng `--qos=mig7-141`, truyền khi `sbatch`) |
| Đánh giá Inspector | `evaluate_inspector_checkpoints.py` | `run_evaluate_inspector_checkpoints.slurm`, normal |
| Đánh giá Student / Teacher | `evaluate_student_checkpoints.py` | `run_evaluate_student_checkpoints.slurm`, normal |
| Tính lại accuracy, margin AUC, McNemar | `recompute_margin_auc.py` | — |
| Hình và bảng một run Inspector | `analyze_inspector_training.py` | — |

- Quy ước tài nguyên (QOS bắt buộc, cùng tên loại MIG): small 1g.18gb `--qos=mig1-18`, normal 2g.35gb `--qos=mig2-35`, large 3g.71gb `--qos=mig3-71`, full 7g.141gb `--qos=mig7-141`. `--qos=normal` bị từ chối.
- Bộ nhớ đỉnh đã đo: Inspector Baseline khoảng 97–121 GiB (full); Student warm-start khoảng 30 GiB (normal); training loop 67.2 GiB (full, không vừa large).
- Kết quả đánh giá: `evaluation_results/inspector/`, `evaluation_results/student/`, `evaluation_results/train_loop/`, thư mục con trùng tên thư mục training.
- Bản sao lưu code trước mỗi lần sửa: `scripts/backups/`.

### 1.9. Nguyên tắc thực nghiệm và tiêu chí giữ một thay đổi

1. Student validation là tiêu chí chọn checkpoint. Teacher và tập ngoài chỉ dùng để chẩn đoán và báo cáo.
2. Mỗi thực nghiệm khác phiên bản tham chiếu đúng một thay đổi, cùng split, seed, batch và số optimizer step.
3. Không thêm module nếu chưa có thực nghiệm chứng minh lợi ích. Không tune trên Test hay Bench.

Một thay đổi chỉ được giữ nếu:

1. Student tốt hơn supervised control (chỉ HumanGT + GemmaGT) với cùng số optimizer step.
2. Cải thiện không chỉ nằm ở Teacher.
3. Không làm giảm đáng kể kết quả trên tập ngoài.
4. Kết quả lặp lại qua nhiều seed, hoặc chênh lệch đủ lớn so với nhiễu của tập đánh giá nhỏ.
5. Lợi ích tương xứng với chi phí MC-Dropout.

---

## 2. Các hướng triển khai trước đây (setting 3k, đã cũ)

Các mục dưới đây dùng 3,000 cặp HumanGT / GemmaGT, Tie Head `independent`. Chỉ giữ setting và kết luận đã chuyển thành cài đặt hiện tại.

### 2.1. Giai đoạn đầu (đến 18/10, checkpoint đã xoá)

| Hướng | Kết luận |
|---|---|
| GemmaGT trên các cặp khác HumanGT (đóng 13/10) | Loss của Metrics and Overall Projection chỉ khoảng 1.5–1.8% total loss → bỏ projection khỏi baseline, dùng Inspector làm supervision trực tiếp. Teacher tốt hơn nhưng không truyền về Student → thêm KL(Teacher ‖ Student), chọn checkpoint theo Student, thêm semantic guard. |
| Split dùng chung có calibration (12–16/10) | Calibration đẩy ngưỡng chấp nhận lên 0.893, Inspector chỉ phủ khoảng 10% Unlabeled; split dùng chung tốn khoảng 31% dữ liệu của Inspector. Từ 16/10: Inspector và Student train độc lập, không calibration, Soft Inspector label. |
| Giai đoạn 2 nhiều round trên Student soft target (job 3151, 3237, 3250, 3575) | Round 1 thường tốt nhất; không chênh lệch nào trên Bench có ý nghĩa. Các run này không có gradient ở ViT LoRA (mục 2.7). |
| Cấu hình Giai đoạn 1 (17–18/10) | Hard label (Majority-vote Gemma target), resize mặc định và lưu checkpoint như Inspector trở thành setting cố định của Student. |

### 2.2. Giai đoạn 1 setting 3k

| Bước | Cách chia | Train / validation |
|---|---|---|
| Overall Inspector | MS-Swift 0.15 trên HumanGT non-tie | khoảng 2,003 / 353 |
| Student warm-start | Ngẫu nhiên theo cặp 0.15, seed 42, giữ tie | 2,550 / 450 |
| Metrics Weights Aggregator | Theo ảnh nguồn 0.20, seed 42, HumanGT non-tie | 1,863 / 493 |
| Giai đoạn 2 | HumanGT theo split của Inspector, GemmaGT theo split của Student | như trên |

- **Overall Inspector** `overall_inspector_r32_llm_lr8e-5`, checkpoint-200 (bản nhị phân): rank 32 chỉ LLM, 8e-5 cosine, 2 epoch, batch 8. Test 74.50 / margin AUC 82.41, Bench 76.52 / 79.81. Cache nhị phân `editreward_inspector_pseudo_labels_10k.jsonl` (ngưỡng 0.5, 5,005 A / 4,995 B, chỉ có `probability_A` và `confidence`).
- **Student-Stage1** `student_hardlabel_r64_lr4e-5`, checkpoint-600: rank 64, 4e-5 cosine, 2 epoch (638 bước), batch 8, loss Gemma validation 0.6687. Test 75.10 / 81.88, Bench 77.39 / 80.72. Aggregator: \(a=[0.2245, 0.2151, 0.2378, 0.0706, 0.1120, 0.1400]\), β = 2.845.
- Giữ Student rank 64, 4e-5 dù rank 32, 8e-5 có loss validation thấp hơn (0.6221): trên tập ngoài không khác biệt (McNemar p = 0.63), và đổi Student kéo theo nhiều thay đổi cùng lúc.
- Phát hiện sau này: 306/353 cặp HumanGT validation nằm trong GemmaGT train. Đây là một lý do chuyển sang hướng hiện tại (mục 3.3).

### 2.3. Giai đoạn 2: ba phiên bản đã chạy

Thành phần chung: Inspector checkpoint-200 và cache ở trên; Student-Stage1 checkpoint-600; Teacher clone; nhánh Unlabeled `inspector_soft_ema_consistency`; batch 5 + 5 + 6; LoRA rank 64 trên LLM + ViT + aligner; chọn `best.pt` trên 353 cặp HumanGT validation (mỗi cặp khoảng 0.28 điểm).

| | `train_loop_single_pass` | `train_loop_single_pass_selected_equal` | `train_loop_curriculum_selected_equal` |
|---|---|---|---|
| Job / ngày | 4481 / 19/10 | 4953 / 21/10 | 5291 / 23/10 |
| Overall prediction khi train và chọn checkpoint | σ(β Σ a_k s_k), 6 metric | σ(β · m), trung bình đều trên S | **(1 + m)/2**, không có β |
| KL(Teacher ‖ Student) | 6 metric | 6 metric | **trên S** |
| Unlabeled | `all_once`, 9,222 mẫu | `all_once` | **3 round `inspector_teacher_rank`**: 3,000 / 3,000 / 3,222 mẫu mới + 750 mẫu ôn lại ở round 2 và 3 |
| EMA decay | 0.99 | 0.99 | **cosine 0.99 → 0.999** |
| Số step / thời gian | 1,537 / 19 giờ 10 phút | 1,537 / 19 giờ 48 phút | 1,787 / 25 giờ 40 phút |
| HumanGT validation tốt nhất | 84.70 (6 metric) | 86.12 | 87.25 |

Kết quả Student và Teacher trên tập ngoài (`selected_equal`, Accuracy / margin AUC):

| Model | Test (1000) | Bench (115) |
|---|---|---|
| Overall Inspector (checkpoint-200) | 74.50 / 82.41 | 76.52 / 79.81 |
| Student-Stage1 (checkpoint-600) | 75.10 / 81.88 | 77.39 / 80.72 |
| Student `train_loop_single_pass` | 76.60 / 83.18 | 77.39 / 79.30 |
| **Student `train_loop_single_pass_selected_equal`** | **77.20** / 83.36 | 75.65 / 81.19 |
| Student `train_loop_curriculum_selected_equal` | 75.30 / 82.96 | 77.39 / **81.73** |
| Teacher (ba phiên bản theo thứ tự trên) | 76.50 / 83.23; 77.00 / 83.54; 75.80 / 82.79 | 75.65 / 78.93; 77.39 / 81.35; 79.13 / 81.43 |

Kết luận:

1. `train_loop_single_pass_selected_equal` tốt nhất trên Test, hơn Student-Stage1 có ý nghĩa (McNemar 52 / 31, p = 0.028) và hơn Inspector (73 / 46, p = 0.017).
2. Chia 3 round kém train một lần trên Test có ý nghĩa (22 / 41, p = 0.023) dù HumanGT validation cao hơn. Phiên bản chia round khác cùng lúc 5 điểm nên chưa tách được nguyên nhân; ablation "giữ setting curriculum nhưng `all_once` và EMA 0.99" chưa chạy.
3. HumanGT validation 353 cặp chọn sai thứ tự so với Test; Bench 115 cặp không phân biệt được các phiên bản.
4. Teacher gần như trùng Student (Test: 21 / 16 cặp khác nhau), kể cả khi EMA lên 0.999.
5. Mức tăng của `train_loop_single_pass` nằm ở ba metric consistency và ở cặp khó (chênh lệch điểm người chấm ≤ 1: 66.8 so với 64.1).
6. Chưa có supervised control cùng số step (chế độ `supervised_control` vẫn gộp 6 metric bằng a_k), nên chưa tách được phần tăng do Inspector và Teacher. Mỗi phiên bản chỉ chạy một seed.

### 2.4. Metrics Weights Aggregator và cách tính AUC

- **Trọng số đã fit không có ích (kết luận 22/10/2026).** Trọng số đều cho kết quả tương đương ở mọi model, chênh lệch chỉ vài cặp và đổi chiều. \(a_k\) bị lẫn với relevance, nên khi Relevance Head đã chọn metric thì phần bù bị tính hai lần. β chỉ đổi thang; vì m bị chặn nên σ(β·m) ≤ 0.945, 9.2% mẫu Unlabeled có Inspector confidence cao hơn mức này không bao giờ khớp được → chuyển sang P(A) = (1 + m)/2.
- **Bỏ AUC gộp cũ.** AUC gộp so điểm ảnh giữa các cặp khác ảnh nguồn và instruction, nên bị chi phối bởi mức điểm riêng của từng ngữ cảnh. Margin AUC cao hơn AUC gộp khoảng 13–16 điểm trên Test hoàn toàn vì lý do này.
- **Dùng `signed_evidence`, không dùng margin thô:** đó là đại lượng model thực sự dùng để quyết định; AUC theo margin thô chỉ lệch tối đa 0.6 điểm. Margin AUC xấp xỉ \(p^2+2p(1-p)q\) (p là accuracy, q là xác suất cặp đúng tự tin hơn cặp sai), nên nhạy hơn accuracy khi so theo cặp.

### 2.5. Tie Head: từ `independent` đến `margin_coupled`

- **Phân tích job 5620 (`independent`, 90k cặp đầu).** Tie Head AUC tăng 0.58 → 0.71 nhưng chỉ ngang margin AUC (−\|d\|) 0.70: Tie Head học lại thông tin đã có trong \(d\). P(Tie) trên cặp Tie chỉ khoảng 0.27, và \(|d|\) của cặp Tie tăng dần (0.90 → 1.22).
- Nguyên nhân chính: \(t\) không phụ thuộc \(d\); cặp Tie không gửi gradient vào Score Head; Tie Head khởi tạo ngẫu nhiên vì không có hướng pretrained cho "hoà".
- **Ablation setting 3k** (`tie_head_ablation.md`, training 5838–5843, đánh giá 5983–5989; 4 dạng cho Inspector, 3 dạng cho Student) → chọn `margin_coupled` (26/10/2026):
  - Student: loss GemmaGT validation 0.6385 so với 0.6687; Tie Head AUC trên Bench(133) 66.4 so với 53.1; A/B trên Test và Bench không đổi.
  - Inspector: loss validation ngang `independent` nhưng ổn định nhất về cuối train.
  - Bị loại: `margin_coupled_detached` và `margin_only`.
- **Lưu ý (28/10/2026):** trong cả 6 run ablation, \(\theta_0\) và \(\kappa\) không được cập nhật do lỗi bf16 (mục 3.5). Kết luận trên là cho dạng \(\theta_0=\operatorname{logit}(0.2)\), \(\kappa=1\) cố định, chỉ residual và Score Head học; chưa kiểm chứng dạng có \(\theta_0,\kappa\) học được.

### 2.6. Inspector Baseline tham khảo (theo setting EditReward)

Hai job chạy trước khi chốt mục tiêu hiện tại: LoRA 1 lượt, 4e-5 cosine, warmup 0.05, batch 16, Tie Head `independent`. EditReward (arXiv 2509.26346) dùng full fine-tune, 2e-6, 2 epoch, Bench 81.69.

| Phiên bản | Job | Khác biệt | Kết quả trên Bench(115) |
|---|---|---|---|
| `inspector_baseline_overall_humanGT_full` | 5620 (20 giờ 39 phút) | rank 32 chỉ LLM, không validation | Đỉnh 75.65 / 78.99 ở 30k–60k cặp, cuối 72.17 / 78.93 |
| `inspector_baseline_overall_humanGT_full_r128_llm_source_val` | 5836 (24 giờ 07 phút) | rank 128 chỉ LLM, validation 3% ảnh nguồn (5,124 cặp) | Eval loss thấp nhất ở khoảng 16k cặp (0.882), sau đó tăng đến 1.114 |

- **Quá tự tin khi train lâu:** ở job 5620, log loss A/B trên Bench tăng 0.58 → 1.05 và \|d\| trung bình 1.58 → 4.71 từ 10k cặp đến cuối, trong khi margin AUC giữ quanh 79.
- **Loss trên cặp "chưa thấy" không đo được độ tổng quát sau khoảng 40k cặp**, vì từ đó mọi cặp mới đều dùng chung ảnh nguồn với cặp đã train. Điểm bão hoà phải đo bằng Bench.

### 2.7. Sự cố đã sửa

| Sự cố | Hậu quả | Cách sửa |
|---|---|---|
| `all_once` trả screening cache rỗng, lỗi cuối round (job 4313) | Mất khoảng 16.6 giờ | Tạo cache cho mọi mẫu được chọn |
| Giai đoạn 2 chỉ đánh giá và lưu cuối round | Không có checkpoint trung gian | Đánh giá và lưu mỗi 200 step |
| Training loop bỏ qua bước chuẩn bị gradient checkpointing của MS-Swift | ViT LoRA không nhận gradient trong mọi run Giai đoạn 2 trước job 4481 | Từ job 4481 cả 216/216 tensor ViT có gradient |
| Hard-label warm-start không cho tạo file gộp; `best_metric` rỗng khi không có validation (28/10) | Không tạo được `student_warmstart.pt` từ checkpoint chọn tay | Cho phép tạo file gộp từ checkpoint chỉ định khi không có validation |
| Launcher Teacher khai partition normal với GPU 1g.18gb (28/10) | Không submit được | Đổi sang small, `--qos=mig1-18` |

Accuracy HumanGT validation phụ thuộc nhẹ vào batch đánh giá do padding (batch 2: 79.04%, batch 8: 79.60%); mọi phiên bản dùng batch 8.

---

## 3. Hướng hiện tại: bám mục tiêu cốt lõi

### 3.1. Mục tiêu cốt lõi (chốt 28/10/2026)

**Với Inspector pipeline là các checkpoint của Inspector Baseline tại các mốc ít cặp đã train, dùng pipeline đề xuất để Student ở các loop của Giai đoạn 2 đạt cao hơn hoặc ngang Inspector Baseline, trong khi chỉ dùng rất ít cặp có nhãn người.** Đây là mục tiêu cuối cùng; mọi thực nghiệm phải bám theo.

- **Mốc cần đạt:** Inspector Baseline (job 6063, mục 3.2) trên Bench(115): tốt nhất 77.39 / margin AUC 80.00 ở mốc 10k cặp, checkpoint cuối 75.65 / 77.10.
- **Inspector pipeline:** checkpoint của Inspector Baseline tại các mốc ít cặp (64, 112, 1008 cặp, …). **Hiện chọn mốc 1008 cặp (checkpoint-63)** của Inspector Baseline bản sửa lỗi bf16 (mục 3.5); mốc 64 và 112 cặp tạm dừng.
- **Student-Stage1:** train trên đúng các cặp mà Inspector pipeline đã train, với Majority-vote Gemma target. Giai đoạn 2 dùng Inspector pipeline, GemmaGT và Unlabeled Data để nâng Student.
- **Tập so sánh:** Bench(115), kèm VLM Bench và Bench(133). Test không dùng để so với Inspector Baseline vì nằm trong dữ liệu train của nó.
- **Giả định nhãn:** chỉ các cặp Inspector pipeline đã train có nhãn người. Student-Stage1 vì vậy train không có validation.

Các mốc hiện tại:

| Model | Số cặp nhãn người | Bench(115): Accuracy / margin AUC |
|---|---:|---|
| Qwen3-VL-Reranker gốc | 0 | 64.35 / 71.97 (đánh giá kèm Student), 61.74 / 71.80 (đánh giá kèm Inspector) |
| Inspector pipeline (checkpoint-4) | 64 | 64.35 / 71.97 |
| Student-Stage1 (checkpoint-12 job 6613) | 64 | 64.35 / 69.26 |
| Inspector pipeline (checkpoint-63, bản lỗi bf16) | 1,008 | 70.43 / 74.18 |
| Student-Stage1 1008 cặp (checkpoint-252, bản sửa lỗi bf16) | 1,008 | 74.78 / 79.35 |
| **Inspector Baseline, mốc tốt nhất** | 10,000 | **77.39 / 80.00** |
| Inspector Baseline, checkpoint cuối | 165,739 | 75.65 / 77.10 |

### 3.2. Inspector Baseline chuẩn (job 6063)

Phiên bản `inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant`, xong sau 26 giờ 02 phút, 10,359 step.

| Tham số | Inspector pipeline (setting 3k đã kiểm chứng) | Inspector Baseline (job 6063) |
|---|---|---|
| LoRA | rank 32 / alpha 64, chỉ LLM | Giống |
| Head, loss, Tie Head | Head ba lớp, `margin_coupled` | Giống |
| max_grad_norm, weight decay, seed | 1.0, 0.01, 42 | Giống |
| Learning rate | 8e-5, cosine, warmup 0.05 | 5e-5, constant, không warmup (các mốc checkpoint so sánh được với nhau) |
| Số lượt dữ liệu | 2 | 1 |
| Batch hiệu dụng | 8 cặp | 16 cặp |
| Validation | MS-Swift 0.15 | MS-Swift chia ngẫu nhiên theo cặp 0.03 (5,033 cặp) |

- 165,739 cặp train (70,931 A / 60,525 B / 34,283 Tie). Thứ tự train cố định, ghi trong `train_order.jsonl`; **3,000 cặp của `editreward_overall_finegrained_gemmaGT_3k.jsonl` đứng đầu** (xáo seed 42) và không nằm trong validation, nên mọi checkpoint đến mốc 3k chỉ train trên các cặp có cả nhãn người và nhãn Gemma.
- 18 mốc lưu checkpoint và đánh giá validation: 50, 100, 200, 300, 500, 1k, 2k, 3k, 5k, 8k, 10k, 15k, 30k, 50k, 70k, 100k, 130k, 150k cặp (số cặp = step × 16), cộng step cuối.
- `progressive_loss.jsonl` ghi loss từng cặp trước khi cập nhật tham số (loss trên cặp chưa thấy).
- Đánh giá Bench: job 6519 và 6683.

| Số cặp đã train | Validation loss | Bench(115) | VLM Bench(115) | Bench(133): margin AUC (−\|d\|) / Tie Head AUC | \|d\| trung bình | Log loss A/B |
|---|---|---|---|---|---|---|
| 0 (model gốc) | — | 61.74 / 71.80 | 81.74 / 91.43 | 63.19 / — | 0.93 | 0.620 |
| 64 (Inspector pipeline) | 1.052 | 64.35 / 71.97 | 83.48 / 91.22 | 60.56 / 60.41 | 0.59 | 0.618 |
| 304 | 1.016 | 65.22 / 73.04 | 88.70 / 96.26 | 61.76 / 61.64 | 0.75 | 0.602 |
| 2,000 | 0.962 | 73.91 / 77.51 | 93.91 / 98.12 | 70.12 / 68.62 | 0.85 | 0.564 |
| 10,000 | 0.904 | **77.39 / 80.00** | 95.65 / 98.62 | 68.62 / 68.26 | 1.38 | **0.542** |
| 30,000 | 0.862 | 76.52 / 78.50 | 94.78 / 98.94 | **74.86 / 74.88** | 2.17 | 0.634 |
| 100,000 | 0.749 | 76.52 / 77.14 | 93.04 / 98.03 | 66.62 / 65.68 | 2.77 | 0.751 |
| 165,744 (cuối) | 0.635 | 75.65 / 77.10 | 92.17 / 97.95 | 69.37 / 68.07 | 3.68 | 0.952 |

Nhận xét:

- Bão hoà ở khoảng 10k cặp; sau đó margin AUC dao động 77–79, nhỏ hơn khoảng tin cậy ±8 điểm.
- Quá tự tin dần (\|d\| 1.38 → 3.68, log loss A/B 0.54 → 0.95), nhẹ hơn job 5620 và 5836.
- Validation loss giảm đến cuối vì ảnh nguồn của cặp validation có trong train; không dùng để phát hiện quá tự tin hay chọn checkpoint.
- Trên Bench(133), Tie Head AUC bám sát margin AUC (−\|d\|) ở mọi mốc (chênh ≤ 1.5 điểm), đúng thiết kế `margin_coupled`.
- Inspector pipeline (64 cặp) gần như chưa khác model gốc (71.97 so với 71.80): khoảng cách tới mốc cần đạt là khoảng 8 điểm margin AUC.

Hình và bảng: `RewardModelEvaluation/OurModel/evaluation_results/inspector/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant/training_analysis/`.

### 3.3. Inspector pipeline = checkpoint-63 của Inspector Baseline (chốt 27/10, mốc 1008 cặp từ 28/10/2026)

- Không train riêng Inspector pipeline: lấy checkpoint của Inspector Baseline tại mốc số cặp đã train (step × 16 cặp). **Mốc hiện tại: checkpoint-63 = 1008 cặp**, là 1008 cặp đầu của `train_order.jsonl` (nhãn người 449 A / 354 B / 205 Tie), đều nằm trong GemmaGT 3k. Mốc 64 cặp (checkpoint-4: 31 A / 24 B / 9 Tie) và 112 cặp (checkpoint-7) tạm dừng.
- Lý do đổi hướng so với setting 3k: Student và Inspector phải dùng cùng một tập cặp có nhãn; ở setting cũ 306/353 cặp HumanGT validation nằm trong GemmaGT train, và khoảng cách giữa Giai đoạn 1 và Giai đoạn 2 quá nhỏ.
- **Lấy checkpoint từ một lần chạy cùng cấu hình trên GPU khác.** Khi GPU full phải chờ lâu, có thể chạy lại Inspector Baseline với cùng dữ liệu, thứ tự cặp và siêu tham số trên GPU nhỏ hơn để có sớm checkpoint-63. Điều kiện:
  - Giữ đúng batch 2 × 8: mặt nạ LoRA dropout và dropout của head được rút theo kích thước batch, đổi batch sẽ cho một đường train khác.
  - Được phép bật gradient checkpointing (khôi phục trạng thái ngẫu nhiên khi tính lại) và đổi batch khi đánh giá validation (không có dropout).
  - Kết quả chỉ tương đương về mặt số học, không giống từng bit (lan truyền ngược của flash attention không xác định, GPU khác loại làm tròn bf16 khác). Khi run trên full có checkpoint-63, so trọng số, dự đoán trên Bench và cache của hai checkpoint.

### 3.4. Student-Stage1 trên 64 cặp (job 6613, tạm dừng, bản \(\theta_0,\kappa\) cố định)

Phiên bản `student_pipeline_overall_finegrained_gemmaGT_64_r64_lr4e-5`:

- Train đúng 64 cặp của Inspector pipeline, Majority-vote Gemma target từ GemmaGT hiện có (không sinh lại Gemma).
- Rank 64, learning rate 4e-5, 2 lượt, 8 cặp mỗi bước, 16 bước, Tie Head `margin_coupled`, **không có validation** (đúng giả định chỉ 64 cặp có nhãn). Lưu checkpoint 4 / 8 / 12 / 16.
- **Chọn checkpoint-12** (96 cặp đã thấy, giữa lượt 2) làm Student-Stage1.

Kết quả (`selected_equal`, đánh giá job 6725–6728):

| Model | Test (1000) | Bench (115) | VLM Bench (115) | Bench(133): margin AUC (−\|d\|) / Tie Head AUC |
|---|---|---|---|---|
| Qwen3-VL-Reranker gốc | 61.50 / 66.28 | 64.35 / 71.97 | 82.61 / 91.81 | 63.89 / — |
| Inspector pipeline (checkpoint-4) | — | 64.35 / 71.97 | 83.48 / 91.22 | 60.56 / 60.41 |
| Student checkpoint-4 | 53.30 / 57.41 | 51.30 / 62.30 | 69.57 / 81.16 | 62.49 / 62.56 |
| Student checkpoint-8 | 56.70 / 61.83 | 55.65 / 64.20 | 72.17 / 83.74 | 56.57 / 58.55 |
| **Student checkpoint-12** | 61.80 / 67.27 | 64.35 / 69.26 | 82.61 / 91.15 | 56.35 / 57.29 |
| Student checkpoint-16 | 61.90 / 67.38 | 59.13 / 67.27 | 80.87 / 91.32 | 51.45 / 53.53 |

- Student-Stage1 thấp hơn Inspector pipeline trên Bench (69.26 so với 71.97 margin AUC), rất thiếu tự tin (\|d\| khoảng 0.12, log loss khoảng 0.67), Tie Head gần ngẫu nhiên trên Bench(133).
- Khoảng cách tới mốc cần đạt: khoảng 13 điểm accuracy và 11 điểm margin AUC trên Bench(115).

Hình và bảng: `RewardModelEvaluation/OurModel/evaluation_results/student/student_pipeline_overall_finegrained_gemmaGT_64_r64_lr4e-5/checkpoint_analysis/`.

### 3.5. Lỗi bf16 ở tham số các head (phát hiện 28/10/2026)

**Hiện tượng.** \(\theta_0\) và \(\kappa\) của Tie Head `margin_coupled` không đổi so với giá trị khởi tạo (\(\theta_0=-1.3828\), \(\kappa_{\text{raw}}=0.5430\), tức logit(0.2) và \(\kappa=1\) ở bf16) trong mọi run đã kiểm tra:

| Run | \(\theta_0\) / \(\kappa_{\text{raw}}\) sau khi train |
|---|---|
| Inspector Baseline job 6063, checkpoint-10359 (10,359 step) | −1.3828 / 0.5430 |
| `inspector_tie_ablation_margin_coupled_3k` (learning rate 8e-5) | −1.3828 / 0.5430 |
| `student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5`, checkpoint-252 (cả 6 metric) | −1.3828 / 0.5430 |

Tỉ lệ phần tử không đổi giữa checkpoint-4 và checkpoint-10359 của job 6063: \(\theta_0,\kappa\) 100%; Score Head phép chiếu khởi tạo từ \(W_\text{yes}-W_\text{no}\) 65.1%; lớp đầu MLP của Tie Head 64.9%; các lớp khác của head 0–2.4%; LoRA 0%.

**Nguyên nhân (đã kiểm chứng).** LoRA được lưu ở fp32, nhưng các head (train qua `modules_to_save`) giữ kiểu bf16 của backbone. Mỗi bước AdamW đổi tham số khoảng bằng learning rate (4e-5 đến 8e-5), nhỏ hơn nửa bước làm tròn của bf16 với các giá trị có độ lớn từ khoảng 0.02–0.03 trở lên (quanh 1.38, bước bf16 là khoảng 0.008), nên cập nhật bị làm tròn mất. Thử trên CPU: 1,000 bước AdamW, tham số bf16 thay đổi 0.0, cùng thí nghiệm ở fp32 thay đổi 0.04–0.08.

**Hệ quả.**
- Dạng Tie Head thực tế của mọi run (ablation, Inspector Baseline, Student-Stage1) là \(\theta_0,\kappa\) cố định; chỉ residual và một phần Score Head được học. Kết luận ablation (mục 2.5) chỉ đúng cho dạng này.
- Score Head của Inspector và Student chỉ học được một phần phép chiếu khởi tạo.

**Vì sao không phát hiện trước khi train.** Kiểm tra trước ablation Tie Head (`tie_head_ablation.md` mục 4) chỉ xác nhận gradient vào \(\theta_0,\kappa\) khác 0, không kiểm tra tham số có thực sự thay đổi sau bước optimizer ở đúng kiểu dữ liệu khi train. Sau mỗi run chỉ xem loss và chỉ số đánh giá, không so trọng số head giữa các checkpoint. Lỗi được thấy lần đầu khi kiểm tra checkpoint-4 cho cache (28/10), nhưng khi đó chỉ được báo lại, không chặn việc train Student 1008 cặp sau đó.

**Kiểm tra bắt buộc từ nay:** trước mỗi run mới, chạy thử vài step và xác nhận mọi tensor có gradient đều thay đổi, gồm \(\theta_0,\kappa\); sau mỗi run, so trọng số head giữa checkpoint đầu và cuối.

**Quyết định (28/10/2026): train lại toàn bộ.**
1. Sửa code để tham số các head ở fp32 (tính toán vẫn bf16 qua autocast), áp dụng cho Inspector, Student-Stage1 và Giai đoạn 2.
2. Train lại Inspector Baseline với cùng setting job 6063 (partition full, khoảng 26 giờ). Inspector pipeline lấy checkpoint-63 (1008 cặp) của run mới ngay khi có, không cần chờ hết run; có thể lấy từ một lần chạy cùng cấu hình trên GPU khác (mục 3.3).
3. Train lại Student-Stage1 trên 1008 cặp, rồi tạo file gộp và Teacher (đã xong Student-Stage1).
4. Sinh lại Overall Pseudo-label Cache từ checkpoint-63 mới.
5. Các kết quả của job 6063, Student 64 / 112 / 1008 cặp bản cũ được giữ để tham khảo, gắn nhãn là \(\theta_0,\kappa\) cố định.

**Kết quả sau khi sửa (Student-Stage1 1008 cặp).**
- Mọi tensor head và LoRA ở fp32 và đều thay đổi giữa checkpoint đầu và cuối, kể cả các phần trước đây đứng yên (phép chiếu khởi tạo của Score Head, lớp đầu MLP của Tie Head).
- \(\theta_0,\kappa\) có học nhưng rất chậm: chỉ dịch khoảng 0.002 sau 252 step với learning rate 4e-5.
- Bias đầu ra của Score Head không nhận gradient: điểm của A và B được trừ cho nhau nên bias triệt tiêu. Đây là đúng thiết kế, không phải lỗi.
- Trên Test và Bench, bản sửa lỗi cao hơn bản lỗi khoảng 0.5 điểm margin AUC, nằm trong nhiễu.

### 3.6. Setting 1008 cặp: kết luận và thay đổi code

Ưu tiên hiện tại là setting **1008 cặp** (checkpoint-63 của Inspector Baseline). Setting 64 và 112 cặp tạm dừng, làm sau.

**Student-Stage1 1008 cặp.** Train trên đúng 1008 cặp của Inspector pipeline, 2 lượt (252 step, 8 cặp mỗi step), không có validation, lưu mỗi 21 step.
- Trước khi hết lượt 1, Student kém checkpoint cuối có ý nghĩa trên Test; từ khi hết lượt 1 (checkpoint-126) trở đi là vùng ổn định, không khác checkpoint cuối có ý nghĩa trên Test và Bench.
- **Chốt (29/10/2026): chọn checkpoint cuối (checkpoint-252, hết 2 lượt), chốt trước khi xem kết quả.** Không chọn theo Bench (Bench là tập so sánh với Inspector Baseline).
- Bản sửa lỗi bf16, checkpoint-252: Bench(115) 74.78 / 79.35, cao hơn Inspector pipeline cùng số cặp (bản lỗi: 70.43 / 74.18). So với Inspector Baseline: đã vượt checkpoint cuối về margin AUC (77.10) nhưng còn dưới mốc tốt nhất (77.39 / 80.00).
- Trên Bench(133), Tie Head AUC (64.3) vẫn thấp hơn margin AUC (−|d|) (70.7).

**Thay đổi code (28–29/10/2026, sao lưu trong `scripts/backups/` và `RewardModelEvaluation/OurModel/backups/`).**

| Thay đổi | Nội dung |
|---|---|
| Head fp32 | Tham số head của Inspector và Student giữ fp32; hidden state bf16 được chuyển sang fp32 khi vào head. Trạng thái EMA của Teacher trong `train_loop.py` cũng giữ fp32 (bước \(1-\alpha\) bị làm tròn mất ở bf16) |
| Sinh cache | `infer_overall_inspector.py` nhận thư mục checkpoint MS-Swift (`checkpoint-<step>`) và báo lỗi khi thiếu khoá LoRA hoặc head (trước đây head thiếu khoá sẽ giữ trọng số khởi tạo mà không báo) |
| Loss ba lớp (cách B, `--overall-loss three_way`) | HumanGT giữ cả cặp hoà, loss \(-\log\bar p[\text{nhãn người}]\); Unlabeled dùng cross-entropy giữa \((P_I(A),P_I(B),P_I(\text{Tie}))\) của Inspector và \(\bar p\) của Student, cộng KL(Teacher ‖ Student). \(\bar p=\frac{1}{|S|}\sum_{k\in S}p_k\), `tie_probs` không detach, relevance detach. Target \(q=P(A)+\tfrac12P(\text{Tie})\) chỉ còn dùng để xếp hạng mẫu và chẩn đoán |
| Không loại mẫu probe (`--exclude-probe-from-training false`) | Train trên toàn bộ 10k Unlabeled. 128 mẫu probe chỉ để theo dõi sự thay đổi của Student và Teacher trên cùng một nhóm mẫu; Unlabeled không có nhãn người nên tách riêng một tập cũng không đánh giá được chất lượng (trước đây loại 778 mẫu cùng ảnh nguồn với probe) |
| Dữ liệu trong 1008 cặp (`--human-split student_train_order`) | HumanGT và GemmaGT = đúng các cặp Student-Stage1 đã train; code kiểm tra số cặp Inspector đã train (step × cặp mỗi step) và các cặp đầu của `train_order.jsonl` khớp với Student |
| Chọn checkpoint không validation (`--checkpoint-selection last`) | Mỗi lần đánh giá đều được nhận, nên `best.pt` là checkpoint cuối; không dừng sớm. Loss và Brier Gemma (soft target) đo trên chính 1008 cặp để theo dõi mức quên |
| `all_once` | Báo lỗi nếu ngân sách Unlabeled mỗi round nhỏ hơn số mẫu được train; trước đây lịch cosine (learning rate, EMA) có thể kết thúc sớm hơn run |
| Launcher `run_train_loop.slurm` | Thêm `REFINEMENT_OVERALL_LOSS`, `REFINEMENT_EXCLUDE_PROBE_FROM_TRAINING`, `REFINEMENT_HUMAN_SPLIT`, `REFINEMENT_CHECKPOINT_SELECTION`; mặc định giữ hành vi cũ. Launcher không khai QOS: submit với `--qos=mig7-141` |
| Đánh giá Student | `evaluate_student_checkpoints.py` lấy trung bình `signed_evidence` trên S thay vì tổng (`AUC-Margin.md` mục 6) |
| Tie Head của Student ở Giai đoạn 2 (30/10) | `train_loop.py` đọc chế độ Tie Head từ checkpoint Student-Stage1, đối chiếu với cấu trúc khoá state dict, báo lỗi nếu `TIE_HEAD_MODE` đặt khác. Launcher Inspector và Student-Stage1 mặc định `TIE_HEAD_MODE=margin_coupled` (trước đây thiếu biến này thì Student bị dựng `independent`) |
| Learning rate riêng cho \(\theta_0,\kappa\) (30/10) | `--tie-coupling-learning-rate` (launcher: `REFINEMENT_TIE_COUPLING_LEARNING_RATE`) đưa \(\theta_0,\kappa_{\text{raw}}\) vào nhóm AdamW riêng, weight decay 0; không đặt thì giữ một nhóm như cũ |
| `config_training.sh` (30/10) | Mọi phiên bản Giai đoạn 2 được định nghĩa ở đây (cấu hình chung + phần khác của từng phiên bản, tên phiên bản = thư mục output). Mỗi lần submit ghi cấu hình đầy đủ, tài nguyên và md5 của launcher và `train_loop.py` vào `experiment_submissions.log` |
| Ghi chú `tie_head_mode` khi đánh giá (chưa sửa) | `evaluate_student_checkpoints.py` ghi `tie_head_mode = independent` cho checkpoint Giai đoạn 2, vì checkpoint này không lưu khoá đó. Model vẫn được dựng đúng `margin_coupled` từ khoá state dict; chỉ trường ghi chú sai |

### 3.7. Cấu hình Giai đoạn 2 đầu tiên (chốt 29/10/2026)

| Thiết lập | Giá trị |
|---|---|
| Inspector pipeline / cache | checkpoint-63 bản sửa lỗi, Overall Pseudo-label Cache ba lớp |
| Student-Stage1 / Teacher | checkpoint-252 bản sửa lỗi, Teacher là bản sao của Student |
| Loss | `three_way` (cách B) + KL(Teacher ‖ Student) **chỉ trên S** (`--consistency-metrics selected`): metric được chọn là các metric được đánh giá và ảnh hưởng tới kết quả của cặp, nên mọi thành phần chỉ tính trên S |
| Dữ liệu | `student_train_order`: 1008 cặp HumanGT (kể cả 205 cặp hoà) và 1008 cặp GemmaGT; 10k Unlabeled, không loại probe |
| Unlabeled | **Train toàn bộ một lần** (`all_once`, 1 round, ngân sách mỗi round 10,000) |
| Mỗi step | **2 HumanGT + 2 GemmaGT + 12 Unlabeled** = 2 microbatch (1 + 1 + 6), gradient accumulation 2; khoảng 834 step, mỗi cặp HumanGT / GemmaGT được thấy khoảng 1.65 lần (với 4 + 4 + 12 là khoảng 3.3 lần). Tỉ lệ HumanGT mỗi microbatch 12.5% nên đặt `--min-supervised-fraction 0.10` |
| EMA Teacher | **cosine 0.99 → 0.9999** như Co-rewarding-II (arXiv 2508.00410): Teacher theo kịp Student trong khoảng 250 step đầu, sau đó gần như đứng yên để làm nhãn ổn định. Với 834 step, cuối run phần Student-Stage1 còn lại trong Teacher khoảng 1.5% (tích các hệ số decay), nên Teacher đi trễ sau Student chứ không neo vào Stage1; con số 63% chỉ đúng với lịch 0.999 → 0.9999. Với 0.99 cố định, phần này còn 37% sau 100 step và 5% sau 300 step |
| Chọn checkpoint | `last`; đánh giá và lưu mỗi 100 step chỉ để theo dõi diễn biến, giữ đủ các checkpoint |
| Còn lại | learning rate 1e-5 cosine, LoRA rank 64 |
| GPU | partition full (đỉnh khoảng 67 GiB do bước forward 6 cặp Unlabeled). Chỉ chuyển xuống large khi full phải chờ quá lâu; khi đó phải giảm microbatch Unlabeled |

- Các checkpoint giữa run chỉ để theo dõi; **không so được với một loop 3k của bản chia loop**, vì mỗi loop đó lọc mẫu theo mức dễ / vừa / khó và thêm 25% mẫu cũ.
- Trước run thật: chạy thử khoảng 10 step với checkpoint-63 và cache thật để kiểm tra toàn bộ phần code mới và đo bộ nhớ.

### 3.8. Vấn đề còn mở liên quan tới mục tiêu

1. **Khoảng cách tới mốc cần đạt:** Student-Stage1 1008 cặp còn dưới mốc tốt nhất của Inspector Baseline khoảng 2.6 điểm accuracy và 0.65 điểm margin AUC trên Bench(115) (với 64 cặp là khoảng 11 điểm margin AUC); ở setting cũ Giai đoạn 2 chỉ tăng khoảng 1–1.5 điểm trên Test.
2. **Inspector pipeline yếu hơn Student-Stage1 cùng số cặp:** ở 1008 cặp (bản sửa lỗi) Inspector đạt 74.62 margin AUC trên Bench(115), Student-Stage1 đạt 79.35; ở 64 cặp Inspector gần như chưa khác model gốc. Sau Giai đoạn 2 đầu tiên (mục 3.9), Student không bị kéo xuống, nhưng bị kéo về phía Inspector (P(Tie), hướng margin) mà không tăng accuracy.
3. **Chọn checkpoint ở Giai đoạn 2** khi gần như không có nhãn người để làm validation; ở setting cũ 353 cặp validation đã chọn sai thứ tự so với Test.
4. **Teacher gần như trùng Student** ở các run cũ (EMA 0.99), nên KL(Teacher ‖ Student) có thể chỉ có tác dụng điều chuẩn; chưa đo đóng góp riêng. Run đầu của setting 1008 cặp dùng EMA cosine 0.99 → 0.9999 (mục 3.7).
5. **Tie Head ở Giai đoạn 2:** đã thêm loss ba lớp (cách B, mục 3.6) để HumanGT tie và P(Tie) của Inspector cùng train Tie Head; chưa chạy. Nhãn hoà của Inspector 1008 cặp còn yếu (Tie Head AUC khoảng 61 ở bản lỗi), và ở Student-Stage1 bản sửa lỗi Tie Head AUC (64.3) vẫn dưới margin AUC (−|d|) (70.7) vì \(\theta_0,\kappa\) học rất chậm. Sau Giai đoạn 2 đầu tiên, Tie Head AUC trên Bench(133) là 63.6, không tăng; \(\theta_0,\kappa\) gần như không đổi. Đang thử learning rate riêng 1e-3 cho \(\theta_0,\kappa\) (mục 3.9).
6. **Supervised control:** setting 1008 cặp có run đối chứng \(\lambda_U=0\) (mục 3.9); setting 64 nhãn chưa có.
7. **Bench nhỏ** (115 cặp, ±8 điểm): kết luận "ngang Inspector Baseline" phải dựa trên kiểm định theo cặp, kèm VLM Bench.
8. **Quyền file khi hai account cùng làm việc:** file adapter do MS-Swift ghi có quyền `-rw-------` ở cả hai account, account kia không đọc được (đã `chmod a+r` cho các run 1008 cặp). Các run sau cần kiểm tra quyền đọc trước khi chuyển bước.
9. **Cân bằng trọng số giữa các nguồn:** loss lấy trung bình theo nguồn (mục 1.6.2), nên với 1 + 1 + 6 mỗi cặp Unlabeled chỉ nặng 1/6 một cặp HumanGT, trong khi setting 3k (5 + 5 + 6) gần cân bằng (1/5 so với 1/6). Khi đổi tỉ lệ cặp giữa các nguồn phải đổi \(\lambda\) theo. Phiên bản \(\lambda_U=6\) (cân bằng theo cặp như setting 3k) cùng EMA 0.99 cố định (mục 3.10) đã bị huỷ trước khi chạy, chưa có kết quả.
10. **Grad clip:** từ 01/11/2026 `train_loop.py` ghi tổng norm gradient trước clip mỗi 10 step (`grad_norm_before_clipping`). Ở run learning rate 1e-3 cho \(\theta_0,\kappa\), norm khoảng 20–570 và mọi step đều bị clip về 1.0. Với AdamW, nhân đều gradient với một hệ số ít ảnh hưởng tới bước cập nhật, và tỉ lệ giữa các nguồn trong cùng step không đổi; clip chủ yếu làm các step đóng góp đều nhau. Chưa coi là nguyên nhân; chưa có số của setting 3k để so.

Ý tưởng chờ đề xuất: dùng Teacher làm nguồn nhãn khi Inspector không chắc chắn; learning rate riêng cho LoRA và các head; class-balanced BCE cho Tie Head.

### 3.9. Kết quả Giai đoạn 2 đầu tiên, setting 1008 cặp (31/10/2026)

Phiên bản `train_loop_pipeline_1008_all_once_three_way_selected_equal` (cấu hình mục 3.7), đánh giá tại `best.pt` = step cuối 834. Acc / margin AUC; Bench có Tie: margin AUC (−|d|) / Tie Head AUC. Inspector không có số Test (Test thuộc dữ liệu train của Inspector).

| Tập | Giai đoạn 2 Student | Giai đoạn 2 Teacher | Student-Stage1 (checkpoint-252) | Inspector pipeline (checkpoint-63) | Inspector Baseline (checkpoint-625) |
|---|---|---|---|---|---|
| Test (1000) | 73.50 / 81.62 | 74.00 / 81.70 | 74.30 / 81.20 | — | — |
| Bench (115) | 74.78 / 78.46 | 73.91 / 78.53 | 74.78 / 79.35 | 73.04 / 74.62 | 76.52 / 79.32 |
| Bench có Tie (133) | 68.79 / 63.62 | 71.64 / 65.65 | 70.68 / 64.30 | 60.68 / 59.66 | 68.99 / 67.83 |
| VLM Bench (115) | 93.04 / 97.61 | 92.17 / 97.77 | 93.04 / 98.38 | 91.30 / 96.20 | 96.52 / 98.44 |
| VLM Bench có Tie (133) | 95.80 / 90.84 | 95.42 / 91.22 | 95.42 / 89.31 | 93.13 / 89.69 | 97.71 / 95.42 |
| Accuracy ba lớp, Bench / VLM Bench (133) | 63.16 / 87.22 | 63.91 / 88.72 | 64.66 / 89.47 | 63.16 / 87.22 | 66.17 / 92.48 |

**Kết luận**
- Không khác Student-Stage1 (McNemar trên Test 34 / 42, p = 0.42; Bench 3 / 3). Margin hai bản tương quan 0.963, cùng chiều ở 92.4% cặp Test.
- Chưa đạt Inspector Baseline checkpoint-625; chênh lệch trên Bench và VLM Bench chưa có ý nghĩa (p = 0.69 và 0.22) vì Bench nhỏ.
- Teacher gần như trùng Student (margin tương quan 0.992 trên Test).
- Tie Head không tốt hơn: P(Tie) tăng ở mọi cặp (cặp Tie 0.164 → 0.183, cặp không Tie 0.123 → 0.140 trên Bench), khả năng tách cặp Tie không đổi.
- |d| của Student khoảng 0.5 trên Test, của Inspector Baseline khoảng 1.05: Student kém tự tin hơn, thể hiện ở log loss (VLM Bench 0.459 so với 0.286).

**Đóng góp của Unlabeled**
- Trọng số: mỗi step 2 HumanGT + 2 GemmaGT + 12 Unlabeled, mỗi nguồn là trung bình trên batch của nó (mục 1.6.2), nên mỗi cặp Unlabeled nặng 1/12, mỗi cặp HumanGT nặng 1/2.
- Mức sàn: entropy trung bình nhãn ba lớp của Inspector pipeline trên 10k Unlabeled là 0.904 (ln 3 = 1.099; 47% mẫu có xác suất lớn nhất dưới 0.5). \(\mathcal L_{\text{unlabeled}}\) đi từ 0.998 lên 1.026 rồi về 1.005, nên phần thực sự kéo Student chỉ khoảng 0.1; HumanGT loss giảm từ 0.81 xuống 0.74.
- Student vẫn tiến về phía Inspector trên các cặp không train (Test / Bench): phần CE vượt mức sàn 0.115 / 0.136 → 0.094 / 0.119; tương quan margin với Inspector 0.832 / 0.856 → 0.849 / 0.880; P(Tie) trung bình trên Test 0.145 → 0.174 (Inspector 0.173). Đây là số liệu quan sát; HumanGT (20% cặp hoà) và GemmaGT replay cũng có thể góp phần.
- Nhận định: nhánh Unlabeled làm đúng việc được thiết kế (bắt chước Inspector), nhưng Inspector pipeline yếu hơn Student-Stage1 nên bắt chước không tăng accuracy.

**Phiên bản tiếp theo (mỗi phiên bản một thay đổi so với mục 3.7)**

| Phiên bản | Thay đổi | Mục đích |
|---|---|---|
| `train_loop_pipeline_1008_inspector_teacher_rank_three_rounds_three_way_selected_equal` | 3 round `inspector_teacher_rank`, 3,334 Unlabeled mỗi round, 25% ôn lại từ round 2 (khoảng 974 step) | So với train một lượt ở setting 1008 cặp |
| `train_loop_pipeline_1008_all_once_three_way_selected_equal_tie_coupling_lr1e-3` | learning rate 1e-3 cho \(\theta_0,\kappa\) | Tie Head học được \(\theta_0,\kappa\) |
| `train_loop_pipeline_1008_all_once_without_unlabeled_loss_three_way_selected_equal` | \(\lambda_U=0\); vẫn forward Unlabeled nên dữ liệu có nhãn, lịch learning rate, EMA và số step giữ nguyên | Đối chứng: Unlabeled có đóng góp vào kết quả cuối không |

- Các checkpoint giữa run (100–800) của phiên bản đầu được đánh giá chỉ để xem diễn biến, không dùng để chọn checkpoint.
- Phép đo gradient theo nguồn (HumanGT, Gemma, mask, Unlabeled CE, Unlabeled KL; cosine giữa Unlabeled và HumanGT) để sau, khi có kết quả đối chứng.

### 3.10. Kết quả phiên bản ba round, setting 1008 cặp (01/11/2026)

Phiên bản `train_loop_pipeline_1008_inspector_teacher_rank_three_rounds_three_way_selected_equal`: cấu hình mục 3.7, chỉ khác ở Unlabeled.
- **Chia round:** 3 round `inspector_teacher_rank`, 3,334 / 4,168 / 4,166 Unlabeled. Từ round 2, 25% là mẫu khó ôn lại.
- **Số step:** 974. Round 1 kết thúc ở step 278, round 2 ở step 626.
- **Đánh giá:** `best.pt` (step cuối 974), các checkpoint mỗi 100 step và trạng thái cuối round 1 và round 2.
- **Cách đọc bảng:** các ô ghi Acc / margin AUC. Ở Bench có Tie, ô ghi margin AUC (−|d|) / Tie Head AUC / accuracy ba lớp.
- **Cột McNemar:** so với Student-Stage1 trên Test, ghi theo dạng (số cặp checkpoint đúng mà Stage1 sai) / (số cặp Stage1 đúng mà checkpoint sai).

| Checkpoint | Test (1000) | McNemar Test | Bench (115) | Bench có Tie (133) | VLM Bench (115) |
|---|---|---|---|---|---|
| Student-Stage1 (checkpoint-252) | 74.30 / 81.20 | — | 74.78 / 79.35 | 70.68 / 64.30 / 64.66 | 93.04 / 98.38 |
| step 100 | 74.10 / 81.29 | 24 / 26 | 72.17 / 78.80 | 72.95 / 66.14 / 62.41 | 90.43 / 98.33 |
| step 200 | 73.70 / 80.88 | 35 / 41 | 72.17 / 78.84 | 73.43 / 68.26 / 62.41 | 90.43 / 98.06 |
| cuối round 1 (278) | 74.60 / 81.14 | 38 / 35 | 74.78 / 78.06 | 72.42 / 67.92 / 64.66 | 93.04 / 98.49 |
| step 300 | 74.50 / 81.28 | 31 / 29 | 73.04 / 77.82 | 71.30 / 68.07 / 63.16 | 93.04 / 98.16 |
| step 400 | 74.50 / 81.09 | 34 / 32 | 74.78 / 77.57 | 71.40 / 66.09 / 64.66 | 93.04 / 98.56 |
| step 500 | 74.20 / 81.17 | 35 / 36 | 73.91 / 78.00 | 69.42 / 64.83 / 63.16 | 90.43 / 98.10 |
| step 600 | 73.80 / 81.37 | 38 / 43 | 74.78 / 78.46 | 69.76 / 64.59 / 63.91 | 91.30 / 98.22 |
| cuối round 2 (626) | 74.80 / 81.11 | 44 / 39 | 75.65 / 78.48 | 69.61 / 66.62 / 63.91 | 90.43 / 98.28 |
| step 700 | 73.10 / 81.08 | 35 / 47 (p = 0.22) | 73.91 / 78.06 | 70.48 / 66.71 / 63.91 | 90.43 / 98.01 |
| step 800 | 72.70 / 80.87 | 37 / 53 (p = 0.11) | 73.91 / 78.43 | 69.76 / 65.46 / 62.41 | 92.17 / 97.91 |
| step 900 | 72.40 / 80.90 | 35 / 54 (p = 0.056) | 73.04 / 78.17 | 69.57 / 65.36 / 62.41 | 89.57 / 97.78 |
| **Student `best.pt` (974)** | 72.70 / 80.95 | 34 / 50 (p = 0.10) | 73.91 / 78.33 | 68.99 / 64.78 / 64.66 | 92.17 / 97.88 |
| Teacher `best.pt` (974) | 73.00 / 81.15 | — | 74.78 / 78.02 | 70.97 / 65.94 / 64.66 | 91.30 / 98.26 |
| Bản một lượt (mục 3.9), Student `best.pt` | 73.50 / 81.62 | 34 / 42 (p = 0.42) | 74.78 / 78.46 | 68.79 / 63.62 / 63.16 | 93.04 / 97.61 |
| Inspector Baseline (checkpoint-625) | — | — | 76.52 / 79.32 | 68.99 / 67.83 / 66.17 | 96.52 / 98.44 |

**Kết luận**
- **So với Student-Stage1:**
  - Không checkpoint nào hơn Stage1.
  - Margin AUC trên Test gần như không đổi suốt run (80.87–81.37).
- **Test giảm trong round 3:**
  - Accuracy giữ ngang Stage1 đến hết round 2 (74.80 ở step 626), rồi giảm khoảng 2 điểm (72.4–72.7 từ step 800).
  - McNemar tiến tới ngưỡng ý nghĩa (p = 0.056 ở step 900).
  - Round 2 cũng có 25% mẫu ôn lại mà không giảm, nên chưa quy được nguyên nhân riêng cho việc ôn lại mẫu khó.
- **So với bản một lượt:** không khác có ý nghĩa (`best.pt`: Test 31 / 39, p = 0.40; Bench 1 / 2). Chia round theo `inspector_teacher_rank` không giải quyết việc Giai đoạn 2 không cải thiện.
- **Bench có Tie:** tín hiệu tăng nhỏ ở step 100–300 (margin AUC 71.3–73.4, Tie Head AUC 66–68), sau đó mất dần. Bench chỉ 133 cặp, nên đây chỉ là gợi ý.
- **Hướng về Inspector, giống bản một lượt:**
  - Phần CE vượt mức sàn so với Inspector pipeline trên Test thấp nhất ở step 400–500 (0.064–0.069). Đúng lúc đó |d| nhỏ nhất (0.43–0.44) và P(Tie) cao nhất (0.18).
  - Đến cuối run, chỉ số này về 0.108 và |d| về khoảng 0.51 như Stage1.
  - Nhánh Unlabeled chủ yếu điều chỉnh độ tự tin (|d|, P(Tie)), không cải thiện thứ tự xếp hạng.
- **Teacher gần như trùng Student:** margin tương quan 0.994 trên Test.

**Hướng tiếp theo**
- **Cách chọn cấu hình:** từ 01/11/2026, cấu hình Giai đoạn 2 dựa trên kết quả các experiment trước và có thể gộp nhiều thay đổi đã có căn cứ trong một phiên bản. Lý do là slot full ít và mỗi run mất 16–19 giờ.
- **Phiên bản mới:** `train_loop_pipeline_1008_all_once_three_way_selected_equal_lambda_unlabeled6_ema_decay_constant0.99`, khác mục 3.7 ở hai điểm:
  - **\(\lambda_U=6\):** với 2 + 2 + 12, mỗi cặp Unlabeled nặng 6/12 = 1/2, bằng mỗi cặp HumanGT, như cân bằng của setting 3k.
  - **EMA 0.99 cố định:** như `train_loop_single_pass_selected_equal` (mục 2.3), phiên bản duy nhất hơn Student-Stage1 có ý nghĩa.
  - **Giữ nguyên:** `all_once`, vì chia round không giúp. Giữ 2 + 2 + 12, vì 5 + 5 + 6 với 1008 cặp làm mỗi cặp HumanGT lặp khoảng 8 lần.
- **Phiên bản bị thay:** phiên bản chỉ có \(\lambda_U=6\) (EMA cosine) bị huỷ trước khi chạy và được thay bằng phiên bản trên.
- **Trạng thái (02/11/2026):** phiên bản trên đã bị huỷ trước khi chạy. Phiên bản learning rate 1e-3 cho \(\theta_0,\kappa\) đã chạy xong, chưa đánh giá; đối chứng \(\lambda_U=0\) đang chạy (mục 3.11).

### 3.11. Tóm tắt các phiên bản Giai đoạn 2 (02/11/2026)

**Khác biệt chung so với proposal gốc (mục 0)**

| Thành phần | Proposal gốc | Đã triển khai |
|---|---|---|
| Nhánh Unlabeled | Metrics and Overall Projection | Soft Inspector label + KL(Teacher ‖ Student) |
| Chọn mẫu | Teacher screening theo ba nhóm, nhiều round | `all_once` hoặc `inspector_teacher_rank` |
| Chọn checkpoint | Theo Teacher | Theo Student (3k) / checkpoint cuối (1008) |
| Inspector | Nhị phân, có Abstain | Ba lớp, không calibration |

**Setting 3k (mục 2.3)**

| Phiên bản | Điểm chính | Kết quả |
|---|---|---|
| `train_loop_single_pass` | `all_once`, gộp 6 metric bằng a_k | Test 76.60, hơn Stage1 chưa có ý nghĩa |
| `train_loop_single_pass_selected_equal` | Gộp `selected_equal` | Test 77.20, hơn Stage1 có ý nghĩa (p = 0.028) |
| `train_loop_curriculum_selected_equal` | 3 round, P(A) tuyến tính, KL trên S, EMA cosine | Test kém bản một lượt có ý nghĩa (p = 0.023) |

**Setting 1008 cặp, khác cấu hình mục 3.7 ở một điểm**

| Phiên bản | Thay đổi | Trạng thái | Kết quả |
|---|---|---|---|
| `train_loop_pipeline_1008_all_once_three_way_selected_equal` | — (cấu hình gốc) | Xong | Ngang Stage1 (Test p = 0.42) |
| `train_loop_pipeline_1008_inspector_teacher_rank_three_rounds_three_way_selected_equal` | 3 round | Xong | Không hơn Stage1; Test giảm khoảng 2 điểm ở round 3 |
| `train_loop_pipeline_1008_all_once_three_way_selected_equal_tie_coupling_lr1e-3` | learning rate 1e-3 cho θ0, κ | Xong, chưa đánh giá | Mọi step bị clip (norm gradient khoảng 30–680) |
| `train_loop_pipeline_1008_all_once_without_unlabeled_loss_three_way_selected_equal` | λ_U = 0 (đối chứng) | Đang chạy | — |

- **Phiên bản bị huỷ trước khi chạy:** bản λ_U = 6, và bản λ_U = 6 + EMA 0.99 cố định.
- **Kết luận:** ở setting 1008 cặp, Giai đoạn 2 chưa cải thiện và chưa tách khỏi Stage1. Nhánh Unlabeled chủ yếu đổi độ tự tin (|d|, P(Tie)), không đổi thứ tự xếp hạng.
- **Giả định chưa chứng minh:**
  - tỉ lệ dữ liệu giữa các nguồn;
  - trọng số λ của từng loss;
  - sự cần thiết của warm-up λ_U;
  - vai trò của KL(Teacher ‖ Student);
  - lợi ích của việc chia round.
- **Hướng tiếp theo:** thiết kế lại các experiment ngắn, chạy được trên partition normal để chạy song song nhiều job. Chi tiết bổ sung sau.

### 3.12. Relevance Head ở Stage 2 (06/11/2026)

Các vấn đề dưới đây đã kiểm tra trên code và log. **Chưa xử lý:** hiện chỉ tập trung vào augmentation ablation và Stage 2; sau khi xong, có thể phải triển khai lại các experiment để giải quyết các vấn đề này.

**1. Chuỗi riêng cho Relevance Head không có Document.**
- Mỗi pair (Student, Teacher) được mã hoá thành 3 chuỗi (`custom_qwen3_vl_reranker.py`): `<Query>: [ảnh nguồn] instruction <Document>: [ảnh A]` → h_A, tương tự với ảnh B → h_B (Score Head, Tie Head); `<Query>: [ảnh nguồn] instruction` (`_format_source`, không có `<Document>`) → h_src (Relevance Head).
- Lý do ngầm của thiết kế: relevance chỉ phụ thuộc ảnh nguồn + instruction, dùng chung cho A và B nên đổi chỗ A ↔ B không làm đổi S. `pipeline_core.md` không ghi lý do; đây là lựa chọn, không bắt buộc.
- Vấn đề: Qwen3-VL-Reranker được pretrain luôn với cặp (Query, Document); chuỗi không có Document nằm ngoài định dạng đó. Relevance Head = `pretrained_projection(h_src) + residual`, tức dựa vào phép chiếu pretrained áp lên chuỗi khác định dạng pretrain.

**2. Relevance Head không nhận gradient từ Unlabeled.**
- Mọi loss chọn S qua `selected_metric_weights`, hàm này detach relevance; vì vậy L_H, CE Inspector và KL Teacher không train Relevance Head ở mọi cặp.
- Loss relevance duy nhất là L_M (BCE với relevance GemmaGT majority-vote 0/1), chỉ trên cặp có nhãn, đúng 1008 cặp Student-Stage1 đã train; cặp có nhãn không bị augmentation (augmentation chỉ áp lên batch Unlabeled). Stage 2 là lượt thứ ba trên cùng dữ liệu relevance; mức giảm BCE relevance (0.218 → 0.181) đo trên cặp đã train, không chứng minh chọn metric tốt hơn.
- Backbone (LoRA LLM + ViT) có học từ Unlabeled qua h_A, h_B, nên μ trên h_src thay đổi gián tiếp; tham số Relevance Head thì không.
- Trên Unlabeled, Student chọn S từ input đã augmentation, Teacher chọn S từ input gốc. Tỉ lệ mẫu có S trùng nhau (cả 6 metric): baseline 0.92, composite 0.78, mask_content 0.83, blur 0.91. Theo ô metric chỉ khác 1.5% (baseline) đến 6.0% (composite), chủ yếu realism, background, structure; aesthetic và fidelity không đổi.
  - Phần chênh có ngay từ step 1–10 (composite 0.80, baseline 0.925) và gần như không đổi theo step: là phản ứng tức thời của Relevance Head Stage 1 với input đã đổi, không phải kết quả học. Ở baseline là do dropout.
  - 99–100% ô khác nhau có relevance của ít nhất một model trong [0.2, 0.8]; 30% số ô có relevance Teacher trong vùng này, nên ngưỡng cứng 0.5 dễ lật.
  - Ô khác nhau chủ yếu là Student thêm metric (composite 61%, mask_content 71%).
  - Khi S khác, Student = Teacher (overall) thấp hơn 3–5 điểm; đây là tương quan (overall hai bên gộp trên hai S khác nhau, mẫu sát ngưỡng vốn khó hơn), không chứng minh S khác gây ra chênh nhãn.
- **Yêu cầu thiết kế (anh, 06/11), chưa có trong code:** Student, kể cả Relevance Head, phải học được từ Unlabeled, cùng mục tiêu với việc làm khó input bằng augmentation; gặp mẫu augmentation, Relevance Head phải chọn lại metric phù hợp. Chưa định nghĩa đích học của relevance trên mẫu augmentation.

**3. KL(Teacher ‖ Student) chỉ so phân phối 3 lớp.**
- Với mỗi metric m, KL(T_m ‖ S_m) trên (A, B, Tie); không có thành phần relevance trong loss.
- Relevance của Student chỉ dùng làm mặt nạ chọn metric (≥ 0.5, detach, không nhận gradient); relevance của Teacher không dùng trong loss, chỉ ghi log. CE Inspector cũng vậy (gộp `selected_equal` theo S của Student, detach).

**4. Không đo được relevance trên mẫu mới bằng nhãn:** nhãn relevance chỉ có trên 1008 cặp đã train; Bench không có nhãn relevance.

**Hướng sau (chưa làm, nhiều hệ quả): bỏ chuỗi source, Relevance Head dùng chung chuỗi A, B.**
- Lý do: chuỗi source không có `<Document>`, nằm ngoài định dạng pretrain của reranker (MS-Swift: mỗi document một chuỗi ghép với query; ở đây document = ảnh A / ảnh B); nhãn relevance của Gemma được gán khi xem cả ảnh nguồn, A, B; giảm khoảng 1/3 chi phí forward.
- Dạng: \(\mu=\sigma(f_{\text{rel}}([\tfrac{h_A+h_B}{2};|h_A-h_B|]))\), đối xứng khi đổi chỗ A ↔ B; bias khởi tạo theo tỉ lệ relevance của GemmaGT, không dùng \(W_\text{yes}-W_\text{no}\).
- Hệ quả: train lại Student-Stage1, Teacher và baseline Stage 2; sửa `custom_qwen3_vl_reranker.py`, `train_student_warmstart.py`, `train_loop.py` và các script đánh giá.
