# Cách tính Accuracy, Margin AUC và Tie Head AUC

Quy chuẩn chung để đánh giá reward model chỉnh sửa ảnh trên EditReward-Data-Test và EditReward-Bench, dùng cho mọi file .md của nhóm (pipeline_process.md, tie_head_ablation.md, ...). Model khác (của nhóm hoặc bên ngoài) phải tính đúng theo các bước dưới đây thì kết quả mới so sánh được. Cập nhật lần cuối 29/10/2026.

Tóm tắt:
- **Tập không có Tie:** Accuracy và margin AUC đối xứng, tức ROC AUC của {d} so với {−d}, với d = điểm ảnh thắng − điểm ảnh thua (mục 3).
- **Tập có Tie:** cặp hoà là lớp dương, mọi cặp còn lại là lớp âm. Margin AUC tính từ −|d|; Tie Head AUC tính từ P(Tie), ghi "—" nếu model không có Tie Head (mục 4).

---

## 1. Tập đánh giá và nhãn

| Tập | Loại | Số cặp | Nhãn |
|---|---|---|---|
| EditReward-Data-Test (1000) | Không có Tie | 1000 | 571 A thắng / 429 B thắng |
| EditReward-Bench (115) | Không có Tie | 115 (133 bỏ 18 cặp hoà) | 41 A thắng / 74 B thắng |
| EditReward-Bench (133) w Tie samples | Có Tie | 133 | 18 cặp hoà |
| EditReward (VLM-Reannotate)-Bench (115) | Không có Tie | 115 cặp không hoà theo nhãn gốc, chấm bằng nhãn VLM | |
| EditReward (VLM-Reannotate)-Bench (133) w Tie samples | Có Tie | 133 | 2 cặp hoà, chỉ để tham khảo |

File (tính từ `/datastore/truongtq/TrainingRewardModel`):
- Test: `RewardModelEvaluation/EvalDatasets/EditReward-Test_pairwise_1k.jsonl`.
- Bench: `RewardModelEvaluation/EvalDatasets/EditReward-Bench_pairwise.json`.
- Nhãn VLM: `RewardModelEvaluation/EvalDatasets/EditReward-Bench_VLM_judge/pairs.csv`, phần `2pair`, cột `majority_overall` (đa số 15 phiếu): 0 → A thắng, 1 → B thắng, 2 → hoà.

Quy ước:
- Nhãn \(y_i = +1\) nếu A tốt hơn, \(-1\) nếu B tốt hơn, 0 nếu hoà.
- Giữ đúng thứ tự A/B như trong file benchmark.
- Báo cáo số cặp đã dùng. Model không chấm được cặp nào thì ghi rõ, không tự bỏ cặp.
- Không calibration hay chọn ngưỡng trên Test hoặc Bench.

## 2. Margin của model

Margin \(m_i\) của cặp i là một số thực: dương nghĩa là model cho A tốt hơn, âm nghĩa là B tốt hơn. **Margin phải là đúng đại lượng mà dấu của nó quyết định lựa chọn A/B của model**, để Accuracy và margin AUC đánh giá cùng một quyết định.

| Loại model | Margin \(m_i\) |
|---|---|
| Chấm điểm từng ảnh (EditReward, ...) | \(m_i = s(A_i) - s(B_i)\), với \(s\) là điểm gốc của model (cùng ảnh nguồn và instruction) |
| Qwen3-VL-Reranker gốc, Inspector của nhóm | \(m_i = \big[\ell_{yes}(A_i) - \ell_{no}(A_i)\big] - \big[\ell_{yes}(B_i) - \ell_{no}(B_i)\big]\), với \(\ell\) là logit của token "yes"/"no" |
| Student / Teacher của nhóm (cách gộp selected_equal) | \(m_i = \frac{1}{\lvert S_i\rvert}\sum_{k \in S_i} e_{ik}\), với \(e_{ik} = (1 - t_{ik})\tanh(\Delta_{ik}/2)\) |
| Trả về xác suất \(P(A \succ B)\) | \(m_i = P(A \succ B) - 0.5\) hoặc logit của nó (cho cùng kết quả) |
| Chỉ trả về A hoặc B | \(m_i \in \{+1, -1\}\); margin AUC khi đó bằng Accuracy |

Ký hiệu của Student / Teacher, với metric k của cặp i:
- \(\Delta_{ik}\) = điểm của A − điểm của B do Score Head cho.
- \(t_{ik}\) = xác suất hoà của Tie Head; \(r_{ik}\) = xác suất liên quan của Relevance Head.
- \(S_i = \{k : r_{ik} \ge 0.5\}\); nếu tập này rỗng thì \(S_i\) chỉ gồm metric có \(r_{ik}\) lớn nhất.
- Margin là **trung bình** trên \(S_i\), không phải tổng. Tổng cho cùng dấu (cùng Accuracy) nhưng khác thứ hạng giữa các cặp, nên khác margin AUC.

Tính chất:
- Margin AUC không đổi khi áp cùng một hàm đồng biến lên margin (sigmoid, tanh, nhân hằng số dương).
- Margin AUC thay đổi nếu biến đổi điểm từng ảnh trước khi lấy hiệu. Model chấm từng ảnh phải dùng điểm gốc.

## 3. Tập không có Tie: Accuracy và margin AUC đối xứng

Áp dụng cho EditReward-Data-Test (1000), EditReward-Bench (115), EditReward (VLM-Reannotate)-Bench (115). Mọi cặp có \(y_i \in \{+1, -1\}\), n là số cặp.

**Margin có hướng.** Đặt

\[
d_i = y_i \cdot m_i = \text{điểm ảnh thắng} - \text{điểm ảnh thua}.
\]

\(d_i > 0\) khi model chọn đúng, \(d_i < 0\) khi chọn sai, \(\lvert d_i\rvert\) là độ tự tin.

**Accuracy.**

\[
\text{Accuracy} = \frac{1}{n}\sum_{i=1}^{n} \mathbb{1}[d_i > 0].
\]

Cặp có \(m_i = 0\) tính là sai.

**Margin AUC (đối xứng).** Mỗi cặp được nhìn theo cả hai thứ tự: thứ tự đúng (ảnh thắng đứng trước) cho mẫu dương có điểm \(d_i\), thứ tự đảo cho mẫu âm có điểm \(-d_i\). Lớp dương \(P = \{d_1, \dots, d_n\}\), lớp âm \(N = \{-d_1, \dots, -d_n\}\). Margin AUC là ROC AUC của P so với N, tức xác suất một mẫu dương ngẫu nhiên có điểm cao hơn một mẫu âm ngẫu nhiên:

\[
\text{margin AUC} = \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\Big(\mathbb{1}[d_i > -d_j] + \tfrac12\,\mathbb{1}[d_i = -d_j]\Big)
= \frac{1}{n^2}\sum_{i=1}^{n}\sum_{j=1}^{n}\Big(\mathbb{1}[d_i + d_j > 0] + \tfrac12\,\mathbb{1}[d_i + d_j = 0]\Big).
\]

Tổng lấy trên mọi \(i, j\), kể cả \(i = j\). Giá trị bằng nhau tính 0.5; cặp có \(m_i = 0\) vẫn giữ lại.

Ý nghĩa và tính chất:
- Với hai cặp i, j: \(d_i + d_j > 0\) khi hoặc cả hai đều đúng, hoặc cặp đúng có độ tự tin lớn hơn cặp sai. Vì vậy margin AUC đo cả việc chọn đúng lẫn việc đặt độ tự tin cao vào đúng các cặp.
- 100 là hoàn hảo (mọi \(d_i > 0\)); 50 là đoán ngẫu nhiên hoặc margin luôn bằng 0.
- Không phụ thuộc thứ tự A/B trong file: đổi chỗ A và B thì cả \(y_i\) và \(m_i\) đổi dấu, \(d_i\) giữ nguyên.
- Mọi so sánh diễn ra bên trong một cặp, nên độ lệch điểm chung của cả cặp (ảnh nguồn hay instruction dễ hoặc khó) bị triệt tiêu.
- Quan hệ với Accuracy p: gọi q là xác suất độ tự tin \(\lvert m\rvert\) ở một cặp đúng lớn hơn ở một cặp sai, thì

\[
\text{margin AUC} \approx p^2 + 2p(1-p)\,q, \qquad \text{margin AUC} - p \approx 2p(1-p)(q - 0.5).
\]

Margin AUC cao hơn Accuracy nghĩa là margin lớn đáng tin hơn margin nhỏ (q > 0.5).

**Ví dụ.** \(y = [+1, -1, +1, -1]\), \(m = [2.0, -0.5, -0.3, 0.0]\) → \(d = [2.0, 0.5, -0.3, 0.0]\). Trong 16 tổng \(d_i + d_j\) có 12 tổng dương và 1 tổng bằng 0, nên margin AUC = (12 + 0.5)/16 = **0.78125**. Accuracy = 2/4 = **0.5** (cặp 4 có margin 0 tính sai).

## 4. Tập có Tie: margin AUC (−|d|) và Tie Head AUC

Áp dụng cho EditReward-Bench (133) w Tie samples và EditReward (VLM-Reannotate)-Bench (133) w Tie samples. Nhãn \(y_i \in \{+1, -1, 0\}\).

**Chuyển cặp hoà thành lớp dương.** Bài toán trở thành phân biệt cặp hoà với cặp không hoà:
- Lớp dương \(T = \{i : y_i = 0\}\) (cặp hoà), \(n_T = \lvert T\rvert\).
- Lớp âm \(N = \{i : y_i \ne 0\}\) (A thắng hoặc B thắng, không phân biệt hướng), \(n_N = \lvert N\rvert\).

Với một điểm \(u_i\) bất kỳ (càng lớn thì model càng cho là hoà), AUC là xác suất một cặp hoà ngẫu nhiên có điểm cao hơn một cặp không hoà ngẫu nhiên:

\[
\text{AUC}(u) = \frac{1}{n_T\, n_N}\sum_{i \in T}\sum_{j \in N}\Big(\mathbb{1}[u_i > u_j] + \tfrac12\,\mathbb{1}[u_i = u_j]\Big).
\]

Mỗi cặp chỉ dùng một lần (khác mục 3, không nhân đôi). 50 là ngẫu nhiên, 100 là tách hoàn hảo cặp hoà khỏi cặp không hoà.

Hai chỉ số khác nhau ở điểm \(u_i\):

| Chỉ số | Điểm \(u_i\) | Ý nghĩa |
|---|---|---|
| **Margin AUC (−\|d\|)** | \(u_i = -\lvert d_i\rvert = -\lvert m_i\rvert\) | Hai ảnh có điểm càng gần nhau thì cặp càng được coi là hoà. Không dùng hướng A/B, nên mọi model có margin đều tính được. |
| **Tie Head AUC** | \(u_i = P_i(\text{Tie})\) | Xác suất hoà do Tie Head dự đoán. Chỉ tính cho model có Tie Head đã được train. |

Margin dùng cho −|d|:
- Model chấm từng ảnh, Qwen3-VL-Reranker gốc, Inspector: \(m_i\) như mục 2.
- Student / Teacher: \(m_i^{\text{tie}} = \frac{1}{\lvert S_i\rvert}\sum_{k \in S_i} \Delta_{ik}\) (trung bình hiệu điểm thô trên \(S_i\)), và \(P_i(\text{Tie}) = \frac{1}{\lvert S_i\rvert}\sum_{k \in S_i} t_{ik}\).

Quy ước:
- Model không có Tie Head (EditReward, Qwen3-VL-Reranker gốc) và model gốc chưa train Tie Head (step 0 của Inspector) ghi "—" ở cột Tie Head AUC. Không thay bằng margin AUC (−|d|).
- **Không** báo margin AUC đối xứng (mục 3) trên phần không hoà như là "margin AUC của tập có Tie". Trên Bench (133), phần không hoà chính là Bench (115), đã có cột riêng.
- VLM Bench (133) chỉ có 2 cặp hoà (\(n_T = 2\)), nên cả hai chỉ số chỉ để tham khảo.

## 5. Khoảng tin cậy và so sánh hai model

- **Khoảng tin cậy 95%:** lấy mẫu lặp có hoàn lại trên các cặp của tập (tập có Tie: trên cả 133 cặp), 1000 lần với seed 42, rồi lấy phân vị 2.5 và 97.5 của chỉ số.
- **So sánh hai model:** ghép hai model theo cùng một cặp, dùng cùng một tập mẫu lặp cho cả hai và tính chênh lệch chỉ số trên từng lần lấy mẫu. Khoảng tin cậy của chênh lệch không chứa 0 thì khác biệt có ý nghĩa. Với Accuracy, dùng kiểm định McNemar chính xác.
- Không kết luận bằng cách so hai khoảng tin cậy riêng lẻ: trên Bench mỗi khoảng rộng khoảng ±8 điểm.

## 6. Những cách tính không dùng

- **AUC gộp (pooled AUC, trước 21/10/2026):** so điểm ảnh thắng và ảnh thua giữa các cặp khác ảnh nguồn và instruction. Thấp hơn margin AUC 13–16 điểm trên Test và từng dẫn đến kết luận sai.
- **AUC một chiều theo nhãn A/B** (lớp dương là cặp A thắng, điểm là \(m_i\)): phụ thuộc thứ tự A/B và tỉ lệ nhãn.
- **Margin AUC đối xứng trên phần không hoà, báo như margin AUC của tập có Tie** (mục 4).
- **Tổng thay cho trung bình** trên \(S_i\) với Student / Teacher (mục 2). `evaluate_student_checkpoints.py` tính theo tổng đến ngày 29/10/2026: trong các file `evaluation.json` của Student / Teacher tạo trước ngày này, `selected_equal_margin` và `margin_auc` của `selected_equal` (tập không có Tie) là theo tổng, Accuracy vẫn đúng. Số đúng tính lại từ dự đoán đã lưu bằng `recompute_margin_auc.py` hoặc `analyze_student_checkpoints.py`. Phần tập có Tie của script này luôn đúng mục 4.
- **−\|trung bình `signed_evidence`\|** thay cho −\|trung bình \(\Delta_{ik}\)\| trên tập có Tie với Student / Teacher (mục 4). Margin AUC (−\|d\|) là mốc chỉ dùng Score Head; ảnh hưởng của Tie Head được đo riêng ở cột Tie Head AUC. `signed_evidence` = \((1-t_{ik})\tanh(\Delta_{ik}/2)\) đã chứa \(t_{ik}\), nên dùng nó sẽ trộn Tie Head vào mốc này: hai cột không còn so được với nhau để biết Tie Head đóng góp thêm bao nhiêu, và không còn so được với model không có Tie Head (EditReward, Qwen3-VL-Reranker gốc) hay với Inspector (−\|d\| từ hiệu điểm thô).

## 7. Bảng số liệu: Inspector pipeline tại 112 training samples

Inspector pipeline = checkpoint của Inspector Baseline (`inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant`, job 6063, 16 cặp mỗi bước). Mốc 112 training samples là **checkpoint-7**; mốc 64 là checkpoint-4. Hàng Inspector Baseline là checkpoint có performance cao nhất của cùng run trên toàn bộ EditReward-Data (165744 cặp, 10359 bước): **checkpoint-625** (10000 cặp), chọn theo margin AUC trên Bench (115), cũng là checkpoint có Accuracy Bench (115) cao nhất. Checkpoint cuối (checkpoint-10359) đạt 75.65 / 77.10 trên Bench (115).

| Model | Bench (115) Acc | Bench (115) Margin AUC | Bench (133) w Tie: Margin AUC (−\|d\|) | Bench (133) w Tie: Tie Head AUC | VLM Bench (115) Acc | VLM Bench (115) Margin AUC | VLM Bench (133) w Tie: Margin AUC (−\|d\|) | VLM Bench (133) w Tie: Tie Head AUC |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| EditReward (`OtherModels/EditReward/evaluation_results_vl_inferencer`, 22/10) | 77.39 | 81.69 | 62.51 | — | 92.17 | 96.95 | 82.82 | — |
| Qwen3-VL-Reranker gốc | 64.35 | 71.97 | 63.89 | — | 82.61 | 91.81 | 90.27 | — |
| Inspector Baseline, full EditReward-Data (checkpoint-625, 10000 cặp) | 77.39 | 80.00 | 68.62 | 68.26 | 95.65 | 98.62 | 98.47 | 98.09 |
| Inspector Baseline, model gốc (step 0, job 6063) | 61.74 | 71.80 | 63.19 | — | 81.74 | 91.43 | 90.84 | — |
| Inspector pipeline, 64 training samples (checkpoint-4) | 64.35 | 71.97 | 60.56 | 60.41 | 83.48 | 91.22 | 86.26 | 86.26 |
| **Inspector pipeline, 112 training samples (checkpoint-7)** | **65.22** | **72.75** | **61.71** | **61.84** | **86.96** | **94.42** | **90.27** | **90.46** |

Khoảng tin cậy 95% của checkpoint-7: Bench (115) margin AUC [63.85, 81.57]; Bench (133) margin AUC (−|d|) [49.15, 74.84], Tie Head AUC [49.20, 74.98]; VLM Bench (115) margin AUC [89.93, 97.58].

Nhận xét:
- So với 64 training samples, checkpoint-7 tăng nhẹ trên mọi cột; rõ nhất ở VLM Bench (115): Accuracy +3.48, margin AUC +3.20.
- Inspector pipeline 112 cặp còn cách Inspector Baseline (checkpoint-625) 12.17 điểm Accuracy và 7.25 điểm margin AUC trên Bench (115), 6.91 / 6.42 điểm trên Bench (133) w Tie.
- Trên Bench (133), Tie Head AUC vẫn bám sát margin AUC (−|d|) (chênh 0.1), và cả hai vẫn thấp hơn mô hình gốc không có Tie Head (63.19 / 63.89). Khoảng tin cậy đều chứa 50: chưa tách được cặp hoà.
- EditReward-Data-Test của checkpoint-4 và checkpoint-7 đang được chấm (output `evaluation_results/inspector/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_with_test`), sẽ bổ sung vào bảng.

Nguồn số liệu:
- Inspector: `RewardModelEvaluation/OurModel/evaluation_results/inspector/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant/training_analysis/training_analysis.json`.
- Muốn kiểm tra cài đặt của mình, tính lại từ `checkpoint-7.json` cùng thư mục Inspector; kết quả phải ra đúng hàng in đậm.
