# Pipeline core: công thức, hàm loss và các phần tính toán

File này ghi lại các công thức đang chạy trong code ở phiên bản mới nhất (setting 1008 pairs, Stage 2 `train_loop_pipeline_1008_*`). Thông tin chung, hướng đi và kết quả experiment nằm trong `pipeline_process.md`.

Mỗi mục ghi rõ nguồn code để đối chiếu:
- `custom_qwen3_vl_reranker.py` = `Qwen3-VL-Embedding/src/models/custom_qwen3_vl_reranker.py`
- `train_overall_inspector.py`, `train_student_warmstart.py`, `train_student_hardlabel.py`, `train_loop.py` nằm trong `Qwen3-VL-Embedding/scripts/`

---

## 0. Ký hiệu

| Ký hiệu | Ý nghĩa |
|---|---|
| $x = (\text{src}, \text{instr}, I_A, I_B)$ | Một pair: ảnh gốc, câu lệnh chỉnh sửa, hai ảnh đã chỉnh sửa A và B |
| $k \in \{1..6\}$ | Metric: aesthetic, fidelity, realism, background consistency, foreground consistency, structure consistency |
| $h_A, h_B, h_{src}$ | Representation của chuỗi A, chuỗi B và chuỗi chỉ có nguồn (src + instr) |
| $s_k(h)$ | Score của một ảnh ở metric $k$ |
| $d_k = s_k(h_A) - s_k(h_B)$ | Margin của metric $k$ |
| $w_k = \sigma(d_k)$ | Xác suất A thắng khi không Tie |
| $t_k$ | Xác suất Tie (Tie Head) |
| $\rho_k$ | Xác suất metric $k$ liên quan tới câu lệnh (Relevance Head) |
| $P_k = (P_{k,A}, P_{k,B}, P_{k,T})$ | Phân phối ba trạng thái của metric $k$ |
| $\bar P$ | Phân phối ba trạng thái overall (gộp các metric) |
| $\sigma$ | Hàm sigmoid |
| $\mathrm{BCE}(y, p) = -y\log p - (1-y)\log(1-p)$ | Binary cross-entropy |

---

## 1. Backbone và representation

Nguồn: `_pool_qwen_last_hidden`, `CustomQwen3VLForMultiMetricReranking.forward` trong `custom_qwen3_vl_reranker.py`.

- Backbone là Qwen3-VL-Reranker-8B với LoRA trên `all-linear` (LLM + ViT + aligner).
- Code chỉ chạy backbone và bỏ qua `lm_head`. Representation của một chuỗi là hidden state cuối của token hợp lệ cuối cùng (last valid token pooling):
$$h = H_{L}[\,\text{last valid position}\,]$$
- Mỗi pair của Student và Teacher cần **3 chuỗi**: (src, instr, $I_A$) → $h_A$, (src, instr, $I_B$) → $h_B$, (src, instr) → $h_{src}$.
- Mỗi pair của Inspector cần **2 chuỗi** ($h_A$, $h_B$) và không có $h_{src}$.

---

## 2. Các head của Student và Teacher (`MultiMetricRewardHeads`)

Head chạy bằng fp32 (bản sửa lỗi head bf16). Ba head có tham số độc lập với nhau. Student và Teacher có cùng kiến trúc.

### 2.1 Score Head dùng chung cho A và B (`_SharedScoreHead`)

$$s_k(h) = \underbrace{v_k^\top h}_{\text{pretrained projection}} + \underbrace{\big[W_2\,\mathrm{Drop}(\mathrm{GELU}(W_1 h + b_1)) + b_2\big]_k}_{\text{residual MLP}}$$

- $W_1 \in \mathbb{R}^{512 \times D}$, $W_2 \in \mathbb{R}^{6 \times 512}$, dropout 0.05.
- **Khởi tạo:** $W_2 = 0$ và $b_2 = 0$, nên lúc bắt đầu score chỉ là phần projection.
- **Phần projection:** gọi $u = W_{\text{yes}} - W_{\text{no}}$ là hướng yes/no lấy từ `lm_head` của reranker. Với Student, mỗi metric có một hàng riêng:
$$v_k = u + 0.35\,\lVert u\rVert\, e_k$$
  trong đó $e_k$ là các vector đơn vị trực giao với $u$ và trực giao với nhau.
- **Dropout dùng chung mask cho A và B:** cùng một mask được áp lên cả $h_A$ và $h_B$. Vì vậy $d_k$ phản đối xứng chính xác: hoán đổi A↔B thì $d_k \to -d_k$, kể cả khi đang bật dropout.

### 2.2 Tie Head, kiểu `margin_coupled` (`_SymmetricTieHead` + `_MarginCoupledTie`)

Đặc trưng đối xứng (không đổi khi hoán đổi A↔B):
$$u_A = W_u h_A + b_u,\quad u_B = W_u h_B + b_u \quad (W_u \in \mathbb{R}^{256\times D})$$
$$\phi = \big[\tfrac12(u_A + u_B),\ |u_A - u_B|\big] \in \mathbb{R}^{512}$$
$$r_k = \big[\mathrm{MLP}(\phi)\big]_k \quad (\text{Linear } 512\to512,\ \mathrm{GELU},\ \mathrm{Drop}\ 0.05,\ \text{Linear } 512\to 6)$$

Gắn với margin:
$$\mathrm{logit}\, t_k = \theta_{0,k} - \kappa_k\,|d_k| + r_k,\qquad \kappa_k = \mathrm{softplus}(\kappa^{raw}_k) > 0$$

**Khởi tạo:**
- $\theta_{0,k} = \mathrm{logit}(0.2) = -1.386$
- $\kappa_k = 1$
- Lớp cuối của MLP bằng 0, nên ban đầu $r_k = 0$.

**Ý nghĩa:** gradient vào $t$ còn đi tiếp vào $|d|$ (không detach). Vì vậy nhãn Tie kéo $|d|$ về 0, còn nhãn A/B làm $|d|$ lớn hơn.

### 2.3 Relevance Head (`_RelevanceHead`)

Head này chỉ nhìn chuỗi nguồn:
$$\rho_k = \sigma\big(v^{\rho\,\top}_k h_{src} + [W^\rho_2\,\mathrm{Drop}(\mathrm{GELU}(W^\rho_1 h_{src}+b^\rho_1)) + b^\rho_2]_k\big)$$

Khởi tạo giống Score Head: các hàng $v^\rho_k$ là cùng $v_k$ ở 2.1, residual bằng 0.

### 2.4 Phân phối ba trạng thái của một metric

$$P_{k,A} = (1 - t_k)\,w_k,\qquad P_{k,B} = (1 - t_k)(1 - w_k),\qquad P_{k,T} = t_k$$

Signed evidence (bằng chứng có dấu):
$$e_k = P_{k,A} - P_{k,B} = (1 - t_k)(2w_k - 1) \in [-1, 1]$$

---

## 3. Gộp metric thành overall: `selected_equal`

Nguồn: `selected_metric_weights`, `overall_three_way_probs` trong `train_loop.py`.

**Tập metric được chọn** (lấy theo Relevance của chính mô hình, có detach):
$$S = \{k : \rho_k \ge 0.5\};\qquad \text{nếu } S = \varnothing \text{ thì } S = \{\arg\max_k \rho_k\}$$
$$\alpha_k = \frac{\mathbb{1}[k \in S]}{|S|}$$

**Phân phối overall:**
$$\bar P = \sum_k \alpha_k\, P_k = \Big(\sum_k \alpha_k P_{k,A},\ \sum_k \alpha_k P_{k,B},\ \sum_k \alpha_k t_k\Big)$$

**Signed evidence overall:**
$$\bar m = \sum_k \alpha_k e_k$$
Margin AUC (chỉ số chính khi đánh giá) được tính trên đại lượng này.

**Gradient:**
- $\alpha$ được detach và chọn cứng, nên nhãn overall **không** train Relevance Head.
- $t_k$ **không** detach, nên nhãn overall có train Tie Head (và qua coupling, train cả $|d|$).

Phương án `all_metrics_fitted` (trọng số cố định $a_k$ từ Metrics Weights Aggregator) và xác suất `sigmoid_beta` vẫn còn trong code nhưng không dùng. Với loss `three_way` thì tham số `linear` không có tác dụng vì loss dùng thẳng $\bar P$. Tham số này chỉ có tác dụng với loss `binary`, khi đó $P(A) = (1+\bar m)/2$.

---

## 4. Inspector (Inspector Baseline, Inspector pipeline = một checkpoint của nó)

Nguồn: `OverallInspectorHead`, `OverallInspectorLoss` trong `custom_qwen3_vl_reranker.py`; `train_overall_inspector.py`.

### 4.1 Head

Giống Student nhưng chỉ có **1 output overall** và không có Relevance Head:
- $v = W_{\text{yes}} - W_{\text{no}}$ đúng bằng hướng yes/no, không thêm thành phần trực giao. Vì vậy lúc khởi tạo, margin bằng đúng margin của reranker gốc.
- Tie Head kiểu `margin_coupled`, cùng công thức với 2.2 với một output.
- $P^I = \big((1-t)w,\ (1-t)(1-w),\ t\big)$, với $w = \sigma(s(h_A) - s(h_B))$.

### 4.2 Loss: three-way cross-entropy trên vote người

Target là one-hot $y = (y_A, y_B, y_T)$, lấy từ vote `leftvote` / `rightvote` / `tie`.

$$\mathcal{L}_I = \mathrm{BCE}(y_T,\ t) + (y_A + y_B)\,\mathrm{BCE}\Big(\frac{y_A}{y_A+y_B},\ w\Big)$$

Với one-hot, công thức này bằng $-\log P^I_{\text{lớp đúng}}$. Code tính trên logit để ổn định số học. Loss được lấy trung bình trên batch.

### 4.3 Thứ tự dữ liệu

- `train_order.jsonl` cố định thứ tự toàn bộ quá trình train, và dataloader không xáo lại.
- 3000 pair của tập GemmaGT 3k được đặt đầu tiên, theo thứ tự xáo với seed 42. Sau đó là phần còn lại của HumanGT full.
- Ở mỗi step, script đối chiếu label của batch với `train_order.jsonl` và dừng với lỗi nếu lệch.
- Batch là 2 × 8 = 16 pair mỗi step, nên checkpoint-$n$ đã train đúng $16n$ pair đầu tiên của file.

| Mốc | Inspector checkpoint | Pair đã train |
|---|---|---|
| 64 | checkpoint-4 | 64 |
| 1008 | checkpoint-63 | 1008 |
| 2000 | checkpoint-125 | 2000 |
| 3000 | checkpoint-188 | 3008 |

Cấu hình train Inspector Baseline:
- LoRA rank 32 trên LLM, learning rate 5e-5 không đổi, warmup 0.
- 1 lượt qua dữ liệu, Tie Head `margin_coupled`.

### 4.4 Pseudo label của Inspector cho Unlabeled

Nguồn: `infer_overall_inspector.py`, `load_inspector_cache` trong `train_loop.py`.

- Chạy Inspector pipeline ở eval mode (không bật dropout) trên 10k Unlabeled. Cache lưu $P^I = (P^I_A, P^I_B, P^I_T)$.
- Với loss `three_way`, Stage 2 dùng thẳng $P^I$ làm soft target.
- Các đại lượng phụ, không dùng trong loss `three_way`:
  - $p^I_A = P^I_A + \tfrac12 P^I_T$
  - nhãn $= \mathrm{sign}(p^I_A - 0.5)$
  - độ tin cậy $= \max(p^I_A, 1-p^I_A)$

---

## 5. Student Stage 1 (huấn luyện trên GemmaGT, hard label)

Nguồn: `_validate_and_aggregate_runs`, `FactorizedWarmStartLoss`, `train_student_hardlabel.py`.

### 5.1 Nhãn mềm từ 3 lần chạy Gemma

Với mỗi metric $k$, gọi $n_k$ là số lần chạy (trong 3 lần) đánh dấu metric là liên quan (`targeted_metric = 1`). Trong $n_k$ lần đó, đếm số lần chọn A, B và Tie:
$$g_{k,A} = \frac{\#A}{n_k},\quad g_{k,B} = \frac{\#B}{n_k},\quad g_{k,T} = \frac{\#\text{Tie}}{n_k},\qquad r_k = \frac{n_k}{3}$$

Nếu $n_k = 0$ thì $g_{k,\cdot} = 0$, $r_k = 0$, và metric đó chỉ supervise Relevance Head.

### 5.2 Hard label theo đa số (dùng cho train ở Stage 1 và Stage 2)

- **Relevance:** $r_k = 1$ nếu $n_k \ge 2$, ngược lại $r_k = 0$. Khi $r_k = 0$ thì toàn bộ target preference bằng 0.
- **Preference:** khi $r_k = 1$, target là one-hot của lựa chọn chiếm **đa số tuyệt đối** (> 50%) trong các lần chạy liên quan. Nếu không có lựa chọn nào đạt đa số tuyệt đối thì chọn Tie.
- Validation (nếu có) vẫn dùng nhãn mềm.

### 5.3 Loss phân rã (factorized)

$$\mathcal{L}_{pref} = \frac{1}{6B}\sum_{i,k} r_{ik}\Big[\mathrm{BCE}(g_{ik,T},\ t_{ik}) + (g_{ik,A} + g_{ik,B})\,\mathrm{BCE}\Big(\frac{g_{ik,A}}{g_{ik,A}+g_{ik,B}},\ w_{ik}\Big)\Big]$$
$$\mathcal{L}_{rel} = \frac{1}{6B}\sum_{i,k}\mathrm{BCE}(r_{ik},\ \rho_{ik})$$
$$\mathcal{L}_{G} = \mathcal{L}_{pref} + \lambda_{rel}\,\mathcal{L}_{rel},\qquad \lambda_{rel} = 1$$

**Ý nghĩa:**
- $r_k$ là trọng số áp dụng. Metric không liên quan không bị biến thành "Tie ngữ nghĩa".
- Phần Tie và phần chọn người thắng được tách riêng. Phần chọn người thắng chỉ có khối lượng $g_A + g_B$.

### 5.4 Cấu hình train Stage 1

- **Dữ liệu:** $N$ pair đầu của `train_order.jsonl` của Inspector, lấy từ file GemmaGT 3k. Không có validation, chọn checkpoint cuối.
- **Mô hình:** LoRA rank 64, alpha 128, dropout 0.05, train trên LLM + ViT + aligner + `metric_heads`.
- **Tối ưu:**
  - learning rate 4e-5, cosine, warmup 5%, weight decay 0.01;
  - 2 lượt qua dữ liệu, 8 pair mỗi step (1 × gradient accumulation 8);
  - MS-Swift xáo thứ tự pair với seed 42.

### 5.5 Khởi tạo Teacher

`initialize_teacher_model.py`: Teacher là bản sao độc lập, chính xác từng tensor, của Student Stage 1 (LoRA và `metric_heads`). Ở bước $0$: $\theta_T = \theta_S$.

---

## 6. Stage 2: Student-Teacher training loop

Nguồn: vòng lặp chính trong `main()`, `inspector_supervision_loss`, `gemma_replay_losses`, `update_teacher_ema`, `ema_decay_at`, `set_loss_weight` trong `train_loop.py`. Backend là MS-Swift online trainer.

### 6.1 Dữ liệu của một optimizer step

Một step gồm **2 microbatch** (gradient accumulation 2). Mỗi microbatch có 1 HumanGT + 1 GemmaGT + 6 Unlabeled. Tổng cộng mỗi step có **2 HumanGT + 2 GemmaGT + 12 Unlabeled**.

| Nguồn | Tập dữ liệu | Cách lấy |
|---|---|---|
| HumanGT | Đúng $N$ pair Student Stage 1 đã train (`student_train_order`), vote người one-hot, có giữ Tie | Xoay vòng, xáo lại sau mỗi lượt (seed 42 + 10000) |
| GemmaGT | Cùng $N$ pair đó, hard label theo đa số (5.2) | Xoay vòng, xáo lại (seed 42 + 20000) |
| Unlabeled | 10k pair, `all_once`: mỗi pair đúng 1 lần, thứ tự xáo ngẫu nhiên | Theo thứ tự đã xáo |

Với $N = 1008$, 10.000 Unlabeled / 12 mỗi step → **834 step**. HumanGT và GemmaGT được lặp khoảng 834 × 2 / 1008 ≈ 1,65 lượt.

### 6.2 Loss HumanGT (`three_way`)

$$\mathcal{L}_H = -\frac{1}{B_H}\sum_i \sum_{c\in\{A,B,T\}} y_{ic}\,\log \bar P^S_{ic}$$

Trong đó $\bar P^S$ là phân phối overall của Student theo `selected_equal` (mục 3), và $y$ là one-hot vote người.

### 6.3 Loss GemmaGT (replay)

$\mathcal{L}_G$ đúng như 5.3, dùng hard label và $\lambda_{gemma} = \lambda_{mask} = 1$:
$$\mathcal{L}_{G,step} = \lambda_{gemma}\,\gamma_g\,\mathcal{L}_{pref} + \lambda_{mask}\,\gamma_m\,\mathcal{L}_{rel}$$

$\gamma_g$ và $\gamma_m$ là hệ số replay. Chúng bằng 1 và chỉ đổi thành 1.5 ở cuối một lượt, nếu Brier trên Gemma bị trôi quá 0.02 so với ban đầu. Với 1 lượt thì chúng luôn bằng 1.

### 6.4 Loss Unlabeled (`inspector_soft_ema_consistency` + `three_way`)

$$\mathcal{L}_U = \underbrace{-\frac{1}{B_U}\sum_i\sum_c P^I_{ic}\,\log \bar P^S_{ic}}_{\text{CE(Inspector} \to \text{Student overall)}} \;+\; \lambda_C\,\underbrace{\frac{1}{B_U}\sum_i \sum_k \alpha^S_{ik}\,\mathrm{KL}\big(\bar P^T_{ik}\,\|\,P^S_{ik}\big)}_{\text{consistency trên tập } S}$$

- $P^I$: phân phối ba trạng thái của Inspector, lấy từ cache (4.4), không có gradient.
- $\bar P^T_{ik} = \frac{1}{J}\sum_{j=1}^{J} P^{T,(j)}_{ik}$: trung bình $J = 4$ lần forward của Teacher với **MC Dropout có chọn lọc**. Chỉ bật `lora_dropout` và dropout trong `metric_heads`, các phần khác ở eval mode. Không có gradient.
- $\mathrm{KL}(\bar P^T \| P^S) = \sum_c \bar P^T_c(\log \bar P^T_c - \log P^S_c)$, trong đó xác suất được chặn dưới ở $10^{-7}$.
- $\alpha^S$: trọng số `selected_equal` lấy từ Relevance của **Student** (detach). Vì vậy KL là trung bình trên các metric thuộc $S$ của từng pair (`consistency_metrics = selected`).
- $\lambda_C = 1$.

Teacher được forward trên 12 Unlabeled của step **trước** khi Student cập nhật. Dự đoán của Teacher chỉ dùng cho đúng step đó và không lưu lại.

### 6.5 Tổng loss và trọng số theo từng pair

Với $G = 2$ microbatch trong một step:
$$\mathcal{L}_{step} = \frac{1}{G}\sum_{g=1}^{G}\Big[\mathcal{L}_H^{(g)} + \mathcal{L}_{G,step}^{(g)} + \lambda_U(t)\,\mathcal{L}_U^{(g)}\Big]$$

Mỗi loss là trung bình trong microbatch của nó (per-source mean). Vì vậy trọng số hiệu dụng của **một pair** là:

| Nguồn | Trọng số mỗi pair |
|---|---|
| HumanGT | $1/2$ |
| GemmaGT | $1/2$ |
| Unlabeled | $\lambda_U / 12$ |

$\lambda_U = 6$ làm một pair Unlabeled nặng bằng một pair HumanGT, giống tỉ lệ ở setting 3k. Khi đổi số pair mỗi step thì phải tính lại $\lambda$ cho phù hợp.

**Ramp-up của $\lambda_U$** (`set_loss_weight`), với $t$ là step tính từ 0:
$$\lambda_U(t) = \lambda_U \cdot \min\Big(1,\ \frac{t+1}{R}\Big),\qquad R = \mathrm{round}(0.2 \cdot T_{total})$$

Với 834 step: $R = 167$, nên $\lambda_U$ đạt giá trị đầy đủ từ step 167. Hệ số `set_scale` thích nghi chỉ dùng cho objective `projection`. Ở đây nó bằng 1.

### 6.6 Optimizer, learning rate, gradient clipping

- **Optimizer:** AdamW, learning rate $10^{-5}$, $\beta = (0.9,\ 0.95)$, $\epsilon = 10^{-8}$, weight decay 0.01.
- **Tham số được train:** LoRA của Student (rank và alpha lấy từ checkpoint Stage 1) và `metric_heads`, gồm cả $\theta_0$ và $\kappa^{raw}$.
- **Tie coupling learning rate riêng** (chỉ ở phiên bản `..._tie_coupling_lr1e-3`): $\theta_0$ và $\kappa^{raw}$ được tách thành một param group riêng với learning rate $10^{-3}$ và weight decay 0, theo cùng dạng warmup + cosine.
- **Learning rate schedule:** warmup tuyến tính 5% số step, sau đó cosine về 0:
$$\eta(t) = \eta_0\cdot\begin{cases}\dfrac{t+1}{W} & t < W\\[4pt] \tfrac12\Big(1+\cos\big(\pi\,\tfrac{t-W}{T-W}\big)\Big) & t \ge W\end{cases},\qquad W = \mathrm{round}(0.05\,T)$$
- **Gradient clipping:** clip theo norm toàn cục $\lVert g\rVert_2 \le 1.0$.

### 6.7 Cập nhật Teacher bằng EMA

Sau mỗi optimizer step, áp dụng cho mọi tensor dạng số thực của LoRA và `metric_heads`:
$$\theta_T \leftarrow d(t)\,\theta_T + \big(1 - d(t)\big)\,\theta_S$$

Có hai lịch cho $d(t)$ (`ema_decay_at`, $t$ tính từ 0, $T$ là tổng số step):
- **constant:** $d(t) = d$ (phiên bản mới nhất dùng $0.99$).
- **cosine:** $d(t) = d_{end} - \tfrac12(d_{end} - d_{start})\big(1 + \cos(\pi\,\tfrac{t}{T-1})\big)$. Lịch này tăng từ $d_{start} = 0.99$ lên $d_{end} = 0.9999$.

Phần Stage 1 còn lại trong Teacher sau $n$ step là $\prod_{t<n} d(t)$:
- constant 0.99: $0.99^{100} \approx 37\%$, $0.99^{300} \approx 5\%$, $0.99^{834} \approx 0.02\%$.
- cosine 0.99 → 0.9999 trên 834 step: khoảng 1,5% ở cuối quá trình train.

Teacher chỉ dùng để tạo target consistency (6.4). Teacher không nhận gradient.

### 6.8 Lưu checkpoint và chọn checkpoint

- Mỗi 100 step lưu `checkpoint-N.pt`, chứa adapter của cả Student và Teacher. Giữ tối đa 10 checkpoint.
- `best.pt` là checkpoint cuối (`checkpoint_selection = last`), vì không có HumanGT validation.
- Đánh giá offline có thể nạp `{role}_adapter_state_dict` của Student hoặc Teacher từ bất kỳ checkpoint nào.

---

## 7. Cấu hình Stage 2 của các phiên bản 1008 pairs

**Cấu hình chung** (`COMMON_CONFIG` trong `config_training.sh`):
- `three_way`, `selected_equal`, KL trên $S$;
- `all_once` 10k Unlabeled, 1 lượt;
- 1/1/6 × gradient accumulation 2;
- learning rate 1e-5, $\lambda_U = 1$, $\lambda_C = 1$, EMA cosine 0.99 → 0.9999;
- $J = 4$, chọn checkpoint `last`.

| Phiên bản (output folder) | Khác với cấu hình chung |
|---|---|
| `train_loop_pipeline_1008_all_once_three_way_selected_equal` | — |
| `train_loop_pipeline_1008_inspector_teacher_rank_three_rounds_three_way_selected_equal` | 3 lượt, 3.334 Unlabeled mới mỗi lượt. Lượt 1 lấy theo độ tin cậy của Inspector, chỉ giữ pair mà margin của Teacher cùng dấu với nhãn Inspector. Từ lượt 2 có thêm 25% replay: các pair của lượt trước có margin Teacher nhỏ nhất theo hướng nhãn Inspector. |
| `train_loop_pipeline_1008_all_once_three_way_selected_equal_tie_coupling_lr1e-3` | Learning rate 1e-3 cho $\theta_0$ và $\kappa^{raw}$, weight decay 0 |
| `train_loop_pipeline_1008_all_once_without_unlabeled_loss_three_way_selected_equal` | $\lambda_U = 0$. Unlabeled và Teacher vẫn được forward nên luồng dữ liệu và lịch train giữ nguyên. |
| `train_loop_pipeline_1008_all_once_three_way_selected_equal_lambda_unlabeled6_ema_decay_constant0.99` | $\lambda_U = 6$ (mỗi pair Unlabeled nặng bằng một pair HumanGT) và EMA constant 0.99 |
