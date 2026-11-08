# Thí nghiệm cấu hình huấn luyện Overall Inspector Model

Cập nhật: 2026-10-17

**Kết luận:** chọn **phiên bản 10, checkpoint-200** (`output/overall_inspector_r32_llm_lr8e-5/checkpoint-200`) làm Overall Inspector Model.

## 1. Bối cảnh

- Model gốc: Qwen3-VL-Reranker-8B, huấn luyện bằng MS-Swift 4.5.3, `task_type generative_reranker`, `loss_type listwise_reranker`.
- Dữ liệu huấn luyện: HumanGT Data (`TrainingDataCreation/editreward_overall_humanGT_3k.jsonl`), chỉ dùng cặp không hòa. MS-Swift tách tập kiểm định với `split_dataset_ratio 0.15`: 353 cặp kiểm định, khoảng 2000 cặp huấn luyện (251 bước tối ưu mỗi epoch với batch 1 × tích lũy gradient 8).
- Phiên bản đầu tiên (gọi là baseline cũ) học quá khớp rõ rệt: loss đánh giá thấp nhất ở bước 200 (0.5925), rồi tăng lên 0.7687 ở bước 300 khi bắt đầu epoch 2 và lên 1.12 ở cuối 4 epoch. Model có 419.95M tham số huấn luyện (4.57%) với rank 128 và LoRA trên cả LLM lẫn ViT.
- Mục tiêu: tìm cấu hình learning rate, lịch learning rate, rank và phạm vi LoRA phù hợp với lượng dữ liệu nhỏ.

## 2. Các điểm kỹ thuật đã kiểm tra

- **Hàm loss:** `ListwiseRerankerLoss` của MS-Swift tính cross-entropy trên từng nhóm, ảnh đúng ở vị trí 0, nhiệt độ 1.0. Với nhóm `[ảnh thắng, ảnh thua]`, loss bằng `-log σ(s_thắng − s_thua)`, tức đúng dạng Bradley-Terry, khớp tài liệu Reranker của MS-Swift.
- **Điểm số:** dùng đúng điểm gốc của MS-Swift, `logit(yes) − logit(no)` tại token hợp lệ cuối cùng.
- **Bỏ scalar head:** phiên bản đầu huấn luyện thêm `overall_inspector_head` (4096 tham số) qua `modules_to_save`. Head này được khởi tạo bằng `W_yes − W_no`, nên khi không huấn luyện nó cho đúng điểm của MS-Swift. Từ phiên bản 1 trở đi đã bỏ head để bám sát MS-Swift.
- **Lịch learning rate:** cosine có warmup 5%, cập nhật theo từng bước tối ưu (không cố định theo epoch). Với 2 epoch (502 bước), epoch 1 nhận khoảng 80% tổng learning rate cộng dồn. Với 4 epoch của baseline cũ, epoch 1 chỉ nhận 44%.
- **Cắt gradient:** độ lớn gradient trước khi cắt của baseline cũ có trung vị 349, nên 100% bước bị cắt về 1.0. Với AdamW, độ lớn mỗi bước cập nhật vẫn tỉ lệ với learning rate. Cắt gradient chỉ làm các bước đóng góp đều nhau. Giữ nguyên batch 1×8 và `max_grad_norm 1.0` như ví dụ của MS-Swift.
- **Loss hiển thị khi huấn luyện:** MS-Swift luôn đặt `compute_loss_func`, nên loss không được chia cho số bước tích lũy. Loss huấn luyện được ghi ra lớn khoảng 8 lần loss của từng cặp. Loss đánh giá không bị ảnh hưởng.
- **Ngân sách cập nhật:** ước lượng thô bằng (alpha / rank) × learning rate × rank. Vì alpha luôn bằng 2 × rank, ngân sách cập nhật = 2 × learning rate × rank.

## 3. Thay đổi code

- `run_overall_inspector_training.slurm`: thêm các biến `INSPECTOR_LORA_RANK`, `INSPECTOR_LEARNING_RATE`, `INSPECTOR_NUM_TRAIN_EPOCHS`, `INSPECTOR_FREEZE_VISUAL`. `lora_alpha` luôn bằng 2 × rank. Bỏ `--modules_to_save overall_inspector_head`.
- `Qwen3-VL-Embedding/scripts/train_overall_inspector.py`: bỏ scalar head. Thêm bộ bọc chỉ để loại `pair_marker` trước forward gốc của MS-Swift. Khi đóng băng ViT thì LoRA chỉ gắn vào 252 lớp Linear của LLM (q/k/v/o/gate/up/down × 36 lớp).
- `Qwen3-VL-Embedding/scripts/evaluate_inspector_checkpoints.py`: không còn yêu cầu trọng số head. EditReward-Bench chỉ giữ 115 mẫu không hòa. Thêm AUC gộp và lưu điểm riêng từng ảnh.
- `Qwen3-VL-Embedding/scripts/infer_overall_inspector.py`: kết quả có thêm `score_A`, `score_B`.

## 4. Thiết lập chung

Giữ nguyên ở mọi phiên bản: batch 1 × tích lũy gradient 8, warmup 0.05, cosine, `max_grad_norm 1.0`, `lora_dropout 0.05`, `weight_decay 0.01`, `target_modules all-linear`, `max_length 16384`, seed 42, đánh giá và lưu mỗi 50 bước. Checkpoint được chọn theo loss đánh giá thấp nhất trên tập kiểm định của MS-Swift. EditReward-Data-Test và EditReward-Bench chỉ dùng để đánh giá ngoại tuyến.

## 5. Các phiên bản

Thư mục của phiên bản 1–9 nằm trong `Qwen3-VL-Embedding/output_training/`. Thư mục của phiên bản 10–12 nằm trong `Qwen3-VL-Embedding/output/`.

| Phiên bản | Thư mục | Rank | Phạm vi LoRA | Learning rate | Epoch | Tham số huấn luyện | Ngân sách cập nhật | Job |
|---|---|---|---|---|---|---|---|---|
| Baseline cũ | `overall_inspector` | 128 | LLM + ViT + head | 5e-6 | 4 | 419.95M | 1.28e-3 | — |
| 1 | `overall_inspector_r32` | 32 | LLM + ViT | 1e-5 | 2 | 104.99M | 6.4e-4 | 4054 |
| 2 | `overall_inspector_r16` | 16 | LLM + ViT | 1e-5 | 2 | 52.49M | 3.2e-4 | 4055 |
| 3 | `overall_inspector_r16_llm` | 16 | LLM | 1e-5 | 2 | 43.65M | 3.2e-4 | 4056 |
| 4 | `overall_inspector_r32_llm_lr2e-5` | 32 | LLM | 2e-5 | 2 | 87.29M | 1.28e-3 | 4057 |
| 5 | `overall_inspector_r32_lr5e-6` | 32 | LLM + ViT | 5e-6 | 2 | 104.99M | 3.2e-4 | 4065 |
| 6 | `overall_inspector_r16_llm_lr2e-5` | 16 | LLM | 2e-5 | 2 | 43.65M | 6.4e-4 | 4066 |
| 7 | `overall_inspector_r32_llm_lr4e-5` | 32 | LLM | 4e-5 | 2 | 87.29M | 2.56e-3 | 4200 |
| 8 | `overall_inspector_r64_llm_lr2e-5` | 64 | LLM | 2e-5 | 2 | 174.59M | 2.56e-3 | 4201 |
| 9 | `overall_inspector_r32_llm_lr2e-5_epoch1` | 32 | LLM | 2e-5 | 1 | 87.29M | 1.28e-3 | 4202 |
| 10 | `overall_inspector_r32_llm_lr8e-5` | 32 | LLM | 8e-5 | 2 | 87.29M | 5.12e-3 | 4260 |
| 11 | `overall_inspector_r32_llm_lr5e-6` | 32 | LLM | 5e-6 | 2 | 87.29M | 3.2e-4 | 4261 |
| 12 | `overall_inspector_r32_llm_lr2e-5_rerun` | 32 | LLM | 2e-5 | 2 | 87.29M | 1.28e-3 | 4262 |

Phiên bản 4, 5 và 6 bị dừng giữa chừng do server sập (lần lượt ở bước 450, 350, 400), nhưng đã lưu checkpoint-200. Riêng phiên bản 5 còn lưu checkpoint-250. Các phiên bản 7, 8, 9 lấy phiên bản 4 làm gốc và mỗi phiên bản chỉ đổi một yếu tố. Phiên bản 10 và 11 cũng lấy phiên bản 4 làm gốc và chỉ đổi learning rate. Phiên bản 12 chạy lại đúng cấu hình và seed của phiên bản 4 để kiểm tra tính lặp lại. Phiên bản 10, 11, 12 được submit từ một account Slurm khác nhưng dùng cùng môi trường, mã nguồn và thư mục dữ liệu.

## 6. Loss đánh giá trên tập kiểm định của MS-Swift

Mỗi ô ghi loss / độ chính xác. In đậm là bước có loss thấp nhất.

| Bước | Baseline cũ | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 50 | 0.6423 / 0.646 | 0.6455 / 0.663 | 0.6507 / 0.660 | 0.6529 / 0.680 | 0.6155 / 0.691 | 0.6531 / 0.663 | 0.6348 / 0.686 | 0.6880 / 0.671 | 0.6578 / 0.674 | 0.6130 / 0.688 | 0.6431 / 0.691 | 0.6526 / 0.683 | 0.6135 / 0.691 |
| 100 | 0.6264 / 0.660 | 0.6228 / 0.680 | 0.6293 / 0.683 | 0.6311 / 0.660 | 0.6098 / 0.680 | 0.6393 / 0.666 | 0.6198 / 0.677 | 0.6034 / 0.669 | 0.6050 / 0.671 | 0.6104 / 0.671 | 0.6145 / 0.703 | 0.6372 / 0.677 | 0.6114 / 0.677 |
| 150 | 0.6070 / 0.671 | 0.6134 / 0.683 | 0.6206 / 0.671 | 0.6174 / 0.677 | 0.6084 / 0.700 | 0.6275 / 0.680 | 0.6124 / 0.691 | 0.6021 / 0.691 | 0.5992 / 0.691 | 0.6247 / 0.691 | 0.6184 / 0.711 | 0.6244 / 0.671 | 0.6126 / 0.691 |
| 200 | **0.5925** / 0.691 | **0.6027** / 0.691 | **0.6070** / 0.683 | **0.6045** / 0.674 | **0.5933** / 0.697 | 0.6146 / 0.674 | **0.6067** / 0.697 | **0.5835** / 0.720 | **0.5877** / 0.703 | **0.6068** / 0.691 | **0.5775** / 0.728 | 0.6135 / 0.669 | **0.5912** / 0.697 |
| 250 | 0.6049 / 0.700 | 0.6244 / 0.691 | 0.6121 / 0.674 | 0.6183 / 0.669 | 0.6188 / 0.708 | **0.6084** / 0.674 | 0.6286 / 0.697 | 0.6143 / 0.711 | 0.6158 / 0.708 | 0.6086 / 0.694 | 0.6111 / 0.722 | **0.6107** / 0.677 | 0.6159 / 0.711 |
| 300 | 0.7687 / 0.708 | 0.6926 / 0.694 | 0.6280 / 0.677 | 0.6345 / 0.680 | 0.7211 / 0.720 | 0.6150 / 0.671 | 0.6987 / 0.697 | 0.8853 / 0.720 | 0.7927 / 0.717 | — | 1.2203 / 0.717 | 0.6128 / 0.680 | 0.7202 / 0.708 |
| 400 | 0.6026 / 0.705 | 0.6362 / 0.703 | 0.6231 / 0.680 | 0.6185 / 0.691 | 0.6292 / 0.714 | — | 0.6348 / 0.708 | 0.6496 / 0.720 | 0.6665 / 0.711 | — | 0.7086 / 0.731 | 0.6146 / 0.680 | 0.6234 / 0.708 |
| 502 | — | 0.6304 / 0.711 | 0.6252 / 0.677 | 0.6181 / 0.691 | — | — | — | 0.6672 / 0.711 | 0.6695 / 0.705 | — | 0.7535 / 0.734 | 0.6130 / 0.674 | 0.6250 / 0.711 |

Ghi chú:
- Phiên bản 9 kết thúc ở bước 251 với loss 0.6086.
- Baseline cũ chạy 4 epoch, tới bước 1004 với loss 1.1199.
- Phiên bản 10, 11, 12 có thêm các bước 350, 450, 500, không ghi vào bảng. Phiên bản 11 chỉ còn lưu checkpoint-250, không còn checkpoint-200.

## 7. Kết quả đánh giá ngoại tuyến

- EditReward-Data-Test: 1000 cặp không hòa.
- EditReward-Bench: 115 cặp không hòa, đã bỏ 18 cặp hòa trong tổng 133 cặp.
- Kết quả nằm trong `RewardModelEvaluation/OurModel/evaluation_results/inspector_checkpoints/<tên thư mục huấn luyện>/`.

Định nghĩa chỉ số:
- **Độ chính xác:** tỉ lệ cặp có dấu của (điểm A − điểm B) khớp nhãn.
- **Margin AUC (AUC chính, từ 21/10/2026):** với mỗi cặp đặt d = nhãn × (điểm A − điểm B), rồi tính ROC AUC của {d} so với {−d}. Kết quả bằng xác suất d_i + d_j > 0 và không phụ thuộc thứ tự A/B. Chỉ số này đo xếp hạng trong cặp và việc độ tự tin có đi cùng độ đúng hay không. Cách tính và lý do nằm ở `pipeline_process.md` mục 3.6. Khoảng tin cậy của margin AUC lấy từ 1000 lần lấy mẫu lặp theo cặp (`RewardModelEvaluation/OurModel/recompute_margin_auc.py`, kết quả ở `evaluation_results/margin_auc_summary.json`).
- **AUC gộp (cũ, chỉ tham khảo):** gom điểm của ảnh thắng trong mọi cặp làm nhóm dương và điểm của ảnh thua làm nhóm âm. Cách này chủ yếu so sánh ảnh khác ảnh nguồn và khác instruction, nên bị chi phối bởi mức điểm riêng của từng ngữ cảnh.
- **Brier và log loss:** tính trên xác suất sigmoid(điểm A − điểm B), nhiệt độ 1.
- **Khoảng tin cậy 95% của độ chính xác:** ước lượng bằng lấy mẫu lặp 2000 lần.

Ở bảng dưới, in đậm là giá trị tốt nhất của mỗi cột.

| Phiên bản | Checkpoint | Test độ chính xác | Test margin AUC | Test Brier | Test log loss | Bench độ chính xác | Bench margin AUC | Bench Brier | AUC gộp cũ (Test / Bench) |
|---|---|---|---|---|---|---|---|---|---|
| Model gốc chưa huấn luyện | — | 61.5 | 66.55 | 0.2329 | 0.6779 | 61.7 | 71.80 | 0.2139 | 57.4 / 64.5 |
| Baseline cũ | 200 | 74.0 [71.4, 76.8] | 79.22 [76.46, 81.72] | 0.1872 | 0.5603 | 68.7 [60.0, 76.5] | 75.97 | 0.2046 | 66.9 / 69.1 |
| 1 | 200 | 72.7 | 77.69 | 0.1937 | 0.5791 | 67.8 | 74.22 | 0.2145 | 65.8 / 68.0 |
| 2 | 200 | 69.3 | 75.55 | 0.2031 | 0.6046 | 68.7 | 74.53 | 0.2091 | 63.2 / 67.4 |
| 3 | 200 | 70.2 | 75.77 | 0.2022 | 0.6024 | 67.8 | 74.82 | 0.2085 | 63.5 / 67.6 |
| 4 | 200 | 73.8 [71.0, 76.5] | 79.48 [76.59, 82.07] | 0.1856 | 0.5549 | 71.3 | 76.23 | 0.2021 | 67.0 / 69.8 |
| 5 | 200 | 66.6 | 73.51 | 0.2108 | 0.6234 | 67.8 | 74.27 | 0.2091 | 61.5 / 66.5 |
| 5 | 250 | 68.8 | 74.94 | 0.2057 | 0.6143 | 68.7 | 74.02 | 0.2150 | 62.8 / 66.9 |
| 6 | 200 | 72.3 | 78.44 | 0.1904 | 0.5700 | 71.3 | 75.41 | 0.2094 | 66.3 / 68.7 |
| 7 | 200 | 74.2 [71.4, 76.9] | 81.39 [78.64, 83.75] | 0.1771 | 0.5320 | 74.8 [66.1, 82.6] | 78.01 | 0.1938 | 67.8 / 71.5 |
| 8 | 200 | 73.7 | 80.55 | 0.1806 | 0.5413 | 73.0 | 78.03 | 0.1956 | 67.7 / 71.3 |
| 9 | 200 | 72.4 | 78.56 | 0.1897 | 0.5708 | 68.7 | 74.30 | 0.2173 | 66.3 / 68.1 |
| 9 | 251 | 72.5 | 78.60 | 0.1895 | 0.5709 | 68.7 | 74.25 | 0.2182 | 66.4 / 68.2 |
| **10** | **200** | **74.5** [71.9, 77.2] | **82.41** [79.73, 84.81] | **0.1713** | **0.5156** | **76.5** [68.7, 84.3] | **79.81** [71.16, 87.71] | **0.1894** | 65.9 / 72.0 |
| 11 | 250 | 69.1 [66.3, 72.0] | 75.15 | 0.2053 | 0.6144 | 67.8 [59.1, 75.7] | 74.74 | 0.2129 | 63.0 / 67.2 |
| 12 | 200 | 73.4 [70.6, 76.2] | 79.51 [76.58, 82.09] | 0.1856 | 0.5546 | 72.2 [64.3, 80.0] | 76.83 | 0.1996 | 67.1 / 70.1 |

### So sánh từng cặp, chỉ đổi một yếu tố (EditReward-Data-Test)

Kiểm định McNemar: xét các mẫu mà hai model trả lời khác nhau, đếm số mẫu chỉ model bên trái đúng và số mẫu chỉ model bên phải đúng. Chênh lệch margin AUC là trái trừ phải, tính trên cùng từng cặp, khoảng tin cậy 95% từ 1000 lần lấy mẫu lặp theo cặp.

| Yếu tố | So sánh | Độ chính xác | Chỉ trái đúng / chỉ phải đúng | Giá trị p | Chênh lệch margin AUC |
|---|---|---|---|---|---|
| Rank 32 so với 16 | 1 với 2 | 72.7 vs 69.3 | 54 / 20 | < 0.001 | **+2.14 [+1.19, +3.08]** |
| Rank 32 so với 16 | 4 với 6 | 73.8 vs 72.3 | 28 / 13 | 0.028 | **+1.03 [+0.67, +1.46]** |
| Learning rate 2e-5 so với 1e-5 | 6 với 3 | 72.3 vs 70.2 | 56 / 35 | 0.035 | **+2.67 [+1.61, +3.76]** |
| Learning rate 1e-5 so với 5e-6 | 1 với 5 | 72.7 vs 66.6 | 88 / 27 | < 0.001 | **+4.18 [+2.76, +5.60]** |
| Chỉ LLM so với LLM + ViT | 3 với 2 | 70.2 vs 69.3 | 19 / 10 | 0.136 | +0.22 [−0.02, +0.51] |
| Learning rate 4e-5 so với 2e-5 | 7 với 4 | 74.2 vs 73.8 | 31 / 27 | 0.694 | **+1.92 [+1.20, +2.65]** |
| Rank 64 so với 32 | 8 với 4 | 73.7 vs 73.8 | 18 / 19 | 1.000 | **+1.08 [+0.53, +1.62]** |
| 1 epoch so với 2 epoch | 9 với 4 | 72.4 vs 73.8 | 16 / 30 | 0.054 | **−0.91 [−1.44, −0.38]** |
| Learning rate 8e-5 so với 4e-5 | 10 với 7 | 74.5 vs 74.2 | 43 / 40 | 0.826 | **+1.02 [+0.12, +1.93]** |
| Learning rate 8e-5 so với 2e-5 | 10 với 4 | 74.5 vs 73.8 | 59 / 52 | 0.569 | **+2.94 [+1.70, +4.13]** |
| Learning rate 2e-5 so với 5e-6 | 4 với 11 | 73.8 vs 69.1 | 82 / 35 | < 0.001 | **+4.33 [+3.05, +5.62]** |
| Chỉ LLM so với LLM + ViT (rank 32, learning rate 5e-6) | 11 với 5 | 69.1 vs 68.8 | 16 / 13 | 0.711 | +0.21 [−0.07, +0.47] |
| Chạy lại cùng cấu hình, cùng seed | 12 với 4 | 73.4 vs 73.8 | 8 / 12 | 0.503 | +0.03 [−0.20, +0.27] |

Phiên bản 10 so với baseline cũ: margin AUC +3.19 [+1.69, +4.65] trên Test, +3.84 [−0.34, +8.38] trên Bench. Phiên bản 7 so với 8 (cùng ngân sách, tăng learning rate so với tăng rank): +0.84 [+0.39, +1.25] trên Test.

Chênh lệch xác suất so với phiên bản 4 trên Test. Model được so trên cùng từng mẫu, khoảng tin cậy 95% lấy từ 5000 lần lấy mẫu lặp:

| Phiên bản | Chênh lệch Brier | Chênh lệch log loss |
|---|---|---|
| 7 | −0.0085 [−0.0117, −0.0053] | −0.0228 [−0.0309, −0.0144] |
| 8 | −0.0050 [−0.0074, −0.0026] | −0.0136 [−0.0197, −0.0072] |
| Baseline cũ | +0.0016 [−0.0011, +0.0043] | +0.0055 [−0.0016, +0.0125] |
| 10 | −0.0144 [−0.0198, −0.0090] | −0.0393 [−0.0533, −0.0249] |
| 11 | +0.0197 [+0.0142, +0.0250] | +0.0596 [+0.0431, +0.0764] |
| 12 | −0.0000 [−0.0010, +0.0010] | −0.0002 [−0.0027, +0.0022] |

So sánh trực tiếp phiên bản 10 với phiên bản 7 trên Test, cùng cách tính: chênh lệch Brier −0.0058 [−0.0098, −0.0018], chênh lệch log loss −0.0165 [−0.0268, −0.0059].

### Tính lặp lại (phiên bản 12 so với phiên bản 4)

| Tập | Cùng dấu margin | Tương quan Pearson của margin | Trung bình \|chênh lệch margin\| | Trung bình \|margin\| của phiên bản 4 |
|---|---|---|---|---|
| Test | 97.9% | 0.997 | 0.074 | 1.030 |
| Bench | 99.1% | 0.998 | 0.088 | 1.466 |

Loss kiểm định ở bước 200: 0.5912 (phiên bản 12) so với 0.5933 (phiên bản 4).

## 8. Kết luận

1. **Checkpoint tốt nhất luôn nằm quanh cuối epoch 1 (bước 200).** Khi dữ liệu bắt đầu lặp lại ở epoch 2, model học quá khớp: loss đánh giá tăng vọt ở bước 300. Learning rate càng lớn thì học quá khớp càng mạnh: phiên bản 10 (learning rate 8e-5) có loss 1.22 ở bước 300. Riêng phiên bản 11 (learning rate 5e-6) đạt loss thấp nhất ở bước 250 và gần như không tăng lại. Chọn checkpoint theo loss kiểm định xử lý được vấn đề này.
2. **Ở bước 200, độ chính xác tăng theo ngân sách cập nhật,** cho tới khoảng 1.28e-3. Giảm rank hoặc giảm learning rate không giúp model tổng quát hóa tốt hơn, chỉ làm model học chưa đủ. Phiên bản 11 (learning rate 5e-6) kém phiên bản 4 rõ rệt (p < 0.001). Từ 1.28e-3 trở lên, độ chính xác trên Test chạm trần khoảng 74%.
3. **Tăng ngân sách lên 2.56e-3 rồi 5.12e-3 không tăng độ chính xác nhưng làm xếp hạng độ tự tin và xác suất tốt hơn rõ rệt.** Margin AUC trên Test tăng đều: phiên bản 4 79.48 → phiên bản 7 81.39 → phiên bản 10 82.41. Mỗi bước tăng có khoảng tin cậy không chứa 0. Brier và log loss cũng giảm một cách đáng tin cậy ở mỗi lần tăng. Model cùng độ chính xác nhưng đặt margin lớn vào đúng các cặp dễ, margin nhỏ vào các cặp khó. Đây là yếu tố quan trọng vì xác suất của Inspector được dùng làm nhãn mềm. Phiên bản 10 tốt hơn phiên bản 7, và khoảng tin cậy của chênh lệch không chứa 0. 8e-5 là learning rate lớn nhất đã thử, nên chưa biết xác suất còn cải thiện khi tăng tiếp hay không. Cùng ngân sách đó, tăng learning rate (phiên bản 7) tốt hơn tăng rank (phiên bản 8), và chỉ cần một nửa số tham số.
4. **LoRA trên ViT không có lợi.** Độ chính xác ngang bản chỉ dùng LLM ở rank 16 (phiên bản 3 với 2, p = 0.136) và ở rank 32 (phiên bản 11 với 5, p = 0.711, Brier gần như trùng), nhưng tốn thêm tham số. Độ lớn gradient của các bản có ViT là 47–260, so với 9–29 khi chỉ dùng LLM.
5. **Lịch 1 epoch kém hơn** lịch 2 epoch có chọn checkpoint ở bước 200, vì nhận ít hơn khoảng 26% tổng learning rate cộng dồn. Phần cuối lịch, khi learning rate giảm về 0, không cải thiện gì thêm.
6. **EditReward-Bench không đủ lớn để phân biệt các phiên bản.** Khoảng tin cậy của độ chính xác rộng khoảng ±8.5 điểm, của margin AUC khoảng ±8 điểm. Chênh lệch margin AUC theo cặp trên Bench cùng chiều với Test nhưng hầu hết khoảng tin cậy chứa 0. Bench chỉ dùng để kiểm tra bổ sung. Mọi phiên bản đã huấn luyện đều tốt hơn model gốc trên cả hai tập.
7. **Margin AUC thay cho AUC gộp (cập nhật 21/10/2026).** Kết luận cũ "AUC gộp thấp hơn độ chính xác, phiên bản 10 có AUC gộp thấp hơn phiên bản 7" là do cách tính gộp. AUC gộp so sánh ảnh của các cặp khác ngữ cảnh, nên mức điểm riêng của từng ảnh nguồn và instruction lấn át khoảng cách trong cặp. Với margin AUC, phiên bản 10 cao nhất (82.41 trên Test, 79.81 trên Bench) và hơn phiên bản 7 +1.02 [+0.12, +1.93]. Margin AUC cao hơn độ chính xác khoảng 8 điểm vì margin lớn đáng tin hơn margin nhỏ. Nhãn mềm theo cặp, sigmoid(điểm A − điểm B), dùng đúng margin này. Không nên dùng điểm riêng của từng ảnh như giá trị tuyệt đối để so giữa các cặp.
8. **Kết quả lặp lại được.** Chạy lại phiên bản 4 với cùng seed (phiên bản 12) cho margin gần như trùng (tương quan 0.997, 97.9% cặp cùng dấu) và Brier, log loss bằng nhau. Hai lần chạy không giống hệt nhau, nhưng độ dao động (khoảng ±0.4 điểm độ chính xác) nhỏ hơn nhiều so với chênh lệch giữa các cấu hình.

## 9. Lựa chọn cuối cùng

**Overall Inspector Model = phiên bản 10, checkpoint-200**

- Cấu hình: rank 32, alpha 64, LoRA chỉ trên 252 lớp Linear của LLM, learning rate 8e-5, cosine 2 epoch, warmup 0.05, batch 1 × 8, không có scalar head.
- Thư mục: `Qwen3-VL-Embedding/output/overall_inspector_r32_llm_lr8e-5/checkpoint-200`.
- Kết quả: loss kiểm định 0.5775, Test 74.5% (margin AUC 82.41 [79.73, 84.81], Brier 0.1713, log loss 0.5156), Bench 76.5% (margin AUC 79.81 [71.16, 87.71]). AUC gộp cũ: Test 65.9, Bench 72.0.
- Lý do: loss kiểm định thấp nhất trong mọi phiên bản (tiêu chí chọn đã thống nhất), và margin AUC, Brier, log loss đều tốt hơn phiên bản 7 một cách đáng tin cậy. Phiên bản 7 (checkpoint-200, loss kiểm định 0.5835) là lựa chọn trước đó.

## 10. Việc còn mở

- Đã kiểm tra: file `overall_inspector_checkpoint.pt` của phiên bản 7 chứa đúng trọng số LoRA của checkpoint-200. Trọng số trùng hoàn toàn với `checkpoint-200/adapter_model.safetensors` và khác bước 502. Trường `selection.best_checkpoint` trong file cũng ghi checkpoint-200.
- Đã kiểm tra: file `overall_inspector_checkpoint.pt` của phiên bản 10 chứa đúng trọng số LoRA của checkpoint-200. 504 tensor trùng hoàn toàn với `checkpoint-200/adapter_model.safetensors` (chênh lệch lớn nhất bằng 0) và khác bước 502 (chênh lệch lớn nhất 0.0034). Không có tensor LoRA nào trên ViT. Trường `selection.best_checkpoint` và `ms_swift_best_checkpoint` đều ghi checkpoint-200 (loss 0.5775); rank 32, alpha 64, nhiệt độ 1.0, `frozen_for_inference` bật. File dùng được trực tiếp cho Overall Pseudo-label Cache.
- Có thể thử thêm learning rate lớn hơn 8e-5 (ví dụ 1.6e-4), vì xác suất vẫn đang cải thiện ở giá trị lớn nhất đã thử.
- Các việc tồn đọng từ trước:
  - lỗi đọc ngưỡng ở `infer_overall_inspector.py`;
  - `train_loop.py` vẫn dùng `human_split_from_inspector` và ngưỡng;
  - cách chia dữ liệu cho Student Model;
  - thí nghiệm với cặp hòa;
  - cách tạo nhãn cho Overall Pseudo-label Cache.
