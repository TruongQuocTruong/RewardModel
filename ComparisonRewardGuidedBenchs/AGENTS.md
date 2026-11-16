# AGENTS.md — ComparisonRewardGuidedBenchs

Hướng dẫn cho agent khi làm bảng **Reward-guided Best-of-N image editing**: dùng reward model chọn ảnh tốt nhất
trong N ảnh rollout, rồi chấm ảnh được chọn bằng giao thức gốc của từng bench. Đây là bằng chứng trực tiếp về
chất lượng tín hiệu reward, tương tự Table 2 trong `GettingSpaceRight.pdf`.
Quy ước chung của repo xem `../AGENTS.md`; quy ước của các bảng so sánh khác xem
`../ComparisonRewardModelsOnOtherBenchs/AGENTS.md` (Table 4) và `../ComparisonRewardModels/AGENTS.md` (AURA-Test).
File này chỉ ghi phần riêng và phần ghi đè.

## Mô tả dự án

Bảng cần điền:

| Selector | GEdit-Bench-EN G_SC | G_PQ | G_O | ImgEdit Instruction Adherence | Editing Quality | Detail Preservation | KRIS-Bench Visual Consistency | Visual Quality | Instruction Following |
|---|---|---|---|---|---|---|---|---|---|
| FLUX.2-Klein-9B | | | | | | | | | |
| Qwen2.5-VL-7B | | | | | | | | | |
| ImageReward | | | | | | | | | |
| CLIP | | | | | | | | | |
| Our Models | | | | | | | | | |

- **Dòng FLUX.2-Klein-9B (không chọn ảnh)**: điểm judge của **seed 0** (chốt lại 2026-11-15, thay cho trung bình 9 seed).
- **Dòng selector**: với mỗi mẫu, selector chấm điểm 9 ảnh và chọn ảnh có điểm cao nhất. Nếu điểm bằng nhau,
  chọn seed nhỏ nhất (quy tắc cố định, ghi trong code). **Một ảnh được chọn cho mỗi mẫu**, và mọi cột của bench
  đó được chấm trên đúng ảnh này (selector không được chọn riêng cho từng cột).
- **Judge chỉ chấm ảnh cần cho bảng** (chốt 2026-11-15): seed 0 + ảnh do 4 selector chọn (tối đa 5 ảnh / mẫu).
  Không chấm cả 9 seed, nên không có dòng Oracle hay trung bình 9 seed.
- **Our Models**: checkpoint và quy tắc điểm cho mỗi ảnh xem phần "Quyết định còn mở".
- Làm **từng bước một**, chờ người dùng duyệt giữa các bước.

## Quy trình (thứ tự bắt buộc)

1. **Manifest**: liệt kê mẫu × seed → `Manifests/<Bench>.jsonl` (đọc 3 file `*_300.json` + đường dẫn ảnh;
   kiểm tra đủ 9 ảnh mỗi mẫu bằng job Slurm, không quét thư mục ảnh trong session).
2. **Selector chấm 9 ảnh** (`scripts/score_selector.py`, `script_slurm/run_score_selector.slurm`, mỗi selector một job)
   → `SelectorScores/<Selector>/<Bench>/scores.jsonl`; **chọn ảnh** (`scripts/select_best.py`; argmax, hoà → seed nhỏ nhất)
   → `results/selections/<Selector>/<Bench>.jsonl`. Selector: Qwen2.5-VL-7B, ImageReward, CLIP, FIRE-Reward-Stage2-Teacher.
3. **Judge** (`scripts/judge_bench.py`) chấm seed 0 + các seed trong `results/selections/` → `JudgeScores/<Judge>/<Bench>/scores.jsonl`.
   Selection chưa có thì job chấm seed 0 trước rồi chờ (kiểm tra mỗi 10 phút, tối đa 48 giờ) để giữ slot large/full;
   thêm selector hay đổi quy tắc chọn → resubmit, job chỉ chấm ảnh còn thiếu.
4. **Tổng hợp**: `scripts/best_of_n_table.py` (chưa viết) → `results/best_of_n_table.{json,md}`.

## Dữ liệu

**Rollout** (chỉ đọc, thuộc nghiatg): `/datastore/nghiatg/Truongtq/RewardGuidedBenchsRolloutsGen/`
- Model sinh ảnh: FLUX.2-klein-9B bản distilled, 4 bước, guidance 1.0, bf16, cạnh dài ≤ 1024
  (giữ tỉ lệ, padding lên bội của 16, crop bỏ padding khi lưu). Script `generate_flux2_9b_{gedit,imgedit,kris}.py`.
- **N = 9**: mỗi mẫu có `seed_{0,5,10,15,20,25,30,35,40}.png` (các run đã gộp lại trong một thư mục mẫu).
- Đường dẫn ảnh: `Results/FLUX2-9B/<Bench>/<category>/<key>/seed_<s>.png`
  - GEdit-Bench-EN: `<task_type>/<key>`, metadata `gedit_300.json` (list; `key`, `task_type`, `instruction`, `image`).
  - ImgEdit-Bench: `singleturn/<thư mục ảnh gốc>/<key>`, metadata `imgedit_singleturn_300.json`
    (dict key → `id`, `prompt`, `edit_type`; trùng 300/300 với `imgedit/Benchmark/Basic/basic_edit.json`).
  - KRIS-Bench: `<category>/<export_key>`, metadata `kris_300.json` (list; `category`, `id`, `instruction`,
    `explanation`, `export_key`, `image`, `gt_image`).
- `Results/random/` là thí nghiệm khác (GEdit, 100 mẫu × 3 seed ngẫu nhiên); **không dùng** trừ khi người dùng yêu cầu.
- Script sinh ảnh vẫn trỏ về `/datastore/danhnt/...`. Dữ liệu đã chuyển sang nghiatg; chỉ đọc kết quả, không chạy lại.

**Ảnh gốc** (chỉ đọc, thuộc anhndt): `/datastore/anhndt/Truongtq/OtherEditingBenchmark/`
- `GEdit-Bench-EN/` (`images/<task_type>/<key>.png`), `ImgEdit-Bench/Benchmark/singleturn/`, `KRIS_Bench/` (`images/<export_key>/image.png`,
  `gt_image.png` cho `viewpoint_change`).

**Thống kê tập 300 mẫu**
- GEdit-Bench-EN, 11 `task_type`: text_change 49, ps_human 36, subject-replace 30, style_change 29, subject-add 29,
  color_alter 25, tone_transfer 24, motion_change 21, material_alter 20, background_change 19, subject-remove 18.
- ImgEdit, 9 `edit_type`: extract 53, background 43, style 39, adjust 38, replace 34, add 32, remove 32, action 20, compose 9.
- KRIS-Bench, 18 category (không có `multi-element_composition`, `temporal_prediction`): 10 category kiến thức
  (196 mẫu có `explanation`), 10 mẫu `viewpoint_change` có ảnh ground truth. 3 chỉ dẫn khác bản trong
  `Kris_Bench/KRIS_Bench/<category>/annotation.json` → dùng chỉ dẫn của `kris_300.json` (chỉ dẫn lúc sinh ảnh), ghi lại khác biệt.

## Giao thức chấm của từng bench (repo gốc, chỉ đọc)

| Bench | Repo (commit) | Giao thức |
|---|---|---|
| GEdit-Bench-EN | `Step1X-Edit/GEdit-Bench` (4cca05d) | VIEScore `task="tie"` (`viescore/__init__.py`, prompt trong `vie_prompts.py`). Ảnh gốc và ảnh sửa được resize về diện tích 512×512, giữ tỉ lệ (`run_gedit_score.py`). Prompt SC nhận 2 ảnh và trả 2 điểm 0–10 (mức độ sửa đúng, mức độ sửa quá tay); prompt PQ chỉ nhận ảnh sửa và trả 2 điểm (tự nhiên, lỗi ảnh). **SC = min**, **PQ = min**, **O = √(SC·PQ)** cho mỗi ảnh. Tổng hợp: trung bình theo `task_type`, sau đó trung bình 11 `task_type` (`calculate_statistics.py`, `language=en`). Judge gốc: GPT-4.1 (cột G_*) hoặc Qwen2.5-VL-72B-Instruct-AWQ (cột Q_*, greedy, `max_new_tokens=512`, seed 42). |
| ImgEdit (Basic, singleturn) | `imgedit/Benchmark/Basic` (b798481) | Một lần gọi judge cho mỗi ảnh với prompt theo `edit_type` (`prompts.json`, thay `<edit_prompt>`), gửi ảnh gốc rồi ảnh sửa. Judge trả 3 điểm 1–5. Overall gốc = trung bình 3 điểm của mỗi ảnh (`step1_get_avgscore.py`), sau đó trung bình theo `edit_type` (`step2_typescore.py`). Judge gốc: GPT-4o. **Thứ tự 3 điểm khác nhau giữa các `edit_type`** (bảng bên dưới). |
| KRIS-Bench | `Kris_Bench` (cf14752) | 3 lần gọi judge cho mỗi ảnh, điểm 1–5 (`utils/prompts.py`). Visual Consistency: `prompt_consist` (ảnh gốc + ảnh sửa). Visual Quality: `prompt_quality` (chỉ ảnh sửa). Instruction Following: `prompt_instruction_following` cho 7 category thường (`metrics_common.py`); `prompt_dual_evaluation` cho 10 category kiến thức, lấy `instruction_score` (kèm `knowledge_score`, lưu lại nhưng không thuộc bảng) (`metrics_knowledge.py`); `prompt_view_instruction_following` cho `viewpoint_change`, có thêm ảnh ground truth (`metrics_view_change.py`). Parse JSON trước, sau đó mới dùng regex (giữ nguyên hàm parse gốc). Judge gốc: GPT-4o. |

Ghép điểm ImgEdit vào 3 cột (theo nghĩa của tên điểm trong `prompts.json`; vị trí 1/2/3 = thứ tự judge trả về):

| `edit_type` | Instruction Adherence | Editing Quality | Detail Preservation |
|---|---|---|---|
| replace, add, adjust, remove, background, compose | 1 (Prompt / Instruction Compliance) | 2 (Visual Naturalness / Seamlessness) | 3 (Physical & Detail …) |
| style | 1 (Style Fidelity) | 3 (Rendering Quality) | 2 (Content Preservation) |
| action | 1 (Action Fidelity) | 3 (Visual & Anatomical Coherence) | 2 (Identity Preservation) |
| extract | 1 (Object Identity) | 3 (Visual Quality) | 2 (Mask Precision) |

Lưu ý giao thức:
- Mọi prompt ImgEdit đều có câu "The second and third score should no higher than first score!!!". Giữ nguyên prompt; không tự sửa điểm.
- KRIS gốc so khớp `"abnormality_correction" in category`, nhưng category thật là `anomaly_correction`, nên code gốc
  **không bao giờ** dùng `prompt_abnormal_instruction_following`. Mặc định theo code gốc (`--kris-anomaly-prompt original_code`).
- Giao thức đã chốt (judge, cách ghép cột, thang điểm, nguồn) được ghi vào `EvaluationConfig/benchmark_protocols.json`
  (chưa viết). Mỗi mục kèm đường dẫn code gốc + số dòng.

## Selector

| Dòng | Model | Weights | Cách tính điểm cho mỗi ảnh |
|---|---|---|---|
| FLUX.2-Klein-9B | (không chọn) | — | Seed 0 |
| Qwen2.5-VL-7B | Qwen2.5-VL-7B-Instruct | `/datastore/tuanld/Truongtq/RewardModels/Qwen2.5-VL-7B` | Chưa chốt prompt (xem phần mở) |
| ImageReward | ImageReward-v1.0 | `RewardModels/ImageReward` (+ `bert-base-uncased`), đang chờ tải | Điểm ImageReward(instruction, ảnh sửa) — chưa chốt |
| CLIP | CLIP ViT-L/14 | `RewardModels/CLIP-ViT-L-14`, đang chờ tải | Cosine(text instruction, ảnh sửa) — chưa chốt |
| Our Models | Qwen3-VL-Reranker-8B + LoRA | Xem `../ComparisonRewardModelsOnOtherBenchs/EvaluationConfig/output_mapping.json` | Chưa chốt (xem phần mở) |

- Our Models: tái dùng code nạp model và forward từ `../RewardModelEvaluation/OurModel/` và `train_loop.forward_pairs`
  (giống `../ComparisonRewardModelsOnOtherBenchs/scripts/score_bench.py`; import, không sao chép).
  Input = (ảnh gốc, chỉ dẫn, ảnh A, ảnh B), dựng prompt giống hệt lúc train. Ở eval mode, điểm của ảnh A chỉ phụ thuộc
  vào (ảnh gốc, chỉ dẫn, ảnh A). Vì vậy 9 ảnh được ghép thành cặp (seed 0–5, 10–15, 20–25, 30–35, 40–0) và chỉ lấy
  điểm từng ảnh. Lưu đủ 6 điểm metric, xác suất relevance và Tie, để đổi quy tắc chọn mà không phải chạy lại.
- Selector **không được** thấy điểm judge và không được dùng prompt của judge.
- Không tune prompt, ngưỡng hay checkpoint của bất kỳ selector nào trên các bench này.

## Môi trường

```bash
source /datastore/truongtq/miniconda3/etc/profile.d/conda.sh
conda activate qwen_vl      # torch 2.4, transformers 4.57.6, peft 0.12, pytest; không có vLLM
```
- Qwen2.5-VL-72B-Instruct-AWQ: `/datastore/shared/models/Qwen2.5-VL-72B-Instruct-AWQ` (đã chạy bằng env `qwen_vl`
  trong `/datastore/danhnt/Truongtq/BenchmarkEvaluation/run_score_gedit_qwen72b_awq.slurm`, partition `full`).
  Gemma-4-31B-it: `/datastore/shared/models/gemma-4-31B-it`.
- Code chấm cũ có thể tham khảo (của danhnt, chỉ đọc, không gọi trực tiếp):
  `/datastore/danhnt/Truongtq/BenchmarkEvaluation/score_gedit_qwen72b_awq.py`, `score_imgedit_gemma4.py`.
  Trước khi dùng lại phải đối chiếu prompt từng chữ với repo gốc ở trên.
- Env riêng (tạo 2026-11-15, clone từ `qwen_vl`; torch 2.4, transformers 4.57.6, peft 0.12 giữ nguyên):
  - `qwen_vl_awq` = + autoawq 0.2.9 (judge Qwen2.5-VL-72B-AWQ; không có kernel `awq_ext`, chạy bằng Triton). transformers 4.57
    chỉ nạp AWQ qua autoawq (`quantizer_awq.py`); env `gemma4` (transformers 5.15) đòi `gptqmodel` → không env cũ nào nạp được.
    autoawq gọi `PytorchGELUTanh` (đã đổi tên `GELUTanh`) → alias trong `judges.patch_transformers_for_autoawq`.
  - `image_reward` = + image-reward 1.5, openai-clip 1.0.1, timm 0.6.13, fairscale 0.4.13, diffusers 0.31.0, setuptools < 81.
    ImageReward import 3 hàm đã chuyển sang `transformers.pytorch_utils` → alias trong
    `best_of_n_selectors.patch_transformers_for_imagereward`.
- Không tạo hay sửa env khi chưa được duyệt. Không tải model hay dataset từ Internet trong job đánh giá;
  weights mới tải theo quy trình `RewardModels/` (`models.txt`, `--dry-run`, Slurm) sau khi được duyệt.
- File `Step1X-Edit/GEdit-Bench/secret.env` thuộc repo clone: không đọc, không in, không chép.

## Lệnh chính

Mọi việc nặng (liệt kê ảnh, judge, selector) **chỉ chạy qua Slurm**, không chạy trên login node.

```bash
sbatch script_slurm/run_build_manifests.slurm
sbatch script_slurm/run_judge_<judge>_<bench>.slurm
sbatch script_slurm/run_select_<selector>.slurm
python scripts/best_of_n_table.py --output results/best_of_n_table   # nhẹ, ghi .json + .md
pytest -q                                                            # chạy trong thư mục này
```

**Slurm**
- Judge 72B-AWQ: khoảng 40–50 GB (ước lượng; đo lại bằng smoke test). `large` (71 GB, `mig3-71`) nếu đủ, còn không thì `full` (141 GB, `mig7-141`).
  Selector Our Models: khoảng 30 GiB → `normal`. Qwen2.5-VL-7B / ImageReward / CLIP: ước lượng trước, chọn tier nhỏ nhất đủ dùng.
- Số lần gọi judge (chưa tính lần thử lại): GEdit 2 × 2.700, ImgEdit 1 × 2.700, KRIS 3 × 2.700 → khoảng 16.200.
  Đo thời gian mỗi lần gọi bằng smoke test 1 mẫu × 9 seed trước khi ước lượng giờ chạy.
- Mỗi user tối đa **3 job** chạy cùng lúc. `sbatch --test-only` không kiểm tra giới hạn QOS
  (so `#SBATCH` với `sacctmgr show qos format=name,maxtres`). Job từ tài khoản khác: kiểm tra quyền hai chiều,
  `umask 002` trước `sbatch`.
- Job phải chạy tiếp được sau khi bị ngắt: ghi `scores.jsonl` từng dòng, bỏ qua `(sample_id, seed)` đã có.
- Đọc ảnh tuần tự trong job, không mở nhiều worker đọc cùng lúc trên `/datastore` (tránh nghẽn I/O).

## Cấu trúc thư mục

| Đường dẫn | Nội dung |
|---|---|
| `AGENTS.md` | File này |
| `GettingSpaceRight.pdf` | Paper tham khảo (Table 2) |
| `Step1X-Edit/`, `imgedit/`, `Kris_Bench/` | Repo gốc của bench (clone 2026-11-15, **chỉ đọc**) |
| `EvaluationConfig/benchmark_protocols.json` | Judge, prompt, ghép cột, tổng hợp của từng bench, kèm nguồn |
| `EvaluationConfig/selectors.json` | Selector → model, weights (commit), cách tính điểm cho mỗi ảnh |
| `Manifests/<Bench>.jsonl` | Một dòng / (mẫu, seed): `sample_id`, `bench`, `category`, `instruction`, `source_image`, `edited_image`, `seed`[, `gt_image`, `explanation`] |
| `JudgeScores/<Judge>/<Bench>/` | Điểm judge của mọi ảnh + output thô + `run_config.json` |
| `SelectorScores/<Selector>/<Bench>/` | Điểm selector của mọi ảnh + `run_config.json` |
| `results/` | `selections/<Selector>/<Bench>.jsonl` (seed được chọn) + `best_of_n_table.{json,md}` |
| `scripts/`, `script_slurm/` | Code của mình / launcher `.slurm` |
| `tests/`, `pytest.ini` | Test pytest |
| `logs/` | Log Slurm |
| `submissions.log` | Nhật ký mọi lần submit |

Thư mục này bị `.gitignore` gốc loại; muốn đưa lên git phải thêm `!/ComparisonRewardGuidedBenchs/<đường dẫn>/`
(chỉ code, config, `AGENTS.md`; không đưa repo clone, ảnh, điểm, kết quả).

## Quy ước

**Đặt tên**
- Tên bench, cột, dòng giữ đúng như bảng (GEdit-Bench-EN, ImgEdit, KRIS-Bench; FLUX.2-Klein-9B, Qwen2.5-VL-7B,
  ImageReward, CLIP). Dòng của mình dùng tên đã thống nhất (FIRE-Reward-Stage1 / FIRE-Reward-Stage2 …).
- Thư mục judge ghi rõ model: `JudgeScores/Qwen2.5-VL-72B-Instruct-AWQ/`, `JudgeScores/GPT-4.1/`…
- Không dùng mã tắt; viết đầy đủ thuật ngữ trong code, log, tài liệu.

**Output**
- Judge, mỗi dòng: `{"sample_id", "bench", "category", "seed", "judge", "scores": {cột: điểm}, "raw_outputs": [...], "parse_ok"}`.
  Lưu nguyên văn output của judge để kiểm tra lỗi parse. Mẫu parse lỗi được ghi `null`, không đoán.
- Selector, mỗi dòng: `{"sample_id", "bench", "seed", "selector", "score"[, "metric_scores", "relevance", "probabilities"]}`.
- Cấu hình đầy đủ mỗi lần chạy: model / checkpoint / commit, prompt (md5), env, md5 code, giao thức đã dùng.
- Bảng chính: GEdit giữ thang 0–10, ImgEdit và KRIS giữ thang 1–5 như code gốc (trừ khi đã chốt cách quy đổi khác),
  làm tròn 2 chữ số. Báo kèm khoảng tin cậy 95% bootstrap theo mẫu (2000 lần, seed cố định) và bootstrap ghép cặp
  của từng selector so với dòng FLUX.2-Klein-9B (trung bình 9 seed).
- Các bước của cùng một việc: một thư mục, mỗi bước một thư mục con. Output job thử (`*_test_*`) được xoá sau khi kiểm tra.

**Logging khi submit**
- Mỗi job ghi một dòng vào `submissions.log`: ngày, tài khoản, job ID, partition / QOS / GPU, bước (judge / selector),
  bench, model / checkpoint, đường dẫn output, md5 của launcher và script.
- Đọc log và số liệu thật trước khi kết luận; báo nguyên văn lỗi.

## Testing

- Mỗi khi thêm feature mới hoặc sửa logic, **phải** viết test tương ứng trong `tests/` (pytest), file
  `test_<module>.py`, tên test mô tả rõ hành vi (vd `test_imgedit_style_maps_second_score_to_detail_preservation`).
- Viết test **trước hoặc cùng lúc** với code: trường hợp bình thường, trường hợp biên (0 mẫu, 1 seed, mọi điểm bằng nhau,
  NaN, parse lỗi) và trường hợp lỗi (input sai phải raise đúng exception).
- Bắt buộc có test cho:
  - manifest: đủ 9 seed, đường dẫn đúng theo từng bench;
  - parse output judge: GEdit lấy min của 2 điểm và O = √(SC·PQ); ImgEdit ghép cột theo `edit_type`;
    KRIS: JSON trước, regex sau, chọn đúng prompt theo category;
  - chọn argmax, với quy tắc phá hoà (seed nhỏ nhất);
  - trung bình 9 seed và seed 0;
  - tổng hợp theo category đúng như code gốc;
  - bootstrap.
- Dùng model giả / data giả, chạy trên CPU, mỗi test < 10 giây. Không tải checkpoint thật, không đọc ảnh thật, không submit Slurm.
  `tests/conftest.py` ẩn GPU, bật chế độ offline của Hugging Face, chặn `sbatch` / `scontrol` / `scancel` / `srun`.
- Chạy `pytest -q` trong thư mục này sau khi sửa; chỉ báo hoàn thành khi tất cả test pass (báo nguyên văn lỗi nếu fail).
- Không được xoá hoặc sửa test cũ để cho pass, trừ khi người dùng đồng ý.

## Không được làm

- Không sửa, ghi đè hay di chuyển rollout (`/datastore/nghiatg/Truongtq/RewardGuidedBenchsRolloutsGen/`) và ảnh gốc
  (`/datastore/anhndt/Truongtq/OtherEditingBenchmark/`).
- Không quét hay đọc toàn bộ thư mục ảnh trong session; mọi việc trên 8.100 ảnh là job Slurm đã duyệt.
- Không sửa code trong repo clone (`Step1X-Edit/`, `imgedit/`, `Kris_Bench/`), `../RewardModelEvaluation/`,
  `../ComparisonRewardModels*/`; cần thay đổi thì viết wrapper trong `scripts/`.
- Không dùng điểm judge để chọn checkpoint, quy tắc điểm hay prompt của selector; không chọn quy tắc riêng cho từng cột.
- Không xoá checkpoint / weights; chỉ người dùng xoá.
- Không submit, huỷ hay sửa job Slurm khi chưa được duyệt; không chạy việc nặng trên login node.
- Không commit hay push khi chưa được yêu cầu; không ghi token, mật khẩu, khoá API, thông tin cá nhân.

## Quyết định đã chốt

- 2026-11-15: dòng FLUX.2-Klein-9B = seed 0; judge chỉ chấm seed 0 + ảnh do 4 selector chọn.
- Our Models = FIRE-Reward-Stage2 (Teacher), `output_training/best_stage2_ckpt/checkpoint-621.pt`.
- N = 9 (seed 0, 5, …, 40), 300 mẫu mỗi bench, rollout FLUX.2-klein-9B có sẵn; không dùng `Results/random/`.
- 2026-11-15, judge (GPT-4o / GPT-4.1 không còn dùng được): **gemma-4-31B-it** (`/datastore/shared/models/gemma-4-31B-it`,
  env `gemma4`) cho GEdit-Bench-EN và KRIS-Bench; **Qwen2.5-VL-72B-Instruct-AWQ** (`/datastore/shared/models/Qwen2.5-VL-72B-Instruct-AWQ`)
  cho ImgEdit. Phải chạy trên `large` (3g.71gb, chỉ có 2 GPU) hoặc `full` (7g.141gb, chỉ có 1 GPU): KRIS → full, GEdit và ImgEdit → large.
  Nhiều job xếp hàng, nên submit sớm để giữ chỗ; script chạy tiếp được sau khi bị ngắt (resubmit cùng launcher).
- Code judge: `scripts/bench_protocols.py` (prompt, ghép cột, parse như code gốc), `scripts/judges.py`, `scripts/judge_bench.py`;
  launcher `script_slurm/run_judge_{gemma4_gedit,gemma4_kris,qwen72b_awq_imgedit}.slurm`.
  Parse: sinh greedy với ngân sách token gốc; nếu parse lỗi thì sinh lại một lần với gấp đôi ngân sách; GEdit vẫn lỗi thì
  dùng quy tắc đoán của VIEScore (seed theo mẫu), bench khác để `null`. Trạng thái parse lưu trong mỗi dòng.
- KRIS `anomaly_correction`: mặc định theo code gốc (prompt IF thường, `--kris-anomaly-prompt original_code`).
- ImageReward và CLIP: tải vào `/datastore/tuanld/Truongtq/RewardModels` (`ImageReward` = THUDM/ImageReward,
  `CLIP-ViT-L-14` = openai/clip-vit-large-patch14, `bert-base-uncased` = tokenizer mà ImageReward gọi cứng).

## Quyết định còn mở

1. (Đã xong 2026-11-15: env `qwen_vl_awq`, `image_reward`.)
2. **FIRE-Reward-Stage2 (Teacher)**: điểm mỗi ảnh để chọn = aesthetic + fidelity (quy tắc Table 4, mặc định hiện tại)
   hay `selected_equal`? Cả 6 metric đều được lưu, đổi quy tắc không cần chạy lại.
3. **Qwen2.5-VL-7B**: prompt chấm điểm cho mỗi ảnh (không được trùng prompt của judge).
4. **ImageReward / CLIP**: text đầu vào = chỉ dẫn (mặc định hiện tại).
5. **KRIS**: thang điểm báo
   1–5 hay quy đổi như paper KRIS? Tổng hợp: trung bình theo category rồi trung bình, hay trung bình mọi mẫu?
6. **ImgEdit `extract`**: chấp nhận "Mask Precision" là Detail Preservation không? Tổng hợp theo `edit_type` rồi trung bình?
