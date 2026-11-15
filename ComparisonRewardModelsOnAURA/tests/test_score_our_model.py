"""Tests for scripts/score_our_model.py with a fake pair runner (no weights, no GPU, no real images)."""

import json

import pytest

import bench_inputs as bi
import evaluation_metrics as em
import score_our_model as so

ROWS = [
    {"sample_id": "A1", "metric": "Aesthetic", "prompt": "add a hat", "src_image": "s.png", "edited_images": ["a", "b"],
     "human_label": "A", "votes": {"A": 3, "B": 0, "Tie": 0}},
    {"sample_id": "F1", "metric": "Fidelity", "prompt": "blur", "src_image": "s.png", "edited_images": ["c", "d"]},
    {"sample_id": "S1", "metric": "Structure Consistency", "prompt": "p", "src_image": "t.png",
     "edited_images": ["e", "f"]},
]


def student_output(index_value=0.0):
    """Six heads; head k has scores (k + index_value, k) and probabilities [0.5, 0.3, 0.2]."""
    return {"heads": list(so.STUDENT_HEADS),
            "scores_a": [k + 1.0 + index_value for k in range(6)], "scores_b": [float(k) for k in range(6)],
            "three_way_probs": [[0.5, 0.3, 0.2]] * 6, "relevance_probs": [0.9, 0.1, 0.8, 0.2, 0.3, 0.7]}


class FakeRunner:
    def __init__(self, make):
        self.make, self.calls = make, []

    def __call__(self, source, edited, instruction):
        self.calls.append((source, edited, instruction))
        return self.make()


def paths(row):
    return f"SRC/{row['src_image']}", [f"ED/{p}" for p in row["edited_images"]]


def run(tmp_path, runner, kind, rows=ROWS):
    path = tmp_path / "scores.jsonl"
    counts = so.score_rows(runner, kind, rows, path, image_paths=paths, log=lambda m: None)
    return counts, bi.load_jsonl(path)


def test_student_reads_the_head_of_each_bench_metric():
    assert {m: so.head_of_metric("student", m) for m in so.STUDENT_HEAD_OF_METRIC} == {
        "Aesthetic": "aesthetic", "Realism": "realism", "Fidelity": "fidelity",
        "Background Consistency": "background_consistency", "Foreground Consistency": "foreground_consistency",
        "Structure Consistency": "structure_consistency"}


def test_head_of_metric_rejects_unknown_metric_or_kind():
    with pytest.raises(ValueError, match="unknown AURA-Test metric"):
        so.head_of_metric("student", "Overall")
    with pytest.raises(ValueError, match="unknown variant kind"):
        so.head_of_metric("teacher", "Aesthetic")


def test_student_head_order_matches_training_metric_names():
    assert so.STUDENT_HEADS == ("aesthetic", "fidelity", "realism", "background_consistency",
                                "foreground_consistency", "structure_consistency")


def test_head_record_picks_the_named_head_and_margin_is_score_a_minus_score_b():
    record = so.head_record(student_output(), so.STUDENT_HEADS, "realism")
    assert record["score_a"] == 3.0 and record["score_b"] == 2.0 and record["margin"] == 1.0
    assert record["probabilities"] == pytest.approx([0.5, 0.3, 0.2])
    assert record["signed_evidence"] == pytest.approx(0.2)


@pytest.mark.parametrize("change", [
    {"scores_a": [1.0] * 5},
    {"scores_b": [float("nan")] * 6},
    {"three_way_probs": [[0.5, 0.3, 0.2]] * 5},
    {"three_way_probs": [[0.5, 0.5, 0.5]] * 6},
    {"three_way_probs": [[1.2, -0.2, 0.0]] * 6},
    {"three_way_probs": [[0.5, 0.5]] * 6},
])
def test_head_record_rejects_wrong_length_nan_or_invalid_probabilities(change):
    with pytest.raises(ValueError):
        so.head_record({**student_output(), **change}, so.STUDENT_HEADS, "aesthetic")


def test_student_score_rows_writes_one_record_per_row_with_the_metric_head(tmp_path):
    runner = FakeRunner(student_output)
    counts, records = run(tmp_path, runner, "student")
    assert counts == {"scored": 3, "skipped_done": 0}
    assert [(r["sample_id"], r["dimension"]) for r in records] == [
        ("A1", "aesthetic"), ("F1", "fidelity"), ("S1", "structure_consistency")]
    assert [r["score_a"] - r["score_b"] for r in records] == [1.0, 1.0, 1.0]
    assert records[2]["score_a"] == 6.0
    assert set(records[0]["all_heads"]) == set(so.STUDENT_HEADS)
    assert records[0]["relevance_probs"]["aesthetic"] == 0.9


def test_runner_gets_source_both_edited_images_in_bench_order_and_instruction(tmp_path):
    runner = FakeRunner(student_output)
    run(tmp_path, runner, "student", ROWS[:1])
    assert runner.calls == [("SRC/s.png", ["ED/a", "ED/b"], "add a hat")]


def test_score_record_is_self_contained_and_readable_by_the_metric_code(tmp_path):
    _, records = run(tmp_path, FakeRunner(student_output), "student", ROWS[:1])
    record = records[0]
    assert record["human_label"] == "A" and record["prompt"] == "add a hat" and "seconds" in record
    pair = em.PairScore(sample_id=record["sample_id"], metric=record["metric"], human_label=record["human_label"],
                        votes=record["votes"], observed_agreement=1.0, score_a=record["score_a"],
                        score_b=record["score_b"], probabilities=record["probabilities"])
    assert pair.margin == 1.0
    assert em.predicted_labels_from_probabilities([record["probabilities"]]) == ["A"]


def test_score_rows_resumes_without_rescoring_done_rows(tmp_path):
    run(tmp_path, FakeRunner(student_output), "student", ROWS[:2])
    runner = FakeRunner(student_output)
    counts, records = run(tmp_path, runner, "student")
    assert counts == {"scored": 1, "skipped_done": 2}
    assert len(runner.calls) == 1 and len(records) == 3


def test_score_rows_with_empty_row_list_writes_nothing(tmp_path):
    counts, records = run(tmp_path, FakeRunner(student_output), "student", [])
    assert counts == {"scored": 0, "skipped_done": 0} and records == []


def test_main_fails_before_loading_a_model_when_weights_are_missing(tmp_path, monkeypatch):
    bench = tmp_path / "bench.jsonl"
    bench.write_text("".join(json.dumps(r) + "\n" for r in ROWS))
    monkeypatch.setitem(so.VARIANTS, "student_stage1",
                        {**so.VARIANTS["student_stage1"], "checkpoint": tmp_path / "missing.pt"})
    with pytest.raises(FileNotFoundError, match="missing local weights"):
        so.main(["--variant", "student_stage1", "--bench", str(bench), "--output-dir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()


def test_head_record_rejects_unknown_head():
    with pytest.raises(ValueError, match="is not one of"):
        so.head_record(student_output(), so.STUDENT_HEADS, "overall")


def test_score_rows_rejects_nan_output_without_writing_the_row(tmp_path):
    bad = FakeRunner(lambda: {**student_output(), "scores_a": [float("nan")] * 6})
    with pytest.raises(ValueError):
        run(tmp_path, bad, "student", ROWS[:1])
    assert bi.load_jsonl(tmp_path / "scores.jsonl") == []


def test_inspector_baseline_is_not_a_variant_and_has_no_mapping_row(comparison_root):
    # 2026-11-13 (user): the Inspector is an overall model (overall system prompt); AURA-Test labels one metric.
    assert set(so.VARIANTS) == {"student_stage1", "student_stage2", "teacher_stage2"}
    assert not any("inspector" in name for name in so.VARIANTS)
    mapping = json.loads((comparison_root / "EvaluationConfig" / "metric_mapping.json").read_text())
    assert "Inspector Baseline" not in {r["model"] for r in mapping["models"]}
    with pytest.raises(ValueError, match="unknown variant kind"):
        so.head_of_metric("inspector", "Aesthetic")


def test_student_stage1_variant_points_to_the_stage1_warm_start():
    student = so.VARIANTS["student_stage1"]
    assert student["kind"] == "student" and student["model"] == "Student-Stage1"
    assert student["checkpoint"].name == "student_warmstart.pt" and student["pairs_trained"] == 1008
    assert student["run_dir"].name.endswith("_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63")


def test_mapping_row_of_student_stage1_matches_the_heads_the_script_writes(comparison_root):
    import comparison_table as ct
    mapping = json.loads((comparison_root / "EvaluationConfig" / "metric_mapping.json").read_text())
    dimensions = bi.dimension_by_bench_metric(mapping, "Student-Stage1", "Qwen3-VL-Reranker-8B")
    assert dimensions == {m: so.head_of_metric("student", m) for m in so.STUDENT_HEAD_OF_METRIC}
    assert ct.result_folder_name("Student-Stage1", "Qwen3-VL-Reranker-8B") == so.VARIANTS["student_stage1"]["result_folder"]


def test_stage2_student_and_teacher_read_the_two_adapters_of_the_same_checkpoint():
    student, teacher = so.VARIANTS["student_stage2"], so.VARIANTS["teacher_stage2"]
    assert student["checkpoint"] == teacher["checkpoint"] == so.STAGE2_CHECKPOINT
    assert so.STAGE2_CHECKPOINT.parent.name == "Stage2_R0RelKL10kDecay98CosineOnceBatch4Eval5" and so.STAGE2_CHECKPOINT.name == "checkpoint-927.pt"
    assert (student["role"], teacher["role"]) == ("student", "teacher")
    assert student["kind"] == teacher["kind"] == "student"  # same six metric heads
    assert (student["model"], teacher["model"]) == ("Student-Stage2", "Teacher-Stage2")
    assert len({v["result_folder"] for v in so.VARIANTS.values()}) == 3


def _fake_training_modules(monkeypatch, loaded_roles):
    """Fake evaluate_reward_model / train_loop / METRIC_NAMES so make_student_runner runs without weights."""
    import sys
    import types

    import torch

    def load_refined_model(checkpoint, role, device, min_pixels, max_pixels, base_model_override=None):
        loaded_roles.append(role)
        return object(), object()

    def forward_pairs(model, processor, pairs, prompt, device):
        return types.SimpleNamespace(scores_A=torch.ones(1, 6), scores_B=torch.zeros(1, 6),
                                     three_way_probs=torch.tensor([[[0.5, 0.3, 0.2]] * 6]),
                                     relevance_probs=torch.full((1, 6), 0.5))

    modules = {"evaluate_reward_model": types.SimpleNamespace(load_refined_model=load_refined_model),
               "train_loop": types.SimpleNamespace(PairRecord=lambda **kw: kw, forward_pairs=forward_pairs),
               "src": types.ModuleType("src"), "src.models": types.ModuleType("src.models"),
               "src.models.custom_qwen3_vl_reranker": types.SimpleNamespace(METRIC_NAMES=list(so.STUDENT_HEADS))}
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(so, "_import_training_code", lambda: None)
    monkeypatch.setattr(torch, "load", lambda *a, **k: {"prompt": {"profile": "six_metric"}, "optimizer_step": 621,
                                                          "model_role": "six_metric_refinement_student_teacher"})


@pytest.mark.parametrize("variant, role", [("student_stage1", "student"), ("student_stage2", "student"),
                                           ("teacher_stage2", "teacher")])
def test_runner_loads_the_adapter_of_the_variant_role(monkeypatch, variant, role):
    loaded_roles = []
    _fake_training_modules(monkeypatch, loaded_roles)
    run_pair, loaded = so.make_student_runner(so.VARIANTS[variant], "cpu")
    assert loaded_roles == [role] and loaded["role"] == role and loaded["optimizer_step"] == 621
    output = run_pair("s.png", ["a.png", "b.png"], "add a hat")
    assert output["scores_a"] == [1.0] * 6 and output["three_way_probs"][0] == pytest.approx([0.5, 0.3, 0.2])


def test_runner_rejects_unknown_role(monkeypatch):
    _fake_training_modules(monkeypatch, [])
    with pytest.raises(ValueError, match="unknown role"):
        so.make_student_runner({**so.VARIANTS["teacher_stage2"], "role": "inspector"}, "cpu")


@pytest.mark.parametrize("variant", ["student_stage2", "teacher_stage2"])
def test_mapping_rows_of_stage2_match_the_heads_and_result_folders(comparison_root, variant):
    import comparison_table as ct
    mapping = json.loads((comparison_root / "EvaluationConfig" / "metric_mapping.json").read_text())
    spec = so.VARIANTS[variant]
    row = next(r for r in mapping["models"] if r["model"] == spec["model"])
    assert "pending" not in row and "checkpoint-927.pt" in row["checkpoint"]
    assert bi.dimension_by_bench_metric(mapping, spec["model"], "Qwen3-VL-Reranker-8B") == {
        m: so.head_of_metric("student", m) for m in so.STUDENT_HEAD_OF_METRIC}
    assert ct.result_folder_name(spec["model"], "Qwen3-VL-Reranker-8B") == spec["result_folder"]


def test_parse_args_rejects_unknown_variant_and_split_options():
    for argv in (["--variant", "inspector_baseline"], ["--variant", "student_stage1", "--shard", "0"]):
        with pytest.raises(SystemExit):
            so.parse_args(argv)
    assert so.parse_args(["--variant", "student_stage1"]).per_metric_limit is None


def test_launcher_uses_qwen_vl_env_normal_partition_and_has_no_shard_or_inspector_option(comparison_root):
    text = (comparison_root / "script_slurm" / "run_score_our_model.slurm").read_text()
    assert "CONDA_ENV=qwen_vl" in text
    assert "#SBATCH --partition=normal" in text and "#SBATCH --gres=gpu:2g.35gb:1" in text
    for name in ("student_stage1", "student_stage2", "teacher_stage2"):
        spec = so.VARIANTS[name]
        assert f'{name})     RESULT_FOLDER={spec["result_folder"]}; MODEL_ROW="{spec["model"]}"' in text
    assert "inspector_baseline)" not in text and "shard" not in text.lower()


def _gpu_type_pipeline(comparison_root):
    """The text-processing part of the launcher line that reads the GPU type from scontrol output."""
    line = next(l for l in (comparison_root / "script_slurm" / "run_score_our_model.slurm").read_text().splitlines()
                if l.startswith("GRES="))
    return line.split("2>/dev/null |", 1)[1].rsplit("|| true", 1)[0]


@pytest.mark.parametrize("scontrol_text, expected", [
    ("ReqTRES=cpu=4,mem=64G,node=1,billing=2,gres/gpu=1,gres/gpu:2g.35gb=1\n   TresPerNode=gres/gpu:2g.35gb:1",
     "2g.35gb"),
    ("TresPerNode=gres/gpu:3g.71gb:1", "3g.71gb"),
    ("AllocTRES=cpu=4,gres/gpu=1", ""),
])
def test_launcher_logs_the_typed_gpu_not_the_untyped_count(comparison_root, scontrol_text, expected):
    import subprocess
    result = subprocess.run(["bash", "-c", _gpu_type_pipeline(comparison_root)], input=scontrol_text,
                            capture_output=True, text=True)
    assert result.stdout.strip() == expected


@pytest.mark.parametrize("launcher", sorted(p.name for p in (bi.COMPARISON_ROOT / "script_slurm").glob("run_score_*.slurm")))
def test_launcher_root_and_log_paths_point_to_the_current_comparison_folder(comparison_root, launcher):
    import re
    text = (comparison_root / "script_slurm" / launcher).read_text()
    root = re.search(r'^ROOT="([^"]+)"', text, re.M).group(1)
    logs = re.findall(r"^#SBATCH --(?:output|error)=(\S+)", text, re.M)
    assert root == str(comparison_root)
    assert len(logs) == 2 and all(path.startswith(f"{comparison_root}/logs/") for path in logs)
