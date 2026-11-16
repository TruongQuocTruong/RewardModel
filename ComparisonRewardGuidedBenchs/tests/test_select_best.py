import json

import pytest

import bench_protocols as bp
import best_of_n_selectors as sel
import judge_bench as jb
import select_best as sb


def _rows(scores, sample="s1", extra=None):
    return [{"sample_id": sample, "seed": seed, "score": v, **(extra(seed) if extra else {})}
            for seed, v in zip(bp.SEEDS, scores)]


# ---------------------------------------------------------------- choose
def test_choose_takes_argmax_seed():
    (row,) = sb.choose(_rows([0, 1, 9, 3, 4, 5, 6, 7, 8]), "CLIP")
    assert row["seed"] == 10 and row["score"] == 9


def test_choose_breaks_ties_with_smallest_seed():
    (row,) = sb.choose(_rows([1, 5, 5, 5, 0, 0, 0, 0, 0]), "CLIP")
    assert row["seed"] == 5
    (row,) = sb.choose(_rows([2.0] * 9), "CLIP")
    assert row["seed"] == 0


def test_choose_rejects_missing_seed_duplicate_and_nan():
    with pytest.raises(ValueError):
        sb.choose(_rows([1] * 8), "CLIP")
    with pytest.raises(ValueError):
        sb.choose(_rows([1] * 9) + _rows([1] * 9)[:1], "CLIP")
    with pytest.raises(ValueError):
        sb.choose(_rows([float("nan")] + [1] * 8), "CLIP")


def test_choose_empty_input_gives_no_selection():
    assert sb.choose([], "CLIP") == []


def test_fire_aesthetic_fidelity_rule_uses_metric_scores_not_selected_equal():
    def metrics(seed):
        m = [0.0] * 6
        m[0] = m[1] = 10.0 if seed == 40 else 0.0
        return {"metric_scores": m}
    rows = _rows([9 if s == 0 else 0 for s in bp.SEEDS], extra=metrics)
    assert sb.choose(rows, sb.FIRE, "aesthetic_fidelity")[0]["seed"] == 40
    assert sb.choose(rows, sb.FIRE, "selected_equal")[0]["seed"] == 0
    with pytest.raises(ValueError):
        sb.choose(rows, sb.FIRE, "best_column")


def test_base_row_selects_seed_zero_for_every_sample():
    rows = sb.base_selection([{"sample_id": "a"}, {"sample_id": "b"}])
    assert [r["seed"] for r in rows] == [0, 0]


def test_metric_order_matches_table4_scripts():
    import sys
    sys.path.insert(0, str(sel.TABLE4_SCRIPTS))
    from compute_bench_metrics import METRIC_NAMES
    assert tuple(METRIC_NAMES) == sb.METRIC_NAMES


# ---------------------------------------------------------------- selector helpers
def test_pair_seeds_covers_all_nine_seeds_in_five_forwards():
    pairs = sel.pair_seeds(bp.SEEDS)
    assert pairs == [(0, 5), (10, 15), (20, 25), (30, 35), (40, 0)]
    assert sel.pair_seeds([]) == [] and sel.pair_seeds([7]) == [(7, 7)]


def test_expected_rating_normalises_probabilities():
    assert sel.expected_rating([0, 0, 0, 0, 1]) == 5
    assert sel.expected_rating([1, 0, 0, 0, 1]) == 3
    with pytest.raises(ValueError):
        sel.expected_rating([0, 0, 0, 0, 0])


def test_fire_selector_maps_pair_sides_to_seeds():
    class FakeUnit:
        def __init__(self, s, i, a, b):
            self.a, self.b = a, b

    def fake_scores(units):
        u = units[0]
        seed = lambda p: int(p.split("_")[-1])
        return [{"image_scores_A": seed(u.a), "image_scores_B": seed(u.b), "metric_scores_A": [seed(u.a)] * 6,
                 "metric_scores_B": [seed(u.b)] * 6, "relevance_probs": [0.5] * 6, "tie_probs": [0.1] * 6}]

    fire = sel.FireSelector.__new__(sel.FireSelector)
    fire.Unit, fire.score_units = FakeUnit, fake_scores
    sample = {"source_image": "s", "instruction": "i"}
    items = [{"seed": s, "edited_image": f"img_{s}"} for s in bp.SEEDS]
    rows = fire.score_sample(sample, items)
    assert [r["seed"] for r in rows] == list(bp.SEEDS)
    assert all(r["score"] == r["seed"] and r["metric_scores"][0] == r["seed"] for r in rows)
    assert rows[0]["paired_with"] == 5 and rows[-1]["paired_with"] == 0


def test_load_selector_rejects_unknown_name():
    with pytest.raises((ValueError, KeyError)):
        sel.load_selector("PickScore", "x")


# ---------------------------------------------------------------- judge image set
def _items(samples=("a", "b")):
    return [{"sample_id": s, "seed": seed} for s in samples for seed in bp.SEEDS]


def test_wanted_items_is_seed_zero_plus_selected_seeds(tmp_path):
    (tmp_path / "CLIP").mkdir()
    (tmp_path / "CLIP" / "ImgEdit.jsonl").write_text(
        json.dumps({"sample_id": "a", "seed": 25}) + "\n" + json.dumps({"sample_id": "b", "seed": 0}) + "\n")
    items, missing = jb.wanted_items(_items(), tmp_path, "ImgEdit", ["CLIP", "ImageReward"])
    assert sorted((i["sample_id"], i["seed"]) for i in items) == [("a", 0), ("a", 25), ("b", 0)]
    assert missing == ["ImageReward"]
    assert [i["chosen_by"] for i in items if i["seed"] == 25] == [["CLIP"]]


def test_wanted_items_without_selections_is_seed_zero_only(tmp_path):
    items, missing = jb.wanted_items(_items(), tmp_path, "ImgEdit", ["CLIP"])
    assert {i["seed"] for i in items} == {0} and len(items) == 2 and missing == ["CLIP"]


def test_wanted_items_rejects_unknown_seed(tmp_path):
    (tmp_path / "CLIP").mkdir()
    (tmp_path / "CLIP" / "ImgEdit.jsonl").write_text(json.dumps({"sample_id": "a", "seed": 3}) + "\n")
    with pytest.raises(ValueError):
        jb.wanted_items(_items(), tmp_path, "ImgEdit", ["CLIP"])


def test_imagereward_patch_exposes_helpers_in_modeling_utils():
    import transformers.modeling_utils as mu
    sel.patch_transformers_for_imagereward()
    assert callable(mu.apply_chunking_to_forward) and callable(mu.prune_linear_layer)


# ---------------------------------------------------------------- EditScore / EditReward / run name
def test_capped_size_keeps_small_images_and_aspect_of_large_ones():
    assert sel.capped_size((800, 600), None) == (800, 600)
    assert sel.capped_size((800, 600), 10**6) == (800, 600)
    w, h = sel.capped_size((6000, 3000), 10**6)
    assert w * h <= 10**6 and abs(w / h - 2) < 0.01
    with pytest.raises(ValueError):
        sel.capped_size((10, 10), 0)


def test_editscore_selector_uses_overall_and_resizes_edited_to_source(tmp_path):
    from PIL import Image
    Image.new("RGB", (64, 32)).save(tmp_path / "src.png")
    for s in bp.SEEDS:
        Image.new("RGB", (16, 16)).save(tmp_path / f"seed_{s}.png")
    seen = []

    class FakeScorer:
        def evaluate(self, images, instruction):
            seen.append((images[0].size, images[1].size, instruction))
            reasoning = "guess_if_cannot_parse" if len(seen) == 2 else "ok"
            return {"overall": len(seen), "prompt_following": 1, "consistency": 2, "perceptual_quality": 3,
                    "SC_reasoning": reasoning, "PQ_reasoning": "ok"}

    es = sel.EditScoreSelector.__new__(sel.EditScoreSelector)
    es.scorer, es.max_source_pixels = FakeScorer(), None
    items = [{"seed": s, "edited_image": str(tmp_path / f"seed_{s}.png"), "instruction": "add a cat"} for s in bp.SEEDS]
    rows = es.score_sample({"source_image": str(tmp_path / "src.png")}, items)
    assert [r["seed"] for r in rows] == list(bp.SEEDS)
    assert [r["score"] for r in rows] == list(range(1, 10))
    assert all(a == b == (64, 32) and i == "add a cat" for a, b, i in seen)
    assert [r["guessed"] for r in rows].count(True) == 1 and rows[1]["guessed"]


def test_editscore_selector_caps_large_source(tmp_path):
    from PIL import Image
    Image.new("RGB", (400, 200)).save(tmp_path / "src.png")
    Image.new("RGB", (16, 16)).save(tmp_path / "e.png")

    class FakeScorer:
        def evaluate(self, images, instruction):
            assert images[0].size == images[1].size
            return {"overall": images[0].size[0], "prompt_following": 0, "consistency": 0, "perceptual_quality": 0}

    es = sel.EditScoreSelector.__new__(sel.EditScoreSelector)
    es.scorer, es.max_source_pixels = FakeScorer(), 20000
    row = es.score_sample({"source_image": str(tmp_path / "src.png")},
                          [{"seed": 0, "edited_image": str(tmp_path / "e.png"), "instruction": "x"}])[0]
    assert row["scored_size"][0] * row["scored_size"][1] <= 20000 and row["score"] == 200


def test_editreward_local_config_replaces_only_model_path_and_output_dir():
    text = 'a: 1\nmodel_name_or_path: "Qwen/Qwen2.5-VL-7B-Instruct"\noutput_dir: "x"\nb: 2\n'
    out = sel.editreward_local_config(text, "/local/qwen", "/out")
    assert out == 'a: 1\nmodel_name_or_path: "/local/qwen"\noutput_dir: "/out"\nb: 2\n'
    with pytest.raises(ValueError):
        sel.editreward_local_config("a: 1\n", "/local/qwen", "/out")


def test_editreward_config_of_repo_has_mean_pooling_and_one_backbone_line():
    text = sel.EDITREWARD_CONFIG.read_text(encoding="utf-8")
    assert 'pooling_strategy: "mean"' in text
    sel.editreward_local_config(text, "/local/qwen", "/out")


def test_editreward_reward_dim_is_read_from_config_text():
    assert sel.editreward_reward_dim("a: 1\nreward_dim: overall_detail\n") == "overall_detail"
    assert sel.editreward_reward_dim('reward_dim: "dim2"  # comment\n') == "dim2"


def test_editreward_reward_dim_rejects_missing_duplicate_or_unknown_value():
    for text in ("a: 1\n", "reward_dim: dim1\nreward_dim: dim2\n", "reward_dim: everything\n"):
        with pytest.raises(ValueError):
            sel.editreward_reward_dim(text)


def test_editreward_repo_config_uses_overall_detail_prompt():
    assert sel.editreward_reward_dim(sel.EDITREWARD_CONFIG.read_text(encoding="utf-8")) == "overall_detail"


def test_editreward_selector_passes_config_reward_dim_to_inferencer(tmp_path, monkeypatch):
    import sys, types
    captured = {}

    class FakeInferencer:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.model = types.SimpleNamespace(pooling_strategy="mean")

    package = types.ModuleType("EditReward")
    module = types.ModuleType("EditReward.inference_edit")
    module.EditRewardInferencer = FakeInferencer
    monkeypatch.setitem(sys.modules, "EditReward", package)
    monkeypatch.setitem(sys.modules, "EditReward.inference_edit", module)
    (tmp_path / "ckpt").mkdir(); (tmp_path / "backbone").mkdir()
    selector = sel.EditRewardSelector(str(tmp_path / "ckpt"), "cpu", backbone_path=str(tmp_path / "backbone"),
                                      work_dir=str(tmp_path / "work"))
    assert captured["reward_dim"] == "overall_detail" and captured["rm_head_type"] == "ranknet_multi_head"
    assert selector.description["reward_dim"] == "overall_detail"


def test_editreward_rows_take_mu_and_reject_bad_outputs():
    items = [{"seed": s} for s in (0, 5)]
    rows = sel.editreward_rows([[1.5, -0.2], [0.5, 0.1]], items)
    assert [(r["seed"], r["score"], r["log_sigma"]) for r in rows] == [(0, 1.5, -0.2), (5, 0.5, 0.1)]
    with pytest.raises(ValueError):
        sel.editreward_rows([[1.0, 0.0]], items)
    with pytest.raises(ValueError):
        sel.editreward_rows([[float("nan"), 0.0], [1.0, 0.0]], items)
    with pytest.raises(ValueError):
        sel.editreward_rows([[1.0], [1.0]], items)


def test_editscore_editreward_default_weights_exist():
    for name in ("EditScore", "EditReward"):
        assert name in sel.SELECTORS
    from pathlib import Path
    assert (Path(sel.DEFAULT_PATHS["EditScore"]) / "adapter_config.json").is_file()
    assert (Path(sel.DEFAULT_PATHS["EditReward"]) / "model.safetensors").exists()


def test_select_best_run_name_reads_and_writes_its_own_folder(tmp_path, monkeypatch):
    samples = [{"sample_id": "s1"}]
    monkeypatch.setattr(bp, "load_samples", lambda bench: samples)
    scores = tmp_path / "scores" / "FIRE-new" / "ImgEdit"
    scores.mkdir(parents=True)
    rows = _rows([1, 2, 9, 3, 4, 5, 6, 7, 8], extra=lambda s: {"metric_scores": [s] * 6})
    (scores / "scores.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    sb.main(["--selector", sb.FIRE, "--run-name", "FIRE-new", "--benches", "ImgEdit",
             "--scores-root", str(tmp_path / "scores"), "--output-root", str(tmp_path / "sel")])
    out = sb.read_jsonl(tmp_path / "sel" / "FIRE-new" / "ImgEdit.jsonl")
    assert out[0]["seed"] == 40 and out[0]["selector"] == "FIRE-new"  # aesthetic_fidelity uses metric_scores
    assert not (tmp_path / "sel" / sb.FIRE).exists()
