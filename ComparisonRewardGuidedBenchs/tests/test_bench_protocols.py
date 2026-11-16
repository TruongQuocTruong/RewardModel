import json
import math
import random

import pytest

import bench_protocols as bp


@pytest.fixture(scope="module")
def prompts():
    return bp.load_prompts()


def _item(bench, category="color_alter", **extra):
    item = {"bench": bench, "sample_id": f"{category}/1", "category": category, "instruction": "Make it red.",
            "explanation": "Rust is red.", "source_image": "/src.png", "edited_image": "/edit.png", "seed": 0}
    item.update(extra)
    return item


# ---------------------------------------------------------------- prompts and calls
def test_gedit_prompts_match_viescore_tie_task(prompts):
    assert prompts["gedit_sc"].startswith("You are a professional digital artist")
    assert "<instruction>" in prompts["gedit_sc"]
    assert "<instruction>" not in prompts["gedit_pq"]


def test_gedit_calls_send_source_and_edited_to_sc_and_only_edited_to_pq(prompts):
    sc, pq = bp.judge_calls(_item("GEdit-Bench-EN"), prompts)
    assert [c[0] for c in sc["content"]] == ["image", "image", "text"]
    assert sc["content"][0][1] == "/src.png" and sc["content"][1][1] == "/edit.png"
    assert "Make it red." in sc["content"][2][1] and "<instruction>" not in sc["content"][2][1]
    assert [c[:2] for c in pq["content"][:1]] == [("image", "/edit.png")]
    assert all(c[2] == "gedit_512_area" for c in sc["content"] + pq["content"] if c[0] == "image")


def test_imgedit_call_puts_text_before_original_then_edited(prompts):
    (call,) = bp.judge_calls(_item("ImgEdit", "style"), prompts)
    assert [c[0] for c in call["content"]] == ["text", "image", "image"]
    assert call["content"][1][1] == "/src.png" and call["content"][2][1] == "/edit.png"
    assert "Make it red." in call["content"][0][1] and "<edit_prompt>" not in call["content"][0][1]


def test_kris_quality_prompt_is_sent_unformatted_with_only_edited_image(prompts):
    calls = {c["name"]: c for c in bp.judge_calls(_item("KRIS-Bench", "color_change"), prompts)}
    quality = calls["quality"]["content"]
    assert quality[0] == ("text", prompts["kris"]["prompt_quality"])
    assert "{{" in quality[0][1]
    assert [c[0] for c in quality] == ["text", "text", "image"] and quality[2][1] == "/edit.png"


def test_kris_knowledge_category_uses_dual_prompt_with_explanation(prompts):
    names = [c["name"] for c in bp.judge_calls(_item("KRIS-Bench", "chemistry"), prompts)]
    assert names == ["consistency", "dual", "quality"]
    dual = bp.judge_calls(_item("KRIS-Bench", "chemistry"), prompts)[1]["content"][0][1]
    assert "Rust is red." in dual


def test_kris_viewpoint_change_adds_ground_truth_image_last(prompts):
    calls = bp.judge_calls(_item("KRIS-Bench", "viewpoint_change", gt_image="/gt.png"), prompts)
    content = calls[1]["content"]
    assert content[-2] == ("text", "This is the ground truth image:") and content[-1][1] == "/gt.png"


def test_kris_viewpoint_change_without_ground_truth_raises(prompts):
    with pytest.raises(ValueError):
        bp.judge_calls(_item("KRIS-Bench", "viewpoint_change"), prompts)


def test_kris_anomaly_correction_uses_normal_prompt_like_original_code(prompts):
    text = bp.judge_calls(_item("KRIS-Bench", "anomaly_correction"), prompts)[1]["content"][0][1]
    assert text == prompts["kris"]["prompt_instruction_following"].format(instruct="Make it red.")
    intended = bp.judge_calls(_item("KRIS-Bench", "anomaly_correction"), prompts, "intended")[1]["content"][0][1]
    assert intended != text


def test_kris_unknown_category_raises():
    with pytest.raises(ValueError):
        bp.kris_category_group("temporal_prediction")


def test_unknown_bench_raises(prompts):
    with pytest.raises(ValueError):
        bp.judge_calls(_item("Other"), prompts)


# ---------------------------------------------------------------- samples
def test_load_samples_builds_paths_and_expand_gives_nine_seeds(tmp_path):
    (tmp_path / "gedit_300.json").write_text(json.dumps([
        {"key": "k1", "task_type": "color_alter", "instruction": "x", "image": "images/color_alter/k1.png"}]))
    (tmp_path / "imgedit_singleturn_300.json").write_text(json.dumps(
        {"7": {"id": "animal/0001.jpg", "prompt": "y", "edit_type": "style"}}))
    (tmp_path / "kris_300.json").write_text(json.dumps([
        {"category": "viewpoint_change", "id": "3", "instruction": "z", "explanation": "", "export_key": "000009",
         "image": "images/000009/image.png", "gt_image": "images/000009/gt_image.png"}]))
    g = bp.load_samples("GEdit-Bench-EN", tmp_path, tmp_path / "src")[0]
    assert g["rollout_dir"].endswith("Results/FLUX2-9B/GEdit-Bench-EN/color_alter/k1")
    assert g["source_image"] == str(tmp_path / "src/GEdit-Bench-EN/images/color_alter/k1.png")
    i = bp.load_samples("ImgEdit", tmp_path, tmp_path / "src")[0]
    assert i["rollout_dir"].endswith("ImgEdit-Bench/singleturn/animal/7")
    k = bp.load_samples("KRIS-Bench", tmp_path, tmp_path / "src")[0]
    assert k["sample_id"] == "viewpoint_change/3" and k["gt_image"].endswith("000009/gt_image.png")
    items = bp.expand_seeds([g])
    assert [it["seed"] for it in items] == list(bp.SEEDS) and len(items) == 9
    assert items[1]["edited_image"].endswith("k1/seed_5.png")


def test_imgedit_nested_compose_id_keeps_full_folder_in_rollout_dir(tmp_path):
    (tmp_path / "imgedit_singleturn_300.json").write_text(json.dumps(
        {"1147": {"id": "compose/human/12.jpg", "prompt": "y", "edit_type": "compose"}}))
    i = bp.load_samples("ImgEdit", tmp_path, tmp_path / "src")[0]
    assert i["rollout_dir"].endswith("ImgEdit-Bench/singleturn/compose/human/1147")
    assert i["source_image"].endswith("Benchmark/singleturn/compose/human/12.jpg")


def test_load_samples_rejects_duplicate_ids(tmp_path):
    row = {"key": "k1", "task_type": "color_alter", "instruction": "x", "image": "images/color_alter/k1.png"}
    (tmp_path / "gedit_300.json").write_text(json.dumps([row, row]))
    with pytest.raises(ValueError):
        bp.load_samples("GEdit-Bench-EN", tmp_path, tmp_path)


def test_expand_seeds_of_empty_list_is_empty():
    assert bp.expand_seeds([]) == []


def test_gedit_resize_keeps_ratio_and_area():
    assert bp.gedit_resize_size(1024, 1024) == (512, 512)
    w, h = bp.gedit_resize_size(2000, 1000)
    assert abs(w * h - 512 * 512) < 1000 and abs(w / h - 2) < 0.01
    with pytest.raises(ValueError):
        bp.gedit_resize_size(0, 10)


# ---------------------------------------------------------------- GEdit parsing
def test_viescore_parses_json_and_takes_min_score():
    parsed = bp.viescore_output_to_dict('Sure.\n{\n"score" : [8, 6],\n"reasoning" : "ok"\n}')
    assert bp.gedit_min_score(parsed) == 6


def test_viescore_parses_bare_list_and_single_number():
    assert bp.gedit_min_score(bp.viescore_output_to_dict("[7, 9]")) == 7
    assert bp.gedit_min_score(bp.viescore_output_to_dict("4")) == 4


def test_viescore_unparsable_returns_false_and_give_up_guesses_reproducibly():
    assert bp.viescore_output_to_dict("no scores here") is False
    a = bp.viescore_output_to_dict("no scores", give_up_parsing=True, rng=random.Random("x"))
    b = bp.viescore_output_to_dict("no scores", give_up_parsing=True, rng=random.Random("x"))
    assert a == b and 0 <= a["score"][0] <= 10


def test_gedit_overall_is_geometric_mean_and_none_propagates():
    assert bp.gedit_columns(9.0, 4.0)["overall"] == pytest.approx(6.0)
    assert bp.gedit_columns(None, 4.0)["overall"] is None
    assert bp.gedit_columns(0.0, 4.0)["overall"] == 0.0
    with pytest.raises(ValueError):
        bp.gedit_columns(-1.0, 4.0)


def test_gedit_min_score_rejects_nan_and_missing():
    assert bp.gedit_min_score({"score": [float("nan")]}) is None
    assert bp.gedit_min_score(False) is None
    assert bp.gedit_min_score({"score": []}) is None


# ---------------------------------------------------------------- ImgEdit parsing
ANSWER = "Brief reasoning: fine.\nStyle Fidelity: 4\nContent Preservation: 2\nRendering Quality: 3"


def test_imgedit_style_maps_second_score_to_detail_preservation():
    cols = bp.imgedit_columns(bp.imgedit_scores(ANSWER), "style")
    assert cols == {"instruction_adherence": 4.0, "editing_quality": 3.0, "detail_preservation": 2.0,
                    "original_average": 3.0}


def test_imgedit_replace_maps_scores_in_order():
    cols = bp.imgedit_columns([5, 4, 3], "replace")
    assert (cols["instruction_adherence"], cols["editing_quality"], cols["detail_preservation"]) == (5, 4, 3)


def test_imgedit_wrong_score_count_gives_none_columns_but_keeps_original_average():
    cols = bp.imgedit_columns([5, 4], "add")
    assert cols["instruction_adherence"] is None and cols["original_average"] == 4.5
    assert bp.imgedit_columns([], "add")["original_average"] is None


def test_imgedit_unknown_edit_type_raises():
    with pytest.raises(ValueError):
        bp.imgedit_columns([1, 2, 3], "warp")


# ---------------------------------------------------------------- KRIS parsing
def test_kris_score_reads_json_first_then_regex_fallbacks():
    assert bp.kris_score('{"reasoning": "x", "consistency_score": 4}', "consistency_score") == 4
    assert bp.kris_score("quality score: 2", "quality_score") == 2
    assert bp.kris_score("I give it 3/5", "instruction_score") == 3
    assert bp.kris_score("nothing", "quality_score") is None


def test_kris_dual_scores_json_and_regex():
    assert bp.kris_dual_scores('{"instruction_score": 5, "knowledge_score": 2}') == \
        {"instruction_score": 5, "knowledge_score": 2}
    assert bp.kris_dual_scores("Instruction score: 3\nKnowledge score: 1") == \
        {"instruction_score": 3, "knowledge_score": 1}


def test_kris_columns_for_common_and_knowledge():
    common = bp.kris_columns({"consistency": '{"consistency_score": 5}', "quality": '{"quality_score": 4}',
                              "instruction": '{"instruction_score": 2}'})
    assert common == {"visual_consistency": 5.0, "visual_quality": 4.0, "instruction_following": 2.0}
    knowledge = bp.kris_columns({"consistency": "x", "quality": "4/5",
                                 "dual": '{"instruction_score": 3, "knowledge_score": 1}'})
    assert knowledge["visual_consistency"] is None and knowledge["instruction_following"] == 3.0
    assert knowledge["knowledge_plausibility"] == 1.0
