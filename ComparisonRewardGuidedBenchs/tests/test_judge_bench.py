import json

import pytest

import bench_protocols as bp
import judge_bench as jb


@pytest.fixture(scope="module")
def prompts():
    return bp.load_prompts()


def _item(bench, category):
    return {"bench": bench, "sample_id": f"{category}/1", "category": category, "instruction": "do",
            "explanation": "", "source_image": "/s.png", "edited_image": "/e.png", "seed": 5}


class Scripted:
    """Fake judge: returns answers in order and records the token budgets."""

    def __init__(self, answers):
        self.answers, self.budgets = list(answers), []

    def __call__(self, content, max_new_tokens):
        self.budgets.append(max_new_tokens)
        return self.answers.pop(0)


def test_gedit_item_strict_parse(prompts):
    fake = Scripted(['{"score": [8, 9], "reasoning": ""}', '{"score": [4, 6], "reasoning": ""}'])
    row = jb.judge_item(_item("GEdit-Bench-EN", "color_alter"), prompts, fake)
    assert row["scores"]["semantic_consistency"] == 8 and row["scores"]["perceptual_quality"] == 4
    assert row["parse"] == {"SC": "strict", "PQ": "strict"} and fake.budgets == [512, 512]


def test_gedit_unparsable_retries_longer_then_guesses(prompts):
    fake = Scripted(["oops", "still oops", '{"score": [5, 5]}'])
    row = jb.judge_item(_item("GEdit-Bench-EN", "color_alter"), prompts, fake)
    assert row["parse"]["SC"] == "guessed" and fake.budgets[:2] == [512, 1024]
    assert 0 <= row["scores"]["semantic_consistency"] <= 10


def test_imgedit_item_maps_columns(prompts):
    fake = Scripted(["Brief reasoning: ok\nPrompt Compliance: 5\nVisual Naturalness: 3\nPhysical & Detail Integrity: 2"])
    row = jb.judge_item(_item("ImgEdit", "remove"), prompts, fake)
    assert row["scores"]["instruction_adherence"] == 5 and row["scores"]["detail_preservation"] == 2


def test_imgedit_failed_parse_keeps_none(prompts):
    fake = Scripted(["no", "no again"])
    row = jb.judge_item(_item("ImgEdit", "remove"), prompts, fake)
    assert row["parse"]["ImgEdit"] == "failed" and row["scores"]["instruction_adherence"] is None


def test_kris_item_uses_three_calls(prompts):
    fake = Scripted(['{"consistency_score": 4}', '{"instruction_score": 3}', '{"quality_score": 5}'])
    row = jb.judge_item(_item("KRIS-Bench", "count_change"), prompts, fake)
    assert row["scores"] == {"visual_consistency": 4.0, "visual_quality": 5.0, "instruction_following": 3.0}
    assert fake.budgets == [1000, 1000, 1000]


def test_read_done_drops_interrupted_last_line(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_bytes(b'{"sample_id": "a", "seed": 0}\n{"sample_id": "a", "se')
    assert jb.read_done(path) == {("a", 0)}
    assert path.read_bytes() == b'{"sample_id": "a", "seed": 0}\n'


def test_read_done_rejects_corrupt_middle_line_and_duplicates(tmp_path):
    path = tmp_path / "scores.jsonl"
    path.write_text('garbage\n{"sample_id": "a", "seed": 0}\n')
    with pytest.raises(ValueError):
        jb.read_done(path)
    path.write_text('{"sample_id": "a", "seed": 0}\n{"sample_id": "a", "seed": 0}\n')
    with pytest.raises(ValueError):
        jb.read_done(path)


def test_read_done_missing_file_is_empty(tmp_path):
    assert jb.read_done(tmp_path / "none.jsonl") == set()


def test_check_files_reports_missing_images(tmp_path):
    (tmp_path / "s.png").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        jb.check_files([{"source_image": str(tmp_path / "s.png"), "edited_image": str(tmp_path / "e.png")}])


def test_default_judges_follow_user_decision():
    assert jb.DEFAULT_JUDGE == {"GEdit-Bench-EN": "gemma-4-31B-it", "KRIS-Bench": "gemma-4-31B-it",
                                "ImgEdit": "Qwen2.5-VL-72B-Instruct-AWQ"}


def test_select_items_rejects_non_positive_max_samples(tmp_path):
    (tmp_path / "gedit_300.json").write_text(json.dumps([
        {"key": "k", "task_type": "color_alter", "instruction": "x", "image": "images/color_alter/k.png"}]))
    with pytest.raises(ValueError):
        jb.select_items("GEdit-Bench-EN", tmp_path, tmp_path, 0)
    assert len(jb.select_items("GEdit-Bench-EN", tmp_path, tmp_path, 1)) == 9


def test_autoawq_patch_aliases_gelu_tanh():
    import transformers.activations as act
    import judges
    judges.patch_transformers_for_autoawq()
    assert act.PytorchGELUTanh is act.GELUTanh or act.PytorchGELUTanh.__name__ == "PytorchGELUTanh"
