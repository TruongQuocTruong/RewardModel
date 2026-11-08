#!/usr/bin/env python3
"""Student warm-start trained on majority-vote hard GemmaGT labels.

Identical to ``train_student_warmstart.py`` (GemmaGT Data, factorized
warm-start loss, same split and hyperparameters) except for one change: the
training targets are hardened by majority vote over the three Gemma runs.
Validation keeps the soft three-run targets, so the validation loss used for
checkpoint selection is comparable with the soft-label Student.

Majority-vote rule per metric:
- Relevance is 1 when at least 2 of the 3 runs mark the metric applicable,
  otherwise 0 (and the preference targets are zero, as in the soft labels).
- When relevant, the preference target is the choice (A, B or Tie) holding a
  strict majority (> 50%) of the applicable runs; without one it is Tie.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import torch
from datasets import Dataset as HfDataset

from train_student_warmstart import (
    CHECKPOINT_FILENAME,
    NUM_METRICS,
    GemmaWarmStartPreprocessor,
    MetricWarmStartSwiftSft,
    logger,
)

MAJORITY_RELEVANCE = 2.0 / 3.0


def _majority_vote_labels(labels: List[float], stats: Counter | None = None) -> List[float]:
    """Harden one soft [g_A | g_B | g_T | relevance] label vector."""
    g_a = labels[0:NUM_METRICS]
    g_b = labels[NUM_METRICS : 2 * NUM_METRICS]
    g_t = labels[2 * NUM_METRICS : 3 * NUM_METRICS]
    relevance = labels[3 * NUM_METRICS : 4 * NUM_METRICS]
    hard_a, hard_b, hard_t, hard_relevance = [], [], [], []
    for a, b, t, m in zip(g_a, g_b, g_t, relevance):
        if m < MAJORITY_RELEVANCE - 1e-6:
            hard_a.append(0.0)
            hard_b.append(0.0)
            hard_t.append(0.0)
            hard_relevance.append(0.0)
            if stats is not None:
                stats["not_relevant"] += 1
            continue
        if a > 0.5:
            choice = (1.0, 0.0, 0.0)
        elif b > 0.5:
            choice = (0.0, 1.0, 0.0)
        elif t > 0.5:
            choice = (0.0, 0.0, 1.0)
        else:
            choice = (0.0, 0.0, 1.0)
            if stats is not None:
                stats["no_majority_to_tie"] += 1
        if stats is not None:
            stats["relevant"] += 1
            if (a, b, t) != choice:
                stats["preference_changed"] += 1
        hard_a.append(choice[0])
        hard_b.append(choice[1])
        hard_t.append(choice[2])
        hard_relevance.append(1.0)
    return hard_a + hard_b + hard_t + hard_relevance


class GemmaHardLabelPreprocessor(GemmaWarmStartPreprocessor):
    """Same encoding as the soft preprocessor with majority-vote targets."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cache_signature = ("gemma_three_run_majority_vote_v1",) + tuple(
            self._cache_signature[1:]
        )

    def preprocess(self, row: Dict[str, Any]) -> Dict[str, Any]:
        result = super().preprocess(row)
        result["label"] = _majority_vote_labels(result["label"])
        return result


class StudentHardLabelSwiftSft(MetricWarmStartSwiftSft):
    def _get_dataset(self):
        _, validation_dataset = super()._get_dataset()

        stats: Counter = Counter()
        for record in self._train_records:
            soft = GemmaWarmStartPreprocessor().preprocess(dict(record))["label"]
            _majority_vote_labels(soft, stats)
        self._hard_label_stats = dict(stats)
        logger.info("Majority-vote GemmaGT training targets: %s", self._hard_label_stats)

        train_dataset = GemmaHardLabelPreprocessor()(
            HfDataset.from_list(self._train_records),
            num_proc=int(self.args.dataset_num_proc),
            load_from_cache_file=bool(self.args.load_from_cache_file),
            strict=True,
        )
        logger.info("Validation keeps the soft three-run GemmaGT targets.")
        return train_dataset, validation_dataset

    def train(self, trainer):
        result = super().train(trainer)

        if trainer.is_world_process_zero():
            checkpoint_path = Path(self.args.output_dir) / CHECKPOINT_FILENAME
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint["gemma_training_targets"] = {
                "mode": "majority_vote_hard",
                "relevance_rule": "relevant if >= 2 of 3 runs have targeted_metric=1",
                "preference_rule": "strict majority of applicable runs; no majority -> Tie",
                "validation_targets": "soft three-run proportions",
                "train_cell_counts": self._hard_label_stats,
            }
            tmp_path = checkpoint_path.with_suffix(".pt.tmp")
            torch.save(checkpoint, tmp_path)
            os.replace(tmp_path, checkpoint_path)
            logger.info("Recorded hard GemmaGT training targets in: %s", checkpoint_path)
        return result


def main():
    return StudentHardLabelSwiftSft().main()


if __name__ == "__main__":
    main()
