#!/usr/bin/env python3
"""Stage 2 training loop with quality-gated adaptive Teacher screening.

The frozen Inspector cache orders candidates. Each round rechecks candidates
with one deterministic EMA Teacher pass only until the round's sample targets
are filled. J=4 MC Dropout is reserved for selected training batches.
Unused candidates remain available for later rounds; stale Teacher predictions
are never reused as training targets. Later-round proportions are soft targets,
with a minimum quality-gated round size and complete checkpoints per round.
"""

from __future__ import annotations

import argparse
import inspect
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from peft import (
    LoraConfig,
    PeftConfig,
    PeftModel,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from qwen_vl_utils import process_vision_info
from swift.tuners import Swift
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.models.custom_qwen3_vl_reranker import (  # noqa: E402
    METRIC_NAMES,
    NUM_METRICS,
    FactorizedWarmStartLoss,
    MultiMetricRewardOutput,
    TIE_HEAD_MODE_ENV,
    _pool_qwen_last_hidden,
    install_multi_metric_heads,
    resolve_tie_head_mode,
    tie_head_mode_from_state_keys,
)


STUDENT_ROLE = "six_metric_student_warmstart"
TEACHER_ROLE = "six_metric_ema_teacher_initialized"
INSPECTOR_ROLE = "overall_inspector"
TIE_VOTES = {"tie", "tievote"}
VOTE_TO_LABEL = {"leftvote": 1.0, "rightvote": 0.0}
# label_a of a HumanGT tie, kept only with --overall-loss three_way.
HUMAN_TIE_LABEL = 0.5
DEFAULT_EDITREWARD_ROOT = (
    "/datastore/shared/datasets/EditReward-Data"
)
PROBABILITY_EPS = 1e-7
UNLABELED_OBJECTIVES = (
    "projection",
    "inspector_direct",
    "inspector_ema_consistency",
    "inspector_soft_ema_consistency",
)
REPLAY_STRATEGIES = ("hard", "random_previous_round")
GEMMA_TARGETS = ("soft", "majority_vote")
UNLABELED_SAMPLINGS = (
    "agreement_strata", "confidence_order", "all_once", "inspector_teacher_rank",
    "previous_iteration_teacher_inspector_agreement",
)
# Iteration 1 trains every sample of the subset once (as all_once); after every iteration the
# Teacher (EMA) scores those samples on the original input and only the samples whose Teacher
# label differs from the Inspector label are left out of the next iteration.
ITERATIVE_SAMPLING = "previous_iteration_teacher_inspector_agreement"
# all_metrics_fitted: six metrics with the fitted a_k; selected_equal: equal mean
# over the metrics with relevance >= 0.5 (else the most relevant one).
OVERALL_AGGREGATIONS = ("all_metrics_fitted", "selected_equal")
# sigmoid_beta: P(A) = sigmoid(beta * m); linear: P(A) = (1 + m) / 2, no constant.
OVERALL_PROBABILITIES = ("sigmoid_beta", "linear")
OVERALL_LOSSES = ("binary", "three_way")
CONSISTENCY_METRICS = ("all", "selected")
EMA_SCHEDULES = ("constant", "cosine")
RELEVANCE_THRESHOLD = 0.5
LINEAR_EVIDENCE_EPS = 1e-6
ACTIVE_SWIFT_TEMPLATE: Any | None = None


@dataclass(frozen=True)
class PairRecord:
    sample_id: str
    source_image: str
    edit_instruction: str
    image_a: str
    image_b: str


@dataclass(frozen=True)
class HumanRecord(PairRecord):
    label_a: float


@dataclass(frozen=True)
class GemmaRecord(PairRecord):
    g_a: Tuple[float, ...]
    g_b: Tuple[float, ...]
    g_t: Tuple[float, ...]
    relevance: Tuple[float, ...]


@dataclass(frozen=True)
class InspectorPseudoLabels:
    labels: torch.Tensor
    confidence: torch.Tensor
    probability_a: torch.Tensor
    # [N, 3] = (P(A), P(B), P(Tie)); NaN for a binary cache.
    three_way: torch.Tensor


@dataclass(frozen=True)
class UnlabeledPool:
    path: Path
    dataset_root: Path
    sample_ids: List[str]
    source_groups: List[str]
    byte_offsets: List[int]


@dataclass
class TeacherCache:
    sample_ids: List[str]
    mean_probs: torch.Tensor
    mean_relevance: torch.Tensor
    var_winner: torch.Tensor
    var_tie: torch.Tensor
    var_relevance: torch.Tensor
    lower: torch.Tensor
    upper: torch.Tensor
    inspector_labels: torch.Tensor
    effective_kappa: torch.Tensor
    valid: torch.Tensor
    status: List[str]


@dataclass(frozen=True)
class RoundSelection:
    indices: List[int]
    fresh_high_agreement: int
    fresh_low_agreement: int
    fresh_box_only: int
    hard_replay: int
    screened: int
    screening_status: Dict[str, int]


class CyclingRecordSampler:
    """Draw shuffled training records indefinitely without using validation."""

    def __init__(self, records: Sequence[Any], seed: int) -> None:
        if not records:
            raise ValueError("Human training split is empty.")
        self.records = list(records)
        self.rng = random.Random(seed)
        self.order: List[int] = []
        self.position = 0
        # Indices drawn since the last take_drawn() call (written to the round summary).
        self.drawn: List[int] = []
        self._reshuffle()

    def _reshuffle(self) -> None:
        self.order = list(range(len(self.records)))
        self.rng.shuffle(self.order)
        self.position = 0

    def draw(self, count: int) -> List[Any]:
        if count <= 0:
            raise ValueError("Human batch size must be positive.")
        output: List[Any] = []
        while len(output) < count:
            if self.position == len(self.order):
                self._reshuffle()
            take = min(count - len(output), len(self.order) - self.position)
            indices = self.order[self.position : self.position + take]
            output.extend(self.records[index] for index in indices)
            self.drawn.extend(indices)
            self.position += take
        return output

    def take_drawn(self) -> List[int]:
        drawn, self.drawn = self.drawn, []
        return drawn


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--student-checkpoint", required=True)
    parser.add_argument(
        "--base-model-path",
        default=None,
        help="Local base-model copy used at runtime; does not modify checkpoint weights.",
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--inspector-checkpoint", required=True)
    parser.add_argument("--inspector-cache", required=True)
    parser.add_argument(
        "--metrics-weights-aggregator", "--preference-aggregator",
        dest="metrics_weights_aggregator", default=None,
        help=(
            "Required unless --overall-probability linear, --overall-aggregation "
            "selected_equal and a non-projection objective are used together; "
            "the file is not read in that case."
        ),
    )
    parser.add_argument("--human-dataset", required=True)
    parser.add_argument("--gemma-dataset", required=True)
    parser.add_argument("--unlabeled-dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--resume-checkpoint", default=None)
    parser.add_argument("--max-rounds", type=int, default=5)
    parser.add_argument(
        "--restart-lr-each-round",
        action="store_true",
        help=(
            "Restart the cosine schedule, including its warmup, over each round's actual "
            "optimizer-step count. Without it one cosine spans all --max-rounds rounds and the "
            "last rounds train with a nearly zero learning rate."
        ),
    )
    parser.add_argument(
        "--replay-round-selections",
        default=None,
        help=(
            "JSON list with one list of Unlabeled sample IDs per round. Every round trains "
            "exactly those IDs (new shuffle and augmentation draws) instead of re-running the "
            "agreement filter, so arms compare on identical data and step budgets. Build it "
            "from a finished run with scripts/build_round_schedule.py."
        ),
    )
    parser.add_argument("--unlabeled-per-round", type=int, default=0)
    parser.add_argument(
        "--unlabeled-subset-size",
        type=int,
        default=0,
        help=(
            "0 trains on every Unlabeled sample; N keeps a fixed random subset of N "
            "samples (seed + 50_000, IDs saved to unlabeled_subset.json) and leaves "
            "the rest untrained."
        ),
    )
    parser.add_argument(
        "--unlabeled-passes",
        type=int,
        default=1,
        help="all_once only: passes over the Unlabeled set, reshuffled for each pass.",
    )
    parser.add_argument(
        "--labeled-source",
        choices=("separate", "merged"),
        default="separate",
        help=(
            "separate: HumanGT and GemmaGT are drawn and forwarded as two sources; "
            "merged: HumanGT and GemmaGT are the same pairs, each labeled pair is "
            "forwarded once and gets the HumanGT, Gemma and mask losses."
        ),
    )
    parser.add_argument(
        "--labeled-pairs-per-step",
        type=int,
        default=0,
        help="merged only: labeled pairs per optimizer step (multiple of --human-batch-size).",
    )
    parser.add_argument(
        "--loss-weighting",
        choices=("per_source", "per_pair"),
        default="per_source",
        help=(
            "per_source: each source is a mean over its microbatches; per_pair (merged "
            "only): every pair is summed and divided by the labeled pairs of the step, "
            "so one Unlabeled pair weighs lambda_U times one labeled pair."
        ),
    )
    parser.add_argument("--lambda-human", type=float, default=1.0)
    parser.add_argument(
        "--selection-seed",
        type=int,
        default=None,
        help=(
            "Seed of the fixed selections (diagnostic probe, Unlabeled subset, Gemma "
            "monitoring pairs); unset = --seed. Keeps the same data when only --seed changes."
        ),
    )
    parser.add_argument(
        "--head-learning-rate",
        type=float,
        default=None,
        help="Own AdamW learning rate for metric_heads (LoRA keeps --learning-rate); unset = shared.",
    )
    parser.add_argument(
        "--lr-scheduler-type",
        choices=("cosine", "constant_with_warmup"),
        default="cosine",
        help="Learning-rate schedule after --warmup-ratio warmup (MS-Swift online backend).",
    )
    parser.add_argument(
        "--step-diagnostics",
        action="store_true",
        help=(
            "Merged labeled source only: print one 'step=N diagnostics {json}' line per optimizer "
            "step with this step's losses (Unlabeled CE split into KL to Inspector + target "
            "entropy, Teacher KL), gradient norms per source and per group before clipping, "
            "cosine labeled vs Unlabeled gradient, update ratio |dW|/|W| and learning rates. "
            "Keeps two fp32 copies of the trainable parameters in CPU memory."
        ),
    )
    parser.add_argument(
        "--gemma-monitor-pairs",
        type=int,
        default=0,
        help=(
            "Without Gemma validation: 0 monitors Gemma losses on every training pair; "
            "N uses a fixed random subset of N training pairs (seed + 70_000)."
        ),
    )
    parser.add_argument("--diagnostic-probe-size", type=int, default=128)
    parser.add_argument(
        "--exclude-probe-from-training",
        choices=("true", "false"),
        default="true",
        help=(
            "true keeps the diagnostic probe and every Unlabeled sample sharing its "
            "source images out of training; false trains on all Unlabeled Data and "
            "only measures the probe."
        ),
    )
    parser.add_argument("--hard-replay-fraction", type=float, default=0.25)
    parser.add_argument("--low-agreement-fraction", type=float, default=0.25)
    parser.add_argument("--max-box-only-fraction", type=float, default=0.25)
    parser.add_argument("--box-only-set-weight", type=float, default=0.25)
    parser.add_argument("--min-round-fraction", type=float, default=0.5)
    parser.add_argument("--min-supervised-fraction", type=float, default=0.15)
    parser.add_argument("--high-inspector-confidence", type=float, default=0.80)
    parser.add_argument("--screening-multiplier", type=int, default=4)
    parser.add_argument("--screening-chunk-size", type=int, default=64)
    parser.add_argument("--max-replays-per-sample", type=int, default=2)
    parser.add_argument("--patience-rounds", type=int, default=2)
    parser.add_argument("--min-validation-improvement", type=float, default=0.002)
    parser.add_argument("--human-batch-size", type=int, default=1)
    parser.add_argument("--gemma-batch-size", type=int, default=1)
    parser.add_argument("--unlabeled-batch-size", type=int, default=1)
    parser.add_argument("--teacher-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument(
        "--tie-coupling-learning-rate",
        type=float,
        default=None,
        help=(
            "Separate learning rate for the margin-coupled Tie Head theta0 and kappa, "
            "without weight decay. Unset keeps them in the shared groups."
        ),
    )
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.95)
    parser.add_argument("--adam-epsilon", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument(
        "--unlabeled-objective",
        choices=UNLABELED_OBJECTIVES,
        default="projection",
        help=(
            "Unlabeled Data term of the Composite Loss Objective: "
            "projection = Metrics and Overall Projection (A3); "
            "inspector_direct = BCE of the Metrics Weights Aggregator logit on "
            "Overall Pseudo-label Cache labels (A1); "
            "inspector_ema_consistency = inspector_direct plus KL(Teacher Model || "
            "Student Model) on every selected sample (A2); "
            "inspector_soft_ema_consistency = BCE of the Metrics Weights Aggregator "
            "logit on the Inspector probability_A of every selected sample plus "
            "KL(Teacher Model || Student Model)."
        ),
    )
    parser.add_argument(
        "--unlabeled-sampling",
        choices=UNLABELED_SAMPLINGS,
        default="agreement_strata",
        help=(
            "Fresh Unlabeled Data of each round: agreement_strata = Inspector "
            "high/low agreement plus capped abstained samples; confidence_order = "
            "every remaining sample in descending Inspector confidence; all_once = "
            "every trainable sample once in random order, without Teacher Model "
            "screening or strata (requires --max-rounds 1); inspector_teacher_rank = "
            "--unlabeled-per-round fresh samples in descending Inspector confidence "
            "(round 1 defers samples the Teacher Model opposes to round 2; the last "
            "round takes every remaining sample) plus, from round 2, "
            "--hard-replay-fraction of the previous round with the smallest "
            "Teacher margin along the Inspector label; "
            "previous_iteration_teacher_inspector_agreement = iteration 1 trains every "
            "sample once (as all_once), every later iteration trains the samples of the "
            "previous iteration on which the Teacher (after EMA, original input, no dropout) "
            "and the Inspector give the same label (iteration_selection_round_XX.jsonl); "
            "stops early when no sample is left or both mean losses stall "
            "(--loss-plateau-tolerance)."
        ),
    )
    parser.add_argument(
        "--loss-plateau-tolerance",
        type=float,
        default=0.01,
        help=(
            "previous_iteration_teacher_inspector_agreement only: stop after an iteration "
            "whose mean Unlabeled loss and mean HumanGT cross-entropy both fell by less than "
            "this fraction of the previous iteration's values."
        ),
    )
    parser.add_argument(
        "--replay-strategy",
        choices=REPLAY_STRATEGIES,
        default="hard",
        help=(
            "Replay share of later rounds: hard = highest-ranked hard samples of "
            "earlier rounds; random_previous_round = uniform random samples trained "
            "in the previous round."
        ),
    )
    parser.add_argument(
        "--gemma-targets",
        choices=GEMMA_TARGETS,
        default="soft",
        help=(
            "GemmaGT Data replay targets: soft = three-run proportions; "
            "majority_vote = hard labels as in train_student_hardlabel.py. "
            "Gemma validation always keeps the soft targets."
        ),
    )
    parser.add_argument(
        "--overall-aggregation",
        choices=OVERALL_AGGREGATIONS,
        default="all_metrics_fitted",
        help=(
            "Overall logit of the HumanGT loss, the Inspector BCE and HumanGT "
            "validation: all_metrics_fitted = beta * sum_k a_k s_k over six metrics; "
            "selected_equal = beta * mean of s_k over the Relevance Head selection. "
            "The same aggregation gives the Teacher Model margin of curriculum screening."
        ),
    )
    parser.add_argument(
        "--overall-loss",
        choices=OVERALL_LOSSES,
        default="binary",
        help=(
            "binary = BCE on the overall P(A), HumanGT ties dropped, tie_probs detached; "
            "three_way = cross-entropy on the overall three-way distribution "
            "(mean of (P(A), P(B), P(Tie)) over the overall metric weights), HumanGT "
            "ties kept, tie_probs not detached, Inspector three-way soft labels on "
            "Unlabeled Data."
        ),
    )
    parser.add_argument(
        "--overall-probability",
        choices=OVERALL_PROBABILITIES,
        default="sigmoid_beta",
        help=(
            "Overall probability that A wins from the aggregated evidence m in [-1, 1]: "
            "sigmoid_beta = sigmoid(beta * m) with the Metrics Weights Aggregator beta; "
            "linear = (1 + m) / 2, without beta."
        ),
    )
    parser.add_argument(
        "--consistency-metrics",
        choices=CONSISTENCY_METRICS,
        default="all",
        help=(
            "Metrics of KL(Teacher Model || Student Model): all = mean over six "
            "metrics; selected = mean over the Student Relevance Head selection."
        ),
    )
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument(
        "--teacher-kl-relevance-weight",
        type=float,
        default=0.0,
        help=(
            "Share w of the relevance term in the Teacher consistency (user, 2026-11-07): "
            "consistency = (1 - w) * KL_pref + w * KL_rel, where KL_pref = the three-way "
            "KL(Teacher || Student) per metric averaged over the Student-selected metrics "
            "(unchanged) and KL_rel = the Bernoulli KL(Teacher relevance || Student "
            "relevance) averaged over all 6 metrics (Student relevance not detached). "
            "0 = the previous loss, unchanged."
        ),
    )
    parser.add_argument(
        "--save-round-checkpoints",
        choices=("all", "first", "none"),
        default="all",
        help=(
            "round_<N>.pt (full state for resume, about 6.9 GB) at the end of: every round "
            "(all), only round 1 (first; it is read by diagnose_gradient_components.py) or "
            "none. round_<N>_summary.json is always written (user, 2026-11-07)."
        ),
    )
    parser.add_argument(
        "--checkpoints-per-iteration",
        type=int,
        default=0,
        help=(
            "Save checkpoint-<step>.pt at K evenly spaced optimizer steps of every "
            "iteration (the last one at the iteration end), instead of every "
            "--eval-save-steps; 0 = use --eval-save-steps (user, 2026-11-07)."
        ),
    )
    parser.add_argument(
        "--human-split",
        choices=("inspector", "student_train_order"),
        default="inspector",
        help=(
            "inspector = HumanGT train/validation IDs stored in the Inspector .pt; "
            "student_train_order = HumanGT uses exactly the Student-Stage1 training pairs "
            "(train_order prefix), which must equal the pairs the Inspector checkpoint "
            "(MS-Swift checkpoint-<step> folder) has trained on; GemmaGT pairs outside "
            "this prefix are not used."
        ),
    )
    parser.add_argument(
        "--checkpoint-selection",
        choices=("validation", "last"),
        default="validation",
        help=(
            "validation = best HumanGT validation accuracy with the semantic guard; "
            "last = no validation: every evaluation becomes the selected checkpoint, "
            "so best.pt is the last one; Gemma losses on the GemmaGT training pairs "
            "(soft targets) are only monitored."
        ),
    )
    parser.add_argument(
        "--selection-model",
        choices=("student", "teacher"),
        default="student",
        help="Model whose HumanGT validation accuracy selects best.pt.",
    )
    parser.add_argument("--lambda-set", type=float, default=1.0)
    parser.add_argument("--set-target-contribution", type=float, default=0.10)
    parser.add_argument("--set-target-projection-rate", type=float, default=0.35)
    parser.add_argument("--set-scale-min", type=float, default=0.5)
    parser.add_argument("--set-scale-max", type=float, default=4.0)
    parser.add_argument("--set-scale-momentum", type=float, default=0.5)
    parser.add_argument("--lambda-gemma", type=float, default=1.0)
    parser.add_argument("--lambda-mask", type=float, default=1.0)
    parser.add_argument("--set-ramp-steps", type=int, default=0)
    parser.add_argument("--semantic-drift-threshold", type=float, default=0.02)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--ema-schedule",
        choices=EMA_SCHEDULES,
        default="constant",
        help=(
            "constant = --ema-decay at every step; cosine = per-step cosine increase "
            "from --ema-decay-start to --ema-decay-end over all planned optimizer steps."
        ),
    )
    parser.add_argument("--ema-decay-start", type=float, default=0.99)
    parser.add_argument("--ema-decay-end", type=float, default=0.999)
    parser.add_argument("--mc-passes", type=int, default=4)
    parser.add_argument(
        "--teacher-dropout",
        choices=("auto", "on"),
        default="auto",
        help=(
            "auto = one Teacher pass (--mc-passes 1) without dropout, MC-Dropout for J >= 2; "
            "on = LoRA and metric_heads dropout also for the single Teacher pass."
        ),
    )
    parser.add_argument(
        "--freeze-visual",
        choices=("false", "true"),
        default="false",
        help=(
            "true = only the LLM LoRA and metric_heads are trained: the ViT and aligner "
            "LoRA keep their Student-Stage1 values and stay out of the optimizer "
            "(as the Inspector, which trains without visual LoRA). false = all LoRA trained."
        ),
    )
    parser.add_argument(
        "--per-sample-log",
        choices=("false", "true"),
        default="false",
        help=(
            "true = append one JSON line per trained pair and step to --per-sample-log-path: "
            "Unlabeled (Inspector, Teacher and Student three-way probabilities, Teacher status, "
            "Inspector CE, Teacher KL) and labeled pairs (Student three-way probabilities, human "
            "label, HumanGT CE, Gemma losses). Values come from the training forward (Student "
            "before the update of that step, train mode); no extra forward. Merged labeled source only."
        ),
    )
    parser.add_argument(
        "--per-sample-log-path",
        default="",
        help="JSONL file for --per-sample-log true (run_train_loop.slurm: <output dir>/per_sample_log_<job>.jsonl).",
    )
    parser.add_argument(
        "--unlabeled-subset-file",
        default="",
        help=(
            "JSON with 'sample_ids' (e.g. selected_unlabeled_1008.json of "
            "score_teacher_inspector_agreement.py): the fixed Unlabeled subset instead of the "
            "random one; --unlabeled-subset-size N keeps its first N IDs (N <= file size)."
        ),
    )
    parser.add_argument(
        "--student-augmentation",
        choices=("none", "online", "combined", "instruction_only", "source_image_only", "extreme", "blank",
                 "single_operation"),
        default="none",
        help=(
            "online = the Student forward of every Unlabeled sample uses one augmentation "
            "drawn for (sample, round) by student_augmentation.py (masking / injection / "
            "spatial / flip, plus A<->B swap); combined = one text operation + one layout + "
            "image degradations + swap together (choose_combined_augmentation); "
            "instruction_only / source_image_only / extreme = choose_part_augmentation of "
            "student_augmentation.py on the combined scenarios file; single_operation = "
            "exactly one operation (--augmentation-operation, choose_single_augmentation) on "
            "the combined scenarios file, no swap; the Teacher "
            "and the Inspector keep the original sample. Swapped Student outputs are swapped "
            "back before every loss and log."
        ),
    )
    parser.add_argument(
        "--augmentation-operation",
        default="",
        help="The operation of --student-augmentation single_operation "
             "(student_augmentation.SINGLE_OPERATIONS), or several operations of "
             "student_augmentation.COMBINABLE_OPERATIONS joined by '+' (all applied together, "
             "choose_combined_operations).",
    )
    parser.add_argument(
        "--inspector-target",
        choices=("soft", "one_hot"),
        default="soft",
        help="Target of CE(Inspector, Student overall) in the Unlabeled term (--overall-loss "
             "three_way): soft = the Inspector three-way probabilities (entropy floor); one_hot = "
             "one-hot of the Inspector argmax label (floor 0). The Teacher KL is unchanged.",
    )
    parser.add_argument(
        "--augment-labeled",
        action="store_true",
        help="Also augment the labeled pairs (HumanGT + GemmaGT, one forward) with the same "
             "--augmentation-operation and the same per-sample draw rule as the Unlabeled samples; "
             "labels are kept. Eligibility of the labeled pairs is built in the run with the rules of "
             "build_augmentation_scenarios.py (combined mode) and saved to "
             "augmentation_labeled_scenarios.json in the run folder (2026-11-06).",
    )
    parser.add_argument(
        "--augmentation-scenarios",
        default="",
        help="augmentation_scenarios_<N>.json of build_augmentation_scenarios.py (--student-augmentation online).",
    )
    parser.add_argument(
        "--augmentation-examples",
        type=int,
        default=0,
        help=(
            "Keep the original and the augmented source image of this many Unlabeled samples "
            "(drawn at random with --seed from the subset) in <output dir>/augmentation_tmp/ "
            "with examples.jsonl (instructions, layout, degradation, swap); 0 keeps none."
        ),
    )
    parser.add_argument(
        "--handwritten-instructions",
        default="",
        help=(
            "Hand-written rewrites {instructions: {instruction: {indirect, convoluted, terse}}} "
            "(--student-augmentation combined only): one rewrite, drawn per (sample, round), "
            "replaces the information-loss text operation; the noise operation, layout, "
            "degradation and swap are unchanged."
        ),
    )
    parser.add_argument(
        "--teacher-update",
        choices=("optimizer_step", "after_round"),
        default="optimizer_step",
        help=(
            "optimizer_step = EMA Teacher update after every optimizer step (--ema-schedule); "
            "after_round = the Teacher stays fixed inside a loop and is updated once at the "
            "end of each loop with --ema-decay."
        ),
    )
    parser.add_argument("--kappa", type=float, default=0.05)
    parser.add_argument("--cache-slack", type=float, default=0.02)
    parser.add_argument("--projection-iterations", type=int, default=40)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument(
        "--eval-save-steps",
        type=int,
        default=0,
        help=(
            "Every N optimizer steps, evaluate the selection model on HumanGT and "
            "GemmaGT validation, update the best snapshot, and save "
            "checkpoint-<step>.pt (0 disables; round-end evaluation is kept)."
        ),
    )
    parser.add_argument(
        "--eval-save-total-limit",
        type=int,
        default=5,
        help="Most recent step checkpoints kept; the best step checkpoint is never removed.",
    )
    parser.add_argument("--seed", type=int, default=42)
    # None keeps the processor defaults, as in MS-Swift training without --max_pixels.
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args(argv)


def uses_metrics_weights_aggregator(args: argparse.Namespace) -> bool:
    """a_k enter all_metrics_fitted and the projection half-space; beta enters sigmoid_beta."""
    return (
        args.overall_probability == "sigmoid_beta"
        or args.overall_aggregation == "all_metrics_fitted"
        or args.unlabeled_objective == "projection"
    )


def validate_args(args: argparse.Namespace) -> None:
    if args.replay_round_selections and args.resume_checkpoint:
        raise ValueError("--replay-round-selections does not support --resume-checkpoint.")
    if args.human_split == "student_train_order" and args.checkpoint_selection != "last":
        raise ValueError("--human-split student_train_order has no HumanGT validation; use --checkpoint-selection last.")
    if args.overall_loss == "three_way" and args.unlabeled_objective != "inspector_soft_ema_consistency":
        raise ValueError(
            "--overall-loss three_way needs --unlabeled-objective inspector_soft_ema_consistency."
        )
    for name in (
        "max_rounds",
        "patience_rounds",
        "human_batch_size",
        "gemma_batch_size",
        "unlabeled_batch_size",
        "teacher_batch_size",
        "eval_batch_size",
        "gradient_accumulation_steps",
        "mc_passes",
        "projection_iterations",
        "logging_steps",
        "max_replays_per_sample",
        "screening_multiplier",
        "screening_chunk_size",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in ("learning_rate", "adam_epsilon", "max_grad_norm"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and positive.")
    if args.tie_coupling_learning_rate is not None and (
        not math.isfinite(args.tie_coupling_learning_rate)
        or args.tie_coupling_learning_rate <= 0.0
    ):
        raise ValueError("--tie-coupling-learning-rate must be finite and positive.")
    for name in ("weight_decay", "lambda_set", "lambda_consistency", "lambda_gemma", "lambda_mask", "kappa", "cache_slack", "min_validation_improvement", "semantic_drift_threshold", "set_target_contribution", "set_target_projection_rate", "set_scale_min", "set_scale_max"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative.")
    if not 0.0 <= args.warmup_ratio < 1.0:
        raise ValueError("--warmup-ratio must be in [0, 1).")
    if not 0.0 <= args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in [0, 1).")
    if args.head_learning_rate is not None and not (
        math.isfinite(args.head_learning_rate) and args.head_learning_rate > 0.0
    ):
        raise ValueError("--head-learning-rate must be finite and positive.")
    if args.diagnostic_probe_size < 0:
        raise ValueError("--diagnostic-probe-size must be >= 0 (0 = no fixed probe).")
    if args.mc_passes < 1:
        raise ValueError("--mc-passes must be >= 1 (1 = one deterministic Teacher pass).")
    if args.step_diagnostics and args.labeled_source != "merged":
        raise ValueError("--step-diagnostics requires --labeled-source merged.")
    if args.per_sample_log == "true":
        if not args.per_sample_log_path:
            raise ValueError("--per-sample-log true requires --per-sample-log-path.")
        if args.labeled_source != "merged":
            raise ValueError("--per-sample-log requires --labeled-source merged.")
        if args.overall_loss != "three_way" or args.unlabeled_objective == "projection":
            raise ValueError("--per-sample-log requires --overall-loss three_way and an Inspector objective.")
    elif args.per_sample_log_path:
        raise ValueError("--per-sample-log-path is set but --per-sample-log is false.")
    if args.unlabeled_subset_file and not args.unlabeled_subset_size:
        raise ValueError("--unlabeled-subset-file requires --unlabeled-subset-size (its size).")
    if args.student_augmentation != "none":
        if not args.augmentation_scenarios:
            raise ValueError("--student-augmentation online requires --augmentation-scenarios.")
        if not args.unlabeled_subset_file:
            raise ValueError("--student-augmentation online requires --unlabeled-subset-file.")
        if args.labeled_source != "merged" or args.unlabeled_objective == "projection":
            raise ValueError("--student-augmentation online requires the merged labeled source and an Inspector objective.")
    elif args.augmentation_scenarios:
        raise ValueError("--augmentation-scenarios is set but --student-augmentation is none.")
    if args.augmentation_examples < 0 or (args.augmentation_examples and args.student_augmentation == "none"):
        raise ValueError("--augmentation-examples must be >= 0 and needs --student-augmentation.")
    if args.handwritten_instructions and args.student_augmentation not in ("combined", "instruction_only"):
        raise ValueError("--handwritten-instructions requires --student-augmentation combined or instruction_only.")
    if args.inspector_target == "one_hot" and (
            args.unlabeled_objective != "inspector_soft_ema_consistency" or args.overall_loss != "three_way"):
        raise ValueError("--inspector-target one_hot needs inspector_soft_ema_consistency and --overall-loss three_way.")
    if (args.student_augmentation == "single_operation") != bool(args.augmentation_operation):
        raise ValueError("--augmentation-operation is set exactly with --student-augmentation single_operation.")
    if args.augment_labeled and (args.student_augmentation != "single_operation" or args.labeled_source != "merged"):
        raise ValueError("--augment-labeled needs --student-augmentation single_operation and --labeled-source merged.")
    if args.student_augmentation == "instruction_only" and not args.handwritten_instructions:
        raise ValueError("--student-augmentation instruction_only requires --handwritten-instructions.")
    if not math.isfinite(args.lambda_human) or args.lambda_human < 0.0:
        raise ValueError("--lambda-human must be finite and non-negative.")
    if args.labeled_source == "separate":
        if args.labeled_pairs_per_step or args.loss_weighting != "per_source":
            raise ValueError(
                "--labeled-pairs-per-step and --loss-weighting per_pair require --labeled-source merged."
            )
    else:
        if args.labeled_pairs_per_step < 0 or args.labeled_pairs_per_step % args.human_batch_size:
            raise ValueError("--labeled-pairs-per-step must be a non-negative multiple of --human-batch-size.")
        if args.labeled_pairs_per_step == 0 and (
            args.lambda_human or args.lambda_gemma or args.lambda_mask
        ):
            raise ValueError("--labeled-pairs-per-step 0 needs lambda_human = lambda_gemma = lambda_mask = 0.")
        unlabeled_per_step = args.unlabeled_batch_size * args.gradient_accumulation_steps
        if args.labeled_pairs_per_step and args.labeled_pairs_per_step / (
            args.labeled_pairs_per_step + unlabeled_per_step
        ) < args.min_supervised_fraction:
            raise ValueError(
                "Labeled pairs fall below --min-supervised-fraction of the pairs per optimizer step."
            )
    if args.unlabeled_per_round < 0 or args.set_ramp_steps < 0:
        raise ValueError("Round budget and ramp steps must be non-negative.")
    if args.eval_save_steps < 0 or args.eval_save_total_limit <= 0:
        raise ValueError("--eval-save-steps must be non-negative and --eval-save-total-limit positive.")
    if not 0.0 <= args.teacher_kl_relevance_weight <= 1.0:
        raise ValueError("--teacher-kl-relevance-weight must be in [0, 1].")
    if args.teacher_kl_relevance_weight > 0.0 and (
        args.unlabeled_objective != "inspector_soft_ema_consistency" or args.lambda_consistency <= 0.0
    ):
        raise ValueError(
            "--teacher-kl-relevance-weight > 0 needs --unlabeled-objective "
            "inspector_soft_ema_consistency and --lambda-consistency > 0."
        )
    if args.checkpoints_per_iteration < 0:
        raise ValueError("--checkpoints-per-iteration must be non-negative.")
    if args.checkpoints_per_iteration and args.eval_save_steps:
        raise ValueError("Use either --checkpoints-per-iteration or --eval-save-steps, not both.")
    if not 0.0 <= args.cache_slack < 1.0:
        raise ValueError("--cache-slack must be in [0, 1).")
    for name in ("hard_replay_fraction", "low_agreement_fraction", "max_box_only_fraction", "set_scale_momentum"):
        value = float(getattr(args, name))
        if not 0.0 <= value < 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1).")
    for name in ("box_only_set_weight", "min_round_fraction", "min_supervised_fraction"):
        value = float(getattr(args, name))
        if not 0.0 < value <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in (0, 1].")
    if args.hard_replay_fraction + args.low_agreement_fraction >= 1.0:
        raise ValueError("Replay and low-agreement fractions must leave room for high agreement.")
    pairs_per_microbatch = (
        args.human_batch_size + args.gemma_batch_size + args.unlabeled_batch_size
    )
    for name in ("human_batch_size", "gemma_batch_size"):
        if args.labeled_source == "merged":
            break
        if getattr(args, name) / pairs_per_microbatch < args.min_supervised_fraction:
            raise ValueError(
                f"--{name.replace('_', '-')} must contribute at least "
                f"{args.min_supervised_fraction:.1%} of pair records per optimizer step."
            )
    if args.unlabeled_sampling == "all_once" and args.max_rounds != 1:
        raise ValueError("--unlabeled-sampling all_once requires --max-rounds 1.")
    if args.unlabeled_sampling == ITERATIVE_SAMPLING:
        if args.resume_checkpoint:
            raise ValueError(f"--unlabeled-sampling {ITERATIVE_SAMPLING} cannot resume.")
        if not 0.0 <= args.loss_plateau_tolerance < 1.0:
            raise ValueError("--loss-plateau-tolerance must be in [0, 1).")
    if args.gemma_monitor_pairs < 0:
        raise ValueError("--gemma-monitor-pairs must be >= 0.")
    if args.unlabeled_subset_size < 0 or args.unlabeled_passes < 1:
        raise ValueError("--unlabeled-subset-size must be >= 0 and --unlabeled-passes >= 1.")
    if args.unlabeled_passes > 1 and args.unlabeled_sampling != "all_once":
        raise ValueError("--unlabeled-passes > 1 requires --unlabeled-sampling all_once.")
    if args.unlabeled_sampling == "inspector_teacher_rank" and args.unlabeled_per_round <= 0:
        raise ValueError("--unlabeled-sampling inspector_teacher_rank requires --unlabeled-per-round.")
    if args.ema_schedule == "cosine" and not (
        0.0 <= args.ema_decay_start <= args.ema_decay_end < 1.0
    ):
        raise ValueError("Cosine EMA requires 0 <= --ema-decay-start <= --ema-decay-end < 1.")
    if uses_metrics_weights_aggregator(args) and not args.metrics_weights_aggregator:
        raise ValueError("This configuration requires --metrics-weights-aggregator.")
    if not 0.5 < args.high_inspector_confidence <= 1.0:
        raise ValueError("--high-inspector-confidence must be in (0.5, 1].")
    if args.set_target_contribution <= 0.0 or args.set_target_projection_rate <= 0.0:
        raise ValueError("Set-loss targets must be positive.")
    if args.set_scale_min <= 0.0 or args.set_scale_max < args.set_scale_min:
        raise ValueError("Set-loss scale bounds are invalid.")


def required_file(raw_path: str, description: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    return path


def required_inspector_checkpoint_dir(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    for name in ("adapter_model.safetensors", "trainer_state.json", "args.json"):
        if not (path / name).is_file():
            raise FileNotFoundError(f"Inspector MS-Swift checkpoint is missing {name}: {path}")
    return path


def inspector_trained_pairs(checkpoint_dir: Path) -> int:
    """Pairs an MS-Swift Inspector checkpoint has trained on: step x pairs per step."""
    state = json.loads((checkpoint_dir / "trainer_state.json").read_text(encoding="utf-8"))
    run_args = json.loads((checkpoint_dir / "args.json").read_text(encoding="utf-8"))
    pairs_per_step = int(run_args["per_device_train_batch_size"]) * int(
        run_args["gradient_accumulation_steps"]
    )
    return int(state["global_step"]) * pairs_per_step


def train_order_prefix_ids(train_order: Path, count: int) -> List[str]:
    """First ``count`` distinct sample_ids of an Inspector train_order.jsonl."""
    ids: List[str] = []
    seen = set()
    with train_order.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            sample_id = json.loads(line)["sample_id"]
            if sample_id in seen:
                continue
            seen.add(sample_id)
            ids.append(sample_id)
            if len(ids) == count:
                return ids
    raise ValueError(f"{train_order} has fewer than {count} distinct pairs.")


def human_split_from_student_train_order(
    records: Sequence[HumanRecord],
    student_checkpoint: Mapping[str, Any],
    inspector_dir: Path,
) -> Tuple[List[HumanRecord], List[HumanRecord]]:
    """HumanGT = the Student-Stage1 training pairs = the Inspector pipeline pairs."""
    data = student_checkpoint.get("data", {})
    if data.get("split_method") != "train_order_prefix_v1":
        raise ValueError("--human-split student_train_order needs a train_order_prefix_v1 Student.")
    train_ids = list(data.get("train_sample_ids") or [])
    validation_ids = list(data.get("validation_sample_ids") or [])
    if not train_ids or set(train_ids) & set(validation_ids):
        raise ValueError("Student training IDs are empty or overlap validation.")
    train_order = inspector_dir.parent / "train_order.jsonl"
    inspector_pairs = inspector_trained_pairs(inspector_dir)
    if inspector_pairs != len(train_ids):
        raise ValueError(
            f"Inspector checkpoint trained on {inspector_pairs} pairs, "
            f"Student-Stage1 on {len(train_ids)}."
        )
    if set(train_order_prefix_ids(train_order, len(train_ids))) != set(train_ids):
        raise ValueError("Student-Stage1 pairs differ from the Inspector train_order prefix.")
    by_id = {record.sample_id: record for record in records}
    missing = [sample_id for sample_id in train_ids + validation_ids if sample_id not in by_id]
    if missing:
        raise ValueError(f"{len(missing)} Student pairs are missing from the Human dataset.")
    return [by_id[i] for i in train_ids], [by_id[i] for i in validation_ids]


def required_output(raw_path: str) -> Path:
    path = Path(raw_path).expanduser().resolve()
    if not path.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {path.parent}")
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite refinement checkpoint: {path}")
    return path


def _required_text(row: Mapping[str, Any], field: str, line_number: int) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Line {line_number}: missing or empty {field!r}.")
    return value.strip()


def resolve_image_path(raw_path: str, dataset_root: Path | None = None) -> str:
    value = raw_path[7:] if raw_path.startswith("file://") else raw_path
    path = Path(value).expanduser()
    if path.is_absolute():
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Image does not exist: {resolved}")
        return str(resolved)
    if dataset_root is None:
        raise ValueError(f"Relative image path has no dataset root: {raw_path}")
    root = dataset_root.expanduser().resolve()
    candidates = [root / path]
    extracted_root = root / "extracted_data"
    if extracted_root.is_dir():
        candidates.append(extracted_root / path)
    for candidate in candidates:
        resolved = candidate.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"Relative image path escapes dataset root: {raw_path}") from exc
        if resolved.is_file():
            return str(resolved)
    raise FileNotFoundError(
        f"Image does not exist below dataset root {root}: {raw_path}"
    )


def _unlabeled_row_to_record(
    row: Mapping[str, Any], line_number: int, dataset_root: Path
) -> PairRecord:
    if "sample_id" in row:
        fields = ("sample_id", "source_image", "edit_instruction", "edited_image_A", "edited_image_B")
    else:
        fields = ("unique_key", "source_img", "instruction", "left_output_image", "right_output_image")
    sample_field, source_field, instruction_field, a_field, b_field = fields
    image_a = resolve_image_path(_required_text(row, a_field, line_number), dataset_root)
    image_b = resolve_image_path(_required_text(row, b_field, line_number), dataset_root)
    if image_a == image_b:
        raise ValueError(f"Line {line_number}: candidates A and B are identical.")
    return PairRecord(
        sample_id=_required_text(row, sample_field, line_number),
        source_image=resolve_image_path(_required_text(row, source_field, line_number), dataset_root),
        edit_instruction=_required_text(row, instruction_field, line_number),
        image_a=image_a,
        image_b=image_b,
    )


def index_unlabeled_pool(path: Path, dataset_root: Path) -> UnlabeledPool:
    sample_ids: List[str] = []
    source_groups: List[str] = []
    byte_offsets: List[int] = []
    seen = set()
    with path.open("rb") as handle:
        line_number = 0
        while True:
            offset = handle.tell()
            line = handle.readline()
            if not line:
                break
            line_number += 1
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            sample_field = "sample_id" if "sample_id" in row else "unique_key"
            source_field = "source_image" if "sample_id" in row else "source_img"
            sample_id = _required_text(row, sample_field, line_number)
            if sample_id in seen:
                raise ValueError(f"Duplicate unlabeled sample_id: {sample_id}")
            seen.add(sample_id)
            sample_ids.append(sample_id)
            source_groups.append(
                resolve_image_path(
                    _required_text(row, source_field, line_number), dataset_root
                )
            )
            byte_offsets.append(offset)
    if not sample_ids:
        raise ValueError("Unlabeled dataset is empty.")
    return UnlabeledPool(path, dataset_root, sample_ids, source_groups, byte_offsets)


def materialize_unlabeled_round(
    pool: UnlabeledPool, selected_indices: Sequence[int]
) -> Dict[int, PairRecord]:
    records: Dict[int, PairRecord] = {}
    with pool.path.open("rb") as handle:
        for index in selected_indices:
            handle.seek(pool.byte_offsets[index])
            row = json.loads(handle.readline())
            record = _unlabeled_row_to_record(row, index + 1, pool.dataset_root)
            if record.sample_id != pool.sample_ids[index]:
                raise RuntimeError("Unlabeled dataset changed after its index was built.")
            records[index] = record
    return records


def load_gemma_records(path: Path) -> List[GemmaRecord]:
    # Reuse the exact three-run target definition from Student warm-start.
    from scripts.train_student_warmstart import _validate_and_aggregate_runs

    records: List[GemmaRecord] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = _required_text(row, "sample_id", line_number)
            if sample_id in seen:
                raise ValueError(f"Duplicate Gemma sample_id: {sample_id}")
            seen.add(sample_id)
            g_a, g_b, g_t, relevance = _validate_and_aggregate_runs(
                row.get("runs"), sample_id
            )
            image_a = resolve_image_path(_required_text(row, "edited_image_A", line_number))
            image_b = resolve_image_path(_required_text(row, "edited_image_B", line_number))
            if image_a == image_b:
                raise ValueError(f"Sample {sample_id}: candidates A and B are identical.")
            records.append(GemmaRecord(
                sample_id=sample_id,
                source_image=resolve_image_path(_required_text(row, "source_image", line_number)),
                edit_instruction=_required_text(row, "edit_instruction", line_number),
                image_a=image_a,
                image_b=image_b,
                g_a=tuple(g_a), g_b=tuple(g_b), g_t=tuple(g_t),
                relevance=tuple(relevance),
            ))
    if not records:
        raise ValueError("Gemma dataset is empty.")
    return records


def majority_vote_gemma_record(record: GemmaRecord) -> GemmaRecord:
    """Harden one GemmaGT record; same rule as train_student_hardlabel.py.

    Relevance is 1 when at least 2 of 3 runs mark the metric applicable. A
    relevant metric takes the strict-majority choice of its applicable runs,
    or Tie without a majority.
    """
    g_a, g_b, g_t, relevance = [], [], [], []
    for a, b, t, m in zip(record.g_a, record.g_b, record.g_t, record.relevance):
        if m < 2.0 / 3.0 - 1e-6:
            choice, relevant = (0.0, 0.0, 0.0), 0.0
        elif a > 0.5:
            choice, relevant = (1.0, 0.0, 0.0), 1.0
        elif b > 0.5:
            choice, relevant = (0.0, 1.0, 0.0), 1.0
        else:
            choice, relevant = (0.0, 0.0, 1.0), 1.0
        g_a.append(choice[0])
        g_b.append(choice[1])
        g_t.append(choice[2])
        relevance.append(relevant)
    return replace(
        record, g_a=tuple(g_a), g_b=tuple(g_b), g_t=tuple(g_t), relevance=tuple(relevance)
    )


def load_human_records(
    path: Path, dataset_root: Path, keep_ties: bool = False
) -> List[HumanRecord]:
    records: List[HumanRecord] = []
    seen = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"Line {line_number}: expected a JSON object.")
            vote = _required_text(row, "vote_type", line_number).lower()
            if vote in TIE_VOTES and not keep_ties:
                continue
            if vote not in VOTE_TO_LABEL and vote not in TIE_VOTES:
                raise ValueError(f"Line {line_number}: unsupported vote_type {vote!r}.")
            sample_id = _required_text(row, "unique_key", line_number)
            if sample_id in seen:
                raise ValueError(f"Duplicate Human sample_id: {sample_id}")
            seen.add(sample_id)
            image_a = resolve_image_path(
                _required_text(row, "left_output_image", line_number), dataset_root
            )
            image_b = resolve_image_path(
                _required_text(row, "right_output_image", line_number), dataset_root
            )
            if image_a == image_b:
                raise ValueError(f"Sample {sample_id}: candidates A and B are identical.")
            records.append(
                HumanRecord(
                    sample_id=sample_id,
                    source_image=resolve_image_path(
                        _required_text(row, "source_img", line_number), dataset_root
                    ),
                    edit_instruction=_required_text(row, "instruction", line_number),
                    image_a=image_a,
                    image_b=image_b,
                    label_a=HUMAN_TIE_LABEL if vote in TIE_VOTES else VOTE_TO_LABEL[vote],
                )
            )
    if not records:
        raise ValueError("No non-tie Human records were found.")
    return records


def human_split_from_inspector(
    records: Sequence[HumanRecord], inspector_checkpoint: Mapping[str, Any]
) -> Tuple[List[HumanRecord], List[HumanRecord]]:
    if inspector_checkpoint.get("model_role") != INSPECTOR_ROLE:
        raise ValueError("Inspector checkpoint role is incompatible.")
    data = inspector_checkpoint.get("data")
    if not isinstance(data, dict):
        raise ValueError("Inspector checkpoint is missing data metadata.")
    train_ids = data.get("train_sample_ids")
    validation_ids = data.get("validation_sample_ids")
    if not isinstance(train_ids, list) or not isinstance(validation_ids, list):
        raise ValueError("Inspector checkpoint is missing train/validation sample IDs.")
    if set(train_ids) & set(validation_ids):
        raise ValueError("Inspector train and validation IDs overlap.")
    by_id = {record.sample_id: record for record in records}
    calibration_ids = data.get("calibration_sample_ids", [])
    if not isinstance(calibration_ids, list):
        raise ValueError("Inspector calibration sample IDs are invalid.")
    if (set(train_ids) & set(calibration_ids)) or (set(validation_ids) & set(calibration_ids)):
        raise ValueError("Inspector train, validation, and calibration IDs overlap.")
    expected = set(train_ids) | set(validation_ids) | set(calibration_ids)
    if set(by_id) != expected:
        raise ValueError("Human dataset does not match the Inspector split.")
    # The Inspector uses MS-Swift's random pair split, so a source image may
    # appear in both partitions with different edits.
    train = [by_id[sample_id] for sample_id in train_ids]
    validation = [by_id[sample_id] for sample_id in validation_ids]
    return train, validation


def load_inspector_cache(path: Path, sample_ids: Sequence[str]) -> InspectorPseudoLabels:
    rows: Dict[str, Tuple[int, float, float, Tuple[float, float, float]]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            sample_id = _required_text(row, "sample_id", line_number)
            if "probability_Tie" in row:
                # Three-way Inspector: the soft label is P(A) + P(Tie)/2 =
                # (1 + signed_evidence)/2, the same quantity as the Student's
                # linear overall probability (1 + m)/2.
                three_way = [
                    float(row.get(key, float("nan")))
                    for key in ("probability_A", "probability_B", "probability_Tie")
                ]
                if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in three_way):
                    raise ValueError(f"Line {line_number}: invalid three-way Inspector probabilities.")
                if abs(sum(three_way) - 1.0) > 1e-4:
                    raise ValueError(f"Line {line_number}: three-way probabilities do not sum to 1.")
                probability_a = min(1.0, max(0.0, three_way[0] + 0.5 * three_way[2]))
                label = 1 if probability_a >= 0.5 else -1
                confidence = max(probability_a, 1.0 - probability_a)
                if sample_id in rows:
                    raise ValueError(f"Duplicate Inspector cache sample_id: {sample_id}")
                rows[sample_id] = (label, confidence, probability_a, tuple(three_way))
                continue
            accepted = row.get("is_accepted")
            label = row.get("pseudo_label")
            if accepted is False and label is None:
                label = 0
            if type(label) is not int or label not in (-1, 0, 1):
                raise ValueError(f"Line {line_number}: pseudo_label must be -1, 1, or null when abstained.")
            if (accepted is True and label == 0) or (accepted is False and label != 0):
                raise ValueError(f"Line {line_number}: Inspector acceptance/label mismatch.")
            confidence = float(row.get("confidence", 0.0))
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                raise ValueError(f"Line {line_number}: invalid Inspector confidence.")
            # probability_A is the Inspector soft label; NaN when an older cache
            # lacks it (only the soft objective requires it).
            probability_a = float(row.get("probability_A", float("nan")))
            if math.isfinite(probability_a):
                if not 0.0 <= probability_a <= 1.0:
                    raise ValueError(f"Line {line_number}: invalid Inspector probability_A.")
                if abs(max(probability_a, 1.0 - probability_a) - confidence) > 1e-4:
                    raise ValueError(f"Line {line_number}: probability_A and confidence disagree.")
                if label * (probability_a - 0.5) < 0.0:
                    raise ValueError(f"Line {line_number}: probability_A and pseudo_label disagree.")
            if sample_id in rows:
                raise ValueError(f"Duplicate Inspector cache sample_id: {sample_id}")
            rows[sample_id] = (label, confidence, probability_a, (float("nan"),) * 3)
    expected = set(sample_ids)
    if set(rows) != expected:
        raise ValueError("Inspector cache IDs do not exactly match unlabeled IDs.")
    return InspectorPseudoLabels(
        labels=torch.tensor([rows[sample_id][0] for sample_id in sample_ids], dtype=torch.int8),
        confidence=torch.tensor([rows[sample_id][1] for sample_id in sample_ids], dtype=torch.float32),
        probability_a=torch.tensor([rows[sample_id][2] for sample_id in sample_ids], dtype=torch.float32),
        three_way=torch.tensor([rows[sample_id][3] for sample_id in sample_ids], dtype=torch.float32),
    )


def select_diagnostic_probe(
    labels: InspectorPseudoLabels,
    source_groups: Sequence[str],
    size: int,
    seed: int,
) -> List[int]:
    """Select a fixed, stratified probe that is never used for optimization."""
    total = len(labels.labels)
    if len(source_groups) != total:
        raise ValueError("Unlabeled source-group index is misaligned with labels.")
    if size >= total:
        raise ValueError("Diagnostic probe must leave Unlabeled Data for training.")
    rng = random.Random(seed)
    strata: Dict[int, List[int]] = {-1: [], 0: [], 1: []}
    for index, label in enumerate(labels.labels.tolist()):
        strata[int(label)].append(index)
    for indices in strata.values():
        rng.shuffle(indices)
    exact = {label: size * len(indices) / total for label, indices in strata.items()}
    quotas = {label: min(len(strata[label]), int(value)) for label, value in exact.items()}
    remainder = size - sum(quotas.values())
    order = sorted(strata, key=lambda label: exact[label] - quotas[label], reverse=True)
    while remainder:
        progressed = False
        for label in order:
            if quotas[label] < len(strata[label]):
                quotas[label] += 1
                remainder -= 1
                progressed = True
                if remainder == 0:
                    break
        if not progressed:
            raise RuntimeError("Could not allocate the diagnostic probe.")
    selected: List[int] = []
    selected_sources = set()
    for label, indices in strata.items():
        selected_for_label = 0
        for index in indices:
            if selected_for_label == quotas[label]:
                break
            source_group = source_groups[index]
            if source_group in selected_sources:
                continue
            selected.append(index)
            selected_sources.add(source_group)
            selected_for_label += 1
    if len(selected) != size:
        raise ValueError(
            "Not enough distinct Unlabeled source groups for the diagnostic probe."
        )
    rng.shuffle(selected)
    return selected


class UnlabeledRoundSampler:
    """Commit screened curriculum samples without consuming rejected candidates."""

    def __init__(
        self,
        labels: InspectorPseudoLabels,
        excluded_indices: Sequence[int],
        seed: int,
        max_replays_per_sample: int,
    ) -> None:
        self.labels = labels
        self.excluded = set(excluded_indices)
        self.remaining = set(range(len(labels.labels))) - self.excluded
        self.rng = random.Random(seed)
        self.replay_counts: Counter[int] = Counter()
        self.max_replays_per_sample = max_replays_per_sample
        # inspector_teacher_rank: samples the Teacher opposed in round 1; they
        # stay in ``remaining`` and come first in round 2.
        self.deferred: set[int] = set()

    def draw(
        self,
        count: int,
        high_candidates: Sequence[int],
        low_candidates: Sequence[int],
        box_only_candidates: Sequence[int],
        hard_candidates: Sequence[int],
        replay_fraction: float,
        low_fraction: float,
        max_box_only_fraction: float,
        first_round: bool,
        screened: int,
        screening_status: Dict[str, int],
    ) -> RoundSelection:
        replay_available = [
            index for index in hard_candidates
            if index not in self.excluded
            and index not in self.remaining
            and self.replay_counts[index] < self.max_replays_per_sample
        ]
        high_available = [index for index in high_candidates if index in self.remaining]
        high_set = set(high_available)
        low_available = [
            index for index in low_candidates
            if index in self.remaining and index not in high_set
        ]
        low_set = set(low_available)
        box_available = [
            index for index in box_only_candidates
            if index in self.remaining and index not in high_set and index not in low_set
        ]
        if first_round:
            high = high_available[:count]
            low: List[int] = []
            box_only: List[int] = []
            replay: List[int] = []
        else:
            # Fractions are targets/caps, not minimum quotas.  A depleted
            # stratum must not shrink an otherwise useful round.
            replay_target = min(round(count * replay_fraction), len(replay_available))
            low_target = min(round(count * low_fraction), len(low_available))
            high_target = max(0, count - replay_target - low_target)
            replay = replay_available[:replay_target]
            high = high_available[:high_target]
            low = low_available[:low_target]
            remaining_slots = count - len(high) - len(low) - len(replay)
            if remaining_slots:
                extra_high = high_available[len(high):len(high) + remaining_slots]
                high += extra_high
                remaining_slots -= len(extra_high)
            if remaining_slots:
                extra_low = low_available[len(low):len(low) + remaining_slots]
                low += extra_low
                remaining_slots -= len(extra_low)
            box_limit = math.floor(count * max_box_only_fraction)
            box_only = box_available[:min(remaining_slots, box_limit)]
        selected = high + low + box_only + replay
        self.rng.shuffle(selected)
        if len(selected) != len(set(selected)):
            raise RuntimeError("Curriculum selection contains duplicate samples.")
        self.remaining.difference_update(high + low + box_only)
        for index in replay:
            self.replay_counts[index] += 1
        return RoundSelection(
            indices=selected,
            fresh_high_agreement=len(high),
            fresh_low_agreement=len(low),
            fresh_box_only=len(box_only),
            hard_replay=len(replay),
            screened=screened,
            screening_status=screening_status,
        )


def load_metrics_weights_aggregator(path: Path) -> Tuple[torch.Tensor, float, Dict[str, Any]]:
    aggregator = json.loads(path.read_text(encoding="utf-8"))
    if aggregator.get("model_role") != "preference_aggregator":
        raise ValueError("Metrics Weights Aggregator role is incompatible.")
    if aggregator.get("frozen") is not True:
        raise ValueError("Preference Aggregator parameters must be frozen.")
    if list(aggregator.get("metric_order", [])) != list(METRIC_NAMES):
        raise ValueError("Preference Aggregator metric order is incompatible.")
    if aggregator.get("feature_definition") not in {
        "signed_evidence", "relevance_weighted_signed_evidence"
    }:
        raise ValueError("Unsupported Preference Aggregator feature definition.")
    weights = torch.tensor(aggregator.get("weights"), dtype=torch.float32)
    beta = float(aggregator.get("beta"))
    if weights.shape != (NUM_METRICS,) or not torch.isfinite(weights).all():
        raise ValueError("Preference Aggregator weights must be a finite [6] vector.")
    if torch.any(weights < 0.0) or not torch.isclose(
        weights.sum(), torch.tensor(1.0), atol=1e-6
    ):
        raise ValueError("Preference Aggregator weights must be non-negative and sum to one.")
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("Preference Aggregator beta must be finite and positive.")
    return weights, beta, aggregator


def _build_trainable_peft_config(config_dict: Dict[str, Any]) -> PeftConfig:
    supported = set(inspect.signature(LoraConfig).parameters)
    filtered = {key: value for key, value in config_dict.items() if key in supported}
    filtered["inference_mode"] = False
    return PeftConfig.from_peft_type(**filtered)


def load_student_model(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    min_pixels: int | None,
    max_pixels: int | None,
    base_model_override: Path | None = None,
) -> Tuple[PeftModel, Any]:
    if checkpoint.get("model_role") != STUDENT_ROLE:
        raise ValueError("Student checkpoint role is incompatible.")
    if list(checkpoint.get("metric_order", [])) != list(METRIC_NAMES):
        raise ValueError("Student metric order is incompatible.")
    base_path = (
        str(base_model_override)
        if base_model_override is not None else checkpoint.get("base_model_path")
    )
    if not isinstance(base_path, str) or not Path(base_path).is_dir():
        raise FileNotFoundError(f"Student base model is unavailable: {base_path}")
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Student checkpoint has no adapter state.")
    pixel_kwargs = {
        key: value
        for key, value in (("min_pixels", min_pixels), ("max_pixels", max_pixels))
        if value is not None
    }
    processor = AutoProcessor.from_pretrained(
        base_path,
        trust_remote_code=True,
        padding_side="left",
        **pixel_kwargs,
    )
    base_model = Qwen3VLForConditionalGeneration.from_pretrained(
        base_path,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # Head structure follows the saved weights (independent or margin-coupled Tie Head).
    install_multi_metric_heads(
        base_model, processor.tokenizer, tie_head_mode=tie_head_mode_from_state_keys(state)
    )
    # Use the same MS-Swift tuner entry point as warm-start.  With a PEFT
    # config this returns a PeftModel while preserving MS-Swift compatibility.
    model = Swift.prepare_model(
        base_model, _build_trainable_peft_config(dict(checkpoint["peft_config"]))
    )
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected Student keys: {result.unexpected_keys[:10]}")
    model.to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if trainable == 0:
        raise RuntimeError("Student model has no trainable parameters.")
    return model, processor


def validate_teacher_checkpoint(
    checkpoint: Mapping[str, Any], student_state: Mapping[str, torch.Tensor]
) -> Mapping[str, torch.Tensor]:
    if checkpoint.get("model_role") != TEACHER_ROLE:
        raise ValueError("Teacher checkpoint role is incompatible.")
    state = checkpoint.get("adapter_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("Teacher checkpoint has no adapter state.")
    if state.keys() != student_state.keys():
        raise ValueError("Student and Teacher parameter names differ.")
    for name, student_tensor in student_state.items():
        teacher_tensor = state[name]
        if student_tensor.shape != teacher_tensor.shape or student_tensor.dtype != teacher_tensor.dtype:
            raise ValueError(f"Student/Teacher tensor metadata differs: {name}")
        if not torch.equal(student_tensor.detach().cpu(), teacher_tensor.detach().cpu()):
            raise ValueError(f"Initialized Teacher differs from warm-start Student: {name}")
    return state


def _format_candidate(
    record: PairRecord, candidate: str, prompt: Mapping[str, Any]
) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": prompt["system_message"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<Instruct>: {prompt['evaluation_instruction']}\n<Query>:",
                },
                {"type": "image", "image": f"file://{record.source_image}"},
                {
                    "type": "text",
                    "text": f" {record.edit_instruction}\n<Document>:",
                },
                {"type": "image", "image": f"file://{candidate}"},
            ],
        },
    ]


def _format_source(record: PairRecord, prompt: Mapping[str, Any]) -> List[Dict[str, Any]]:
    return [
        {"role": "system", "content": prompt["system_message"]},
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": f"<Instruct>: {prompt['evaluation_instruction']}\n<Query>:",
                },
                {"type": "image", "image": f"file://{record.source_image}"},
                {"type": "text", "text": f" {record.edit_instruction}"},
            ],
        },
    ]


def prepare_pair_inputs(
    processor: Any,
    records: Sequence[PairRecord],
    prompt: Mapping[str, Any],
    device: torch.device,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not records:
        raise ValueError("Cannot forward an empty pair batch.")
    if ACTIVE_SWIFT_TEMPLATE is not None:
        from scripts.train_student_warmstart import (
            _format_candidate as swift_format_candidate,
            _format_source as swift_format_source,
        )

        encoded = []
        for record in records:
            row = {
                "messages": swift_format_candidate(
                    record.source_image, record.edit_instruction, record.image_a
                ),
                "rejected_messages": swift_format_candidate(
                    record.source_image, record.edit_instruction, record.image_b
                ),
                "positive_messages": [swift_format_source(
                    record.source_image, record.edit_instruction
                )],
                "label": [0.0] * (4 * NUM_METRICS),
            }
            encoded.append(ACTIVE_SWIFT_TEMPLATE.encode(row, return_length=True))
        collated = ACTIVE_SWIFT_TEMPLATE.data_collator(encoded)
        starts = collated.pop("pair_indices")[:, 0].to(device)
        source_indices = collated.pop("source_indices")
        if not torch.equal(source_indices, starts.cpu() + 2):
            raise RuntimeError("MS-Swift A/B/source row order changed.")
        collated.pop("labels")
        inputs = {
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in collated.items()
        }
        return inputs, starts
    conversations: List[List[Dict[str, Any]]] = []
    for record in records:
        conversations.extend(
            [
                _format_candidate(record, record.image_a, prompt),
                _format_candidate(record, record.image_b, prompt),
                _format_source(record, prompt),
            ]
        )
    texts = processor.apply_chat_template(
        conversations, tokenize=False, add_generation_prompt=True
    )
    images, videos = process_vision_info(conversations)
    inputs = processor(
        text=texts,
        images=images,
        videos=videos,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}
    starts = torch.arange(len(records), device=device, dtype=torch.long) * 3
    return inputs, starts


def forward_prepared_pairs(
    model: PeftModel,
    inputs: Dict[str, torch.Tensor],
    starts: torch.Tensor,
) -> MultiMetricRewardOutput:
    if ACTIVE_SWIFT_TEMPLATE is not None:
        pairs = torch.stack((starts, starts + 1), dim=1)
        return model(
            **inputs,
            pair_indices=pairs,
            source_indices=starts + 2,
        )
    hidden = _pool_qwen_last_hidden(model, inputs, inputs.get("attention_mask"))
    return model.get_base_model().metric_heads(
        hidden[starts], hidden[starts + 1], hidden[starts + 2]
    )


def forward_pairs(
    model: PeftModel,
    processor: Any,
    records: Sequence[PairRecord],
    prompt: Mapping[str, Any],
    device: torch.device,
) -> MultiMetricRewardOutput:
    inputs, starts = prepare_pair_inputs(processor, records, prompt, device)
    return forward_prepared_pairs(model, inputs, starts)


def batched_indices(length: int, batch_size: int) -> Iterable[List[int]]:
    for start in range(0, length, batch_size):
        yield list(range(start, min(length, start + batch_size)))


def capture_adapter_state_cpu(model: PeftModel) -> Dict[str, torch.Tensor]:
    return {
        name: tensor.detach().cpu().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }


def capture_adapter_state_for_teacher_swap(model: PeftModel) -> Dict[str, torch.Tensor]:
    """Keep online Student snapshots on the GPU when the Swift path has room."""
    if ACTIVE_SWIFT_TEMPLATE is None:
        return capture_adapter_state_cpu(model)
    return {
        name: tensor.detach().clone()
        for name, tensor in get_peft_model_state_dict(model).items()
    }


def load_adapter_state(model: PeftModel, state: Mapping[str, torch.Tensor]) -> None:
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if result.unexpected_keys:
        raise RuntimeError(f"Unexpected adapter state keys: {result.unexpected_keys[:10]}")


def enable_selected_mc_dropout(model: nn.Module) -> Dict[str, int]:
    model.eval()
    counts = {"lora": 0, "heads": 0}
    for name, module in model.named_modules():
        if not isinstance(module, nn.Dropout):
            continue
        if "lora_dropout" in name:
            module.train()
            counts["lora"] += 1
        elif "metric_heads" in name:
            module.train()
            counts["heads"] += 1
    if counts["lora"] == 0 or counts["heads"] == 0:
        raise RuntimeError(f"Selective MC Dropout modules were not found: {counts}")
    return counts


def _max_linear_over_boxed_simplex(
    lower: torch.Tensor, upper: torch.Tensor, coefficients: torch.Tensor
) -> torch.Tensor:
    """Maximize a linear function over independent [6,3] boxed simplexes."""
    point = lower.clone()
    residual = 1.0 - point.sum(dim=-1)
    capacity = upper - lower
    order = torch.argsort(coefficients, dim=-1, descending=True)
    for position in range(3):
        index = order[..., position : position + 1]
        available = torch.gather(capacity, -1, index).squeeze(-1)
        addition = torch.minimum(residual, available).clamp_min(0.0)
        point.scatter_add_(-1, index, addition.unsqueeze(-1))
        residual = residual - addition
    if torch.max(torch.abs(point.sum(dim=-1) - 1.0)).item() > 1e-5:
        raise RuntimeError("Failed to construct a point on a boxed simplex.")
    return (point * coefficients).sum(dim=(-1, -2))


def diagnostic_pair_weights(
    aggregator_weights: torch.Tensor | None, relevance: torch.Tensor
) -> torch.Tensor:
    """Metric weights [N, 6] of the feasible-set half-space."""
    if aggregator_weights is None:
        return selected_metric_weights(relevance)
    return aggregator_weights.to(device=relevance.device, dtype=torch.float32)[None, :].expand(
        relevance.size(0), -1
    )


def build_feasible_cache(
    sample_ids: List[str],
    probability_samples: torch.Tensor,
    winner_samples: torch.Tensor,
    tie_samples: torch.Tensor,
    relevance_samples: torch.Tensor,
    inspector_labels: torch.Tensor,
    aggregator_weights: torch.Tensor | None,
    kappa: float,
    slack: float,
) -> TeacherCache:
    # All tensors are CPU float32 with leading MC dimension J.
    if probability_samples.shape[1:] != (len(sample_ids), NUM_METRICS, 3):
        raise ValueError("MC probability samples have an invalid shape.")
    mean_probs = probability_samples.mean(dim=0)
    lower = (probability_samples.amin(dim=0) - slack).clamp(0.0, 1.0)
    upper = (probability_samples.amax(dim=0) + slack).clamp(0.0, 1.0)
    lower_sum = lower.sum(dim=-1)
    upper_sum = upper.sum(dim=-1)
    if torch.any(lower_sum > 1.0 + 1e-6) or torch.any(upper_sum < 1.0 - 1e-6):
        raise RuntimeError("Teacher bounds do not intersect the probability simplex.")

    labels = inspector_labels.to(dtype=torch.float32)
    # Without the Metrics Weights Aggregator the half-space uses the equal
    # weights over the metrics selected by the mean Teacher relevance.
    weights = diagnostic_pair_weights(
        aggregator_weights, relevance_samples.mean(dim=0)
    )
    direction = torch.tensor([1.0, -1.0, 0.0], dtype=torch.float32)
    coefficients = (
        labels[:, None, None] * weights[:, :, None] * direction[None, None, :]
    )
    maximum_margin = _max_linear_over_boxed_simplex(lower, upper, coefficients)
    inspector_valid = labels != 0.0
    constrained_valid = inspector_valid & (maximum_margin >= kappa - 1e-7)
    box_only_valid = ~inspector_valid
    valid = constrained_valid | box_only_valid
    effective_kappa = torch.zeros(len(sample_ids), dtype=torch.float32)
    effective_kappa[constrained_valid] = float(kappa)
    status = []
    for index in range(len(sample_ids)):
        if box_only_valid[index]:
            status.append("valid_box_only")
        elif constrained_valid[index]:
            status.append("valid_inspector_constraint")
        else:
            status.append("constraint_conflict")
    return TeacherCache(
        sample_ids=sample_ids,
        mean_probs=mean_probs,
        mean_relevance=relevance_samples.mean(dim=0),
        var_winner=winner_samples.var(dim=0, unbiased=False),
        var_tie=tie_samples.var(dim=0, unbiased=False),
        var_relevance=relevance_samples.var(dim=0, unbiased=False),
        lower=lower,
        upper=upper,
        inspector_labels=inspector_labels.clone(),
        effective_kappa=effective_kappa,
        valid=valid,
        status=status,
    )


@torch.no_grad()
def teacher_cache_for_batch(
    model: PeftModel,
    processor: Any,
    records: Sequence[PairRecord],
    prompt: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor],
    inspector_labels: torch.Tensor,
    aggregator_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    mc_seed: int | None = None,
    progress_every: int = 0,
) -> TeacherCache:
    student_state = capture_adapter_state_for_teacher_swap(model)
    try:
        load_adapter_state(model, teacher_state)
        if args.mc_passes == 1 and args.teacher_dropout == "auto":
            # One deterministic Teacher pass: dropout stays off.
            model.eval()
        else:
            dropout_counts = enable_selected_mc_dropout(model)
            if not all(dropout_counts.values()):
                raise RuntimeError("Teacher MC Dropout was not enabled.")
        fork_devices = [torch.cuda.current_device()] if device.type == "cuda" else []
        with torch.random.fork_rng(devices=fork_devices, enabled=mc_seed is not None):
            if mc_seed is not None:
                torch.manual_seed(mc_seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(mc_seed)
            all_probs: List[List[torch.Tensor]] = [[] for _ in range(args.mc_passes)]
            all_winners: List[List[torch.Tensor]] = [[] for _ in range(args.mc_passes)]
            all_ties: List[List[torch.Tensor]] = [[] for _ in range(args.mc_passes)]
            all_relevance: List[List[torch.Tensor]] = [[] for _ in range(args.mc_passes)]
            for batch_number, indices in enumerate(
                batched_indices(len(records), args.teacher_batch_size), start=1
            ):
                # Image decoding and tokenization are identical across MC
                # passes; keep this chunk's tensors for all four forwards.
                inputs, starts = prepare_pair_inputs(
                    processor, [records[index] for index in indices], prompt, device
                )
                for pass_index in range(args.mc_passes):
                    output = forward_prepared_pairs(model, inputs, starts)
                    all_probs[pass_index].append(output.three_way_probs.float().cpu())
                    all_winners[pass_index].append(output.cond_win_prob_A.float().cpu())
                    all_ties[pass_index].append(output.tie_probs.float().cpu())
                    all_relevance[pass_index].append(output.relevance_probs.float().cpu())
                    del output
                del inputs, starts
                if progress_every and batch_number % progress_every == 0:
                    print(
                        f"Teacher MC: {args.mc_passes} passes, processed "
                        f"{min(batch_number * args.teacher_batch_size, len(records))}/{len(records)}",
                        flush=True,
                    )
    finally:
        load_adapter_state(model, student_state)
        model.eval()
    cache = build_feasible_cache(
        [record.sample_id for record in records],
        torch.stack([torch.cat(items) for items in all_probs]),
        torch.stack([torch.cat(items) for items in all_winners]),
        torch.stack([torch.cat(items) for items in all_ties]),
        torch.stack([torch.cat(items) for items in all_relevance]),
        inspector_labels,
        aggregator_weights,
        args.kappa,
        args.cache_slack,
    )
    return cache


@contextmanager
def teacher_screen_session(model: PeftModel, teacher_state: Mapping[str, torch.Tensor]):
    """Swap adapters once for all incremental screening chunks in a round."""
    student_state = capture_adapter_state_for_teacher_swap(model)
    try:
        load_adapter_state(model, teacher_state)
        yield
    finally:
        load_adapter_state(model, student_state)
        model.eval()


@torch.no_grad()
def teacher_screen_batch(
    model: PeftModel,
    processor: Any,
    records: Sequence[PairRecord],
    prompt: Mapping[str, Any],
    batch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One deterministic Teacher pass; MC validation is deferred to training.

    Returns three-state probabilities [N, 6, 3] and relevance [N, 6].
    """
    model.eval()
    probabilities: List[torch.Tensor] = []
    relevance: List[torch.Tensor] = []
    for indices in batched_indices(len(records), batch_size):
        output = forward_pairs(
            model, processor, [records[index] for index in indices], prompt, device
        )
        probabilities.append(output.three_way_probs.detach().float().cpu())
        relevance.append(output.relevance_probs.detach().float().cpu())
    return torch.cat(probabilities, dim=0), torch.cat(relevance, dim=0)


def screening_signed_margin(
    probabilities: torch.Tensor,
    relevance: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    aggregation: str,
) -> torch.Tensor:
    """Teacher overall evidence along the Inspector label (positive = agrees)."""
    evidence = probabilities[:, :, 0] - probabilities[:, :, 1]
    metric_weights = aggregation_weights_from_relevance(relevance, weights, aggregation)
    return labels.float() * (evidence * metric_weights).sum(dim=-1)


def screen_curriculum_candidates(
    sampler: UnlabeledRoundSampler,
    pool: UnlabeledPool,
    hard_candidates: Sequence[int],
    round_budget: int,
    first_round: bool,
    model: PeftModel,
    processor: Any,
    prompt: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor],
    inspector_labels: InspectorPseudoLabels,
    calibrated_acceptance_threshold: float,
    aggregator_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    teacher_optimizer_step: int,
) -> Tuple[List[int], List[int], List[int], List[int], Dict[str, int], Dict[str, Dict[str, Any]]]:
    """Screen incrementally with one deterministic Teacher pass per candidate.

    Inspector decisions are frozen; every visited sample is re-evaluated by
    the current EMA Teacher. J=4 MC bounds are computed only immediately
    before an optimizer step on selected records. Screening scores are never
    reused as training targets after an EMA update.
    """
    if args.unlabeled_sampling in ("all_once", ITERATIVE_SAMPLING):
        # Single pass: every remaining sample, no Teacher Model screening (the iterative
        # sampling uses this for iteration 1 only; later iterations do not call it).
        everything = sorted(sampler.remaining)
        sampler.rng.shuffle(everything)
        cache = {pool.sample_ids[index]: {"classification": "all_once"} for index in everything}
        return everything, [], [], [], {"screened": 0, "all_once": len(everything)}, cache
    scan_limit = round_budget * args.screening_multiplier
    replay_budget = 0 if first_round else round(round_budget * args.hard_replay_fraction)
    low_budget = 0 if first_round else round_budget
    box_budget = 0 if first_round else math.floor(round_budget * args.max_box_only_fraction)
    high_budget = (
        round_budget if first_round else
        max(0, round_budget - replay_budget - round(round_budget * args.low_agreement_fraction))
    )
    confidence_order: List[int] = []
    if args.unlabeled_sampling == "confidence_order":
        # Fresh samples come from one stratum ordered by Inspector confidence;
        # the Teacher only excludes accepted samples it opposes.
        low_budget = box_budget = 0
        high_budget = round_budget - replay_budget
        confidence_order = sorted(
            sampler.remaining,
            key=lambda i: (-float(inspector_labels.confidence[i]), i),
        )
    accepted = sorted(
        (i for i in sampler.remaining if inspector_labels.labels[i].item() != 0),
        key=lambda i: (-float(inspector_labels.confidence[i]), i),
    )
    high_cutoff = args.high_inspector_confidence
    if not first_round and len(accepted) >= high_budget > 0:
        # Relax only as far as the Inspector's HumanGT-calibrated acceptance
        # threshold when the original high-confidence stratum is depleted.
        high_cutoff = max(
            calibrated_acceptance_threshold,
            min(high_cutoff, float(inspector_labels.confidence[accepted[high_budget - 1]])),
        )
    high_order = [
        i for i in accepted
        if float(inspector_labels.confidence[i]) >= high_cutoff
    ]
    abstained = [
        i for i in sampler.remaining if inspector_labels.labels[i].item() == 0
    ]
    sampler.rng.shuffle(abstained)
    # Only the most difficult replay samples are inspected. Reserve a small
    # replacement pool for contradictions, rather than screening all 1500.
    replay_order = [
        i for i in hard_candidates
        if i not in sampler.excluded
        and i not in sampler.remaining
        and sampler.replay_counts[i] < sampler.max_replays_per_sample
    ][:min(scan_limit, max(replay_budget * 2, args.screening_chunk_size))]
    low_order = list(reversed(accepted))
    high, low, box_only, hard = [], [], [], []
    screened: set[int] = set()
    sample_cache: Dict[str, Dict[str, Any]] = {}
    counts: Counter[str] = Counter()

    def inspect(order: Sequence[int], target_group: str, target_count: int) -> None:
        cursor = 0
        while cursor < len(order) and len(screened) < scan_limit:
            if len(high) + len(low) + len(box_only) + len(hard) >= round_budget:
                return
            group_size = len({"high": high, "fresh": high, "low": low, "box": box_only, "hard": hard}[target_group])
            if group_size >= target_count:
                return
            batch_capacity = min(
                args.screening_chunk_size,
                target_count - group_size,
                scan_limit - len(screened),
            )
            batch_indices: List[int] = []
            while cursor < len(order) and len(batch_indices) < batch_capacity:
                index = order[cursor]
                cursor += 1
                if index not in screened:
                    screened.add(index)
                    batch_indices.append(index)
            if not batch_indices:
                continue
            records_by_index = materialize_unlabeled_round(pool, batch_indices)
            records = [records_by_index[i] for i in batch_indices]
            probabilities, relevance = teacher_screen_batch(
                model, processor, records, prompt, args.teacher_batch_size, device
            )
            labels = inspector_labels.labels[batch_indices]
            signed_margin = screening_signed_margin(
                probabilities, relevance, labels, aggregator_weights,
                args.overall_aggregation,
            )
            entropy = -(
                probabilities.clamp_min(PROBABILITY_EPS)
                * probabilities.clamp_min(PROBABILITY_EPS).log()
            ).sum(dim=-1).mean(dim=-1)
            for position, index in enumerate(batch_indices):
                label = int(labels[position])
                margin = float(signed_margin[position])
                confidence = float(inspector_labels.confidence[index])
                classification = "excluded"
                if label != 0 and margin < 0.0:
                    classification = "teacher_opposes_inspector"
                    counts[classification] += 1
                elif target_group == "fresh":
                    high.append(index)
                    classification = "confidence_order"
                elif target_group == "hard" and (
                    label != 0 or args.replay_strategy == "random_previous_round"
                ):
                    hard.append(index)
                    classification = (
                        "hard_replay" if args.replay_strategy == "hard"
                        else "previous_round_replay"
                    )
                elif (
                    label != 0
                    and confidence >= high_cutoff
                    and margin >= args.kappa
                ):
                    high.append(index)
                    classification = "high_agreement"
                elif not first_round and label != 0:
                    low.append(index)
                    classification = "low_agreement"
                elif not first_round and label == 0:
                    box_only.append(index)
                    classification = "box_only_consistency"
                sample_cache[pool.sample_ids[index]] = {
                    "index": index,
                    "teacher_optimizer_step": teacher_optimizer_step,
                    "inspector_label": label,
                    "inspector_confidence": confidence,
                    "mean_signed_margin": margin,
                    "predictive_entropy": float(entropy[position]),
                    "feasibility": "pending_mc_validation",
                    "classification": classification,
                }
            print(
                f"Teacher deterministic screening: screened={len(screened)} "
                f"high={len(high)} low={len(low)} box={len(box_only)} hard={len(hard)} "
                f"targets=({high_budget},{low_budget},{box_budget},{replay_budget})",
                flush=True,
            )

    with teacher_screen_session(model, teacher_state):
        if replay_budget:
            inspect(replay_order, "hard", replay_budget)
        if args.unlabeled_sampling == "confidence_order":
            inspect(confidence_order, "fresh", high_budget)
        else:
            inspect(high_order, "high", high_budget)
        if low_budget:
            inspect(low_order, "low", low_budget)
        if box_budget:
            inspect(abstained, "box", box_budget)
    high.sort(
        key=lambda i: (
            -float(inspector_labels.confidence[i]),
            -float(sample_cache[pool.sample_ids[i]]["mean_signed_margin"]),
            float(sample_cache[pool.sample_ids[i]]["predictive_entropy"]),
        )
    )
    low.sort(
        key=lambda i: (
            int(inspector_labels.labels[i].item() == 0),
            abs(float(sample_cache[pool.sample_ids[i]]["mean_signed_margin"])),
            float(sample_cache[pool.sample_ids[i]]["predictive_entropy"]),
        )
    )
    box_only.sort(key=lambda i: float(sample_cache[pool.sample_ids[i]]["predictive_entropy"]))
    counts.update({
        "screened": len(screened),
        "high": len(high),
        "low": len(low),
        "box": len(box_only),
        "hard": len(hard),
        "high_cutoff_milli": round(high_cutoff * 1000),
    })
    return high, low, box_only, hard, dict(counts), sample_cache


def inspector_teacher_rank_round_sizes(
    trainable_unlabeled: int, args: argparse.Namespace
) -> List[int]:
    """Planned Unlabeled occurrences of each inspector_teacher_rank round."""
    sizes: List[int] = []
    remaining = trainable_unlabeled
    previous = 0
    for round_index in range(args.max_rounds):
        last = round_index == args.max_rounds - 1
        fresh = remaining if last else min(args.unlabeled_per_round, remaining)
        replay = 0 if round_index == 0 else min(
            round(args.unlabeled_per_round * args.hard_replay_fraction), previous
        )
        remaining -= fresh
        previous = fresh + replay
        sizes.append(fresh + replay)
    return sizes


def select_inspector_teacher_rank_round(
    sampler: UnlabeledRoundSampler,
    pool: UnlabeledPool,
    previous_round_indices: Sequence[int],
    round_index: int,
    model: PeftModel,
    processor: Any,
    prompt: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor],
    inspector_labels: InspectorPseudoLabels,
    aggregator_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    teacher_optimizer_step: int,
) -> Tuple[RoundSelection, Dict[str, Dict[str, Any]]]:
    """Fresh samples by Inspector confidence; replay by Teacher margin.

    Round 1 keeps only samples whose deterministic Teacher margin agrees with
    the Inspector label and defers the others to round 2. Later rounds take
    deferred samples first, then the next samples by Inspector confidence,
    without Teacher filtering; the last round takes every remaining sample.
    From round 2, replay takes the previous-round samples with the smallest
    current Teacher margin along the Inspector label.
    """
    first_round = round_index == 0
    last_round = round_index == args.max_rounds - 1
    fresh_budget = (
        len(sampler.remaining) if last_round
        else min(args.unlabeled_per_round, len(sampler.remaining))
    )
    replay_budget = 0 if first_round else round(
        args.unlabeled_per_round * args.hard_replay_fraction
    )

    def by_confidence(indices: Iterable[int]) -> List[int]:
        return sorted(indices, key=lambda i: (-float(inspector_labels.confidence[i]), i))

    def teacher_margins(indices: Sequence[int]) -> List[float]:
        margins: List[float] = []
        for start in range(0, len(indices), args.screening_chunk_size):
            chunk = list(indices[start : start + args.screening_chunk_size])
            records_by_index = materialize_unlabeled_round(pool, chunk)
            probabilities, relevance = teacher_screen_batch(
                model, processor, [records_by_index[i] for i in chunk], prompt,
                args.teacher_batch_size, device,
            )
            margins.extend(screening_signed_margin(
                probabilities, relevance, inspector_labels.labels[chunk],
                aggregator_weights, args.overall_aggregation,
            ).tolist())
        return margins

    sample_cache: Dict[str, Dict[str, Any]] = {}

    def record(index: int, classification: str, margin: float | None) -> None:
        sample_cache[pool.sample_ids[index]] = {
            "index": index,
            "teacher_optimizer_step": teacher_optimizer_step,
            "inspector_label": int(inspector_labels.labels[index]),
            "inspector_confidence": float(inspector_labels.confidence[index]),
            "teacher_signed_margin": margin,
            "classification": classification,
        }

    fresh: List[int] = []
    deferred_taken: List[int] = []
    replay: List[int] = []
    screened = 0
    with teacher_screen_session(model, teacher_state):
        if first_round:
            order = by_confidence(sampler.remaining)
            cursor = 0
            while cursor < len(order) and len(fresh) < fresh_budget:
                chunk = order[cursor : cursor + args.screening_chunk_size]
                cursor += len(chunk)
                screened += len(chunk)
                for index, margin in zip(chunk, teacher_margins(chunk)):
                    if len(fresh) >= fresh_budget:
                        break
                    if margin > 0.0:
                        fresh.append(index)
                        record(index, "teacher_agrees_with_inspector", margin)
                    else:
                        sampler.deferred.add(index)
                        record(index, "deferred_teacher_opposes_inspector", margin)
                print(
                    f"Teacher deterministic screening: screened={screened} "
                    f"fresh={len(fresh)}/{fresh_budget} deferred={len(sampler.deferred)}",
                    flush=True,
                )
        else:
            deferred = by_confidence(sampler.deferred & sampler.remaining)
            deferred_taken = deferred[:fresh_budget]
            deferred_set = set(deferred_taken)
            rest = by_confidence(sampler.remaining - deferred_set)
            fresh = deferred_taken + rest[: fresh_budget - len(deferred_taken)]
            for index in fresh:
                record(
                    index,
                    "deferred_teacher_opposes_inspector" if index in deferred_set
                    else "inspector_confidence_rank",
                    None,
                )
            candidates = [
                index for index in previous_round_indices
                if index not in sampler.excluded
                and index not in sampler.remaining
                and sampler.replay_counts[index] < sampler.max_replays_per_sample
            ]
            if replay_budget and candidates:
                margins = teacher_margins(candidates)
                screened += len(candidates)
                ranked = sorted(
                    zip(candidates, margins), key=lambda item: (item[1], item[0])
                )[:replay_budget]
                replay = [index for index, _ in ranked]
                for index, margin in ranked:
                    record(index, "hard_replay", margin)
                print(
                    f"Teacher replay ranking: candidates={len(candidates)} "
                    f"replay={len(replay)} largest_selected_margin="
                    f"{ranked[-1][1] if ranked else float('nan'):.6f}",
                    flush=True,
                )
    selected = fresh + replay
    sampler.rng.shuffle(selected)
    if len(selected) != len(set(selected)):
        raise RuntimeError("Curriculum selection contains duplicate samples.")
    sampler.remaining.difference_update(fresh)
    sampler.deferred.difference_update(fresh)
    for index in replay:
        sampler.replay_counts[index] += 1
    status = {
        "screened": screened,
        "fresh": len(fresh),
        "deferred_taken": len(deferred_taken),
        "deferred_pending": len(sampler.deferred),
        "hard_replay": len(replay),
    }
    return RoundSelection(
        indices=selected,
        fresh_high_agreement=len(fresh) - len(deferred_taken),
        fresh_low_agreement=len(deferred_taken),
        fresh_box_only=0,
        hard_replay=len(replay),
        screened=screened,
        screening_status=status,
    ), sample_cache


def _boxed_simplex_kl_projection(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    tilt: torch.Tensor,
    iterations: int,
) -> torch.Tensor:
    """KL projection with box and simplex constraints for every metric."""
    logits = torch.log(q.clamp_min(PROBABILITY_EPS)) + tilt
    low = logits.amin(dim=-1) - 50.0
    high = logits.amax(dim=-1) + 50.0
    for _ in range(iterations):
        midpoint = 0.5 * (low + high)
        candidate = torch.exp(
            (logits - midpoint.unsqueeze(-1)).clamp(-80.0, 80.0)
        ).clamp(min=lower, max=upper)
        too_large = candidate.sum(dim=-1) > 1.0
        low = torch.where(too_large, midpoint, low)
        high = torch.where(too_large, high, midpoint)
    normalizer = 0.5 * (low + high)
    projected = torch.exp(
        (logits - normalizer.unsqueeze(-1)).clamp(-80.0, 80.0)
    ).clamp(min=lower, max=upper)
    if torch.max(torch.abs(projected.sum(dim=-1) - 1.0)).item() > 2e-4:
        raise RuntimeError("Boxed-simplex KL projection failed to normalize.")
    return projected


@torch.no_grad()
def project_to_feasible_set(
    q: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    labels: torch.Tensor,
    kappa: torch.Tensor,
    valid: torch.Tensor,
    weights: torch.Tensor,
    iterations: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if q.ndim != 3 or q.shape[1:] != (NUM_METRICS, 3):
        raise ValueError("Student probabilities must have shape [B,6,3].")
    labels_f = labels.to(dtype=q.dtype)
    weights_f = weights.to(device=q.device, dtype=q.dtype)
    # Shared [6] Metrics Weights Aggregator weights or per-pair [B, 6] weights.
    weights_f = weights_f[None, :].expand(q.size(0), -1) if weights_f.ndim == 1 else weights_f
    direction = torch.tensor([1.0, -1.0, 0.0], device=q.device, dtype=q.dtype)

    def point_for_eta(eta: torch.Tensor) -> torch.Tensor:
        tilt = (
            eta[:, None, None]
            * labels_f[:, None, None]
            * weights_f[:, :, None]
            * direction[None, None, :]
        )
        return _boxed_simplex_kl_projection(q, lower, upper, tilt, iterations)

    q_margin = labels_f * (
        weights_f * (q[:, :, 0] - q[:, :, 1])
    ).sum(dim=-1)
    inside_box = ((q >= lower - 1e-6) & (q <= upper + 1e-6)).all(dim=(-1, -2))
    inside = valid & inside_box & (q_margin >= kappa - 1e-6)

    zero_eta = torch.zeros(q.size(0), device=q.device, dtype=q.dtype)
    base = point_for_eta(zero_eta)
    base_margin = labels_f * (
        weights_f * (base[:, :, 0] - base[:, :, 1])
    ).sum(dim=-1)
    needs_halfspace = valid & (base_margin < kappa - 1e-6)

    low_eta = torch.zeros_like(zero_eta)
    high_eta = torch.ones_like(zero_eta)
    for _ in range(30):
        high_point = point_for_eta(high_eta)
        high_margin = labels_f * (
            weights_f
            * (high_point[:, :, 0] - high_point[:, :, 1])
        ).sum(dim=-1)
        unresolved = needs_halfspace & (high_margin < kappa - 1e-6)
        if not torch.any(unresolved):
            break
        high_eta = torch.where(unresolved, high_eta * 2.0, high_eta)
    else:
        raise RuntimeError("Could not bracket the feasible-set KL projection.")

    for _ in range(iterations):
        midpoint = 0.5 * (low_eta + high_eta)
        midpoint_point = point_for_eta(midpoint)
        midpoint_margin = labels_f * (
            weights_f
            * (midpoint_point[:, :, 0] - midpoint_point[:, :, 1])
        ).sum(dim=-1)
        below = needs_halfspace & (midpoint_margin < kappa)
        low_eta = torch.where(below, midpoint, low_eta)
        high_eta = torch.where(needs_halfspace & ~below, midpoint, high_eta)

    constrained = point_for_eta(high_eta)
    target = torch.where(needs_halfspace[:, None, None], constrained, base)
    target = torch.where(inside[:, None, None], q, target)
    target = torch.where(valid[:, None, None], target, q)
    return target, inside


def feasible_set_loss(
    student_probs: torch.Tensor,
    cache: TeacherCache,
    indices: Sequence[int],
    aggregator_weights: torch.Tensor,
    projection_iterations: int,
    box_only_weight: float,
    device: torch.device,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    index = torch.tensor(indices, dtype=torch.long)
    lower = cache.lower[index].to(device)
    upper = cache.upper[index].to(device)
    labels = cache.inspector_labels[index].to(device)
    kappa = cache.effective_kappa[index].to(device)
    valid = cache.valid[index].to(device)
    if not torch.any(valid):
        return student_probs.sum() * 0.0, {
            "valid": 0,
            "inside": 0,
            "projected": 0,
            "valid_mask": valid.tolist(),
            "inside_mask": [False] * len(indices),
            "projection_kl": [0.0] * len(indices),
        }
    with torch.no_grad():
        target, inside = project_to_feasible_set(
            student_probs.detach().float(),
            lower.float(),
            upper.float(),
            labels,
            kappa.float(),
            valid,
            aggregator_weights,
            projection_iterations,
        )
    log_target = torch.log(target.clamp_min(PROBABILITY_EPS))
    log_student = torch.log(student_probs.float().clamp_min(PROBABILITY_EPS))
    per_sample = (target * (log_target - log_student)).sum(dim=(-1, -2))
    per_sample = torch.where(inside, torch.zeros_like(per_sample), per_sample)
    per_sample = per_sample / NUM_METRICS
    # Inspector abstentions have only self-consistency targets; they must not
    # carry the same influence as externally constrained accepted labels.
    sample_weights = torch.where(
        labels == 0,
        torch.full_like(per_sample, box_only_weight),
        torch.ones_like(per_sample),
    )
    loss = (per_sample[valid] * sample_weights[valid]).sum() / valid.sum()
    return loss, {
        "valid": int(valid.sum().item()),
        "inside": int((valid & inside).sum().item()),
        "projected": int((valid & ~inside).sum().item()),
        "valid_mask": valid.cpu().tolist(),
        "inside_mask": inside.cpu().tolist(),
        "projection_kl": per_sample.detach().cpu().tolist(),
    }


def inspector_supervision_loss(
    student_output: MultiMetricRewardOutput,
    cache: TeacherCache,
    indices: Sequence[int],
    aggregator_weights: torch.Tensor | None,
    aggregator_beta: float | None,
    consistency_weight: float,
    device: torch.device,
    soft_targets: Optional[torch.Tensor] = None,
    aggregation: str = "all_metrics_fitted",
    probability: str = "sigmoid_beta",
    consistency_metrics: str = "all",
    three_way_targets: Optional[torch.Tensor] = None,
    relevance_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """A1/A2 Unlabeled Data term without the Metrics and Overall Projection.

    ``relevance_weight`` w > 0 (2026-11-07): the Teacher consistency becomes
    (1 - w) * KL_pref + w * KL_rel (see teacher_relevance_kl); w = 0 keeps KL_pref only.

    Accepted Overall Pseudo-label Cache labels supervise the Metrics Weights
    Aggregator logit directly. With ``soft_targets`` (Inspector probability_A)
    every sample is supervised, abstained ones included. With
    ``consistency_weight > 0`` every sample also matches the J=4 Teacher Model
    mean three-state probabilities.
    """
    index = torch.tensor(indices, dtype=torch.long)
    labels = cache.inspector_labels[index].to(device)
    accepted = labels != 0
    zero = student_output.three_way_probs.sum() * 0.0
    per_sample = torch.zeros(len(indices), device=device, dtype=torch.float32)
    loss = zero
    # Per-sample sums for logging: Inspector cross-entropy, entropy of the Inspector
    # target (cross-entropy - entropy = KL(Inspector || Student)) and Teacher KL.
    inspector_cross_entropy_sum = 0.0
    inspector_target_entropy_sum = 0.0
    teacher_consistency_sum = 0.0
    teacher_preference_kl_sum = 0.0
    teacher_relevance_kl_sum = 0.0
    if three_way_targets is not None:
        # Inspector (P(A), P(B), P(Tie)) against the Student overall three-way
        # distribution; tie_probs are not detached.
        if three_way_targets.shape != (len(indices), 3) or not torch.all(
            torch.isfinite(three_way_targets)
        ):
            raise ValueError("Inspector three-way targets must be finite with shape [N, 3].")
        cross_entropy = three_way_cross_entropy(
            overall_three_way_probs(
                student_output,
                None if aggregator_weights is None else aggregator_weights.to(device),
                aggregation,
            ),
            three_way_targets.to(device=device, dtype=torch.float32),
        )
        per_sample = per_sample + cross_entropy
        loss = loss + cross_entropy.mean()
        accepted = torch.ones_like(accepted)
        target_probabilities = three_way_targets.to(device=device, dtype=torch.float32)
        inspector_cross_entropy_sum = float(cross_entropy.detach().sum())
        inspector_target_entropy_sum = float(
            -(target_probabilities * torch.log(target_probabilities.clamp_min(PROBABILITY_EPS)))
            .sum(dim=-1).sum()
        )
    elif soft_targets is not None:
        if soft_targets.shape != (len(indices),) or not torch.all(torch.isfinite(soft_targets)):
            raise ValueError("Inspector soft targets must be finite with shape [N].")
        logits = human_preference_aggregation_logits(
            student_output,
            None if aggregator_weights is None else aggregator_weights.to(device),
            aggregator_beta, aggregation, probability,
        )
        bce = F.binary_cross_entropy_with_logits(
            logits, soft_targets.to(device=device, dtype=logits.dtype), reduction="none"
        )
        per_sample = per_sample + bce
        loss = loss + bce.mean()
        accepted = torch.ones_like(accepted)
    elif torch.any(accepted):
        logits = human_preference_aggregation_logits(
            student_output,
            None if aggregator_weights is None else aggregator_weights.to(device),
            aggregator_beta, aggregation, probability,
        )
        targets = (labels.float() + 1.0) / 2.0
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        bce = torch.where(accepted, bce, torch.zeros_like(bce))
        per_sample = per_sample + bce
        loss = loss + bce.sum() / accepted.sum()
    used = accepted.clone()
    if consistency_weight > 0.0:
        # Each term keeps its own denominator so A2 differs from A1 only by
        # the added Teacher Model consistency.
        teacher_probs = cache.mean_probs[index].to(device).float()
        log_teacher = torch.log(teacher_probs.clamp_min(PROBABILITY_EPS))
        log_student = torch.log(
            student_output.three_way_probs.float().clamp_min(PROBABILITY_EPS)
        )
        metric_kl = (teacher_probs * (log_teacher - log_student)).sum(dim=-1)
        if consistency_metrics == "selected":
            # Same Student selection as the overall terms; mean over S per pair.
            consistency = (
                metric_kl * selected_metric_weights(student_output.relevance_probs)
            ).sum(dim=-1)
        elif consistency_metrics == "all":
            consistency = metric_kl.sum(dim=-1) / NUM_METRICS
        else:
            raise ValueError(f"Unsupported consistency metrics: {consistency_metrics}")
        teacher_preference_kl_sum = float(consistency.detach().sum())
        if relevance_weight > 0.0:
            relevance_kl = teacher_relevance_kl(
                cache.mean_relevance[index].to(device), student_output.relevance_probs
            )
            teacher_relevance_kl_sum = float(relevance_kl.detach().sum())
            consistency = (1.0 - relevance_weight) * consistency + relevance_weight * relevance_kl
        per_sample = per_sample + consistency_weight * consistency
        loss = loss + consistency_weight * consistency.mean()
        used = torch.ones_like(accepted)
        teacher_consistency_sum = float(consistency.detach().sum())
    return loss, {
        "inspector_cross_entropy_sum": inspector_cross_entropy_sum,
        "inspector_target_entropy_sum": inspector_target_entropy_sum,
        "teacher_consistency_sum": teacher_consistency_sum,
        "teacher_preference_kl_sum": teacher_preference_kl_sum,
        "teacher_relevance_kl_sum": teacher_relevance_kl_sum,
        "valid": int(used.sum().item()),
        "inside": 0,
        "projected": 0,
        "valid_mask": used.cpu().tolist(),
        "inside_mask": [False] * len(indices),
        "projection_kl": per_sample.detach().cpu().tolist(),
    }


THREE_WAY_CLASSES = ("A", "B", "Tie")


def _rounded(values: torch.Tensor) -> List[float]:
    return [round(float(value), 6) for value in values]


def _three_way_label(probabilities: torch.Tensor) -> str:
    """Argmax class (A / B / Tie) of one three-way probability vector."""
    return THREE_WAY_CLASSES[int(probabilities.argmax())]


def unswap_student_output(
    output: MultiMetricRewardOutput, swapped: Sequence[bool]
) -> MultiMetricRewardOutput:
    """Swap A and B back for the rows whose Student input was (B, A).

    The permutation keeps autograd, so every loss and log sees the Student prediction
    in the original A / B order: CE(target, unswapped Student) equals CE(swapped target,
    Student). Tie and relevance do not depend on the order.
    """
    if not any(swapped):
        return output
    mask = torch.tensor(list(swapped), device=output.three_way_probs.device, dtype=torch.bool)
    rows = mask.view(-1, 1)
    three_way = output.three_way_probs
    return MultiMetricRewardOutput(
        tie_probs=output.tie_probs,
        relevance_probs=output.relevance_probs,
        scores_A=torch.where(rows, output.scores_B, output.scores_A),
        scores_B=torch.where(rows, output.scores_A, output.scores_B),
        margins=torch.where(rows, -output.margins, output.margins),
        cond_win_prob_A=torch.where(rows, 1.0 - output.cond_win_prob_A, output.cond_win_prob_A),
        three_way_probs=torch.where(rows.unsqueeze(-1), three_way[..., [1, 0, 2]], three_way),
        signed_evidence=torch.where(rows, -output.signed_evidence, output.signed_evidence),
    )


def per_sample_unlabeled_rows(
    step: int,
    student_output: MultiMetricRewardOutput,
    cache: TeacherCache,
    local_indices: Sequence[int],
    inspector_three_way: torch.Tensor,
    aggregator_weights: torch.Tensor | None,
    aggregation: str,
    consistency_metrics: str,
) -> List[Dict[str, Any]]:
    """Per-sample log rows of one Unlabeled microbatch (read-only, no gradient).

    Student = the training forward (before this step's update); Teacher = the cached
    Teacher pass, aggregated over the metrics the Teacher relevance selects; the
    Teacher KL is the same quantity as the consistency term of the loss.
    """
    with torch.no_grad():
        index = torch.tensor(list(local_indices), dtype=torch.long)
        weights = None if aggregator_weights is None else aggregator_weights.detach().float().cpu()
        student_metric = student_output.three_way_probs.detach().float().cpu()
        student_relevance = student_output.relevance_probs.detach().float().cpu()
        student = (
            student_metric
            * aggregation_weights_from_relevance(student_relevance, weights, aggregation).unsqueeze(-1)
        ).sum(dim=1)
        teacher_metric = cache.mean_probs[index].detach().float().cpu()
        teacher_relevance = cache.mean_relevance[index].detach().float().cpu()
        teacher = (
            teacher_metric
            * aggregation_weights_from_relevance(teacher_relevance, weights, aggregation).unsqueeze(-1)
        ).sum(dim=1)
        inspector = inspector_three_way.detach().float().cpu()
        inspector_cross_entropy = three_way_cross_entropy(student, inspector)
        inspector_entropy = -(inspector * torch.log(inspector.clamp_min(PROBABILITY_EPS))).sum(dim=-1)
        metric_kl = (
            teacher_metric
            * (
                torch.log(teacher_metric.clamp_min(PROBABILITY_EPS))
                - torch.log(student_metric.clamp_min(PROBABILITY_EPS))
            )
        ).sum(dim=-1)
        if consistency_metrics == "selected":
            teacher_kl = (metric_kl * selected_metric_weights(student_relevance)).sum(dim=-1)
        else:
            teacher_kl = metric_kl.sum(dim=-1) / NUM_METRICS
        # Soft-label detail (user, 2026-11-05): besides the one-hot agreement, how confident
        # the Teacher is in its label and how confident the (augmented) Student is in the
        # SAME label, overall and per metric; per-metric KL / CE(Teacher, Student).
        teacher_metric_entropy = -(teacher_metric * torch.log(teacher_metric.clamp_min(PROBABILITY_EPS))).sum(dim=-1)
        metric_cross_entropy = metric_kl + teacher_metric_entropy
        overall_kl = (teacher * (torch.log(teacher.clamp_min(PROBABILITY_EPS))
                                 - torch.log(student.clamp_min(PROBABILITY_EPS)))).sum(dim=-1)
        teacher_selected = selected_metric_weights(teacher_relevance) > 0
        student_selected = selected_metric_weights(student_relevance) > 0
        # Logged in every run (2026-11-07); a loss term only with --teacher-kl-relevance-weight > 0.
        relevance_kl = teacher_relevance_kl(teacher_relevance, student_relevance)
    rows = []
    for position, local_index in enumerate(local_indices):
        teacher_label = _three_way_label(teacher[position])
        student_label = _three_way_label(student[position])
        inspector_label = _three_way_label(inspector[position])
        rows.append({
            "kind": "unlabeled",
            "step": step,
            "sample_id": cache.sample_ids[local_index],
            "teacher_status": cache.status[local_index],
            "teacher_label": teacher_label,
            "student_label": student_label,
            "inspector_label": inspector_label,
            "student_same_as_teacher": student_label == teacher_label,
            "student_same_as_inspector": student_label == inspector_label,
            "teacher_same_as_inspector": teacher_label == inspector_label,
            "inspector": _rounded(inspector[position]),
            "teacher": _rounded(teacher[position]),
            "student": _rounded(student[position]),
            "inspector_cross_entropy": round(float(inspector_cross_entropy[position]), 6),
            "inspector_kl": round(float(inspector_cross_entropy[position] - inspector_entropy[position]), 6),
            "teacher_kl": round(float(teacher_kl[position]), 6),
            "teacher_kl_relevance": round(float(relevance_kl[position]), 6),
            "student_selected_same_as_teacher_selected": bool(
                torch.equal(student_selected[position], teacher_selected[position])
            ),
            "student_selected_metrics": int((selected_metric_weights(student_relevance[position : position + 1]) > 0).sum()),
            "teacher_selected_metrics": int((selected_metric_weights(teacher_relevance[position : position + 1]) > 0).sum()),
        })
        teacher_class = THREE_WAY_CLASSES.index(teacher_label)
        teacher_confidence = float(teacher[position, teacher_class])
        student_confidence = float(student[position, teacher_class])
        metrics = {}
        for metric_index, metric_name in enumerate(METRIC_NAMES):
            t_probs, s_probs = teacher_metric[position, metric_index], student_metric[position, metric_index]
            t_class = int(t_probs.argmax())
            metrics[metric_name] = {
                "teacher": _rounded(t_probs),
                "student": _rounded(s_probs),
                "teacher_label": THREE_WAY_CLASSES[t_class],
                "student_label": _three_way_label(s_probs),
                "student_same_as_teacher": int(s_probs.argmax()) == t_class,
                "teacher_confidence": round(float(t_probs[t_class]), 6),
                "student_confidence_on_teacher_label": round(float(s_probs[t_class]), 6),
                "confidence_gap": round(float(t_probs[t_class] - s_probs[t_class]), 6),
                "kl_teacher_student": round(float(metric_kl[position, metric_index]), 6),
                "cross_entropy_teacher_student": round(float(metric_cross_entropy[position, metric_index]), 6),
                "teacher_relevance": round(float(teacher_relevance[position, metric_index]), 6),
                "student_relevance": round(float(student_relevance[position, metric_index]), 6),
                "teacher_selected": bool(teacher_selected[position, metric_index]),
                "student_selected": bool(student_selected[position, metric_index]),
            }
        rows[-1].update({
            "teacher_confidence": round(teacher_confidence, 6),
            "student_confidence_on_teacher_label": round(student_confidence, 6),
            "confidence_gap": round(teacher_confidence - student_confidence, 6),
            "overall_kl_teacher_student": round(float(overall_kl[position]), 6),
            "metrics": metrics,
        })
    return rows


def per_sample_labeled_rows(
    step: int,
    output: MultiMetricRewardOutput,
    batch: Sequence[HumanRecord],
    aggregator_weights: torch.Tensor | None,
    aggregation: str,
    gemma_losses: Dict[str, torch.Tensor],
) -> List[Dict[str, Any]]:
    """Per-pair log rows of one labeled microbatch (read-only, no gradient).

    Gemma losses are microbatch means, so they are written only for one-pair microbatches.
    """
    with torch.no_grad():
        weights = None if aggregator_weights is None else aggregator_weights.detach().float().cpu()
        relevance = output.relevance_probs.detach().float().cpu()
        student = (
            output.three_way_probs.detach().float().cpu()
            * aggregation_weights_from_relevance(relevance, weights, aggregation).unsqueeze(-1)
        ).sum(dim=1)
        targets = human_three_way_targets(
            torch.tensor([record.label_a for record in batch], dtype=torch.float32)
        )
        cross_entropy = three_way_cross_entropy(student, targets)
    single = len(batch) == 1
    rows = []
    for position, record in enumerate(batch):
        human_label = _three_way_label(targets[position])
        student_label = _three_way_label(student[position])
        rows.append({
            "kind": "labeled",
            "step": step,
            "sample_id": record.sample_id,
            "human": human_label,
            "student_label": student_label,
            "student_same_as_human": student_label == human_label,
            "student": _rounded(student[position]),
            "human_cross_entropy": round(float(cross_entropy[position]), 6),
            "gemma_preference_loss": round(float(gemma_losses["preference_loss"]), 6) if single else None,
            "gemma_relevance_loss": round(float(gemma_losses["relevance_loss"]), 6) if single else None,
            "student_selected_metrics": int((selected_metric_weights(relevance[position : position + 1]) > 0).sum()),
        })
    return rows


def unlabeled_sample_trainable(
    cache: TeacherCache, local_index: int, args: argparse.Namespace
) -> bool:
    if args.unlabeled_objective == "projection":
        return bool(cache.valid[local_index])
    if args.unlabeled_objective == "inspector_direct":
        return int(cache.inspector_labels[local_index]) != 0
    return True


def _ema_dtype(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.float() if tensor.is_floating_point() else tensor


@torch.no_grad()
def teacher_max_absolute_change(
    teacher_state: Mapping[str, torch.Tensor],
    reference_state: Mapping[str, torch.Tensor],
) -> float:
    """Largest absolute difference between the Teacher and a reference copy."""
    largest = 0.0
    for name, reference in reference_state.items():
        current = teacher_state[name].detach().to(device="cpu", dtype=reference.dtype)
        if current.is_floating_point():
            largest = max(largest, float((current - reference).abs().max()))
        elif not torch.equal(current, reference):
            largest = float("inf")
    return largest


def update_teacher_ema(
    model: PeftModel,
    teacher_state: Dict[str, torch.Tensor],
    decay: float,
) -> None:
    student_state = get_peft_model_state_dict(model)
    if student_state.keys() != teacher_state.keys():
        raise RuntimeError("Student and EMA Teacher parameter names diverged.")
    for name, teacher_tensor in teacher_state.items():
        student_tensor = student_state[name].detach().to(
            device=teacher_tensor.device, dtype=teacher_tensor.dtype
        )
        if teacher_tensor.is_floating_point():
            teacher_tensor.mul_(decay).add_(student_tensor, alpha=1.0 - decay)
        else:
            teacher_tensor.copy_(student_tensor)


def teacher_relevance_kl(teacher_relevance: torch.Tensor, student_relevance: torch.Tensor) -> torch.Tensor:
    """Per-pair KL_rel [B]: Bernoulli KL(mu_T || mu_S) per metric, mean over all 6 metrics.

    The Teacher relevance is a target (no gradient); the Student relevance keeps its
    gradient, so the Relevance Head (and the backbone through h_src) learns from Unlabeled.
    Floor 0 (equal relevance)."""
    teacher = teacher_relevance.detach().float().clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    student = student_relevance.float().clamp(PROBABILITY_EPS, 1.0 - PROBABILITY_EPS)
    per_metric = teacher * (torch.log(teacher) - torch.log(student)) + (1.0 - teacher) * (
        torch.log1p(-teacher) - torch.log1p(-student)
    )
    return per_metric.mean(dim=-1)


def selected_metric_weights(relevance: torch.Tensor) -> torch.Tensor:
    """Equal weights [B, 6] over relevance >= 0.5, else the most relevant metric."""
    relevance = relevance.detach().float()
    selected = relevance >= RELEVANCE_THRESHOLD
    fallback = F.one_hot(relevance.argmax(dim=-1), NUM_METRICS).bool()
    selected = torch.where(selected.any(dim=-1, keepdim=True), selected, fallback).float()
    return selected / selected.sum(dim=-1, keepdim=True)


def aggregation_weights_from_relevance(
    relevance: torch.Tensor,
    weights: torch.Tensor | None,
    aggregation: str,
) -> torch.Tensor:
    """Per-pair metric weights [B, 6] for a relevance tensor [B, 6]."""
    if aggregation == "all_metrics_fitted":
        if weights is None:
            raise ValueError("all_metrics_fitted requires the Metrics Weights Aggregator.")
        return weights.to(device=relevance.device, dtype=torch.float32)[None, :].expand(
            relevance.size(0), -1
        )
    if aggregation != "selected_equal":
        raise ValueError(f"Unsupported overall aggregation: {aggregation}")
    return selected_metric_weights(relevance)


def overall_metric_weights(
    output: MultiMetricRewardOutput,
    weights: torch.Tensor | None,
    aggregation: str,
) -> torch.Tensor:
    """Per-pair metric weights [B, 6] of the overall logit."""
    # Hard selection as in the inference wrapper. Relevance is detached so the
    # overall labels never train the Relevance Head.
    return aggregation_weights_from_relevance(output.relevance_probs, weights, aggregation)


def overall_logits(
    evidence: torch.Tensor, beta: float | None, probability: str = "sigmoid_beta"
) -> torch.Tensor:
    """Logit of P(A) for aggregated evidence m in [-1, 1].

    linear: P(A) = (1 + m) / 2, whose logit is 2 * atanh(m); the clamp keeps
    BCE finite when every selected metric saturates.
    """
    if probability == "sigmoid_beta":
        if beta is None:
            raise ValueError("sigmoid_beta requires the Metrics Weights Aggregator beta.")
        return float(beta) * evidence
    if probability != "linear":
        raise ValueError(f"Unsupported overall probability: {probability}")
    bound = 1.0 - LINEAR_EVIDENCE_EPS
    return 2.0 * torch.atanh(evidence.clamp(-bound, bound))


def human_preference_aggregation_logits(
    output: MultiMetricRewardOutput,
    weights: torch.Tensor | None,
    beta: float | None,
    aggregation: str = "all_metrics_fitted",
    probability: str = "sigmoid_beta",
) -> torch.Tensor:
    # HumanGT contains only non-tie pairs in this phase. Preserve the forward
    # definition fitted by Preference Aggregator, but stop this binary loss
    # from lowering Tie Head probabilities merely to enlarge the margin.
    evidence = (
        (1.0 - output.tie_probs.float().detach())
        * (2.0 * output.cond_win_prob_A.float() - 1.0)
    )
    metric_weights = overall_metric_weights(
        output, None if weights is None else weights.to(evidence.device), aggregation
    )
    return overall_logits((evidence * metric_weights).sum(dim=-1), beta, probability)


def human_preference_aggregation_loss(
    output: MultiMetricRewardOutput,
    labels: torch.Tensor,
    weights: torch.Tensor | None,
    beta: float | None,
    aggregation: str = "all_metrics_fitted",
    probability: str = "sigmoid_beta",
) -> torch.Tensor:
    logits = human_preference_aggregation_logits(output, weights, beta, aggregation, probability)
    return F.binary_cross_entropy_with_logits(logits, labels.float())


def overall_three_way_probs(
    output: MultiMetricRewardOutput,
    weights: torch.Tensor | None,
    aggregation: str,
) -> torch.Tensor:
    """Overall (P(A), P(B), P(Tie)) [B, 3]: metric-weighted mean of three_way_probs.

    tie_probs are not detached, so overall labels also train the Tie Head;
    relevance stays detached through the metric weights.
    """
    metric_weights = overall_metric_weights(output, weights, aggregation)
    return (output.three_way_probs.float() * metric_weights.unsqueeze(-1)).sum(dim=1)


def three_way_cross_entropy(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Per-pair -sum_c target_c * log p_c for [B, 3] tensors."""
    return -(targets * torch.log(probabilities.clamp_min(PROBABILITY_EPS))).sum(dim=-1)


def human_three_way_targets(labels: torch.Tensor) -> torch.Tensor:
    """One-hot (A, B, Tie) from label_a in {1.0, 0.0, HUMAN_TIE_LABEL}."""
    is_tie = labels == HUMAN_TIE_LABEL
    is_a = (labels == 1.0) & ~is_tie
    is_b = (labels == 0.0) & ~is_tie
    if not bool(torch.all(is_a | is_b | is_tie)):
        raise ValueError("HumanGT labels must be 1.0 (A), 0.0 (B) or the tie label.")
    return torch.stack((is_a, is_b, is_tie), dim=-1).float()


def ema_decay_at(step: int, total_steps: int, args: argparse.Namespace) -> float:
    """EMA decay of the update after optimizer step ``step`` (0-based)."""
    if args.ema_schedule == "constant":
        return args.ema_decay
    progress = min(1.0, max(0.0, step / max(1, total_steps - 1)))
    span = args.ema_decay_end - args.ema_decay_start
    return args.ema_decay_end - 0.5 * span * (1.0 + math.cos(math.pi * progress))


def gemma_replay_losses(
    output: MultiMetricRewardOutput,
    records: Sequence[GemmaRecord],
    objective: FactorizedWarmStartLoss,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    # The factorized Gemma objective supplies explicit BCE(g_T, tie_probs)
    # on applicable metrics. Human overall ties are not six metric-tie labels.
    def targets(name: str) -> torch.Tensor:
        return torch.tensor(
            [getattr(record, name) for record in records],
            device=device,
            dtype=torch.float32,
        )

    return objective.components(
        output,
        targets("g_a"),
        targets("g_b"),
        targets("g_t"),
        targets("relevance"),
    )


def set_loss_weight(step: int, target: float, ramp_steps: int) -> float:
    if ramp_steps == 0:
        return target
    return target * min(1.0, float(step + 1) / float(ramp_steps))


def adaptive_set_scale(
    round_report: Mapping[str, Any], current_scale: float, args: argparse.Namespace
) -> float:
    """Bounded response to projection pressure and relative loss magnitude."""
    valid = max(1, int(round_report["valid_unlabeled_occurrences"]))
    projection_rate = float(round_report["projected_occurrences"]) / valid
    losses = round_report["losses"]
    supervised = float(losses["human"] + losses["gemma"] + losses["mask"])
    raw_set = max(float(losses["set"]), PROBABILITY_EPS)
    projection_factor = projection_rate / args.set_target_projection_rate
    contribution_factor = (
        args.set_target_contribution * supervised / raw_set
    )
    desired = math.sqrt(max(PROBABILITY_EPS, projection_factor * contribution_factor))
    desired = min(args.set_scale_max, max(args.set_scale_min, desired))
    updated = (
        args.set_scale_momentum * current_scale
        + (1.0 - args.set_scale_momentum) * desired
    )
    return min(args.set_scale_max, max(args.set_scale_min, updated))


def rank_hard_samples(
    diagnostics: Mapping[int, Mapping[str, Any]]
) -> List[int]:
    """Replay learnable projected samples, never infeasible contradictions."""
    eligible = {
        index: values for index, values in diagnostics.items()
        if values.get("valid")
        and values.get("status") == "valid_inspector_constraint"
    }
    if not eligible:
        return []

    def percentile(field: str) -> Dict[int, float]:
        ordered = sorted(
            eligible,
            key=lambda index: float(eligible[index].get(field, 0.0)),
        )
        denominator = max(1, len(ordered) - 1)
        return {index: rank / denominator for rank, index in enumerate(ordered)}

    projection_rank = percentile("projection_kl")
    width_rank = percentile("box_width")
    uncertainty_rank = percentile("combined_variance")
    scores = {}
    for index, values in eligible.items():
        scores[index] = (
            2.0 * bool(values.get("projected", False))
            + projection_rank[index]
            + 0.25 * width_rank[index]
            + 0.25 * uncertainty_rank[index]
        )
    return sorted(scores, key=lambda index: scores[index], reverse=True)


@torch.no_grad()
def evaluate_constraint_probe(
    model: PeftModel,
    processor: Any,
    records: Sequence[PairRecord],
    prompt: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor],
    inspector_labels: torch.Tensor,
    aggregator_weights: torch.Tensor | None,
    args: argparse.Namespace,
    device: torch.device,
    mc_seed: int,
) -> Dict[str, Any]:
    """Measure identical Unlabeled samples without using them for gradients."""
    cache = teacher_cache_for_batch(
        model,
        processor,
        records,
        prompt,
        teacher_state,
        inspector_labels,
        aggregator_weights,
        args,
        device,
        mc_seed=mc_seed,
    )
    student_probabilities = []
    model.eval()
    for indices in batched_indices(len(records), args.eval_batch_size):
        output = forward_pairs(
            model, processor, [records[index] for index in indices], prompt, device
        )
        student_probabilities.append(output.three_way_probs.float())
    q = torch.cat(student_probabilities)
    pair_weights = diagnostic_pair_weights(aggregator_weights, cache.mean_relevance)
    target, inside = project_to_feasible_set(
        q,
        cache.lower.to(device),
        cache.upper.to(device),
        cache.inspector_labels.to(device),
        cache.effective_kappa.to(device),
        cache.valid.to(device),
        pair_weights.to(device),
        args.projection_iterations,
    )
    projection_kl = (
        target * (
            torch.log(target.clamp_min(PROBABILITY_EPS))
            - torch.log(q.clamp_min(PROBABILITY_EPS))
        )
    ).sum(dim=(-1, -2)) / NUM_METRICS
    valid = cache.valid
    widths = (cache.upper - cache.lower).mean(dim=(-1, -2))
    winner_variance = cache.var_winner.mean(dim=-1)
    tie_variance = cache.var_tie.mean(dim=-1)
    relevance_variance = cache.var_relevance.mean(dim=-1)
    entropy = -(
        cache.mean_probs
        * torch.log(cache.mean_probs.clamp_min(PROBABILITY_EPS))
    ).sum(dim=-1).mean(dim=-1)
    direction_margin = (
        cache.inspector_labels.float()
        * (
            pair_weights
            * (cache.mean_probs[:, :, 0] - cache.mean_probs[:, :, 1])
        ).sum(dim=-1)
    )
    per_sample = {}
    for index, sample_id in enumerate(cache.sample_ids):
        per_sample[sample_id] = {
            "box_width": float(widths[index]),
            "winner_variance": float(winner_variance[index]),
            "tie_variance": float(tie_variance[index]),
            "relevance_variance": float(relevance_variance[index]),
            "teacher_entropy": float(entropy[index]),
            "inspector_margin": float(direction_margin[index]),
            "status": cache.status[index],
            "student_inside": bool(inside[index].item()),
            "projection_kl": float(projection_kl[index]),
        }
    valid_count = int(valid.sum())
    status_counts = dict(Counter(cache.status))
    return {
        "samples": len(records),
        "status_counts": status_counts,
        "valid_rate": valid_count / len(records),
        "inside_rate": int((inside.cpu() & valid).sum()) / max(1, valid_count),
        "mean_box_width": float(widths.mean()),
        "mean_winner_variance": float(winner_variance.mean()),
        "mean_tie_variance": float(tie_variance.mean()),
        "mean_relevance_variance": float(relevance_variance.mean()),
        "mean_teacher_entropy": float(entropy.mean()),
        "mean_projection_kl": float(projection_kl[valid.to(device)].mean())
        if valid_count else 0.0,
        "per_sample": per_sample,
    }


def compare_probe_reports(
    current: Mapping[str, Any], reference: Mapping[str, Any]
) -> Dict[str, float | int]:
    current_samples = current["per_sample"]
    reference_samples = reference["per_sample"]
    if current_samples.keys() != reference_samples.keys():
        raise RuntimeError("Diagnostic probe membership changed between rounds.")
    width_delta = []
    projection_delta = []
    tightened = widened = recovered = new_conflicts = 0
    for sample_id, now in current_samples.items():
        before = reference_samples[sample_id]
        delta = float(now["box_width"] - before["box_width"])
        width_delta.append(delta)
        projection_delta.append(
            float(now["projection_kl"] - before["projection_kl"])
        )
        tightened += delta < -1e-6
        widened += delta > 1e-6
        recovered += (
            before["status"] == "constraint_conflict"
            and now["status"] != "constraint_conflict"
        )
        new_conflicts += (
            before["status"] != "constraint_conflict"
            and now["status"] == "constraint_conflict"
        )
    count = len(width_delta)
    return {
        "mean_box_width_delta": sum(width_delta) / count,
        "tightened_fraction": tightened / count,
        "widened_fraction": widened / count,
        "mean_projection_kl_delta": sum(projection_delta) / count,
        "recovered_conflicts": recovered,
        "new_conflicts": new_conflicts,
    }


def separate_head_parameters(
    model: nn.Module, optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    """Move the trainable metric_heads parameters into their own AdamW groups.

    Each original group keeps its weight decay; the head part of it gets
    ``learning_rate``. LoRA parameters keep the base learning rate.
    """
    head_ids = {
        id(parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "metric_heads" in name
    }
    if not head_ids:
        raise ValueError("--head-learning-rate found no trainable metric_heads parameters.")
    new_groups = []
    moved = 0
    for group in optimizer.param_groups:
        head = [parameter for parameter in group["params"] if id(parameter) in head_ids]
        if head:
            group["params"] = [parameter for parameter in group["params"] if id(parameter) not in head_ids]
            new_groups.append({"params": head, "lr": learning_rate, "weight_decay": group["weight_decay"]})
            moved += len(head)
    if moved != len(head_ids):
        raise RuntimeError("Not all metric_heads parameters were in the optimizer groups.")
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
    for group in new_groups:
        optimizer.add_param_group(group)
    summary = [
        (len(group["params"]), group["lr"], group["weight_decay"])
        for group in optimizer.param_groups
    ]
    print(f"Optimizer groups after the head split (params, lr, weight decay): {summary}", flush=True)


def separate_tie_coupling_parameters(
    model: nn.Module, optimizer: torch.optim.Optimizer, learning_rate: float
) -> None:
    """Move the trainable theta0/kappa_raw into their own AdamW group.

    At learning rate 1e-5 AdamW moves them by about 0.01 over a whole run.
    They get no weight decay, which would pull theta0 towards a 0.5 tie prior.
    """
    tie_parameters = [
        parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and "tie_coupling" in name
    ]
    if not tie_parameters:
        raise ValueError("--tie-coupling-learning-rate needs a margin-coupled Tie Head.")
    tie_ids = {id(parameter) for parameter in tie_parameters}
    removed = 0
    for group in optimizer.param_groups:
        kept = [parameter for parameter in group["params"] if id(parameter) not in tie_ids]
        removed += len(group["params"]) - len(kept)
        group["params"] = kept
    if removed != len(tie_parameters):
        raise RuntimeError("theta0/kappa_raw were not all in the optimizer groups.")
    optimizer.param_groups[:] = [group for group in optimizer.param_groups if group["params"]]
    optimizer.add_param_group(
        {"params": tie_parameters, "lr": learning_rate, "weight_decay": 0.0}
    )
    summary = [
        (len(group["params"]), group["lr"], group["weight_decay"])
        for group in optimizer.param_groups
    ]
    print(f"Optimizer groups (tensors, lr, weight decay): {summary}", flush=True)


def cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer, total_steps: int, warmup_ratio: float
) -> torch.optim.lr_scheduler.LambdaLR:
    warmup_steps = round(total_steps * warmup_ratio)

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        remaining = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / remaining))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


@torch.no_grad()
def evaluate_on_validation(
    model: PeftModel,
    processor: Any,
    records: Sequence[HumanRecord],
    prompt: Mapping[str, Any],
    teacher_state: Mapping[str, torch.Tensor] | None,
    weights: torch.Tensor | None,
    beta: float | None,
    batch_size: int,
    device: torch.device,
    gemma_records: Sequence[GemmaRecord] | None = None,
    aggregation: str = "all_metrics_fitted",
    probability: str = "sigmoid_beta",
) -> Tuple[Dict[str, float], List[float]]:
    """Evaluate the Teacher Model (``teacher_state``) or, with None, the Student Model.

    Returns the metrics and the per-pair HumanGT probability that A wins, in
    ``records`` order, so arms can be compared with a paired test.
    """
    student_state = (
        capture_adapter_state_for_teacher_swap(model) if teacher_state is not None else None
    )
    logits_list: List[torch.Tensor] = []
    labels_list: List[torch.Tensor] = []
    gemma_pref_total = gemma_mask_total = 0.0
    gemma_tie_brier_total = gemma_tie_count = 0.0
    gemma_tie_positive_brier = gemma_tie_positive_count = 0.0
    gemma_tie_negative_brier = gemma_tie_negative_count = 0.0
    gemma_winner_brier_total = gemma_winner_count = 0.0
    gemma_mask_brier_total = 0.0
    try:
        if teacher_state is not None:
            load_adapter_state(model, teacher_state)
        model.eval()
        for indices in batched_indices(len(records), batch_size):
            batch = [records[index] for index in indices]
            output = forward_pairs(model, processor, batch, prompt, device)
            logits = overall_logits(
                (
                    output.signed_evidence.float()
                    * overall_metric_weights(
                        output, None if weights is None else weights.to(device), aggregation
                    )
                ).sum(dim=-1),
                beta,
                probability,
            )
            logits_list.append(logits.cpu())
            labels_list.append(
                torch.tensor([record.label_a for record in batch], dtype=torch.float32)
            )
        if gemma_records:
            objective = FactorizedWarmStartLoss()
            for indices in batched_indices(len(gemma_records), batch_size):
                batch = [gemma_records[index] for index in indices]
                output = forward_pairs(model, processor, batch, prompt, device)
                components = gemma_replay_losses(output, batch, objective, device)
                g_t = torch.tensor([record.g_t for record in batch], device=device)
                g_a = torch.tensor([record.g_a for record in batch], device=device)
                g_b = torch.tensor([record.g_b for record in batch], device=device)
                relevance = torch.tensor([record.relevance for record in batch], device=device)
                applicable = relevance > 0.0
                non_tie_mass = g_a + g_b
                winner_applicable = applicable & (non_tie_mass > 0.0)
                winner_target = g_a / non_tie_mass.clamp_min(PROBABILITY_EPS)
                gemma_pref_total += float(components["preference_loss"]) * len(batch)
                gemma_mask_total += float(components["relevance_loss"]) * len(batch)
                gemma_tie_brier_total += float(
                    (((output.tie_probs.float() - g_t) ** 2) * applicable).sum()
                )
                gemma_tie_count += int(applicable.sum())
                tie_error = (output.tie_probs.float() - g_t) ** 2
                tie_positive = applicable & (g_t >= 0.5)
                tie_negative = applicable & (g_t == 0.0)
                gemma_tie_positive_brier += float(tie_error[tie_positive].sum())
                gemma_tie_positive_count += int(tie_positive.sum())
                gemma_tie_negative_brier += float(tie_error[tie_negative].sum())
                gemma_tie_negative_count += int(tie_negative.sum())
                gemma_winner_brier_total += float(
                    (((output.cond_win_prob_A.float() - winner_target) ** 2)
                     * winner_applicable).sum()
                )
                gemma_winner_count += int(winner_applicable.sum())
                gemma_mask_brier_total += float(
                    ((output.relevance_probs.float() - relevance) ** 2).sum()
                )
    finally:
        if student_state is not None:
            load_adapter_state(model, student_state)
    if not logits_list:
        # No HumanGT validation (--checkpoint-selection last).
        nan = float("nan")
        metrics = {
            "samples": 0, "bce": nan, "accuracy": nan,
            "balanced_accuracy": nan, "brier_score": nan,
        }
        probabilities = torch.zeros(0)
    else:
        logits = torch.cat(logits_list).double()
        labels = torch.cat(labels_list).double()
        probabilities = torch.sigmoid(logits)
        predictions = probabilities >= 0.5
        class_accuracy = []
        for label in (0.0, 1.0):
            mask = labels == label
            class_accuracy.append(float((predictions[mask] == bool(label)).double().mean()))
        metrics = {
            "samples": int(labels.numel()),
            "bce": float(F.binary_cross_entropy_with_logits(logits, labels)),
            "accuracy": float((predictions == labels.bool()).double().mean()),
            "balanced_accuracy": float(sum(class_accuracy) / 2.0),
            "brier_score": float(((probabilities - labels) ** 2).mean()),
        }
    if gemma_records:
        metrics.update({
            "gemma_validation_samples": len(gemma_records),
            "gemma_preference_loss": gemma_pref_total / len(gemma_records),
            "gemma_mask_bce": gemma_mask_total / len(gemma_records),
            "gemma_tie_brier": gemma_tie_brier_total / max(1.0, gemma_tie_count),
            "gemma_tie_positive_brier": gemma_tie_positive_brier / max(1.0, gemma_tie_positive_count),
            "gemma_tie_positive_cells": int(gemma_tie_positive_count),
            "gemma_tie_negative_brier": gemma_tie_negative_brier / max(1.0, gemma_tie_negative_count),
            "gemma_tie_negative_cells": int(gemma_tie_negative_count),
            "gemma_winner_brier": gemma_winner_brier_total / max(1.0, gemma_winner_count),
            "gemma_mask_brier": gemma_mask_brier_total / (len(gemma_records) * NUM_METRICS),
        })
    return metrics, probabilities.float().tolist()


def recursive_to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: recursive_to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [recursive_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(recursive_to_cpu(item) for item in value)
    return value


def save_checkpoint_atomically(checkpoint: Mapping[str, Any], path: Path, *, overwrite: bool = False) -> None:
    """Keep a complete round checkpoint recoverable if writing is interrupted."""
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite checkpoint: {path}")
    writing_path = path.with_name(path.name + ".writing")
    if writing_path.exists():
        raise FileExistsError(f"Incomplete checkpoint needs inspection: {writing_path}")
    torch.save(checkpoint, writing_path)
    writing_path.replace(path)


def trainable_group_of(name: str) -> str:
    if "metric_heads" in name:
        return "heads"
    if "visual" in name and "merger" in name:
        return "aligner"
    if "visual" in name:
        return "vit"
    return "llm"


STEP_DIAGNOSTIC_GROUPS = ("vit", "aligner", "llm", "heads")


def cpu_float_copy(tensors: Sequence[Optional[torch.Tensor]]) -> List[Optional[torch.Tensor]]:
    # copy=True: on a CPU fp32 tensor .float().cpu() would return the same storage.
    return [
        None if tensor is None else tensor.detach().to("cpu", dtype=torch.float32, copy=True)
        for tensor in tensors
    ]


def _ratio(numerator: float, denominator: float) -> Optional[float]:
    return numerator / denominator if denominator > 0.0 else None


def gradient_source_report(
    named_parameters: Sequence[Tuple[str, str, torch.Tensor]],
    labeled_gradients: Optional[List[Optional[torch.Tensor]]],
) -> Dict[str, Any]:
    """Norms of labeled, Unlabeled (total - labeled) and total gradient, before clipping.

    Gradients already carry the loss weights (lambda, per-pair weight), so the
    Unlabeled / labeled norm ratio is the effective weight of the Unlabeled term.
    """
    sums = {group: [0.0, 0.0, 0.0, 0.0] for group in STEP_DIAGNOSTIC_GROUPS}
    # Relevance Head alone (2026-11-07): a part of "heads", reported separately, not added to "all".
    relevance_sums = [0.0, 0.0, 0.0, 0.0]
    for index, (name, group, parameter) in enumerate(named_parameters):
        if parameter.grad is None:
            continue
        total = parameter.grad.detach().float()
        labeled = (
            None if labeled_gradients is None or labeled_gradients[index] is None
            else labeled_gradients[index].to(total.device, non_blocking=True)
        )
        unlabeled = total if labeled is None else total - labeled
        targets = [sums[group]] + ([relevance_sums] if "relevance_head" in name else [])
        for entry in targets:
            entry[2] += float(unlabeled.pow(2).sum())
            entry[3] += float(total.pow(2).sum())
            if labeled is not None:
                entry[0] += float(labeled.pow(2).sum())
                entry[1] += float((labeled * unlabeled).sum())
    report: Dict[str, Any] = {}
    all_sums = [0.0, 0.0, 0.0, 0.0]
    for group, (labeled_sq, dot, unlabeled_sq, total_sq) in sums.items():
        for position, value in enumerate((labeled_sq, dot, unlabeled_sq, total_sq)):
            all_sums[position] += value
        report[group] = {
            "labeled": math.sqrt(labeled_sq),
            "unlabeled": math.sqrt(unlabeled_sq),
            "total": math.sqrt(total_sq),
            "cosine": _ratio(dot, math.sqrt(labeled_sq * unlabeled_sq)),
        }
    labeled_sq, dot, unlabeled_sq, total_sq = all_sums
    report["all"] = {
        "labeled": math.sqrt(labeled_sq),
        "unlabeled": math.sqrt(unlabeled_sq),
        "total": math.sqrt(total_sq),
        "cosine": _ratio(dot, math.sqrt(labeled_sq * unlabeled_sq)),
        "unlabeled_over_labeled": _ratio(math.sqrt(unlabeled_sq), math.sqrt(labeled_sq)),
    }
    labeled_sq, dot, unlabeled_sq, total_sq = relevance_sums
    report["relevance_head"] = {
        "labeled": math.sqrt(labeled_sq),
        "unlabeled": math.sqrt(unlabeled_sq),
        "total": math.sqrt(total_sq),
        "cosine": _ratio(dot, math.sqrt(labeled_sq * unlabeled_sq)),
        "unlabeled_over_labeled": _ratio(math.sqrt(unlabeled_sq), math.sqrt(labeled_sq)),
    }
    return report


@torch.no_grad()
def update_ratio_report(
    named_parameters: Sequence[Tuple[str, str, torch.Tensor]],
    before: List[Optional[torch.Tensor]],
) -> Dict[str, Optional[float]]:
    """|W_after - W_before| / |W_before| per group (LoRA A and B and heads together)."""
    sums = {group: [0.0, 0.0] for group in STEP_DIAGNOSTIC_GROUPS}
    for index, (_, group, parameter) in enumerate(named_parameters):
        previous = before[index].to(parameter.device, non_blocking=True)
        current = parameter.detach().float()
        sums[group][0] += float((current - previous).pow(2).sum())
        sums[group][1] += float(previous.pow(2).sum())
    report = {group: _ratio(math.sqrt(delta), math.sqrt(norm)) for group, (delta, norm) in sums.items()}
    delta = sum(value[0] for value in sums.values())
    norm = sum(value[1] for value in sums.values())
    report["all"] = _ratio(math.sqrt(delta), math.sqrt(norm))
    return report


def trainable_gradient_report(model: nn.Module) -> Dict[str, Dict[str, float]]:
    """Gradient presence per trainable group after a backward pass.

    A LoRA block behind reentrant gradient checkpointing whose inputs do not
    require gradients receives None here although it is marked trainable.
    """
    report: Dict[str, Dict[str, float]] = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = trainable_group_of(name)
        entry = report.setdefault(
            group, {"tensors": 0, "without_gradient": 0, "zero_gradient": 0, "squared_norm": 0.0}
        )
        entry["tensors"] += 1
        if parameter.grad is None:
            entry["without_gradient"] += 1
            continue
        squared = float(parameter.grad.detach().float().pow(2).sum())
        entry["squared_norm"] += squared
        if squared == 0.0:
            entry["zero_gradient"] += 1
    for entry in report.values():
        entry["norm"] = math.sqrt(entry.pop("squared_norm"))
    return report


def save_step_checkpoint(
    path: Path,
    model: PeftModel,
    teacher_state: Mapping[str, torch.Tensor],
    base_model_path: str,
    peft_config: Mapping[str, Any],
    prompt: Mapping[str, Any],
    aggregator_record: Mapping[str, Any],
    step: int,
    validation: Mapping[str, Any],
    is_best: bool,
    *,
    overwrite: bool = False,
) -> None:
    """Model-only snapshot (Student and Teacher adapters) for offline evaluation."""
    save_checkpoint_atomically(
        {
            "format_version": 4,
            "model_role": "six_metric_refinement_student_teacher",
            "checkpoint_kind": "step",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model_path": base_model_path,
            "peft_config": peft_config,
            "metric_order": list(METRIC_NAMES),
            "prompt": prompt,
            "student_adapter_state_dict": capture_adapter_state_cpu(model),
            "teacher_adapter_state_dict": {
                name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()
            },
            "preference_aggregator": dict(aggregator_record),
            "optimizer_step": step,
            "validation": dict(validation),
            "is_best_so_far": is_best,
        },
        path,
        overwrite=overwrite,
    )


def main(
    args: argparse.Namespace | None = None,
    *,
    model_override: PeftModel | None = None,
    processor_override: Any | None = None,
    swift_trainer: Any | None = None,
    swift_template: Any | None = None,
) -> None:
    global ACTIVE_SWIFT_TEMPLATE
    args = args or parse_args()
    validate_args(args)
    if (model_override is None) != (processor_override is None):
        raise ValueError("Model and processor overrides must be provided together.")
    if swift_trainer is not None and (model_override is None or swift_template is None):
        raise ValueError("MS-Swift online training needs its model and pairwise template.")
    ACTIVE_SWIFT_TEMPLATE = swift_template
    framework_tag = (
        "ms-swift_online_trainer" if swift_trainer is not None
        else "ms-swift_peft_adaptive_teacher_screening"
    )
    paths = {
        "student": required_file(args.student_checkpoint, "Student checkpoint"),
        "teacher": required_file(args.teacher_checkpoint, "Teacher checkpoint"),
        "inspector": (
            required_file(args.inspector_checkpoint, "Inspector checkpoint")
            if args.human_split == "inspector"
            else required_inspector_checkpoint_dir(args.inspector_checkpoint)
        ),
        "inspector_cache": required_file(args.inspector_cache, "Inspector cache"),
        "aggregator": (
            required_file(args.metrics_weights_aggregator, "Metrics Weights Aggregator")
            if uses_metrics_weights_aggregator(args) else None
        ),
        "human": required_file(args.human_dataset, "Human dataset"),
        "gemma": required_file(args.gemma_dataset, "Gemma dataset"),
        "unlabeled": required_file(args.unlabeled_dataset, "Unlabeled dataset"),
    }
    output_path = required_output(args.output)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if paths["aggregator"] is not None:
        aggregator_weights_cpu, aggregator_beta, aggregator = (
            load_metrics_weights_aggregator(paths["aggregator"])
        )
    else:
        # linear + selected_equal: neither a_k nor beta enters any loss or margin.
        aggregator_weights_cpu, aggregator_beta, aggregator = None, None, None
    inspector_checkpoint = (
        torch.load(paths["inspector"], map_location="cpu", weights_only=False)
        if args.human_split == "inspector"
        # An MS-Swift checkpoint folder: no calibration, default EditReward root.
        else {}
    )
    calibration = inspector_checkpoint.get("acceptance_calibration", {})
    # No calibration: 0.5 accepts the more likely winner of every pair.
    calibrated_acceptance_threshold = float(calibration.get("threshold", 0.5))
    if not 0.5 <= calibrated_acceptance_threshold <= 1.0:
        raise ValueError("Inspector checkpoint has an invalid acceptance threshold.")
    inspector_data = inspector_checkpoint.get("data", {})
    human_root = Path(
        inspector_data.get("editreward_root", DEFAULT_EDITREWARD_ROOT)
    ).expanduser().resolve()
    human_records = load_human_records(
        paths["human"],
        human_root,
        # The train_order split names tie pairs too; the binary loss drops them below.
        keep_ties=args.overall_loss == "three_way" or args.human_split == "student_train_order",
    )
    if args.human_split == "inspector":
        human_train, human_validation = human_split_from_inspector(
            human_records, inspector_checkpoint
        )
    else:
        human_train, human_validation = human_split_from_student_train_order(
            human_records,
            torch.load(paths["student"], map_location="cpu", weights_only=False, mmap=True),
            paths["inspector"],
        )
    if args.overall_loss != "three_way":
        human_train = [record for record in human_train if record.label_a != HUMAN_TIE_LABEL]
    # Validation accuracy is A-versus-B; HumanGT ties only enter the training loss.
    human_validation = [
        record for record in human_validation if record.label_a != HUMAN_TIE_LABEL
    ]
    unlabeled_pool = index_unlabeled_pool(paths["unlabeled"], human_root)
    inspector_pseudo_labels = load_inspector_cache(
        paths["inspector_cache"], unlabeled_pool.sample_ids
    )
    if args.unlabeled_objective == "inspector_soft_ema_consistency" and not bool(
        torch.all(torch.isfinite(inspector_pseudo_labels.probability_a))
    ):
        raise ValueError("The soft Inspector objective requires probability_A for every cache row.")
    if args.overall_loss == "three_way" and not bool(
        torch.all(torch.isfinite(inspector_pseudo_labels.three_way))
    ):
        raise ValueError("--overall-loss three_way requires a three-way Overall Pseudo-label Cache.")
    inspector_counts = Counter(int(value) for value in inspector_pseudo_labels.labels.tolist())
    print(f"Inspector labels: {dict(inspector_counts)}", flush=True)
    print(
        f"Overall aggregation: {args.overall_aggregation}; overall probability: "
        f"{args.overall_probability}"
        + (f" (beta={aggregator_beta:.4f})" if args.overall_probability == "sigmoid_beta" else "")
        + f"; consistency metrics: {args.consistency_metrics}; Metrics Weights Aggregator: "
        + (str(paths["aggregator"]) if paths["aggregator"] is not None else "not used"),
        flush=True,
    )
    print(
        f"Inspector calibrated acceptance threshold={calibrated_acceptance_threshold:.6f}; "
        f"initial high-confidence threshold={args.high_inspector_confidence:.6f}",
        flush=True,
    )
    selection_seed = args.seed if args.selection_seed is None else args.selection_seed
    # --diagnostic-probe-size 0: no fixed probe (its box / projection metrics belong
    # to the projection objective).
    probe_indices = (
        select_diagnostic_probe(
            inspector_pseudo_labels,
            unlabeled_pool.source_groups,
            args.diagnostic_probe_size,
            selection_seed + 40_000,
        )
        if args.diagnostic_probe_size > 0
        else []
    )
    probe_source_groups = {
        unlabeled_pool.source_groups[index] for index in probe_indices
    }
    probe_excluded_indices = (
        [
            index
            for index, source_group in enumerate(unlabeled_pool.source_groups)
            if source_group in probe_source_groups
        ]
        if args.exclude_probe_from_training == "true"
        else []
    )
    probe_records_by_index = materialize_unlabeled_round(unlabeled_pool, probe_indices)
    probe_records = [probe_records_by_index[index] for index in probe_indices]
    print(
        f"Fixed diagnostic probe: {len(probe_indices)} samples from "
        f"{len(probe_source_groups)} source groups; "
        f"excluded training samples={len(probe_excluded_indices)}",
        flush=True,
    )
    # Optional fixed Unlabeled subset: the rest joins the excluded indices, so the
    # sampler never trains it; it stays available as untrained Unlabeled for diagnostics.
    untrained_unlabeled_indices: List[int] = []
    if args.unlabeled_subset_size:
        subset_candidates = sorted(
            set(range(len(unlabeled_pool.sample_ids))) - set(probe_excluded_indices)
        )
        if args.unlabeled_subset_size > len(subset_candidates):
            raise ValueError(
                f"--unlabeled-subset-size {args.unlabeled_subset_size} exceeds the "
                f"{len(subset_candidates)} available Unlabeled samples."
            )
        if args.unlabeled_subset_file:
            # Fixed subset from a file (e.g. the Teacher-Inspector agreement selection).
            with Path(args.unlabeled_subset_file).open("r", encoding="utf-8") as handle:
                file_ids = list(json.load(handle)["sample_ids"])
            index_of_id = {sample_id: index for index, sample_id in enumerate(unlabeled_pool.sample_ids)}
            missing_ids = [sample_id for sample_id in file_ids if sample_id not in index_of_id]
            if missing_ids or len(set(file_ids)) != len(file_ids):
                raise ValueError(
                    f"--unlabeled-subset-file: {len(missing_ids)} unknown IDs, "
                    f"{len(file_ids) - len(set(file_ids))} duplicates."
                )
            if args.unlabeled_subset_size > len(file_ids):
                raise ValueError(
                    f"--unlabeled-subset-size {args.unlabeled_subset_size} exceeds the "
                    f"{len(file_ids)} IDs of --unlabeled-subset-file."
                )
            # A smaller size keeps the first IDs in file order (code checks).
            file_ids = file_ids[: args.unlabeled_subset_size]
            unlabeled_subset = sorted(index_of_id[sample_id] for sample_id in file_ids)
            if set(unlabeled_subset) - set(subset_candidates):
                raise ValueError("--unlabeled-subset-file contains diagnostic probe samples.")
        else:
            unlabeled_subset = sorted(
                random.Random(selection_seed + 50_000).sample(
                    subset_candidates, args.unlabeled_subset_size
                )
            )
        subset_lookup = set(unlabeled_subset)
        untrained_unlabeled_indices = [
            index for index in subset_candidates if index not in subset_lookup
        ]
        probe_excluded_indices = sorted(
            set(probe_excluded_indices) | set(untrained_unlabeled_indices)
        )
        subset_record = {
            "seed": selection_seed + 50_000,
            "source_file": str(Path(args.unlabeled_subset_file).resolve()) if args.unlabeled_subset_file else None,
            "subset_size": len(unlabeled_subset),
            "passes": args.unlabeled_passes,
            "sample_ids": [unlabeled_pool.sample_ids[index] for index in unlabeled_subset],
        }
        subset_path = output_path.parent / "unlabeled_subset.json"
        if subset_path.exists():
            with subset_path.open("r", encoding="utf-8") as handle:
                if json.load(handle)["sample_ids"] != subset_record["sample_ids"]:
                    raise RuntimeError(f"Existing Unlabeled subset differs: {subset_path}")
        else:
            subset_path.parent.mkdir(parents=True, exist_ok=True)
            with subset_path.open("w", encoding="utf-8") as handle:
                json.dump(subset_record, handle, indent=1)
        print(
            f"Unlabeled subset: {len(unlabeled_subset)} samples ("
            + (f"file {args.unlabeled_subset_file}" if args.unlabeled_subset_file else f"seed {selection_seed + 50_000}")
            + "), "
            f"untrained Unlabeled: {len(untrained_unlabeled_indices)}, "
            f"passes: {args.unlabeled_passes}; IDs saved to {subset_path}",
            flush=True,
        )
    # Online augmentation of the Student input (Unlabeled only): scenarios, distractors.
    augmentation = None
    augmentation_samples: Dict[str, Any] = {}
    augmentation_distractors: List[Dict[str, Any]] = []
    augmentation_distractor_paths: Dict[int, str] = {}
    augmentation_info: Dict[int, Dict[str, Any]] = {}
    augmentation_handwritten: Optional[Dict[str, Dict[str, str]]] = None
    augmentation_example_ids: set = set()
    augmentation_examples_saved: set = set()
    # Example rows wait for the Student / Teacher / Inspector labels of the same forward
    # (per-sample log) before they are written to examples.jsonl.
    augmentation_example_rows: Dict[str, Dict[str, Any]] = {}
    augmentation_work_dir = output_path.parent / "augmentation_tmp"
    if args.student_augmentation != "none":
        from scripts import student_augmentation as augmentation

        with Path(args.augmentation_scenarios).open("r", encoding="utf-8") as handle:
            scenarios_file = json.load(handle)
        augmentation_samples = scenarios_file["samples"]
        scenarios_mode = scenarios_file.get("rules", {}).get("mode", "single")
        if (scenarios_mode == "combined") != (args.student_augmentation != "online"):
            raise ValueError(
                f"--student-augmentation {args.student_augmentation} does not match the scenarios "
                f"file mode '{scenarios_mode}'."
            )
        if args.student_augmentation == "single_operation":
            names = args.augmentation_operation.split("+")
            if len(names) == 1 and names[0] not in augmentation.SINGLE_OPERATIONS:
                raise ValueError(f"Unknown --augmentation-operation {args.augmentation_operation}; "
                                 f"choose from {augmentation.SINGLE_OPERATIONS}.")
            if len(names) > 1:
                augmentation.validate_combined_operations(names)
        augmentation_distractors = scenarios_file["distractors"]
        subset_ids = {unlabeled_pool.sample_ids[index] for index in unlabeled_subset}
        missing_scenarios = sorted(subset_ids - set(augmentation_samples))
        if missing_scenarios:
            raise ValueError(f"{len(missing_scenarios)} subset samples have no augmentation scenario.")
        distractor_sources = {row["source_key"] for row in augmentation_distractors}
        subset_sources = {augmentation_samples[sample_id]["source_key"] for sample_id in subset_ids}
        if distractor_sources & subset_sources:
            raise ValueError("A distractor shares its source image with a subset sample.")
        if args.handwritten_instructions:
            handwritten_bytes = Path(args.handwritten_instructions).read_bytes()
            augmentation_handwritten = json.loads(handwritten_bytes.decode("utf-8"))["instructions"]
            subset_instructions = {
                augmentation_samples[sample_id]["instruction"].strip() for sample_id in subset_ids
            }
            missing_rewrites = sorted(
                instruction for instruction in subset_instructions
                if instruction not in augmentation_handwritten
                or any(not str(augmentation_handwritten[instruction].get(style, "")).strip()
                       for style in augmentation.HANDWRITTEN_STYLES)
            )
            if missing_rewrites:
                raise ValueError(f"{len(missing_rewrites)} subset instructions have no complete hand-written rewrite.")
            print(
                f"Hand-written instructions: {len(subset_instructions)} subset instructions covered, "
                f"file {args.handwritten_instructions}, md5 {hashlib.md5(handwritten_bytes).hexdigest()}",
                flush=True,
            )
        augmentation_work_dir.mkdir(parents=True, exist_ok=True)
        if args.augmentation_examples:
            augmentation_example_ids = set(random.Random(args.seed).sample(
                sorted(subset_ids), min(args.augmentation_examples, len(subset_ids))))
            print(f"Augmentation examples: {len(augmentation_example_ids)} samples kept in "
                  f"{augmentation_work_dir} (examples.jsonl)", flush=True)
        print(
            f"Student online augmentation: {len(augmentation_samples)} scenarios, "
            f"{len(augmentation_distractors)} distractors, scenarios {args.augmentation_scenarios}, "
            f"work dir {augmentation_work_dir}",
            flush=True,
        )
    # Labeled-pair augmentation (2026-11-06): eligibility of the labeled pairs with the same
    # rules as build_augmentation_scenarios.py (combined mode); same distractor list as the
    # Unlabeled samples (so their draws do not change), checked to share no source image.
    labeled_augmentation_samples: Dict[str, Any] = {}
    labeled_augmentation_info: Dict[str, Dict[str, Any]] = {}
    if args.augment_labeled:
        labeled_sources = {Path(record.source_image).parent.name for record in human_train}
        shared = len(labeled_sources & {row["source_key"] for row in augmentation_distractors}) + len(
            {str(Path(record.source_image).resolve()) for record in human_train}
            & {str(Path(resolve_image_path(row["source_img"], human_root)).resolve())
               for row in augmentation_distractors})
        if shared:
            raise ValueError(f"{len(shared)} distractors share a source image with a labeled pair.")
        for record in human_train:
            instruction = record.edit_instruction
            reason = augmentation.flip_exclusion(instruction)
            degradations, is_mild, excluded = augmentation.degradation_eligibility(instruction)
            labeled_augmentation_samples[record.sample_id] = {
                "instruction": instruction,
                "source_key": Path(record.source_image).parent.name,
                "text_eligible": augmentation.text_eligibility(instruction),
                "layout_eligible": ["none", "spatial"] + ([] if reason else ["flip"]),
                "flip_excluded_reason": reason,
                "degradation_eligible": degradations,
                "degradation_mild": is_mild,
                "degradation_excluded": excluded,
            }
        labeled_summary = {
            "samples": len(labeled_augmentation_samples),
            "text_eligible_counts": dict(Counter(n for v in labeled_augmentation_samples.values()
                                                 for n in v["text_eligible"])),
            "degradation_eligible_counts": dict(Counter(n for v in labeled_augmentation_samples.values()
                                                        for n in v["degradation_eligible"])),
            "degradation_mild_samples": sum(v["degradation_mild"] for v in labeled_augmentation_samples.values()),
            "distractors": len(augmentation_distractors),
        }
        (output_path.parent / "augmentation_labeled_scenarios.json").write_text(json.dumps({
            "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "operation": args.augmentation_operation, "seed": args.seed,
            "rules": "build_augmentation_scenarios.py combined mode (text_eligibility, flip_exclusion, "
                     "degradation_eligibility); distractors = those of --augmentation-scenarios",
            "summary": labeled_summary, "samples": labeled_augmentation_samples,
        }, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"Labeled augmentation: {args.augmentation_operation} on {len(human_train)} labeled pairs; "
              f"eligibility {json.dumps(labeled_summary)}", flush=True)

    def augmentation_distractor_path(index: int) -> str:
        if index not in augmentation_distractor_paths:
            augmentation_distractor_paths[index] = resolve_image_path(
                augmentation_distractors[index]["source_img"], human_root
            )
        return augmentation_distractor_paths[index]

    del inspector_checkpoint, human_records
    print(
        f"Human split: train={len(human_train)}, validation={len(human_validation)}",
        flush=True,
    )

    student_checkpoint = torch.load(
        paths["student"], map_location="cpu", weights_only=False
    )
    prompt = student_checkpoint.get("prompt")
    if not isinstance(prompt, dict) or not all(
        isinstance(prompt.get(key), str)
        for key in ("system_message", "evaluation_instruction")
    ):
        raise ValueError("Student checkpoint prompt is incomplete.")
    gemma_records = load_gemma_records(paths["gemma"])
    gemma_data = student_checkpoint.get("data", {})
    gemma_train_ids = gemma_data.get("train_sample_ids")
    gemma_validation_ids = gemma_data.get("validation_sample_ids")
    if not isinstance(gemma_train_ids, list) or not isinstance(gemma_validation_ids, list):
        raise ValueError("Student checkpoint is missing Gemma split IDs.")
    gemma_by_id = {record.sample_id: record for record in gemma_records}
    if not set(gemma_train_ids) | set(gemma_validation_ids) <= set(gemma_by_id):
        raise ValueError("Gemma dataset differs from Student warm-start split.")
    unused_gemma = set(gemma_by_id) - set(gemma_train_ids) - set(gemma_validation_ids)
    if unused_gemma and gemma_data.get("split_method") not in (
        "shared_source_grouped_three_way_v1", "train_order_prefix_v1"
    ):
        raise ValueError("Gemma dataset differs from Student warm-start split.")
    if unused_gemma:
        # The shared split reserves its calibration partition for the Inspector.
        print(f"Gemma pairs outside the Student split (not used): {len(unused_gemma)}", flush=True)
    gemma_train = [gemma_by_id[sample_id] for sample_id in gemma_train_ids]
    if not gemma_train or set(gemma_train_ids) & set(gemma_validation_ids):
        raise ValueError("Gemma training split is empty or overlaps validation.")
    gemma_validation = [gemma_by_id[sample_id] for sample_id in gemma_validation_ids]
    if not gemma_validation and args.checkpoint_selection == "last":
        # No held-out GemmaGT: monitor Gemma losses (soft targets) on the
        # training pairs, which measures forgetting, not generalization.
        gemma_validation = list(gemma_train)
        if args.gemma_monitor_pairs:
            if args.gemma_monitor_pairs > len(gemma_validation):
                raise ValueError(
                    f"--gemma-monitor-pairs {args.gemma_monitor_pairs} exceeds the "
                    f"{len(gemma_validation)} GemmaGT training pairs."
                )
            gemma_validation = random.Random(selection_seed + 70_000).sample(
                gemma_validation, args.gemma_monitor_pairs
            )
        print(
            f"Gemma monitoring on the {len(gemma_validation)} GemmaGT training pairs "
            "(soft targets; no Gemma validation)",
            flush=True,
        )
    del gemma_records, gemma_by_id
    if args.gemma_targets == "majority_vote":
        hardened = [majority_vote_gemma_record(record) for record in gemma_train]
        changed = sum(
            1
            for soft, hard in zip(gemma_train, hardened)
            for k in range(NUM_METRICS)
            if (soft.g_a[k], soft.g_b[k], soft.g_t[k], soft.relevance[k])
            != (hard.g_a[k], hard.g_b[k], hard.g_t[k], hard.relevance[k])
        )
        gemma_train = hardened
        print(f"Gemma replay targets: majority vote; changed metric cells={changed}", flush=True)
    trainable_unlabeled = len(unlabeled_pool.sample_ids) - len(probe_excluded_indices)
    # all_once with several passes trains every sample once per pass.
    trained_unlabeled_occurrences = trainable_unlabeled * args.unlabeled_passes
    round_budget = min(
        trained_unlabeled_occurrences,
        args.unlabeled_per_round or max(1, len(gemma_train) // 2),
    )
    if args.unlabeled_sampling in ("all_once", ITERATIVE_SAMPLING) and round_budget != trained_unlabeled_occurrences:
        # all_once trains every sample, but the cosine schedules end after the budget.
        raise ValueError(
            f"--unlabeled-sampling all_once trains {trainable_unlabeled} samples x "
            f"{args.unlabeled_passes} passes; set --unlabeled-per-round to at least "
            f"{trained_unlabeled_occurrences} (now {round_budget})."
        )
    print(
        f"Gemma replay train={len(gemma_train)}; unlabeled budget/round={round_budget}",
        flush=True,
    )
    original_base_model_path = student_checkpoint["base_model_path"]
    runtime_base_model_path = (
        Path(args.base_model_path).expanduser().resolve()
        if args.base_model_path is not None else None
    )
    if model_override is None:
        model, processor = load_student_model(
            student_checkpoint, device, args.min_pixels, args.max_pixels,
            base_model_override=runtime_base_model_path,
        )
    else:
        model, processor = model_override, processor_override
        if not isinstance(model, PeftModel):
            raise TypeError("MS-Swift must provide a PEFT Student model.")
        expected = student_checkpoint["adapter_state_dict"]
        actual = get_peft_model_state_dict(model)
        if expected.keys() != actual.keys() or any(
            not torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu())
            for name in expected
        ):
            raise RuntimeError("MS-Swift Student differs from the full warm-start checkpoint.")
    base_model_path = str(runtime_base_model_path or original_base_model_path)
    peft_config = student_checkpoint["peft_config"]
    current_student_state = get_peft_model_state_dict(model)
    teacher_checkpoint = torch.load(
        paths["teacher"], map_location="cpu", weights_only=False
    )
    teacher_initial_state = validate_teacher_checkpoint(
        teacher_checkpoint, current_student_state
    )
    teacher_state = {
        # The full 141 GB partition can keep the EMA adapter on GPU and avoid
        # repeated CPU/GPU transfers at every online Teacher update. The EMA is
        # kept in fp32: a (1 - decay) step rounds away in bf16.
        name: _ema_dtype(tensor.detach()).to(device if swift_trainer is not None else "cpu")
        for name, tensor in teacher_initial_state.items()
    }
    del teacher_checkpoint, teacher_initial_state, student_checkpoint

    # --freeze-visual true: freeze before the optimizer is built, so the visual LoRA is
    # neither updated nor weight-decayed. A CPU copy lets every checkpoint log the largest
    # change of the visual LoRA (must be 0).
    visual_groups = ("vit", "aligner")
    visual_start_state = None
    if args.freeze_visual == "true":
        frozen_counts = {group: 0 for group in visual_groups}
        for name, parameter in model.named_parameters():
            group = trainable_group_of(name)
            if parameter.requires_grad and group in visual_groups:
                parameter.requires_grad_(False)
                frozen_counts[group] += parameter.numel()
        if not all(frozen_counts.values()):
            raise RuntimeError(
                f"--freeze-visual true expects trainable ViT and aligner LoRA: {frozen_counts}"
            )
        visual_start_state = {
            name: parameter.detach().cpu().clone()
            for name, parameter in model.named_parameters()
            if "lora_" in name and trainable_group_of(name) in visual_groups
        }
        print(f"Frozen visual parameters (kept at Student-Stage1 values): {frozen_counts}", flush=True)
    trainable_counts: Dict[str, int] = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            group = trainable_group_of(name)
            trainable_counts[group] = trainable_counts.get(group, 0) + parameter.numel()
    print(f"Trainable parameters per group: {trainable_counts}", flush=True)

    trainable_parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    diagnostic_parameters = [
        (name, trainable_group_of(name), parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if swift_trainer is None:
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            eps=args.adam_epsilon,
            weight_decay=args.weight_decay,
        )
    else:
        optimizer = swift_trainer.create_optimizer()
    if args.head_learning_rate is not None:
        # Before the scheduler is built, so the head groups follow the same schedule
        # shape from their own base learning rate.
        separate_head_parameters(model, optimizer, args.head_learning_rate)
    if args.lr_scheduler_type != "cosine" and swift_trainer is None:
        raise ValueError("--lr-scheduler-type other than cosine needs the MS-Swift online backend.")
    if args.tie_coupling_learning_rate is not None:
        # Before the scheduler is built, so the group follows the same warmup
        # and cosine shape from its own base learning rate.
        separate_tie_coupling_parameters(model, optimizer, args.tie_coupling_learning_rate)
    if visual_start_state is not None:
        visual_parameter_ids = {
            id(parameter) for name, parameter in model.named_parameters()
            if trainable_group_of(name) in visual_groups
        }
        if any(
            id(parameter) in visual_parameter_ids
            for group in optimizer.param_groups for parameter in group["params"]
        ):
            raise RuntimeError("--freeze-visual true, but the optimizer holds visual parameters.")
    if args.unlabeled_sampling == "inspector_teacher_rank":
        # Rounds differ in size (replay from round 2, remainder in the last one),
        # so the cosine schedules use the exact planned total.
        planned_round_sizes = inspector_teacher_rank_round_sizes(trainable_unlabeled, args)
        total_optimizer_steps = sum(
            math.ceil(math.ceil(size / args.unlabeled_batch_size) / args.gradient_accumulation_steps)
            for size in planned_round_sizes
        )
        print(
            f"Planned Unlabeled per round={planned_round_sizes}; "
            f"planned optimizer steps={total_optimizer_steps}",
            flush=True,
        )
    else:
        batches_per_round = math.ceil(round_budget / args.unlabeled_batch_size)
        optimizer_steps_per_round = math.ceil(
            batches_per_round / args.gradient_accumulation_steps
        )
        total_optimizer_steps = args.max_rounds * optimizer_steps_per_round
    if args.ema_schedule == "cosine":
        print(
            f"EMA decay: cosine {args.ema_decay_start} -> {args.ema_decay_end} "
            f"over {total_optimizer_steps} steps",
            flush=True,
        )
    else:
        print(f"EMA decay: constant {args.ema_decay}", flush=True)
    scheduler = (
        cosine_schedule_with_warmup(optimizer, total_optimizer_steps, args.warmup_ratio)
        if swift_trainer is None
        else swift_trainer.create_scheduler(total_optimizer_steps, optimizer)
    )
    aggregator_weights = (
        aggregator_weights_cpu.to(device) if aggregator_weights_cpu is not None else None
    )
    human_sampler = CyclingRecordSampler(human_train, args.seed + 10_000)
    gemma_sampler = CyclingRecordSampler(gemma_train, args.seed + 20_000)
    labeled_sampler: CyclingRecordSampler | None = None
    if args.labeled_source == "merged":
        # HumanGT and GemmaGT must be exactly the same pairs with the same A/B order,
        # so one forward serves the human and the Gemma labels.
        gemma_lookup = {record.sample_id: record for record in gemma_train}
        if len(gemma_lookup) != len(gemma_train) or {
            record.sample_id for record in human_train
        } != set(gemma_lookup):
            raise ValueError("--labeled-source merged needs identical HumanGT and GemmaGT pair IDs.")
        labeled_pairs = []
        for human_record in human_train:
            gemma_record = gemma_lookup[human_record.sample_id]
            if (
                human_record.source_image, human_record.edit_instruction,
                human_record.image_a, human_record.image_b,
            ) != (
                gemma_record.source_image, gemma_record.edit_instruction,
                gemma_record.image_a, gemma_record.image_b,
            ):
                raise ValueError(
                    f"HumanGT and GemmaGT differ in images or A/B order: {human_record.sample_id}"
                )
            labeled_pairs.append((human_record, gemma_record))
        labeled_sampler = CyclingRecordSampler(labeled_pairs, args.seed + 10_000)
        print(
            f"Labeled source: merged, {len(labeled_pairs)} pairs (HumanGT + GemmaGT, one forward); "
            f"{args.labeled_pairs_per_step} labeled pairs per step; loss weighting: {args.loss_weighting}",
            flush=True,
        )
    unlabeled_sampler = UnlabeledRoundSampler(
        inspector_pseudo_labels,
        probe_excluded_indices,
        args.seed + 30_000,
        args.max_replays_per_sample,
    )
    gemma_objective = FactorizedWarmStartLoss()
    ramp_steps = args.set_ramp_steps or max(1, round(total_optimizer_steps * 0.2))
    autocast_enabled = device.type == "cuda"
    global_optimizer_step = 0
    history: List[Dict[str, Any]] = []
    baseline_metrics, baseline_teacher_probabilities = evaluate_on_validation(
        model,
        processor,
        human_validation,
        prompt,
        teacher_state,
        aggregator_weights,
        aggregator_beta,
        args.eval_batch_size,
        device,
        gemma_validation,
        args.overall_aggregation,
        args.overall_probability,
    )
    print(f"Initial Teacher Human validation: {baseline_metrics}", flush=True)
    baseline_student_metrics, baseline_student_probabilities = evaluate_on_validation(
        model,
        processor,
        human_validation,
        prompt,
        None,
        aggregator_weights,
        aggregator_beta,
        args.eval_batch_size,
        device,
        gemma_validation,
        args.overall_aggregation,
        args.overall_probability,
    )
    print(f"Initial Student Human validation: {baseline_student_metrics}", flush=True)
    human_validation_sample_ids = [record.sample_id for record in human_validation]
    baseline_predictions = {
        "teacher_probability_A": baseline_teacher_probabilities,
        "student_probability_A": baseline_student_probabilities,
    }
    baseline_probe = None
    if probe_indices:
        baseline_probe = evaluate_constraint_probe(
            model,
            processor,
            probe_records,
            prompt,
            teacher_state,
            inspector_pseudo_labels.labels[probe_indices],
            aggregator_weights_cpu,
            args,
            device,
            args.seed + 50_000,
        )
        print(
            "Initial fixed-probe diagnostics: "
            f"width={baseline_probe['mean_box_width']:.6f} "
            f"inside={baseline_probe['inside_rate']:.4f} "
            f"projection_kl={baseline_probe['mean_projection_kl']:.6f} "
            f"statuses={baseline_probe['status_counts']}",
            flush=True,
        )
    selection_baseline_metrics = (
        baseline_student_metrics if args.selection_model == "student" else baseline_metrics
    )
    best_round = 0
    best_metrics = selection_baseline_metrics
    best_teacher_metrics = baseline_metrics
    best_student_metrics = baseline_student_metrics
    best_student_state = capture_adapter_state_cpu(model)
    best_teacher_state = {
        name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()
    }
    best_optimizer_state = recursive_to_cpu(optimizer.state_dict())
    best_scheduler_state = scheduler.state_dict()
    best_rng_state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    rounds_without_improvement = 0
    best_optimizer_step = 0
    step_checkpoint_paths: List[Path] = []
    best_step_checkpoint: Path | None = None
    step_timing: Counter = Counter()
    previous_probe = baseline_probe
    hard_replay_candidates: List[int] = []
    set_scale = 1.0
    gemma_replay_scale = 1.0
    mask_replay_scale = 1.0
    stop_reason = "max_rounds"
    start_round = 0
    per_sample_file = None
    if args.per_sample_log == "true":
        per_sample_path = Path(args.per_sample_log_path)
        if per_sample_path.exists():
            raise FileExistsError(f"Refusing to append to an existing per-sample log: {per_sample_path}")
        per_sample_path.parent.mkdir(parents=True, exist_ok=True)
        per_sample_file = per_sample_path.open("w", encoding="utf-8", buffering=1)
        print(f"Per-sample log: {per_sample_path}", flush=True)

    # Per-step training log next to the per-sample log (training_log_<job>.jsonl): one JSON
    # row per optimizer step (losses, gradient norms by group and source, update ratios,
    # learning rates, clip coefficient, Student / Teacher / Inspector agreement, timing,
    # memory), written and flushed at every step, plus one row per validation.
    step_log_file = None
    if per_sample_file is not None:
        step_log_path = Path(args.per_sample_log_path).with_name(
            Path(args.per_sample_log_path).name.replace("per_sample_log", "training_log", 1)
        )
        if step_log_path == Path(args.per_sample_log_path) or step_log_path.exists():
            raise FileExistsError(f"Refusing to reuse the training log path: {step_log_path}")
        step_log_file = step_log_path.open("w", encoding="utf-8", buffering=1)
        print(f"Training log (per step): {step_log_path}", flush=True)

    def write_step_log(row: Dict[str, Any]) -> None:
        if step_log_file is not None:
            step_log_file.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            step_log_file.flush()

    def write_per_sample(rows: List[Dict[str, Any]]) -> None:
        for row in rows:
            per_sample_file.write(json.dumps(row, ensure_ascii=False) + "\n")

    if args.resume_checkpoint:
        resume_path = required_file(args.resume_checkpoint, "Round checkpoint")
        saved = torch.load(resume_path, map_location="cpu", weights_only=False)
        if (
            saved.get("model_role") != "six_metric_refinement_student_teacher"
            or saved.get("training_framework") != framework_tag
            or saved.get("base_model_path") not in {
                base_model_path, original_base_model_path
            }
            or saved.get("metric_order") != list(METRIC_NAMES)
        ):
            raise ValueError("Round checkpoint is incompatible with this training loop.")
        saved_training = saved["training"]
        saved_args = saved_training["arguments"]
        # Options added later are absent from older checkpoints; absent means their default.
        defaults_of_later_options = {
            "freeze_visual": "false", "per_sample_log": "false", "unlabeled_subset_file": "",
            "student_augmentation": "none", "augmentation_scenarios": "",
            "handwritten_instructions": "", "augmentation_examples": 0,
            "teacher_kl_relevance_weight": 0.0, "checkpoints_per_iteration": 0,
            "save_round_checkpoints": "all",
        }
        for name, value in vars(args).items():
            if name not in {
                "output", "resume_checkpoint", "device", "base_model_path", "per_sample_log_path"
            } and saved_args.get(
                name, defaults_of_later_options.get(name)
            ) != value:
                raise ValueError(f"Resume argument changed: {name}")
        start_round = int(saved_training["completed_round"])
        if not 0 < start_round <= args.max_rounds:
            raise ValueError("Invalid completed round in checkpoint.")
        load_adapter_state(model, saved["student_adapter_state_dict"])
        teacher_state = {
            name: _ema_dtype(tensor.detach()).to(
                device if swift_trainer is not None else "cpu"
            ).clone()
            for name, tensor in saved["teacher_adapter_state_dict"].items()
        }
        optimizer.load_state_dict(saved["optimizer_state_dict"])
        scheduler.load_state_dict(saved["scheduler_state_dict"])
        sampler_state = saved["samplers"]
        unlabeled_sampler.remaining = set(sampler_state["unlabeled_remaining"])
        unlabeled_sampler.replay_counts = Counter(sampler_state["unlabeled_replay_counts"])
        unlabeled_sampler.deferred = set(sampler_state.get("unlabeled_deferred", []))
        unlabeled_sampler.rng.setstate(sampler_state["unlabeled_rng"])
        human_sampler.order = list(sampler_state["human_order"])
        human_sampler.position = int(sampler_state["human_position"])
        human_sampler.rng.setstate(sampler_state["human_rng"])
        gemma_sampler.order = list(sampler_state["gemma_order"])
        gemma_sampler.position = int(sampler_state["gemma_position"])
        gemma_sampler.rng.setstate(sampler_state["gemma_rng"])
        hard_replay_candidates = list(sampler_state["hard_replay_candidates"])
        history = list(saved_training["history"])
        global_optimizer_step = int(saved_training["global_optimizer_step"])
        best_round = int(saved_training["best_round"])
        best_optimizer_step = int(saved_training.get(
            "best_optimizer_step",
            history[best_round - 1]["optimizer_step"] if best_round else 0,
        ))
        best_metrics = saved_training["best_metrics"]
        best_teacher_metrics = saved_training["best_teacher_metrics"]
        best_student_metrics = saved_training["best_student_metrics"]
        baseline_metrics = saved_training["baseline_metrics"]
        baseline_student_metrics = saved_training["baseline_student_metrics"]
        selection_baseline_metrics = (
            baseline_student_metrics if args.selection_model == "student" else baseline_metrics
        )
        baseline_probe = saved_training["baseline_probe"]
        previous_probe = saved_training["previous_probe"]
        rounds_without_improvement = int(saved_training["rounds_without_improvement"])
        set_scale = float(saved_training["set_scale"])
        gemma_replay_scale = float(saved_training["gemma_replay_scale"])
        mask_replay_scale = float(saved_training["mask_replay_scale"])
        best_snapshot = saved["best_snapshot"]
        best_student_state = best_snapshot["student_adapter_state_dict"]
        best_teacher_state = best_snapshot["teacher_adapter_state_dict"]
        best_optimizer_state = best_snapshot["optimizer_state_dict"]
        best_scheduler_state = best_snapshot["scheduler_state_dict"]
        best_rng_state = best_snapshot["rng_state"]
        rng = saved["rng_state"]
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.set_rng_state(rng["torch_cpu"])
        if torch.cuda.is_available() and rng["torch_cuda"]:
            torch.cuda.set_rng_state_all(rng["torch_cuda"])
        print(f"Resumed after round {start_round} from {resume_path}", flush=True)
        del saved
    if swift_trainer is not None:
        swift_trainer.state.global_step = global_optimizer_step

    aggregator_record = (
        {
            "path": str(paths["aggregator"]),
            "weights": aggregator_weights_cpu.tolist(),
            "beta": aggregator_beta,
            "feature_definition": aggregator["feature_definition"],
            "frozen": True,
        }
        if aggregator is not None
        else {"used_in_training": False}
    )

    # --teacher-update after_round: the Teacher must not change inside a loop; a CPU copy
    # of the starting Teacher lets every checkpoint log the largest change (must be 0).
    teacher_loop_start_state = (
        {name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()}
        if args.teacher_update == "after_round"
        else None
    )
    print(
        f"Teacher update: {args.teacher_update}; Teacher dropout: {args.teacher_dropout} "
        f"(MC passes {args.mc_passes})",
        flush=True,
    )

    def synchronized_time() -> float:
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        return time.perf_counter()

    def timed_forward(batch: Sequence[PairRecord]) -> MultiMetricRewardOutput:
        started = synchronized_time()
        inputs, starts = prepare_pair_inputs(processor, batch, prompt, device)
        step_timing["prepare"] += synchronized_time() - started
        return forward_prepared_pairs(model, inputs, starts)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    # previous_iteration_teacher_inspector_agreement: samples kept for the next iteration,
    # mean losses of the previous iteration, and the last training row of every sample
    # (Student on the augmented input) for the selection file.
    next_iteration_indices: Optional[List[int]] = None
    previous_iteration_losses: Optional[Dict[str, float]] = None
    last_training_rows: Dict[str, Dict[str, Any]] = {}
    replay_selections = None
    if args.replay_round_selections:
        replay_selections = json.loads(Path(args.replay_round_selections).read_text())
        if len(replay_selections) != args.max_rounds:
            raise ValueError("Replay schedule must contain exactly --max-rounds selections.")
        excluded = set(probe_excluded_indices)
        available = {
            sample_id: index for index, sample_id in enumerate(unlabeled_pool.sample_ids)
            if index not in excluded
        }
        for sample_ids in replay_selections:
            if not sample_ids or len(set(sample_ids)) != len(sample_ids) or set(sample_ids) - available.keys():
                raise ValueError("Replay schedule contains empty, duplicate, or unavailable sample IDs.")
    for round_index in range(start_round, args.max_rounds):
        round_improved_by_step = False
        stop_after_round = False
        if replay_selections is not None:
            # Fixed per-round IDs from a previous run; new order and augmentation draws.
            selected = [available[sample_id] for sample_id in replay_selections[round_index]]
            random.Random(args.seed + 80_000 + round_index).shuffle(selected)
            requested_unlabeled = len(selected)
            screening_status = {"replayed_selection": len(selected)}
            screening_cache = {
                unlabeled_pool.sample_ids[index]: {"classification": "replayed_selection"}
                for index in selected
            }
            selection = RoundSelection(
                indices=selected, fresh_high_agreement=len(selected), fresh_low_agreement=0,
                fresh_box_only=0, hard_replay=0, screened=len(selected),
                screening_status=screening_status,
            )
            high_candidates = low_candidates = box_candidates = replay_candidates = []
        elif args.unlabeled_sampling == ITERATIVE_SAMPLING and round_index > 0:
            # Samples of the previous iteration on which the Teacher (after EMA) and the
            # Inspector agree, in a new random order; fresh augmentation draws (round index).
            previous_count = len(history[-1]["selected_sample_ids"])
            selected = list(next_iteration_indices)
            random.Random(args.seed + 80_000 + round_index).shuffle(selected)
            requested_unlabeled = len(selected)
            screening_status = {
                "screened": previous_count,
                "kept_teacher_same_as_inspector": len(selected),
                "left_out_teacher_differs_from_inspector": previous_count - len(selected),
            }
            screening_cache = {
                unlabeled_pool.sample_ids[index]: {"classification": "teacher_same_as_inspector"}
                for index in selected
            }
            selection = RoundSelection(
                indices=selected, fresh_high_agreement=len(selected), fresh_low_agreement=0,
                fresh_box_only=0, hard_replay=0, screened=previous_count,
                screening_status=screening_status,
            )
            high_candidates = low_candidates = box_candidates = replay_candidates = []
            if augmentation_example_ids:
                # A new random set of saved examples in every iteration.
                augmentation_example_ids = set(random.Random(args.seed + round_index).sample(
                    sorted(unlabeled_pool.sample_ids[index] for index in selected),
                    min(args.augmentation_examples, len(selected)),
                ))
                augmentation_examples_saved.clear()
        elif args.unlabeled_sampling == "inspector_teacher_rank":
            requested_unlabeled = planned_round_sizes[round_index]
            selection, screening_cache = select_inspector_teacher_rank_round(
                unlabeled_sampler,
                unlabeled_pool,
                hard_replay_candidates,
                round_index,
                model,
                processor,
                prompt,
                teacher_state,
                inspector_pseudo_labels,
                aggregator_weights_cpu,
                args,
                device,
                global_optimizer_step,
            )
            screening_status = selection.screening_status
            high_candidates = low_candidates = box_candidates = replay_candidates = []
        else:
            requested_unlabeled = round_budget
            high_candidates, low_candidates, box_candidates, replay_candidates, screening_status, screening_cache = (
                screen_curriculum_candidates(
                    unlabeled_sampler,
                    unlabeled_pool,
                    hard_replay_candidates,
                    round_budget,
                    round_index == 0,
                    model,
                    processor,
                    prompt,
                    teacher_state,
                    inspector_pseudo_labels,
                    calibrated_acceptance_threshold,
                    aggregator_weights_cpu,
                    args,
                    device,
                    global_optimizer_step,
                )
            )
            selection = unlabeled_sampler.draw(
                round_budget,
                high_candidates,
                low_candidates,
                box_candidates,
                replay_candidates,
                args.hard_replay_fraction,
                args.low_agreement_fraction,
                args.max_box_only_fraction,
                round_index == 0,
                screening_status["screened"],
                screening_status,
            )
        selected_indices = selection.indices
        if args.unlabeled_passes > 1:
            # Pass 1 keeps the sampler order; each later pass is a fresh shuffle of
            # the same samples, so every sample appears exactly once per pass.
            if len(selected_indices) != trainable_unlabeled:
                raise RuntimeError("all_once did not select every trainable Unlabeled sample.")
            pass_rng = random.Random(args.seed + 60_000)
            expanded = list(selected_indices)
            for _ in range(args.unlabeled_passes - 1):
                one_pass = list(selection.indices)
                pass_rng.shuffle(one_pass)
                expanded.extend(one_pass)
            selected_indices = expanded
        if len(selected_indices) < requested_unlabeled:
            print(
                f"Round {round_index + 1}: quality-gated selection "
                f"to {len(selected_indices)}/{requested_unlabeled}; "
                f"available high={len(high_candidates)} low={len(low_candidates)} "
                f"box={len(box_candidates)} hard={len(replay_candidates)}",
                flush=True,
            )
        screening_path = output_path.parent / f"screening_round_{round_index + 1:02d}.pt"
        if screening_path.exists():
            if not args.resume_checkpoint:
                raise FileExistsError(f"Refusing to overwrite screening cache: {screening_path}")
            attempt = 2
            while screening_path.exists():
                screening_path = output_path.parent / (
                    f"screening_round_{round_index + 1:02d}_attempt_{attempt:02d}.pt"
                )
                attempt += 1
        torch.save({
            "round": round_index + 1,
            "teacher_optimizer_step": global_optimizer_step,
            "screening_status": screening_status,
            "sample_cache": screening_cache,
            "selected_sample_ids": [unlabeled_pool.sample_ids[i] for i in selected_indices],
            "note": "One-pass priority snapshot only; J=4 Teacher targets are refreshed before training after EMA.",
        }, screening_path)
        min_selected = math.ceil(requested_unlabeled * args.min_round_fraction)
        if len(selected_indices) < min_selected:
            if round_index == 0:
                raise RuntimeError(
                    "Too few first-round samples satisfy the Inspector/Teacher "
                    "agreement criteria; inspect screening_status before "
                    "changing thresholds."
                )
            stop_reason = "insufficient_quality_gated_samples"
            break
        print(
            f"Round {round_index + 1}/{args.max_rounds}: "
            f"selected={len(selected_indices)}, remaining={len(unlabeled_sampler.remaining)}, "
            f"high_agreement={selection.fresh_high_agreement}, "
            f"low_agreement={selection.fresh_low_agreement}, "
            f"box_only={selection.fresh_box_only}, "
            f"replay={selection.hard_replay}, screening={selection.screening_status}",
            flush=True,
        )
        round_records = materialize_unlabeled_round(unlabeled_pool, selected_indices)
        microbatches = [
            selected_indices[start : start + args.unlabeled_batch_size]
            for start in range(0, len(selected_indices), args.unlabeled_batch_size)
        ]
        loss_totals = Counter()
        # Total gradient norm before clipping, per optimizer step of the current logging window.
        grad_norm_window: List[float] = []
        status_counts = Counter()
        sample_status: Dict[str, str] = {}
        sample_diagnostics: Dict[int, Dict[str, Any]] = {}
        trained_unlabeled_indices: set[int] = set()
        uncertainty_totals = Counter()
        round_valid = round_inside = round_projected = 0
        # --checkpoints-per-iteration K: K evenly spaced save steps in this iteration (global
        # step numbers), the last one at the iteration end.
        round_planned_steps = math.ceil(len(microbatches) / args.gradient_accumulation_steps)
        if args.restart_lr_each_round:
            scheduler = cosine_schedule_with_warmup(optimizer, round_planned_steps, args.warmup_ratio)
            if swift_trainer is not None:
                swift_trainer.lr_scheduler = scheduler
            print(f"Round {round_index + 1}: cosine restart over {round_planned_steps} steps", flush=True)
        round_save_steps = (
            {
                global_optimizer_step + math.ceil(part * round_planned_steps / args.checkpoints_per_iteration)
                for part in range(1, args.checkpoints_per_iteration + 1)
            }
            if args.checkpoints_per_iteration
            else set()
        )
        if round_save_steps:
            print(f"Round {round_index + 1}: checkpoints at steps {sorted(round_save_steps)}", flush=True)

        for group_start in range(0, len(microbatches), args.gradient_accumulation_steps):
            group = microbatches[group_start : group_start + args.gradient_accumulation_steps]
            group_indices = [index for batch in group for index in batch]
            group_records = [round_records[index] for index in group_indices]
            teacher_started = synchronized_time()
            teacher_cache = teacher_cache_for_batch(
                model,
                processor,
                group_records,
                prompt,
                teacher_state,
                inspector_pseudo_labels.labels[group_indices],
                aggregator_weights_cpu,
                args,
                device,
            )
            step_timing["teacher"] += synchronized_time() - teacher_started
            status_counts.update(teacher_cache.status)
            sample_status.update(zip(teacher_cache.sample_ids, teacher_cache.status))
            cache_width = (teacher_cache.upper - teacher_cache.lower).mean(dim=(-1, -2))
            cache_winner_variance = teacher_cache.var_winner.mean(dim=-1)
            cache_tie_variance = teacher_cache.var_tie.mean(dim=-1)
            cache_relevance_variance = teacher_cache.var_relevance.mean(dim=-1)
            for local_index, pool_index in enumerate(group_indices):
                combined_variance = float(
                    cache_winner_variance[local_index]
                    + cache_tie_variance[local_index]
                    + cache_relevance_variance[local_index]
                )
                sample_diagnostics[pool_index] = {
                    "sample_id": teacher_cache.sample_ids[local_index],
                    "status": teacher_cache.status[local_index],
                    "box_width": float(cache_width[local_index]),
                    "winner_variance": float(cache_winner_variance[local_index]),
                    "tie_variance": float(cache_tie_variance[local_index]),
                    "relevance_variance": float(cache_relevance_variance[local_index]),
                    "combined_variance": combined_variance,
                    "valid": bool(teacher_cache.valid[local_index]),
                }
            uncertainty_totals["box_width"] += float(
                (teacher_cache.upper - teacher_cache.lower).mean()
            ) * len(group_records)
            uncertainty_totals["winner_variance"] += float(
                teacher_cache.var_winner.mean()
            ) * len(group_records)
            uncertainty_totals["tie_variance"] += float(
                teacher_cache.var_tie.mean()
            ) * len(group_records)
            uncertainty_totals["relevance_variance"] += float(
                teacher_cache.var_relevance.mean()
            ) * len(group_records)
            valid_group: List[Tuple[List[int], List[int]]] = []
            local_start = 0
            for unlabeled_indices in group:
                aligned = [
                    (pool_index, local_start + offset)
                    for offset, pool_index in enumerate(unlabeled_indices)
                ]
                local_start += len(unlabeled_indices)
                retained = [
                    (pool_index, local_index)
                    for pool_index, local_index in aligned
                    if unlabeled_sample_trainable(teacher_cache, local_index, args)
                ]
                if retained:
                    valid_group.append((
                        [pool_index for pool_index, _ in retained],
                        [local_index for _, local_index in retained],
                    ))
            if local_start != len(group_records):
                raise RuntimeError("Teacher/Student unlabeled group alignment failed.")
            if not valid_group:
                del teacher_cache
                continue
            step_unlabeled = sum(len(indices) for indices, _ in valid_group)
            if args.labeled_source == "separate":
                step_human = len(valid_group) * args.human_batch_size
                step_gemma = len(valid_group) * args.gemma_batch_size
                step_total = step_human + step_gemma + step_unlabeled
                if min(step_human, step_gemma) / step_total < args.min_supervised_fraction:
                    raise RuntimeError("HumanGT or GemmaGT fraction fell below the configured minimum.")
            elif args.labeled_pairs_per_step and args.labeled_pairs_per_step / (
                args.labeled_pairs_per_step + step_unlabeled
            ) < args.min_supervised_fraction:
                raise RuntimeError("Labeled fraction fell below the configured minimum.")
            student_started = synchronized_time()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            lambda_set_now = set_loss_weight(
                global_optimizer_step, args.lambda_set, ramp_steps
            ) * set_scale
            def backward_scaled(loss: torch.Tensor) -> None:
                if swift_trainer is None:
                    loss.backward()
                else:
                    swift_trainer.accelerator.backward(loss)

            def human_loss_of(output: MultiMetricRewardOutput, batch: Sequence[HumanRecord]) -> torch.Tensor:
                labels = torch.tensor(
                    [record.label_a for record in batch], device=device, dtype=torch.float32
                )
                if args.overall_loss == "three_way":
                    loss = three_way_cross_entropy(
                        overall_three_way_probs(output, aggregator_weights, args.overall_aggregation),
                        human_three_way_targets(labels),
                    ).mean()
                else:
                    loss = human_preference_aggregation_loss(
                        output, labels, aggregator_weights, aggregator_beta,
                        args.overall_aggregation, args.overall_probability,
                    )
                if not torch.isfinite(loss):
                    raise RuntimeError("HumanGT loss became non-finite.")
                return loss

            def save_augmentation_example(sample_id, record, choice, augmented_source, augmented_a, augmented_b):
                """Copies of the original and the Student images (source, A, B) and one line in
                examples.jsonl with every path, the instructions and the full choice."""
                from PIL import Image

                stem = f"example_{len(augmentation_examples_saved) + 1:02d}_round{round_index + 1}"
                saved = {}
                for key, path in (("source_original", record.source_image), ("source_augmented", augmented_source),
                                  ("a_original", record.image_a), ("b_original", record.image_b),
                                  ("a_augmented", augmented_a), ("b_augmented", augmented_b)):
                    if key in ("a_augmented", "b_augmented") and path == {"a_augmented": record.image_a,
                                                                         "b_augmented": record.image_b}[key]:
                        saved[key] = saved[key.replace("augmented", "original")]  # unchanged image
                        continue
                    name = f"{stem}_{key}.png"
                    with Image.open(path) as image:
                        image.convert("RGB").save(augmentation_work_dir / name)
                    saved[key] = name
                distractor_index = choice.get("distractor_index")
                other_index = choice.get("instruction_distractor_index")
                row = {
                    "sample_id": sample_id, "round": round_index + 1, "mode": args.student_augmentation,
                    "original_instruction": record.edit_instruction,
                    "student_instruction": choice["instruction"],
                    "original_paths": {"source": str(record.source_image), "image_a": str(record.image_a),
                                       "image_b": str(record.image_b)},
                    "saved_files": saved,
                    "student_image_a_is": "B" if choice["swap"] else "A",
                    "student_image_b_is": "A" if choice["swap"] else "B",
                    "distractor_image": (
                        {"sample_id": augmentation_distractors[distractor_index]["sample_id"],
                         "path": augmentation_distractor_path(distractor_index)}
                        if distractor_index is not None else None),
                    "distractor_instruction_of": (
                        augmentation_distractors[other_index]["sample_id"] if other_index is not None else None),
                    "choice": {key: value for key, value in choice.items() if key != "instruction"},
                }
                augmentation_examples_saved.add(sample_id)
                if per_sample_file is None:
                    write_augmentation_example(row)
                else:
                    augmentation_example_rows[sample_id] = row

            def write_augmentation_example(row):
                with (augmentation_work_dir / "examples.jsonl").open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")

            def inspector_target_of(unlabeled_indices):
                """Inspector three-way target of the Unlabeled CE: soft probabilities or one-hot argmax."""
                soft = inspector_pseudo_labels.three_way[unlabeled_indices]
                if args.inspector_target == "soft":
                    return soft
                return F.one_hot(soft.argmax(dim=-1), num_classes=3).to(dtype=soft.dtype)

            def augment_for_student(unlabeled_indices):
                """Student records of one microbatch (Teacher / Inspector keep the originals)."""
                student_batch, swapped, created = [], [], []
                for pool_index in unlabeled_indices:
                    record = round_records[pool_index]
                    sample_id = record.sample_id
                    if args.student_augmentation == "combined":
                        choice = augmentation.choose_combined_augmentation(
                            augmentation_samples[sample_id], sample_id, round_index, args.seed,
                            augmentation_distractors, augmentation_handwritten,
                        )
                        make_images = augmentation.combined_augmented_images
                    elif args.student_augmentation == "single_operation" and "+" in args.augmentation_operation:
                        choice = augmentation.choose_combined_operations(
                            args.augmentation_operation.split("+"), augmentation_samples[sample_id], sample_id,
                            round_index, args.seed, augmentation_distractors,
                        )
                        make_images = augmentation.combined_operations_images
                    elif args.student_augmentation == "single_operation":
                        choice = augmentation.choose_single_augmentation(
                            args.augmentation_operation, augmentation_samples[sample_id], sample_id,
                            round_index, args.seed, augmentation_distractors,
                        )
                        make_images = augmentation.single_augmented_images
                    elif args.student_augmentation in augmentation.PART_MODES:
                        choice = augmentation.choose_part_augmentation(
                            args.student_augmentation, augmentation_samples[sample_id], sample_id,
                            round_index, args.seed, augmentation_distractors, augmentation_handwritten,
                        )
                        make_images = augmentation.part_augmented_images
                    else:
                        choice = augmentation.choose_augmentation(
                            augmentation_samples[sample_id], sample_id, round_index, args.seed,
                            len(augmentation_distractors),
                        )
                        make_images = augmentation.augmented_images
                    distractor = (
                        augmentation_distractor_path(choice["distractor_index"])
                        if choice.get("distractor_index") is not None else None
                    )
                    source, image_a, image_b = make_images(
                        choice, record.source_image, record.image_a, record.image_b, distractor,
                        augmentation_work_dir, f"round{round_index + 1}_{pool_index}",
                    )
                    created += [path for path in (source, image_a, image_b)
                                if path.startswith(str(augmentation_work_dir))]
                    if sample_id in augmentation_example_ids and sample_id not in augmentation_examples_saved:
                        save_augmentation_example(sample_id, record, choice, source, image_a, image_b)
                    if choice["swap"]:
                        image_a, image_b = image_b, image_a
                    student_batch.append(replace(
                        record, source_image=source, edit_instruction=choice["instruction"],
                        image_a=image_a, image_b=image_b,
                    ))
                    swapped.append(bool(choice["swap"]))
                    augmentation_info[pool_index] = {
                        "augmentation_round": round_index + 1,
                        "augmentation_scenario": choice["scenario"],
                        "augmentation_swap": bool(choice["swap"]),
                        "student_instruction": choice["instruction"],
                        "original_instruction": record.edit_instruction,
                        "augmentation_position": choice.get("position"),
                        "augmentation_distractor": (
                            augmentation_distractors[choice["distractor_index"]]["sample_id"]
                            if choice.get("distractor_index") is not None else None
                        ),
                        "augmentation_fallback": choice.get("fallback"),
                    }
                    if args.student_augmentation != "online":
                        augmentation_info[pool_index].update({
                            "augmentation_text_operation": choice["text_operation"],
                            "augmentation_layout": choice["layout"],
                            "augmentation_degradation": choice["degradation"],
                            "augmentation_instruction_distractor": (
                                augmentation_distractors[choice["instruction_distractor_index"]]["sample_id"]
                                if choice.get("instruction_distractor_index") is not None else None
                            ),
                        })
                        if args.student_augmentation == "single_operation":
                            augmentation_info[pool_index].update({
                                "augmentation_operation": choice["operation"],
                                "augmentation_applied": bool(choice["applied"]),
                                "augmentation_not_applied_reason": choice["not_applied_reason"],
                                "augmentation_strength": choice["strength"],
                                "augmentation_rotation": choice.get("rotation"),
                                "augmentation_mask": choice.get("mask"),
                                "augmentation_applied_operations": choice.get("applied_operations"),
                            })
                        if args.student_augmentation in augmentation.PART_MODES:
                            augmentation_info[pool_index].update({
                                "augmentation_rotation": choice.get("rotation"),
                                "augmentation_mask": choice.get("mask"),
                                "augmentation_mask_area": choice.get("mask_area"),
                            })
                return student_batch, swapped, created

            def augment_labeled_pairs(human_batch, gemma_batch):
                """--augment-labeled: same operation(s) and draw rule as the Unlabeled samples; labels kept."""
                names = args.augmentation_operation.split("+")
                new_human, new_gemma, created = [], [], []
                for human_record, gemma_record in zip(human_batch, gemma_batch):
                    sample_id = human_record.sample_id
                    scenario = labeled_augmentation_samples[sample_id]
                    if len(names) > 1:
                        choice = augmentation.choose_combined_operations(
                            names, scenario, sample_id, round_index, args.seed, augmentation_distractors)
                        make_images = augmentation.combined_operations_images
                    else:
                        choice = augmentation.choose_single_augmentation(
                            names[0], scenario, sample_id, round_index, args.seed, augmentation_distractors)
                        make_images = augmentation.single_augmented_images
                    if choice["swap"]:
                        raise RuntimeError("Labeled augmentation must not swap A and B.")
                    distractor = (
                        augmentation_distractor_path(choice["distractor_index"])
                        if choice.get("distractor_index") is not None else None
                    )
                    source, image_a, image_b = make_images(
                        choice, human_record.source_image, human_record.image_a, human_record.image_b,
                        distractor, augmentation_work_dir, f"labeled_round{round_index + 1}_{sample_id}",
                    )
                    created += [path for path in (source, image_a, image_b)
                                if path.startswith(str(augmentation_work_dir))]
                    changes = dict(source_image=source, edit_instruction=choice["instruction"],
                                   image_a=image_a, image_b=image_b)
                    new_human.append(replace(human_record, **changes))
                    new_gemma.append(replace(gemma_record, **changes))
                    labeled_augmentation_info[sample_id] = {
                        "augmentation_operation": choice["operation"],
                        "augmentation_applied": bool(choice["applied"]),
                        "augmentation_applied_operations": choice.get("applied_operations"),
                        "augmentation_not_applied_reason": choice["not_applied_reason"],
                        "augmentation_strength": choice["strength"],
                        "augmentation_position": choice.get("position"),
                        "augmentation_distractor": (
                            augmentation_distractors[choice["distractor_index"]]["sample_id"]
                            if choice.get("distractor_index") is not None else None
                        ),
                        "original_instruction": human_record.edit_instruction,
                        "student_instruction": choice["instruction"],
                    }
                return new_human, new_gemma, created

            def unlabeled_microbatch_loss(unlabeled_indices, local_indices):
                unlabeled_batch = [round_records[index] for index in unlabeled_indices]
                swapped: List[bool] = []
                created: List[str] = []
                if augmentation is not None:
                    unlabeled_batch, swapped, created = augment_for_student(unlabeled_indices)
                with torch.autocast(
                    device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                ):
                    unlabeled_output = timed_forward(unlabeled_batch)
                for path in created:
                    # The processor has read the images; the composites are not needed again.
                    Path(path).unlink(missing_ok=True)
                if any(swapped):
                    unlabeled_output = unswap_student_output(unlabeled_output, swapped)
                if args.unlabeled_objective == "projection":
                    loss_set, set_stats = feasible_set_loss(
                        unlabeled_output.three_way_probs,
                        teacher_cache,
                        local_indices,
                        aggregator_weights_cpu,
                        args.projection_iterations,
                        args.box_only_set_weight,
                        device,
                    )
                else:
                    loss_set, set_stats = inspector_supervision_loss(
                        unlabeled_output,
                        teacher_cache,
                        local_indices,
                        aggregator_weights_cpu,
                        aggregator_beta,
                        args.lambda_consistency
                        if args.unlabeled_objective in (
                            "inspector_ema_consistency",
                            "inspector_soft_ema_consistency",
                        )
                        else 0.0,
                        device,
                        inspector_pseudo_labels.probability_a[unlabeled_indices]
                        if args.unlabeled_objective == "inspector_soft_ema_consistency"
                        and args.overall_loss != "three_way"
                        else None,
                        args.overall_aggregation,
                        args.overall_probability,
                        args.consistency_metrics,
                        inspector_target_of(unlabeled_indices)
                        if args.overall_loss == "three_way"
                        else None,
                        relevance_weight=args.teacher_kl_relevance_weight,
                    )
                if not torch.isfinite(loss_set):
                    raise RuntimeError("Unlabeled feasible-set loss became non-finite.")
                return loss_set, set_stats, unlabeled_output, len(unlabeled_batch)

            def record_unlabeled_diagnostics(unlabeled_indices, local_indices, set_stats) -> None:
                round_totals[0] += set_stats["valid"]
                round_totals[1] += set_stats["inside"]
                round_totals[2] += set_stats["projected"]
                for position, local_index in enumerate(local_indices):
                    pool_index = unlabeled_indices[position]
                    values = sample_diagnostics[pool_index]
                    if bool(set_stats["valid_mask"][position]):
                        trained_unlabeled_indices.add(pool_index)
                    values["inside"] = bool(set_stats["inside_mask"][position])
                    values["projected"] = bool(
                        set_stats["valid_mask"][position]
                        and not set_stats["inside_mask"][position]
                    )
                    values["projection_kl"] = float(
                        set_stats["projection_kl"][position]
                    )

            round_totals = [0, 0, 0]
            if args.labeled_source == "merged":
                labeled_microbatch_count = args.labeled_pairs_per_step // args.human_batch_size
                # per_pair: divide every pair by the labeled pairs of the step
                # (by the Unlabeled pairs when the step has no labeled pairs).
                pair_normalizer = args.labeled_pairs_per_step or step_unlabeled
                step_total_loss = 0.0
                step_sums = Counter()
                # Per-step monitor of the per-sample log (Student label vs Teacher / Inspector).
                step_agreement = Counter()
                for _ in range(labeled_microbatch_count):
                    pairs = labeled_sampler.draw(args.human_batch_size)
                    human_batch = [human_record for human_record, _ in pairs]
                    gemma_batch = [gemma_record for _, gemma_record in pairs]
                    labeled_created: List[str] = []
                    if args.augment_labeled:
                        human_batch, gemma_batch, labeled_created = augment_labeled_pairs(human_batch, gemma_batch)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                    ):
                        labeled_output = timed_forward(human_batch)
                    for path in labeled_created:
                        Path(path).unlink(missing_ok=True)
                    loss_human = human_loss_of(labeled_output, human_batch)
                    gemma_losses = gemma_replay_losses(
                        labeled_output, gemma_batch, gemma_objective, device
                    )
                    loss_gemma = gemma_losses["preference_loss"]
                    loss_mask = gemma_losses["relevance_loss"]
                    labeled_loss = (
                        args.lambda_human * loss_human
                        + args.lambda_gemma * gemma_replay_scale * loss_gemma
                        + args.lambda_mask * mask_replay_scale * loss_mask
                    )
                    if not torch.isfinite(labeled_loss):
                        raise RuntimeError("Labeled loss became non-finite.")
                    if per_sample_file is not None:
                        labeled_rows = per_sample_labeled_rows(
                            global_optimizer_step + 1, labeled_output, human_batch,
                            aggregator_weights, args.overall_aggregation, gemma_losses,
                        )
                        if args.augment_labeled:
                            for row in labeled_rows:
                                row.update(labeled_augmentation_info.get(row["sample_id"], {}))
                        write_per_sample(labeled_rows)
                    weight = (
                        len(pairs) / pair_normalizer
                        if args.loss_weighting == "per_pair"
                        else 1.0 / labeled_microbatch_count
                    )
                    backward_scaled(labeled_loss * weight)
                    step_total_loss += float(labeled_loss.detach()) * weight
                    step_sums["labeled_microbatches"] += 1
                    step_sums["human"] += float(loss_human.detach())
                    step_sums["gemma"] += float(loss_gemma.detach())
                    step_sums["mask"] += float(loss_mask.detach())
                    step_sums["tie"] += float(gemma_losses["tie_loss"].detach())
                    step_sums["winner"] += float(gemma_losses["conditional_winner_loss"].detach())
                    loss_totals["human"] += float(loss_human.detach())
                    loss_totals["gemma"] += float(loss_gemma.detach())
                    loss_totals["mask"] += float(loss_mask.detach())
                    loss_totals["tie"] += float(gemma_losses["tie_loss"].detach())
                    loss_totals["winner"] += float(gemma_losses["conditional_winner_loss"].detach())
                    loss_totals["labeled_microbatches"] += 1
                    loss_totals["human_records"] += len(human_batch)
                    loss_totals["gemma_records"] += len(gemma_batch)
                    del labeled_output, gemma_losses
                labeled_gradients = (
                    cpu_float_copy([parameter.grad for _, _, parameter in diagnostic_parameters])
                    if args.step_diagnostics and step_sums["labeled_microbatches"] > 0
                    else None
                )
                for unlabeled_indices, local_indices in valid_group:
                    loss_set, set_stats, unlabeled_output, unlabeled_count = unlabeled_microbatch_loss(
                        unlabeled_indices, local_indices
                    )
                    if per_sample_file is not None:
                        unlabeled_rows = per_sample_unlabeled_rows(
                            global_optimizer_step + 1, unlabeled_output, teacher_cache, local_indices,
                            inspector_target_of(unlabeled_indices),
                            aggregator_weights_cpu, args.overall_aggregation, args.consistency_metrics,
                        )
                        if args.inspector_target == "one_hot":
                            # "inspector" / its CE and KL above are the one-hot target used by the loss.
                            soft_rows = inspector_pseudo_labels.three_way[unlabeled_indices].detach().float().cpu()
                            for row, soft in zip(unlabeled_rows, soft_rows):
                                row["inspector_target"] = "one_hot"
                                row["inspector_soft"] = [round(float(v), 6) for v in soft]
                        for row, pool_index in zip(unlabeled_rows, unlabeled_indices):
                            row.update(augmentation_info.pop(pool_index, {}))
                            last_training_rows[row["sample_id"]] = row
                            example = augmentation_example_rows.pop(row["sample_id"], None)
                            if example is not None:
                                # Labels of this Student forward (outputs swapped back, so A / B are
                                # the original images) and of the Teacher / Inspector on the original.
                                example.update({"step": row["step"], **{key: row[key] for key in (
                                    "student_label", "teacher_label", "inspector_label",
                                    "student_same_as_teacher", "student_same_as_inspector",
                                    "teacher_same_as_inspector", "student", "teacher", "inspector",
                                    "teacher_kl", "inspector_kl")}, "labels_refer_to_original_A_B": True})
                                write_augmentation_example(example)
                            step_agreement["unlabeled"] += 1
                            step_agreement["student_same_as_teacher"] += int(row["student_same_as_teacher"])
                            step_agreement["student_same_as_inspector"] += int(row["student_same_as_inspector"])
                            step_agreement["teacher_kl_sum"] += row["teacher_kl"]
                            step_agreement["inspector_kl_sum"] += row["inspector_kl"]
                            step_agreement["teacher_kl_relevance_sum"] += row["teacher_kl_relevance"]
                            step_agreement["selected_same"] += int(row["student_selected_same_as_teacher_selected"])
                        write_per_sample(unlabeled_rows)
                    step_sums["unlabeled_microbatches"] += 1
                    step_sums["unlabeled_records"] += unlabeled_count
                    step_sums["set"] += float(loss_set.detach())
                    for key in (
                        "inspector_cross_entropy_sum",
                        "inspector_target_entropy_sum",
                        "teacher_consistency_sum",
                        "teacher_preference_kl_sum",
                        "teacher_relevance_kl_sum",
                    ):
                        step_sums[key] += float(set_stats.get(key, 0.0))
                    weight = (
                        unlabeled_count / pair_normalizer
                        if args.loss_weighting == "per_pair"
                        else 1.0 / len(valid_group)
                    )
                    backward_scaled(lambda_set_now * loss_set * weight)
                    del unlabeled_output
                    step_total_loss += float(loss_set.detach()) * lambda_set_now * weight
                    loss_totals["set"] += float(loss_set.detach())
                    loss_totals["unlabeled_microbatches"] += 1
                    loss_totals["unlabeled_records"] += unlabeled_count
                    record_unlabeled_diagnostics(unlabeled_indices, local_indices, set_stats)
                loss_totals["total"] += step_total_loss
                loss_totals["total_count"] += 1
            else:
                for unlabeled_indices, local_indices in valid_group:
                    human_batch = human_sampler.draw(args.human_batch_size)
                    gemma_batch = gemma_sampler.draw(args.gemma_batch_size)
                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                    ):
                        human_output = timed_forward(human_batch)
                    loss_human = human_loss_of(human_output, human_batch)
                    backward_scaled(args.lambda_human * loss_human / len(valid_group))
                    del human_output

                    with torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16, enabled=autocast_enabled
                    ):
                        gemma_output = timed_forward(gemma_batch)
                    gemma_losses = gemma_replay_losses(
                        gemma_output, gemma_batch, gemma_objective, device
                    )
                    loss_gemma = gemma_losses["preference_loss"]
                    loss_mask = gemma_losses["relevance_loss"]
                    weighted_gemma = (
                        args.lambda_gemma * gemma_replay_scale * loss_gemma
                        + args.lambda_mask * mask_replay_scale * loss_mask
                    )
                    tie_loss_value = float(gemma_losses["tie_loss"].detach())
                    winner_loss_value = float(
                        gemma_losses["conditional_winner_loss"].detach()
                    )
                    if not torch.isfinite(weighted_gemma):
                        raise RuntimeError("GemmaGT replay loss became non-finite.")
                    backward_scaled(weighted_gemma / len(valid_group))
                    del gemma_output, gemma_losses

                    loss_set, set_stats, unlabeled_output, unlabeled_count = unlabeled_microbatch_loss(
                        unlabeled_indices, local_indices
                    )
                    backward_scaled(lambda_set_now * loss_set / len(valid_group))
                    del unlabeled_output

                    loss_totals["human"] += float(loss_human.detach())
                    loss_totals["gemma"] += float(loss_gemma.detach())
                    loss_totals["mask"] += float(loss_mask.detach())
                    loss_totals["tie"] += tie_loss_value
                    loss_totals["winner"] += winner_loss_value
                    loss_totals["set"] += float(loss_set.detach())
                    loss_totals["total"] += float(
                        (args.lambda_human * loss_human + weighted_gemma + lambda_set_now * loss_set).detach()
                    )
                    loss_totals["total_count"] += 1
                    loss_totals["microbatches"] += 1
                    loss_totals["labeled_microbatches"] += 1
                    loss_totals["unlabeled_microbatches"] += 1
                    loss_totals["human_records"] += len(human_batch)
                    loss_totals["gemma_records"] += len(gemma_batch)
                    loss_totals["unlabeled_records"] += unlabeled_count
                    record_unlabeled_diagnostics(unlabeled_indices, local_indices, set_stats)
            round_valid += round_totals[0]
            round_inside += round_totals[1]
            round_projected += round_totals[2]

            del teacher_cache
            optimizer_started = synchronized_time()
            step_timing["student"] += optimizer_started - student_started
            if global_optimizer_step == 0:
                print(
                    f"First-step gradient check: {trainable_gradient_report(model)}",
                    flush=True,
                )
            step_diagnostics = None
            step_record: Dict[str, Any] = {}
            if args.step_diagnostics:
                step_diagnostics = {
                    "gradient_norm_before_clipping": gradient_source_report(
                        diagnostic_parameters, labeled_gradients
                    )
                }
                del labeled_gradients
                parameters_before_step = cpu_float_copy(
                    [parameter for _, _, parameter in diagnostic_parameters]
                )
            if swift_trainer is None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm, error_if_nonfinite=True
                )
            else:
                grad_norm = swift_trainer.accelerator.clip_grad_norm_(
                    trainable_parameters, args.max_grad_norm
                )
            if grad_norm is not None:
                grad_norm_window.append(float(grad_norm))
            learning_rates_used = [group["lr"] for group in optimizer.param_groups]
            optimizer.step()
            if step_diagnostics is not None:
                labeled_count = max(1, step_sums["labeled_microbatches"])
                unlabeled_records = max(1, step_sums["unlabeled_records"])
                step_diagnostics["update_ratio"] = update_ratio_report(
                    diagnostic_parameters, parameters_before_step
                )
                del parameters_before_step
                step_diagnostics["losses"] = {
                    "labeled_microbatches": int(step_sums["labeled_microbatches"]),
                    "human": step_sums["human"] / labeled_count if step_sums["labeled_microbatches"] else None,
                    "gemma": step_sums["gemma"] / labeled_count if step_sums["labeled_microbatches"] else None,
                    "mask": step_sums["mask"] / labeled_count if step_sums["labeled_microbatches"] else None,
                    "tie": step_sums["tie"] / labeled_count if step_sums["labeled_microbatches"] else None,
                    "winner": step_sums["winner"] / labeled_count if step_sums["labeled_microbatches"] else None,
                    "unlabeled_records": int(step_sums["unlabeled_records"]),
                    "unlabeled": step_sums["set"] / max(1, step_sums["unlabeled_microbatches"]),
                    "unlabeled_inspector_cross_entropy": step_sums["inspector_cross_entropy_sum"] / unlabeled_records,
                    "unlabeled_inspector_target_entropy": step_sums["inspector_target_entropy_sum"] / unlabeled_records,
                    "unlabeled_inspector_kl": (
                        step_sums["inspector_cross_entropy_sum"] - step_sums["inspector_target_entropy_sum"]
                    ) / unlabeled_records,
                    # = the whole Teacher consistency term of the loss (before lambda_C); equals
                    # KL_pref when --teacher-kl-relevance-weight is 0.
                    "unlabeled_teacher_kl": step_sums["teacher_consistency_sum"] / unlabeled_records,
                    "unlabeled_teacher_kl_preference": step_sums["teacher_preference_kl_sum"] / unlabeled_records,
                    "unlabeled_teacher_kl_relevance": step_sums["teacher_relevance_kl_sum"] / unlabeled_records,
                    "teacher_kl_relevance_weight": args.teacher_kl_relevance_weight,
                }
                step_diagnostics["learning_rates"] = learning_rates_used
                step_diagnostics["lambda_unlabeled"] = lambda_set_now
                step_diagnostics["clip_coefficient"] = (
                    min(1.0, args.max_grad_norm / (float(grad_norm) + 1e-6))
                    if grad_norm is not None else None
                )
                print(
                    f"step={global_optimizer_step + 1} diagnostics {json.dumps(step_diagnostics)}",
                    flush=True,
                )
                step_record["diagnostics"] = step_diagnostics
                if per_sample_file is not None and step_agreement["unlabeled"]:
                    count = step_agreement["unlabeled"]
                    agreement_row = {
                        "unlabeled": count,
                        "student_same_as_teacher": step_agreement["student_same_as_teacher"] / count,
                        "student_same_as_inspector": step_agreement["student_same_as_inspector"] / count,
                        "teacher_kl_mean": step_agreement["teacher_kl_sum"] / count,
                        "inspector_kl_mean": step_agreement["inspector_kl_sum"] / count,
                        "teacher_kl_relevance_mean": step_agreement["teacher_kl_relevance_sum"] / count,
                        "student_selected_same_as_teacher_selected": step_agreement["selected_same"] / count,
                    }
                    print(
                        f"step={global_optimizer_step + 1} per_sample_agreement "
                        + json.dumps(agreement_row),
                        flush=True,
                    )
                    step_record["agreement"] = agreement_row
            scheduler.step()
            if args.teacher_update == "optimizer_step":
                ema_decay_now = ema_decay_at(global_optimizer_step, total_optimizer_steps, args)
                update_teacher_ema(model, teacher_state, ema_decay_now)
            else:
                # Teacher fixed inside the loop: logged as decay 1 (Teacher fully kept).
                ema_decay_now = 1.0
            step_timing["optimizer"] += synchronized_time() - optimizer_started
            step_timing["steps"] += 1
            global_optimizer_step += 1
            if swift_trainer is not None:
                swift_trainer.state.global_step = global_optimizer_step
            if global_optimizer_step % args.logging_steps == 0:
                labeled_denominator = max(1, loss_totals["labeled_microbatches"])
                unlabeled_denominator = max(1, loss_totals["unlabeled_microbatches"])
                print(
                    f"round={round_index + 1} step={global_optimizer_step} "
                    f"human={loss_totals['human']/labeled_denominator:.5f} "
                    f"gemma={loss_totals['gemma']/labeled_denominator:.5f} "
                    f"mask={loss_totals['mask']/labeled_denominator:.5f} "
                    f"set={loss_totals['set']/unlabeled_denominator:.5f} "
                    f"lambda_set={lambda_set_now:.4f} set_scale={set_scale:.4f} "
                    f"ema_decay={ema_decay_now:.5f} "
                    f"statuses={dict(status_counts)}",
                    flush=True,
                )
                if grad_norm_window:
                    clipped = sum(norm > args.max_grad_norm for norm in grad_norm_window)
                    print(
                        f"step={global_optimizer_step} grad_norm_before_clipping "
                        f"(last {len(grad_norm_window)} steps): "
                        f"mean={sum(grad_norm_window) / len(grad_norm_window):.4f} "
                        f"min={min(grad_norm_window):.4f} max={max(grad_norm_window):.4f} "
                        f"clipped={clipped}/{len(grad_norm_window)} "
                        f"(max_grad_norm={args.max_grad_norm})",
                        flush=True,
                    )
                    grad_norm_window.clear()
                timed_steps = max(1, step_timing["steps"])
                peak_memory = (
                    torch.cuda.max_memory_allocated(device) / 2**30
                    if device.type == "cuda" else 0.0
                )
                print(
                    f"step={global_optimizer_step} seconds_per_step "
                    f"teacher={step_timing['teacher'] / timed_steps:.2f} "
                    f"prepare={step_timing['prepare'] / timed_steps:.2f} "
                    f"student_including_prepare={step_timing['student'] / timed_steps:.2f} "
                    f"optimizer_and_ema={step_timing['optimizer'] / timed_steps:.2f} "
                    f"peak_memory_gib={peak_memory:.2f}",
                    flush=True,
                )
                step_record["seconds_per_step"] = {
                    key: step_timing[key] / timed_steps
                    for key in ("teacher", "prepare", "student", "optimizer")
                }
                step_record["peak_memory_gib"] = peak_memory
                step_timing.clear()
            write_step_log({
                "kind": "step",
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "round": round_index + 1,
                "step": global_optimizer_step,
                "gradient_norm_before_clipping": None if grad_norm is None else float(grad_norm),
                "max_grad_norm": args.max_grad_norm,
                "learning_rates": learning_rates_used,
                "lambda_unlabeled": lambda_set_now,
                "ema_decay": ema_decay_now,
                "teacher_statuses": dict(status_counts),
                **step_record,
            })
            if (
                args.eval_save_steps and global_optimizer_step % args.eval_save_steps == 0
            ) or global_optimizer_step in round_save_steps:
                step_validation, _ = evaluate_on_validation(
                    model, processor, human_validation, prompt,
                    teacher_state if args.selection_model == "teacher" else None,
                    aggregator_weights, aggregator_beta, args.eval_batch_size, device,
                    gemma_validation, args.overall_aggregation, args.overall_probability,
                )
                step_guard = all(
                    step_validation[name]
                    <= selection_baseline_metrics[name] + args.semantic_drift_threshold
                    for name in (
                        "gemma_tie_positive_brier",
                        "gemma_winner_brier",
                        "gemma_mask_brier",
                    )
                )
                write_step_log({
                    "kind": "validation",
                    "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "round": round_index + 1,
                    "step": global_optimizer_step,
                    "semantic_guard": bool(step_guard),
                    "validation": step_validation,
                })
                step_improved = args.checkpoint_selection == "last" or (
                    step_guard
                    and step_validation["accuracy"]
                    > best_metrics["accuracy"] + args.min_validation_improvement
                )
                if step_improved:
                    round_improved_by_step = True
                    best_round = round_index + 1
                    best_optimizer_step = global_optimizer_step
                    best_metrics = step_validation
                    # Only the selection model is evaluated at a step.
                    if args.selection_model == "student":
                        best_student_metrics = step_validation
                        best_teacher_metrics = None
                    else:
                        best_teacher_metrics = step_validation
                        best_student_metrics = None
                    best_student_state = capture_adapter_state_cpu(model)
                    best_teacher_state = {
                        name: tensor.detach().cpu().clone()
                        for name, tensor in teacher_state.items()
                    }
                    best_optimizer_state = recursive_to_cpu(optimizer.state_dict())
                    best_scheduler_state = scheduler.state_dict()
                    best_rng_state = {
                        "python": random.getstate(),
                        "numpy": np.random.get_state(),
                        "torch_cpu": torch.get_rng_state(),
                        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                    }
                step_path = output_path.parent / f"checkpoint-{global_optimizer_step}.pt"
                save_step_checkpoint(
                    step_path, model, teacher_state, base_model_path, peft_config,
                    prompt, aggregator_record, global_optimizer_step, step_validation,
                    step_improved,
                )
                if step_improved:
                    best_step_checkpoint = step_path
                step_checkpoint_paths.append(step_path)
                if teacher_loop_start_state is not None:
                    print(
                        f"Step {global_optimizer_step} Teacher max absolute change since loop "
                        f"start: {teacher_max_absolute_change(teacher_state, teacher_loop_start_state):.3e}",
                        flush=True,
                    )
                if visual_start_state is not None:
                    parameters_by_name = dict(model.named_parameters())
                    visual_change = max(
                        float((parameters_by_name[name].detach().cpu() - start).abs().max())
                        for name, start in visual_start_state.items()
                    )
                    print(
                        f"Step {global_optimizer_step} visual LoRA max absolute change since "
                        f"Student-Stage1: {visual_change:.3e}",
                        flush=True,
                    )
                removable =[path for path in step_checkpoint_paths if path != best_step_checkpoint]
                while len(removable) > args.eval_save_total_limit:
                    oldest = removable.pop(0)
                    oldest.unlink()
                    step_checkpoint_paths.remove(oldest)
                print(
                    f"Step {global_optimizer_step} {args.selection_model} validation: "
                    f"accuracy={step_validation['accuracy']:.4f} bce={step_validation['bce']:.4f} "
                    f"semantic_guard={step_guard} improved={step_improved} "
                    f"best_step={best_optimizer_step} "
                    f"best_accuracy={best_metrics['accuracy']:.4f} saved={step_path.name}",
                    flush=True,
                )

        del round_records
        if not trained_unlabeled_indices:
            raise RuntimeError(
                "Every selected Unlabeled sample became infeasible under the "
                "current EMA Teacher; refusing to report an empty training round."
            )
        untrained_selected = set(selected_indices) - trained_unlabeled_indices
        for index in untrained_selected:
            classification = screening_cache[unlabeled_pool.sample_ids[index]]["classification"]
            if classification in ("hard_replay", "previous_round_replay"):
                unlabeled_sampler.replay_counts[index] -= 1
            else:
                # A sample screened with one pass may fail fresh J=4
                # feasibility; it remains available for a later EMA Teacher.
                unlabeled_sampler.remaining.add(index)
        probe_report = None
        if probe_indices:
            probe_report = evaluate_constraint_probe(
                model,
                processor,
                probe_records,
                prompt,
                teacher_state,
                inspector_pseudo_labels.labels[probe_indices],
                aggregator_weights_cpu,
                args,
                device,
                args.seed + 50_000,
            )
            probe_report["delta_from_previous"] = compare_probe_reports(
                probe_report, previous_probe
            )
            probe_report["delta_from_baseline"] = compare_probe_reports(
                probe_report, baseline_probe
            )
        validation, teacher_probabilities = evaluate_on_validation(
            model, processor, human_validation, prompt, teacher_state,
            aggregator_weights, aggregator_beta, args.eval_batch_size, device,
            gemma_validation, args.overall_aggregation, args.overall_probability,
        )
        student_validation, student_probabilities = evaluate_on_validation(
            model, processor, human_validation, prompt, None,
            aggregator_weights, aggregator_beta, args.eval_batch_size, device,
            gemma_validation, args.overall_aggregation, args.overall_probability,
        )
        loss_denominators = {
            "human": "labeled_microbatches", "gemma": "labeled_microbatches",
            "tie": "labeled_microbatches", "winner": "labeled_microbatches",
            "mask": "labeled_microbatches", "set": "unlabeled_microbatches",
            "total": "total_count",
        }
        trained_mix = Counter(
            screening_cache[unlabeled_pool.sample_ids[index]]["classification"]
            for index in trained_unlabeled_indices
        )
        report = {
            "round": round_index + 1,
            "selected_unlabeled": len(selected_indices),
            "requested_unlabeled": requested_unlabeled,
            "quality_gated_minimum_satisfied": len(selected_indices) >= min_selected,
            "selected_sample_ids": [unlabeled_pool.sample_ids[index] for index in selected_indices],
            "trained_unlabeled": len(trained_unlabeled_indices),
            "selected_but_not_trained": len(untrained_selected),
            "trained_mix": dict(trained_mix),
            "trained_sample_ids": [
                unlabeled_pool.sample_ids[index]
                for index in sorted(trained_unlabeled_indices)
            ],
            "remaining_unlabeled": len(unlabeled_sampler.remaining),
            # Merged labeled source: sample_id -> times drawn in this round.
            "trained_labeled_sample_ids": (
                dict(Counter(
                    labeled_sampler.records[index][0].sample_id
                    for index in labeled_sampler.take_drawn()
                ))
                if labeled_sampler is not None
                else {}
            ),
            "selection_mix": {
                "fresh_high_agreement": selection.fresh_high_agreement,
                "fresh_low_agreement": selection.fresh_low_agreement,
                "fresh_box_only": selection.fresh_box_only,
                "hard_replay": selection.hard_replay,
                "screened": selection.screened,
                "screening_status": selection.screening_status,
            },
            "optimizer_step": global_optimizer_step,
            "training_pair_counts": {
                "human": loss_totals["human_records"],
                "gemma": loss_totals["gemma_records"],
                "unlabeled": loss_totals["unlabeled_records"],
            },
            "losses": {
                name: loss_totals[name] / max(1, loss_totals[loss_denominators[name]])
                for name in ("human", "gemma", "tie", "winner", "mask", "set", "total")
            },
            "valid_unlabeled_occurrences": round_valid,
            "inside_occurrences": round_inside,
            "projected_occurrences": round_projected,
            "constraint_status": dict(status_counts),
            "sample_status": sample_status,
            "hard_sample_diagnostics": {
                unlabeled_pool.sample_ids[index]: values
                for index, values in sample_diagnostics.items()
            },
            "gemma_replay_scale": gemma_replay_scale,
            "mask_replay_scale": mask_replay_scale,
            "set_loss_scale": set_scale,
            "teacher_uncertainty": {
                name: uncertainty_totals[name] / len(selected_indices)
                for name in ("box_width", "winner_variance", "tie_variance", "relevance_variance")
            },
            "fixed_probe": probe_report,
            "validation": validation,
            "student_validation": student_validation,
            "validation_predictions": {
                "sample_ids": human_validation_sample_ids,
                "teacher_probability_A": teacher_probabilities,
                "student_probability_A": student_probabilities,
            },
            "learning_rate": scheduler.get_last_lr()[0],
        }
        history.append(report)
        print(
            f"Completed round {round_index + 1}: "
            f"losses={report['losses']} statuses={report['constraint_status']} "
            + (
                f"probe_width={probe_report['mean_box_width']:.6f} "
                f"probe_width_delta={probe_report['delta_from_previous']['mean_box_width_delta']:.6f} "
                f"probe_inside={probe_report['inside_rate']:.4f} "
                if probe_report is not None
                else "probe=none "
            )
            + f"validation={validation} remaining={report['remaining_unlabeled']}",
            flush=True,
        )
        print(f"Round {round_index + 1} Student validation: {student_validation}", flush=True)
        selected_validation = (
            student_validation if args.selection_model == "student" else validation
        )
        semantic_guard = all(
            selected_validation[name]
            <= selection_baseline_metrics[name] + args.semantic_drift_threshold
            for name in (
                "gemma_tie_positive_brier",
                "gemma_winner_brier",
                "gemma_mask_brier",
            )
        )
        report["semantic_guard_passed"] = semantic_guard
        improved = args.checkpoint_selection == "last" or (
            semantic_guard
            and selected_validation["accuracy"]
            > best_metrics["accuracy"] + args.min_validation_improvement
        )
        if improved:
            best_round = round_index + 1
            best_optimizer_step = global_optimizer_step
            best_metrics = selected_validation
            best_teacher_metrics = validation
            best_student_metrics = student_validation
            best_student_state = capture_adapter_state_cpu(model)
            best_teacher_state = {
                name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()
            }
            best_optimizer_state = recursive_to_cpu(optimizer.state_dict())
            best_scheduler_state = scheduler.state_dict()
            best_rng_state = {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            rounds_without_improvement = 0
        elif round_improved_by_step:
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        preference_drift = any(
            validation[name]
            > baseline_metrics[name] + args.semantic_drift_threshold
            for name in (
                "gemma_tie_brier",
                "gemma_tie_positive_brier",
                "gemma_winner_brier",
            )
        )
        mask_drift = (
            validation["gemma_mask_brier"]
            > baseline_metrics["gemma_mask_brier"] + args.semantic_drift_threshold
        )
        gemma_replay_scale = 1.5 if preference_drift else 1.0
        mask_replay_scale = 1.5 if mask_drift else 1.0
        if args.unlabeled_objective == "projection":
            set_scale = adaptive_set_scale(report, set_scale, args)
        if args.unlabeled_sampling == "inspector_teacher_rank":
            # Ranked by the Teacher margin at the start of the next round.
            hard_replay_candidates = sorted(trained_unlabeled_indices)
        elif args.replay_strategy == "hard":
            hard_replay_candidates = rank_hard_samples(sample_diagnostics)
        else:
            # Uniform random order over every sample trained in this round.
            hard_replay_candidates = sorted(trained_unlabeled_indices)
            random.Random(args.seed + 50_000 + round_index).shuffle(hard_replay_candidates)
        previous_probe = probe_report
        if teacher_loop_start_state is not None:
            print(
                f"Round {round_index + 1} Teacher max absolute change inside the loop: "
                f"{teacher_max_absolute_change(teacher_state, teacher_loop_start_state):.3e}",
                flush=True,
            )
            # Original design: the EMA Teacher is updated once per loop, not per step.
            update_teacher_ema(model, teacher_state, args.ema_decay)
            teacher_loop_start_state = {
                name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()
            }
            print(
                f"Round {round_index + 1} Teacher updated after the loop with EMA decay "
                f"{args.ema_decay}",
                flush=True,
            )
            # The round-end step checkpoint and the best-state snapshot were taken before this
            # update; refresh both so exported Teachers include every completed EMA update.
            validation, teacher_probabilities = evaluate_on_validation(
                model, processor, human_validation, prompt, teacher_state,
                aggregator_weights, aggregator_beta, args.eval_batch_size, device,
                gemma_validation, args.overall_aggregation, args.overall_probability,
            )
            report["validation"] = validation
            report["validation_predictions"]["teacher_probability_A"] = teacher_probabilities
            if best_round == round_index + 1 and best_optimizer_step == global_optimizer_step:
                best_teacher_state = {
                    name: tensor.detach().cpu().clone() for name, tensor in teacher_state.items()
                }
                best_teacher_metrics = validation
            if global_optimizer_step in round_save_steps:
                save_step_checkpoint(
                    output_path.parent / f"checkpoint-{global_optimizer_step}.pt",
                    model, teacher_state, base_model_path, peft_config, prompt,
                    aggregator_record, global_optimizer_step, selected_validation, improved,
                    overwrite=True,
                )
        if args.unlabeled_sampling == ITERATIVE_SAMPLING:
            # The Teacher after this iteration's EMA updates scores every sample of the
            # iteration on the original input (one pass, eval mode, no dropout); only the
            # samples whose Teacher label differs from the Inspector label are left out.
            scored_indices = list(dict.fromkeys(selected_indices))
            scored_records = materialize_unlabeled_round(unlabeled_pool, scored_indices)
            scoring_started = time.perf_counter()
            with teacher_screen_session(model, teacher_state):
                teacher_metric, teacher_relevance = teacher_screen_batch(
                    model, processor, [scored_records[index] for index in scored_indices],
                    prompt, args.eval_batch_size, device,
                )
            weights_cpu = None if aggregator_weights_cpu is None else aggregator_weights_cpu.detach().float().cpu()
            teacher_overall = (
                teacher_metric
                * aggregation_weights_from_relevance(
                    teacher_relevance, weights_cpu, args.overall_aggregation
                ).unsqueeze(-1)
            ).sum(dim=1)
            inspector_overall = inspector_pseudo_labels.three_way[scored_indices].detach().float().cpu()
            kept_indices: List[int] = []
            selection_rows: List[Dict[str, Any]] = []
            kept_certainty: List[float] = []
            for position, pool_index in enumerate(scored_indices):
                sample_id = unlabeled_pool.sample_ids[pool_index]
                teacher_probabilities = teacher_overall[position]
                inspector_probabilities = inspector_overall[position]
                teacher_label = _three_way_label(teacher_probabilities)
                inspector_label = _three_way_label(inspector_probabilities)
                same = teacher_label == inspector_label
                inspector_class = THREE_WAY_CLASSES.index(inspector_label)
                certainty = (
                    min(float(teacher_probabilities[inspector_class]), float(inspector_probabilities[inspector_class]))
                    if same else None
                )
                training_row = last_training_rows.get(sample_id, {})
                if same:
                    kept_indices.append(pool_index)
                    kept_certainty.append(certainty)
                selection_rows.append({
                    "iteration": round_index + 1,
                    "optimizer_step": global_optimizer_step,
                    "sample_id": sample_id,
                    "teacher": _rounded(teacher_probabilities),
                    "teacher_label": teacher_label,
                    "teacher_largest_probability": round(float(teacher_probabilities.max()), 6),
                    "teacher_probability_of_inspector_label": round(float(teacher_probabilities[inspector_class]), 6),
                    "inspector": _rounded(inspector_probabilities),
                    "inspector_label": inspector_label,
                    "inspector_largest_probability": round(float(inspector_probabilities.max()), 6),
                    "teacher_same_as_inspector": same,
                    "certainty_shared_label_min_teacher_inspector": None if certainty is None else round(certainty, 6),
                    "teacher_selected_metrics": int((selected_metric_weights(teacher_relevance[position : position + 1]) > 0).sum()),
                    "last_training_step": training_row.get("step"),
                    "last_training_student_augmented": training_row.get("student"),
                    "last_training_student_label": training_row.get("student_label"),
                    "last_training_teacher_label": training_row.get("teacher_label"),
                    "last_training_student_same_as_teacher": training_row.get("student_same_as_teacher"),
                    "last_training_student_same_as_inspector": training_row.get("student_same_as_inspector"),
                    "kept_for_next_iteration": same,
                    "reason": "teacher_same_as_inspector" if same else "teacher_differs_from_inspector",
                })
            selection_path = output_path.parent / f"iteration_selection_round_{round_index + 1:02d}.jsonl"
            with selection_path.open("x", encoding="utf-8") as handle:
                for row in selection_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            current_losses = {name: float(report["losses"][name]) for name in ("set", "human")}
            relative_decrease = (
                {
                    name: (previous_iteration_losses[name] - current_losses[name]) / previous_iteration_losses[name]
                    if previous_iteration_losses[name] > 0 else None
                    for name in current_losses
                }
                if previous_iteration_losses is not None else None
            )
            loss_plateau = relative_decrease is not None and all(
                value is not None and value < args.loss_plateau_tolerance
                for value in relative_decrease.values()
            )
            last_iteration = round_index + 1 == args.max_rounds
            if not kept_indices:
                stop_after_round, stop_reason = True, "no_samples_left_teacher_differs_from_inspector"
            elif loss_plateau:
                stop_after_round, stop_reason = True, "loss_plateau"
            sorted_certainty = sorted(kept_certainty)
            iteration_summary = {
                "iteration": round_index + 1,
                "optimizer_step": global_optimizer_step,
                "scored": len(scored_indices),
                "kept_teacher_same_as_inspector": len(kept_indices),
                "left_out_teacher_differs_from_inspector": len(scored_indices) - len(kept_indices),
                "teacher_same_as_inspector_rate": len(kept_indices) / max(1, len(scored_indices)),
                "teacher_labels": dict(Counter(row["teacher_label"] for row in selection_rows)),
                "inspector_labels": dict(Counter(row["inspector_label"] for row in selection_rows)),
                "teacher_largest_probability_median": (
                    sorted(row["teacher_largest_probability"] for row in selection_rows)[len(selection_rows) // 2]
                    if selection_rows else None
                ),
                "certainty_of_kept_min_teacher_inspector": {
                    "mean": sum(sorted_certainty) / len(sorted_certainty) if sorted_certainty else None,
                    "median": sorted_certainty[len(sorted_certainty) // 2] if sorted_certainty else None,
                    "minimum": sorted_certainty[0] if sorted_certainty else None,
                },
                "last_training_student_same_as_teacher_rate": (
                    sum(bool(row["last_training_student_same_as_teacher"]) for row in selection_rows)
                    / max(1, len(selection_rows))
                ),
                "mean_losses": {"unlabeled": current_losses["set"], "human_cross_entropy": current_losses["human"]},
                "previous_iteration_mean_losses": (
                    None if previous_iteration_losses is None else
                    {"unlabeled": previous_iteration_losses["set"], "human_cross_entropy": previous_iteration_losses["human"]}
                ),
                "relative_decrease": (
                    None if relative_decrease is None else
                    {"unlabeled": relative_decrease["set"], "human_cross_entropy": relative_decrease["human"]}
                ),
                "loss_plateau_tolerance": args.loss_plateau_tolerance,
                "loss_plateau": loss_plateau,
                "next_iteration": (
                    None if stop_after_round or last_iteration else round_index + 2
                ),
                "stop_reason": stop_reason if stop_after_round else ("max_rounds" if last_iteration else None),
                "scoring_seconds": time.perf_counter() - scoring_started,
                "selection_file": str(selection_path),
            }
            report["next_iteration_selection"] = iteration_summary
            write_step_log({
                "kind": "iteration_selection",
                "time_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "round": round_index + 1,
                "step": global_optimizer_step,
                **iteration_summary,
            })
            print(f"Iteration {round_index + 1} selection {json.dumps(iteration_summary)}", flush=True)
            next_iteration_indices = kept_indices
            previous_iteration_losses = current_losses
            del teacher_metric, teacher_relevance, scored_records
        round_checkpoint_path = output_path.parent / f"round_{round_index + 1:02d}.pt"
        round_summary_path = output_path.parent / f"round_{round_index + 1:02d}_summary.json"
        if round_checkpoint_path.exists() or round_summary_path.exists():
            raise FileExistsError(
                f"Refusing to overwrite files for round {round_index + 1}"
            )
        round_checkpoint = {
            "format_version": 4,
            "model_role": "six_metric_refinement_student_teacher",
            "training_framework": framework_tag,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "base_model_path": base_model_path,
            "peft_config": peft_config,
            "metric_order": list(METRIC_NAMES),
            "prompt": prompt,
            "student_adapter_state_dict": capture_adapter_state_cpu(model),
            "teacher_adapter_state_dict": {
                name: tensor.detach().cpu().clone()
                for name, tensor in teacher_state.items()
            },
            "optimizer_state_dict": recursive_to_cpu(optimizer.state_dict()),
            "scheduler_state_dict": scheduler.state_dict(),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch_cpu": torch.get_rng_state(),
                "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            },
            "samplers": {
                "unlabeled_remaining": sorted(unlabeled_sampler.remaining),
                "unlabeled_replay_counts": dict(unlabeled_sampler.replay_counts),
                "unlabeled_deferred": sorted(unlabeled_sampler.deferred),
                "unlabeled_rng": unlabeled_sampler.rng.getstate(),
                "human_order": human_sampler.order,
                "human_position": human_sampler.position,
                "human_rng": human_sampler.rng.getstate(),
                "gemma_order": gemma_sampler.order,
                "gemma_position": gemma_sampler.position,
                "gemma_rng": gemma_sampler.rng.getstate(),
                "hard_replay_candidates": hard_replay_candidates,
            },
            "preference_aggregator": dict(aggregator_record),
            "training": {
                "completed_round": round_index + 1,
                "global_optimizer_step": global_optimizer_step,
                "history": history,
                "best_round": best_round,
                "best_optimizer_step": best_optimizer_step,
                "best_metrics": best_metrics,
                "best_teacher_metrics": best_teacher_metrics,
                "best_student_metrics": best_student_metrics,
                "baseline_metrics": baseline_metrics,
                "baseline_student_metrics": baseline_student_metrics,
                "baseline_predictions": baseline_predictions,
                "baseline_probe": baseline_probe,
                "previous_probe": previous_probe,
                "rounds_without_improvement": rounds_without_improvement,
                "set_scale": set_scale,
                "gemma_replay_scale": gemma_replay_scale,
                "mask_replay_scale": mask_replay_scale,
                "arguments": vars(args),
            },
            "best_snapshot": {
                "student_adapter_state_dict": best_student_state,
                "teacher_adapter_state_dict": best_teacher_state,
                "optimizer_state_dict": best_optimizer_state,
                "scheduler_state_dict": best_scheduler_state,
                "rng_state": best_rng_state,
            },
        }
        save_round_file = args.save_round_checkpoints == "all" or (
            args.save_round_checkpoints == "first" and round_index == 0
        )
        if save_round_file:
            save_checkpoint_atomically(round_checkpoint, round_checkpoint_path)
        round_summary_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        print(
            f"Saved complete round {round_index + 1} checkpoint: {round_checkpoint_path}"
            if save_round_file
            else f"Round {round_index + 1} checkpoint not written (--save-round-checkpoints "
            f"{args.save_round_checkpoints}); summary: {round_summary_path}",
            flush=True,
        )
        del round_checkpoint
        if stop_after_round and replay_selections is None:
            break
        if rounds_without_improvement >= args.patience_rounds:
            stop_reason = "validation_plateau"
            break

    print(
        f"Best {args.selection_model} round={best_round} step={best_optimizer_step}, "
        f"Student validation={best_student_metrics}, "
        f"Teacher validation={best_teacher_metrics}, stop={stop_reason}",
        flush=True,
    )
    checkpoint = {
        "format_version": 4,
        "model_role": "six_metric_refinement_student_teacher",
        "training_framework": framework_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_model_path": base_model_path,
        "student_adapter_state_dict": best_student_state,
        "teacher_adapter_state_dict": best_teacher_state,
        "peft_config": peft_config,
        "metric_order": list(METRIC_NAMES),
        "prompt": prompt,
        "preference_aggregator": dict(aggregator_record),
        "loss": {
            "formula": "human_bce_with_detached_tie + lambda_gemma * factorized_gemma_preference + lambda_mask * gemma_relevance_bce + adaptive_lambda_set(step,round) * feasible_set_kl_per_metric",
            "unlabeled_objective": args.unlabeled_objective,
            "overall_aggregation": args.overall_aggregation,
            "overall_probability": args.overall_probability,
            "overall_loss": args.overall_loss,
            "human_split": args.human_split,
            "checkpoint_selection": args.checkpoint_selection,
            "exclude_probe_from_training": args.exclude_probe_from_training,
            "consistency_metrics": args.consistency_metrics,
            "lambda_consistency": args.lambda_consistency,
            "teacher_kl_relevance_weight": args.teacher_kl_relevance_weight,
            "lambda_set": args.lambda_set,
            "lambda_gemma": args.lambda_gemma,
            "lambda_mask": args.lambda_mask,
            "set_ramp_steps": ramp_steps,
            "semantic_drift_threshold": args.semantic_drift_threshold,
            "semantic_replay_boost": 1.5,
            "gemma_replay_enabled": True,
            "human_tie_gradient": "detached",
            "box_only_set_weight": args.box_only_set_weight,
            "adaptive_set_weighting": {
                "target_contribution": args.set_target_contribution,
                "target_projection_rate": args.set_target_projection_rate,
                "minimum_scale": args.set_scale_min,
                "maximum_scale": args.set_scale_max,
                "momentum": args.set_scale_momentum,
            },
        },
        "teacher": {
            "ema_decay": args.ema_decay,
            "ema_schedule": args.ema_schedule,
            "ema_decay_start": args.ema_decay_start,
            "ema_decay_end": args.ema_decay_end,
            "planned_optimizer_steps": total_optimizer_steps,
            "ema_steps": best_optimizer_step,
            "mc_passes": args.mc_passes,
            "cache_slack": args.cache_slack,
            "kappa": args.kappa,
            "receives_gradients": False,
            "inference_timing": "deterministic_candidate_screening_then_J4_before_each_optimizer_group",
        },
        "curriculum": {
            "first_round": "high_inspector_teacher_agreement_only",
            "strict_selected_mix": False,
            "strict_trained_mix": False,
            "infeasible_sample_policy": "drop_only_infeasible_samples",
            "quota_shortfall_policy": "redistribute_to_reliable_fresh_samples_then_stop_if_below_minimum",
            "later_round_target_fractions": {
                "high_agreement": 1.0 - args.low_agreement_fraction - args.hard_replay_fraction,
                "low_agreement": args.low_agreement_fraction,
                "hard_replay": args.hard_replay_fraction,
            },
            "replay_strategy": args.replay_strategy,
            "gemma_targets": args.gemma_targets,
            "unlabeled_sampling": args.unlabeled_sampling,
            "maximum_box_only_fraction": args.max_box_only_fraction,
            "minimum_round_fraction": args.min_round_fraction,
            "minimum_human_and_gemma_fraction_each": args.min_supervised_fraction,
            "high_inspector_confidence": args.high_inspector_confidence,
            "calibrated_acceptance_threshold_floor": calibrated_acceptance_threshold,
            "high_teacher_condition": "deterministic_mean_signed_margin_ge_kappa_then_J4_feasibility_at_training",
            "low_teacher_condition": "accepted_inspector_and_nonnegative_deterministic_signed_margin_then_J4_feasibility_at_training",
            "screening_multiplier": args.screening_multiplier,
            "screening_chunk_size": args.screening_chunk_size,
            "screening_stop": "stop_when_soft_targets_or_candidate_budget_are_filled",
            "infeasible_replay": False,
        },
        "data": {
            "human_dataset": str(paths["human"]),
            "gemma_dataset": str(paths["gemma"]),
            "unlabeled_dataset": str(paths["unlabeled"]),
            "inspector_cache": str(paths["inspector_cache"]),
            "human_train_samples": len(human_train),
            "human_validation_samples": len(human_validation),
            "gemma_replay_samples": len(gemma_train),
            "gemma_validation_samples": len(gemma_validation),
            "unlabeled_pool_samples": len(unlabeled_pool.sample_ids),
            "diagnostic_probe_samples": len(probe_indices),
            "diagnostic_probe_source_groups": len(probe_source_groups),
            "source_group_excluded_unlabeled_samples": len(probe_excluded_indices),
            "unlabeled_subset_size": args.unlabeled_subset_size,
            "unlabeled_passes": args.unlabeled_passes,
            "untrained_unlabeled_samples": len(untrained_unlabeled_indices),
            "diagnostic_probe_sample_ids": [
                unlabeled_pool.sample_ids[index] for index in probe_indices
            ],
            "unlabeled_training_occurrences": sum(
                item["valid_unlabeled_occurrences"] for item in history
            ),
            "unique_unlabeled_training_samples": len({
                sample_id
                for item in history
                for sample_id in item["trained_sample_ids"]
            }),
            "inspector_label_counts": dict(inspector_counts),
        },
        "training": {
            "max_rounds": args.max_rounds,
            "selection_metric": f"HumanGT_validation_accuracy_{args.selection_model}",
            "unlabeled_objective": args.unlabeled_objective,
            "human_validation_sample_ids": human_validation_sample_ids,
            "baseline_predictions": baseline_predictions,
            "best_round": best_round,
            "best_optimizer_step": best_optimizer_step,
            "executed_optimizer_steps": global_optimizer_step,
            "stop_reason": stop_reason,
            "history": history,
            "baseline_fixed_probe": baseline_probe,
            "arguments": vars(args),
        },
        "baseline_teacher_human_validation": baseline_metrics,
        "best_teacher_human_validation": best_teacher_metrics,
        "baseline_student_human_validation": baseline_student_metrics,
        "best_student_human_validation": best_student_metrics,
        "optimizer_state_dict": best_optimizer_state,
        "scheduler_state_dict": best_scheduler_state,
        "rng_state": best_rng_state,
    }
    save_checkpoint_atomically(checkpoint, output_path)
    if per_sample_file is not None:
        per_sample_file.close()
    if step_log_file is not None:
        step_log_file.close()
    print(f"Refinement checkpoint: {output_path}", flush=True)


def configure_student_tie_head_mode(checkpoint: Mapping[str, Any]) -> str:
    """Build the MS-Swift Student with the Tie Head of the Stage-1 Student.

    MS-Swift installs the Student heads from ``TIE_HEAD_MODE`` before the
    warm-start weights are restored, whereas the Teacher follows the saved
    weights. The Stage-1 checkpoint records its mode; it must match the saved
    structure, and a ``TIE_HEAD_MODE`` set by the launcher must equal it.
    """
    structure = tie_head_mode_from_state_keys(checkpoint["adapter_state_dict"])
    recorded = checkpoint.get("tie_head_mode")
    mode = resolve_tie_head_mode(recorded) if recorded else structure
    if (mode != "independent") != (structure != "independent"):
        raise ValueError(
            f"Student checkpoint records Tie Head {mode} but its weights are {structure}."
        )
    requested = os.environ.get(TIE_HEAD_MODE_ENV, "").strip()
    if requested and resolve_tie_head_mode(requested) != mode:
        raise ValueError(
            f"{TIE_HEAD_MODE_ENV}={requested} differs from the Stage-1 Student Tie Head {mode}."
        )
    os.environ[TIE_HEAD_MODE_ENV] = mode
    print(f"Student Tie Head (from Stage-1 checkpoint): {mode}", flush=True)
    return mode


def run_ms_swift_online(stage2_args: argparse.Namespace) -> Any:
    """Run the online coordinator on a real MS-Swift model/template/trainer.

    MS-Swift owns model loading, PEFT preparation, pairwise encoding, the
    optimizer, scheduler, and Accelerate backward. The coordinator remains
    custom because Teacher targets depend on the just-updated EMA at each step.
    """
    validate_args(stage2_args)
    required_output(stage2_args.output)
    from datasets import Dataset as HfDataset
    from swift.loss.base import BaseLoss
    from swift.loss.mapping import loss_map
    from swift.trainers import RerankerTrainer
    from swift.trainers.trainer_factory import TrainerFactory

    from scripts.train_student_warmstart import (
        GemmaWarmStartPreprocessor,
        MetricWarmStartSwiftSft,
    )

    loss_type = "online_metric_refinement"

    class OnlineLossMarker(BaseLoss):
        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise RuntimeError("Online losses are assembled after the current Teacher MC pass.")

    loss_map[loss_type] = OnlineLossMarker

    class OnlineRerankerTrainer(RerankerTrainer):
        def create_loss_and_eval_metric(self, args):
            if args.loss_type != loss_type:
                raise ValueError("Online refinement requires its dedicated loss type.")
            return {}

        def train(self, *args, **kwargs):
            if args or kwargs:
                raise ValueError("Online refinement does not accept static Trainer.train arguments.")
            if self.accelerator.num_processes != 1:
                raise ValueError("Online EMA training currently requires one GPU process.")
            if not hasattr(self, "online_processor") or not hasattr(self, "online_template"):
                raise RuntimeError("MS-Swift online processor/template was not attached.")
            # The dynamic coordinator replaces SwiftMixin.train, so run its
            # gradient checkpointing preparation: it also makes the vision
            # tower inputs require gradients, without which the ViT LoRA
            # behind reentrant checkpointing receives no gradient.
            base_model = self.template.get_base_model(
                self.accelerator.unwrap_model(self.model)
            )
            self._prepare_gradient_checkpointing(base_model)
            main(
                stage2_args,
                model_override=self.model,
                processor_override=self.online_processor,
                swift_trainer=self,
                swift_template=self.online_template,
            )
            return {"final_checkpoint": stage2_args.output}

    previous_factory = TrainerFactory.get_trainer_cls.__func__

    @classmethod
    def online_trainer_factory(cls, args):
        if (
            getattr(args, "task_type", None) == "generative_reranker"
            and getattr(args, "loss_type", None) == loss_type
        ):
            return OnlineRerankerTrainer
        return previous_factory(cls, args)

    TrainerFactory.get_trainer_cls = online_trainer_factory

    class OnlineSwiftSft(MetricWarmStartSwiftSft):
        def _get_dataset(self):
            # The dynamic online batches are supplied by the coordinator.
            # This single strict row only initializes MS-Swift's template and
            # trainer without materializing Unlabeled Data up front.
            dataset_path = required_file(stage2_args.gemma_dataset, "GemmaGT Data")
            with dataset_path.open("r", encoding="utf-8") as handle:
                raw = next(json.loads(line) for line in handle if line.strip())
            dataset = GemmaWarmStartPreprocessor()(
                HfDataset.from_list([raw]),
                num_proc=None,
                load_from_cache_file=False,
                strict=True,
            )
            return dataset, None

        def prepare_model(self, args, model, **kwargs):
            model = super().prepare_model(args, model, **kwargs)
            if not isinstance(model, PeftModel):
                raise TypeError("Online MS-Swift model must be a PEFT Student.")
            checkpoint = torch.load(
                required_file(stage2_args.student_checkpoint, "Student checkpoint"),
                map_location="cpu", weights_only=False,
            )
            if checkpoint.get("model_role") != STUDENT_ROLE:
                raise ValueError("Online MS-Swift initialization requires Stage-1 Student.")
            expected = checkpoint["adapter_state_dict"]
            if get_peft_model_state_dict(model).keys() != expected.keys():
                raise ValueError("MS-Swift adapter structure differs from Stage-1 Student.")
            result = set_peft_model_state_dict(model, expected, adapter_name="default")
            if result.unexpected_keys:
                raise RuntimeError(f"Unexpected restored Student keys: {result.unexpected_keys[:10]}")
            actual = get_peft_model_state_dict(model)
            if any(
                not torch.equal(actual[name].detach().cpu(), expected[name].detach().cpu())
                for name in expected
            ):
                raise RuntimeError("MS-Swift did not restore all warm-start weights exactly.")
            del checkpoint, expected, actual
            return model

        def train(self, trainer):
            trainer.online_processor = self.processor
            trainer.online_template = self.template
            return trainer.train()

    checkpoint = torch.load(
        required_file(stage2_args.student_checkpoint, "Student checkpoint"),
        map_location="cpu", mmap=True, weights_only=False,
    )
    configure_student_tie_head_mode(checkpoint)
    peft_config = checkpoint["peft_config"]
    rank = int(peft_config["r"])
    alpha = float(peft_config["lora_alpha"])
    dropout = float(peft_config["lora_dropout"])
    if peft_config.get("modules_to_save") != ["metric_heads"]:
        raise ValueError("Warm-start modules_to_save must contain metric_heads only.")
    del checkpoint
    model_path = str(Path(stage2_args.base_model_path).expanduser().resolve())
    output_dir = str(Path(stage2_args.output).expanduser().resolve().parent)
    planned_steps = stage2_args.max_rounds * math.ceil(
        math.ceil(max(1, stage2_args.unlabeled_per_round) / stage2_args.unlabeled_batch_size)
        / stage2_args.gradient_accumulation_steps
    )
    ms_args = [
        "--model", model_path,
        "--dataset", str(Path(stage2_args.gemma_dataset).resolve()),
        "--task_type", "generative_reranker",
        "--template", "qwen3_vl_reranker",
        "--loss_type", loss_type,
        "--target_modules", "all-linear",
        "--tuner_type", "lora",
        "--tuner_backend", "peft",
        "--freeze_llm", "false",
        "--freeze_vit", "false",
        "--freeze_aligner", "false",
        "--lora_rank", str(rank),
        "--lora_alpha", str(alpha),
        "--lora_dropout", str(dropout),
        "--modules_to_save", "metric_heads",
        "--learning_rate", str(stage2_args.learning_rate),
        "--optim", "adamw_torch",
        "--lr_scheduler_type", stage2_args.lr_scheduler_type,
        "--weight_decay", str(stage2_args.weight_decay),
        "--adam_beta1", str(stage2_args.adam_beta1),
        "--adam_beta2", str(stage2_args.adam_beta2),
        "--adam_epsilon", str(stage2_args.adam_epsilon),
        "--max_grad_norm", str(stage2_args.max_grad_norm),
        "--warmup_ratio", str(stage2_args.warmup_ratio),
        "--attn_impl", "sdpa",
        "--torch_dtype", "bfloat16",
        "--bf16", "true",
        "--output_dir", output_dir,
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--max_steps", str(planned_steps),
        "--eval_strategy", "no",
        "--save_strategy", "no",
        "--train_dataloader_shuffle", "false",
        "--remove_unused_columns", "false",
        "--packing", "false",
        "--padding_free", "false",
        "--sequence_parallel_size", "1",
        "--max_length", "16384",
        "--dataloader_num_workers", "0",
        "--dataset_num_proc", "1",
        "--load_from_cache_file", "false",
        "--strict", "true",
        "--gradient_checkpointing", "true",
        "--seed", str(stage2_args.seed),
        "--data_seed", str(stage2_args.seed),
        "--add_version", "false",
    ]
    return OnlineSwiftSft(ms_args).main()


def run_ms_swift_supervised_control() -> Any:
    """Matched HumanGT/GemmaGT control, trained by the MS-Swift reranker trainer.

    The reference round summaries determine the exact number of optimizer
    steps. No Inspector cache, Unlabeled Data, Teacher, or set loss enters this
    arm. The data order and cycling samplers match the refinement run.
    """
    from copy import deepcopy

    from datasets import Dataset as HfDataset
    from swift.loss.base import BaseLoss
    from swift.loss.mapping import loss_map
    from swift.pipelines import SwiftSft
    from swift.template.register import TEMPLATE_MAPPING, register_template
    from swift.template.template_inputs import TemplateInputs
    from swift.trainers import RerankerTrainer
    from swift.trainers.trainer_factory import TrainerFactory
    from transformers import TrainerCallback

    from scripts.train_student_warmstart import (
        MetricWarmStartSwiftSft,
        MetricWarmStartTemplate,
        _format_candidate as swift_format_candidate,
        _format_source as swift_format_source,
        _restore_peft_checkpoint,
    )
    from src.prompt_config import DEFAULT_EVALUATION_INSTRUCTION, DEFAULT_SYSTEM_MESSAGE

    control_loss_type = "factorized_metric_supervised_control"
    control_template_type = "qwen3_vl_metric_supervised_control_pairwise"
    control_label_width = 4 * NUM_METRICS + 2

    class ControlLossMarker(BaseLoss):
        def __call__(self, *args: Any, **kwargs: Any) -> torch.Tensor:
            raise RuntimeError("The supervised control loss is owned by its MS-Swift trainer.")

    loss_map[control_loss_type] = ControlLossMarker

    class SupervisedControlTemplate(MetricWarmStartTemplate):
        """Keep A/B/source together; collate HumanGT and GemmaGT separately."""

        def _reranker_encode(self, inputs: TemplateInputs) -> Dict[str, Any]:
            if inputs.rejected is None or len(inputs.positive) != 1 or inputs.negative:
                raise ValueError("Each control row requires A, B, and one source representation.")
            labels = inputs.chosen.label
            if labels is None or len(labels) != control_label_width:
                raise ValueError("Control labels must have 26 elements.")
            candidate_a = self._encode_truncated(inputs.chosen)
            candidate_b = self._encode_truncated(inputs.rejected)
            source = self._encode_truncated(inputs.positive[0])
            return {
                "candidate_a": candidate_a,
                "candidate_b": candidate_b,
                "source": source,
                "labels": labels[: 4 * NUM_METRICS],
                "human_target": float(labels[4 * NUM_METRICS]),
                "is_human": bool(labels[4 * NUM_METRICS + 1]),
                "length": [candidate_a["length"], candidate_b["length"], source["length"]],
            }

        def data_collator(self, batch: List[Dict[str, Any]], *, padding_to=None) -> Dict[str, Any]:
            human = [item for item in batch if item["is_human"]]
            gemma = [item for item in batch if not item["is_human"]]
            if not human:
                raise ValueError("Every control batch needs HumanGT pairs.")
            human_targets = torch.tensor(
                [item["human_target"] for item in human], dtype=torch.float32
            )
            return {
                "labels": human_targets,
                "human": super().data_collator(human, padding_to=padding_to),
                "gemma": (
                    super().data_collator(gemma, padding_to=padding_to)
                    if gemma else None
                ),
            }

    control_meta = deepcopy(TEMPLATE_MAPPING["qwen3_vl_reranker"])
    control_meta.template_type = control_template_type
    control_meta.template_cls = SupervisedControlTemplate
    register_template(control_meta, exist_ok=True)

    def control_metrics(prediction: Any) -> Dict[str, float]:
        logits = np.asarray(prediction.predictions).reshape(-1)
        labels = np.asarray(prediction.label_ids).reshape(-1)
        if len(logits) != len(labels) or not len(logits):
            raise ValueError("Control validation predictions lost pair alignment.")
        probability = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        bce = -np.mean(
            labels * np.log(np.clip(probability, 1e-7, 1.0))
            + (1.0 - labels) * np.log(np.clip(1.0 - probability, 1e-7, 1.0))
        )
        return {
            "accuracy": float(np.mean((logits >= 0.0) == labels)),
            "bce": float(bce),
            "brier": float(np.mean((probability - labels) ** 2)),
        }

    class ControlFinalMilestoneCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            # A quality-gated online round may end before the regular
            # 250-step cadence. Evaluate and save at every actual boundary.
            round_ends = json.loads(os.environ["CONTROL_ROUND_ENDS"])
            if state.global_step in round_ends:
                control.should_evaluate = True
                control.should_save = True
            return control

    class SupervisedControlTrainer(RerankerTrainer):
        def create_loss_and_eval_metric(self, args):
            if args.loss_type != control_loss_type:
                raise ValueError("The control requires its dedicated loss type.")
            return {"compute_metrics": control_metrics}

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            aggregator_path = required_file(
                os.environ["CONTROL_AGGREGATOR"], "Preference Aggregator"
            )
            weights, beta, _ = load_metrics_weights_aggregator(aggregator_path)
            self.control_weights = weights
            self.control_beta = beta
            self.gemma_objective = FactorizedWarmStartLoss()
            self.model_accepts_loss_kwargs = False
            self.label_names = ["labels"]

        def _get_callbacks(self, args):
            return super()._get_callbacks(args) + [ControlFinalMilestoneCallback()]

        def create_optimizer(self):
            if self.optimizer is None:
                self.optimizer = torch.optim.AdamW(
                    [parameter for parameter in self.model.parameters() if parameter.requires_grad],
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                    weight_decay=self.args.weight_decay,
                )
            return self.optimizer

        def create_scheduler(self, num_training_steps, optimizer=None):
            del num_training_steps
            if self.lr_scheduler is None:
                self.lr_scheduler = cosine_schedule_with_warmup(
                    optimizer or self.optimizer,
                    int(os.environ["CONTROL_SCHEDULER_STEPS"]),
                    self.args.warmup_ratio,
                )
            return self.lr_scheduler

        def _human_logits(self, output: MultiMetricRewardOutput) -> torch.Tensor:
            evidence = (
                (1.0 - output.tie_probs.float().detach())
                * (2.0 * output.cond_win_prob_A.float() - 1.0)
            )
            return self.control_beta * (
                evidence * self.control_weights.to(evidence.device)[None, :]
            ).sum(dim=-1)

        def training_step(self, model, inputs, num_items_in_batch=None):
            del num_items_in_batch
            model.train()
            inputs = self._prepare_inputs(inputs)
            human_inputs = dict(inputs["human"])
            human_inputs.pop("labels", None)
            gemma_inputs = inputs["gemma"]
            if gemma_inputs is None:
                raise ValueError("A training step lost its GemmaGT examples.")
            if len(inputs["labels"]) != int(os.environ["CONTROL_HUMAN_BATCH_SIZE"]):
                raise ValueError("Control HumanGT batch size changed unexpectedly.")
            if len(gemma_inputs["labels"]) != int(os.environ["CONTROL_GEMMA_BATCH_SIZE"]):
                raise ValueError("Control GemmaGT batch size changed unexpectedly.")
            with self.compute_loss_context_manager():
                human_output = model(**human_inputs)
                human_loss = F.binary_cross_entropy_with_logits(
                    self._human_logits(human_output), inputs["labels"].float()
                )
            if not torch.isfinite(human_loss):
                raise RuntimeError("Control HumanGT loss became non-finite.")
            self.accelerator.backward(human_loss)
            del human_output

            gemma_labels = gemma_inputs["labels"].float()
            gemma_model_inputs = dict(gemma_inputs)
            gemma_model_inputs.pop("labels")
            with self.compute_loss_context_manager():
                gemma_output = model(**gemma_model_inputs)
                gemma_losses = self.gemma_objective.components(
                    gemma_output,
                    gemma_labels[:, :NUM_METRICS],
                    gemma_labels[:, NUM_METRICS : 2 * NUM_METRICS],
                    gemma_labels[:, 2 * NUM_METRICS : 3 * NUM_METRICS],
                    gemma_labels[:, 3 * NUM_METRICS : 4 * NUM_METRICS],
                )
                gemma_loss = gemma_losses["loss"]
            if not torch.isfinite(gemma_loss):
                raise RuntimeError("Control GemmaGT loss became non-finite.")
            self.accelerator.backward(gemma_loss)
            return (human_loss.detach() + gemma_loss.detach())

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            del num_items_in_batch
            human_inputs = dict(inputs["human"])
            human_inputs.pop("labels", None)
            output = model(**human_inputs)
            logits = self._human_logits(output)
            loss = F.binary_cross_entropy_with_logits(logits, inputs["labels"].float())
            if not return_outputs:
                return loss
            return loss, {"logits": logits[:, None]}

    previous_trainer_factory = TrainerFactory.get_trainer_cls.__func__

    @classmethod
    def control_trainer_factory(cls, args):
        if (
            getattr(args, "task_type", None) == "generative_reranker"
            and getattr(args, "loss_type", None) == control_loss_type
        ):
            return SupervisedControlTrainer
        return previous_trainer_factory(cls, args)

    TrainerFactory.get_trainer_cls = control_trainer_factory

    class SupervisedControlSwiftSft(MetricWarmStartSwiftSft):
        def _prepare_template(self):
            args = self.args
            if args.packing or args.padding_free or args.sequence_parallel_size > 1:
                raise ValueError("Control requires unpacked, non-parallel pair batches.")
            args.template = control_template_type
            args.template_meta = TEMPLATE_MAPPING[control_template_type]
            SwiftSft._prepare_template(self)

        def _get_dataset(self):
            reference_dir = Path(os.environ["CONTROL_REFERENCE_DIR"]).resolve()
            summary_paths = sorted(reference_dir.glob("round_*_summary.json"))
            if not summary_paths:
                raise FileNotFoundError("The online reference has no completed round summaries.")
            summaries = [json.loads(path.read_text()) for path in summary_paths]
            if [item["round"] for item in summaries] != list(range(1, len(summaries) + 1)):
                raise ValueError("Online reference round summaries are not consecutive.")
            step_counts = [
                int(summaries[0]["optimizer_step"]),
                *(
                    int(summaries[index]["optimizer_step"])
                    - int(summaries[index - 1]["optimizer_step"])
                    for index in range(1, len(summaries))
                ),
            ]
            human_batch_size = int(os.environ["CONTROL_HUMAN_BATCH_SIZE"])
            gemma_batch_size = int(os.environ["CONTROL_GEMMA_BATCH_SIZE"])
            if any(steps <= 0 for steps in step_counts):
                raise ValueError(f"Reference optimizer-step schedule is invalid: {step_counts}")
            if any(
                summary["training_pair_counts"]["human"] != steps * human_batch_size
                or summary["training_pair_counts"]["gemma"] != steps * gemma_batch_size
                for summary, steps in zip(summaries, step_counts)
            ):
                raise ValueError("Control pair exposure does not match the reference run.")
            if int(self.args.max_steps) != sum(step_counts):
                raise ValueError("MS-Swift max_steps must match all reference optimizer steps.")
            if int(self.args.per_device_train_batch_size) != human_batch_size + gemma_batch_size:
                raise ValueError("MS-Swift pair batch size does not match H+G exposure.")
            if int(self.args.gradient_accumulation_steps) != 1:
                raise ValueError("Control requires one optimizer step per 5H+5G batch.")
            if self.args.train_dataloader_shuffle or self.args.group_by_length:
                raise ValueError("MS-Swift must preserve contiguous 5H+5G control batches.")
            self.control_steps_per_round = step_counts
            os.environ["CONTROL_ROUND_ENDS"] = json.dumps(
                [int(summary["optimizer_step"]) for summary in summaries]
            )

            inspector = torch.load(
                required_file(os.environ["CONTROL_INSPECTOR_CHECKPOINT"], "Inspector checkpoint"),
                map_location="cpu", weights_only=False,
            )
            human_root = Path(
                inspector.get("data", {}).get("editreward_root", DEFAULT_EDITREWARD_ROOT)
            ).expanduser().resolve()
            human_records = load_human_records(
                required_file(os.environ["CONTROL_HUMAN_DATASET"], "HumanGT Data"),
                human_root,
            )
            human_train, human_val = human_split_from_inspector(human_records, inspector)
            del inspector, human_records

            student = torch.load(
                required_file(os.environ["CONTROL_STUDENT_CHECKPOINT"], "Student checkpoint"),
                map_location="cpu", weights_only=False,
            )
            prompt = student.get("prompt") or {}
            if (
                prompt.get("system_message") != DEFAULT_SYSTEM_MESSAGE
                or prompt.get("evaluation_instruction") != DEFAULT_EVALUATION_INSTRUCTION
            ):
                raise ValueError("Control prompt differs from the Student warm-start prompt.")
            gemma_by_id = {
                record.sample_id: record
                for record in load_gemma_records(
                    required_file(os.environ["CONTROL_GEMMA_DATASET"], "GemmaGT Data")
                )
            }
            gemma_data = student.get("data", {})
            gemma_train_ids = gemma_data.get("train_sample_ids", [])
            gemma_val_ids = gemma_data.get("validation_sample_ids", [])
            if set(gemma_by_id) != set(gemma_train_ids) | set(gemma_val_ids):
                raise ValueError("GemmaGT Data differs from the Student warm-start split.")
            gemma_train = [gemma_by_id[sample_id] for sample_id in gemma_train_ids]
            del student, gemma_by_id

            def make_row(record: PairRecord, label: List[float]) -> Dict[str, Any]:
                return {
                    "messages": swift_format_candidate(
                        record.source_image, record.edit_instruction, record.image_a
                    ),
                    "rejected_messages": swift_format_candidate(
                        record.source_image, record.edit_instruction, record.image_b
                    ),
                    "positive_messages": [swift_format_source(
                        record.source_image, record.edit_instruction
                    )],
                    "label": label,
                }

            human_sampler = CyclingRecordSampler(human_train, 42 + 10_000)
            gemma_sampler = CyclingRecordSampler(gemma_train, 42 + 20_000)
            train_rows: List[Dict[str, Any]] = []
            for _ in range(sum(step_counts)):
                for record in human_sampler.draw(human_batch_size):
                    train_rows.append(make_row(
                        record, [0.0] * (4 * NUM_METRICS) + [record.label_a, 1.0]
                    ))
                for record in gemma_sampler.draw(gemma_batch_size):
                    train_rows.append(make_row(
                        record,
                        list(record.g_a + record.g_b + record.g_t + record.relevance)
                        + [0.0, 0.0],
                    ))
            validation_rows = [
                make_row(record, [0.0] * (4 * NUM_METRICS) + [record.label_a, 1.0])
                for record in human_val
            ]
            self.control_data_summary = {
                "reference_dir": str(reference_dir),
                "steps_per_round": step_counts,
                "human_train": len(human_train),
                "human_validation": len(human_val),
                "gemma_train": len(gemma_train),
                "human_presentations": human_batch_size * sum(step_counts),
                "gemma_presentations": gemma_batch_size * sum(step_counts),
                "unlabeled_presentations": 0,
            }
            return HfDataset.from_list(train_rows), HfDataset.from_list(validation_rows)

        def prepare_model(self, args, model, **kwargs):
            model = super().prepare_model(args, model, **kwargs)
            if not isinstance(model, PeftModel):
                raise TypeError("MS-Swift control must use the PEFT Student adapter.")
            checkpoint = torch.load(
                required_file(os.environ["CONTROL_STUDENT_CHECKPOINT"], "Student checkpoint"),
                map_location="cpu", weights_only=False,
            )
            if checkpoint.get("model_role") != STUDENT_ROLE:
                raise ValueError("Control initialization must be the Stage-1 Student.")
            expected = checkpoint["adapter_state_dict"]
            current = get_peft_model_state_dict(model)
            if current.keys() != expected.keys():
                raise ValueError("MS-Swift control adapter keys differ from warm-start.")
            result = set_peft_model_state_dict(model, expected, adapter_name="default")
            if result.unexpected_keys:
                raise ValueError(f"Warm-start restore had unexpected keys: {result.unexpected_keys[:10]}")
            restored = get_peft_model_state_dict(model)
            if any(
                not torch.equal(restored[name].cpu(), expected[name].cpu())
                for name in expected
            ):
                raise RuntimeError("The MS-Swift control did not restore every Student weight.")
            self.control_peft_config = checkpoint["peft_config"]
            del checkpoint, current, restored
            return model

        def train(self, trainer):
            result = SwiftSft.train(self, trainer)
            checkpoint_dir = trainer.state.best_model_checkpoint
            if not checkpoint_dir:
                raise RuntimeError("MS-Swift control did not select a validation checkpoint.")
            model = trainer.accelerator.unwrap_model(trainer.model)
            _restore_peft_checkpoint(model, Path(checkpoint_dir))
            state = capture_adapter_state_cpu(model)
            if not any("metric_heads" in name for name in state):
                raise RuntimeError("Control checkpoint is missing the three metric heads.")
            checkpoint = {
                "format_version": 1,
                "model_role": "six_metric_supervised_control_student",
                "training_framework": "ms-swift",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "base_model_path": str(Path(self.args.model).resolve()),
                "initial_student_checkpoint": str(
                    Path(os.environ["CONTROL_STUDENT_CHECKPOINT"]).resolve()
                ),
                "student_adapter_state_dict": state,
                "peft_config": self.control_peft_config,
                "metric_order": list(METRIC_NAMES),
                "preference_aggregator": json.loads(Path(os.environ["CONTROL_AGGREGATOR"]).read_text()),
                "data": self.control_data_summary,
                "selection": {
                    "best_checkpoint": str(checkpoint_dir),
                    "best_metric": trainer.state.best_metric,
                    "metric": "HumanGT_validation_accuracy",
                    "executed_optimizer_steps": trainer.state.global_step,
                },
            }
            output_path = Path(self.args.output_dir) / "best.pt"
            save_checkpoint_atomically(checkpoint, output_path)
            print(f"MS-Swift supervised-control checkpoint: {output_path}", flush=True)
            return result

    configure_student_tie_head_mode(torch.load(
        required_file(os.environ["CONTROL_STUDENT_CHECKPOINT"], "Student checkpoint"),
        map_location="cpu", mmap=True, weights_only=False,
    ))
    return SupervisedControlSwiftSft().main()


if __name__ == "__main__":
    mode = os.environ.get("TRAIN_LOOP_MODE", "online_swift")
    if mode == "supervised_control":
        run_ms_swift_supervised_control()
    elif mode == "online_swift":
        run_ms_swift_online(parse_args())
    else:
        raise ValueError(f"Unsupported TRAIN_LOOP_MODE: {mode}")
