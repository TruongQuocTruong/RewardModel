#!/usr/bin/env python3
"""Gradient of every Stage 2 loss component, measured without training.

Replays the first steps of a finished Stage 2 run (``train_loop.py``, merged labeled
source, per_pair weighting, round 1) with the same data order, the same Teacher and
the same loss code, at fixed Student weights (no optimizer step). Each microbatch is
forwarded once; each weighted loss component is backpropagated separately:

    labeled (per pair, weight 1/labeled_pairs_per_step):
        human            lambda_H * L_H                (three-way CE, HumanGT)
        gemma_tie        lambda_G * r * BCE(tie)        (L_G tie part)
        gemma_winner     lambda_G * r * (gA+gB) * BCE(winner)   (L_G winner part)
        mask             lambda_M * L_M                (relevance BCE)
    Unlabeled (per pair, weight 1/labeled_pairs_per_step, times lambda_U):
        unlabeled_inspector   Inspector three-way CE
        unlabeled_teacher     lambda_C * KL(Teacher || Student)

Per step and per parameter group (vit, aligner, llm, heads, all) it reports each
component's gradient norm, the cosine between every pair of components, the norm of
the summed gradient, per-microbatch norms and image token counts; over the steps, the
cosine between consecutive step gradients and ||mean gradient|| / mean ||gradient||.

Checks: on the first ``--verify-total-steps`` steps the sum of the component gradients
must equal the gradient of the summed loss (one extra backward); with
``--reference-log`` the step-1 losses must match the training log.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_PROJECT_ROOT = _HERE.parent
for entry in (str(_PROJECT_ROOT), str(_HERE)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import train_loop as tl  # noqa: E402

GROUPS = ("vit", "aligner", "llm", "heads")
LABELED_COMPONENTS = ("human", "gemma_tie", "gemma_winner", "mask")
UNLABELED_COMPONENTS = ("unlabeled_inspector", "unlabeled_teacher")
COMPONENTS = LABELED_COMPONENTS + UNLABELED_COMPONENTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run-dir", required=True, help="Stage 2 run folder (round_01.pt, screening_round_01.pt)")
    parser.add_argument("--student-weights", choices=("stage1", "step_checkpoint"), required=True,
                        help="stage1: the run's Student-Stage1; step_checkpoint: --step-checkpoint")
    parser.add_argument("--step-checkpoint", default=None, help="checkpoint-N.pt of the run")
    parser.add_argument("--steps", type=int, required=True, help="replay the data of steps 1..steps")
    parser.add_argument("--verify-total-steps", type=int, default=1)
    parser.add_argument("--reference-log", default=None, help="training log (step=1 diagnostics) to compare")
    parser.add_argument("--output", required=True, help="output JSON file")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_training_arguments(run_dir: Path) -> argparse.Namespace:
    saved = torch.load(run_dir / "round_01.pt", map_location="cpu", weights_only=False, mmap=True)
    arguments = dict(saved["training"]["arguments"])
    del saved
    return argparse.Namespace(**arguments)


def check_supported(args: argparse.Namespace) -> None:
    expected = {
        "labeled_source": "merged",
        "loss_weighting": "per_pair",
        "overall_loss": "three_way",
        "unlabeled_objective": "inspector_soft_ema_consistency",
        "unlabeled_sampling": "all_once",
        "human_split": "student_train_order",
        "human_batch_size": 1,
        "unlabeled_batch_size": 1,
        "unlabeled_passes": 1,
    }
    wrong = {k: getattr(args, k) for k, v in expected.items() if getattr(args, k) != v}
    if wrong:
        raise ValueError(f"Unsupported training configuration for this diagnostic: {wrong}")
    if tl.uses_metrics_weights_aggregator(args):
        raise ValueError("A Metrics Weights Aggregator is not supported here.")


def group_parameters(model) -> Dict[str, List[torch.nn.Parameter]]:
    groups: Dict[str, List[torch.nn.Parameter]] = {g: [] for g in GROUPS}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            groups[tl.trainable_group_of(name)].append(parameter)
    return groups


def named_trainable(model) -> List[tuple]:
    return [(name, p) for name, p in model.named_parameters() if p.requires_grad]


def flat_gradient(parameters: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    """fp32 CPU copy of the gradients of a group (zeros where a gradient is None)."""
    pieces = []
    for parameter in parameters:
        if parameter.grad is None:
            pieces.append(torch.zeros(parameter.numel(), dtype=torch.float32))
        else:
            pieces.append(parameter.grad.detach().float().flatten().cpu())
    return torch.cat(pieces)


def zero_gradients(model) -> None:
    for parameter in model.parameters():
        parameter.grad = None


# Reductions over ~10^8 fp32 entries lose precision (a first test gave cosines > 1),
# so norms and dot products are accumulated in float64.
def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x.double()))


def cosine(a: torch.Tensor, b: torch.Tensor) -> float | None:
    a64, b64 = a.double(), b.double()
    na, nb = float(torch.linalg.vector_norm(a64)), float(torch.linalg.vector_norm(b64))
    if na == 0.0 or nb == 0.0:
        return None
    value = float(torch.dot(a64, b64)) / (na * nb)
    if abs(value) > 1.0 + 1e-6:
        raise RuntimeError(f"Cosine outside [-1, 1]: {value}")
    return value


def all_group(buffers: Dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([buffers[g] for g in GROUPS])


def image_token_count(inputs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    grid = inputs.get("image_grid_thw")
    patches = [int(torch.prod(row)) for row in grid.cpu()] if grid is not None else []
    return {
        "image_patches_per_image": patches,
        "image_patches_total": int(sum(patches)),
        "sequence_length": int(inputs["input_ids"].shape[1]),
    }


def reference_losses(log_path: Path) -> Dict[str, float]:
    with log_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            match = re.match(r"step=1 diagnostics (\{.*\})$", line.strip())
            if match:
                record = json.loads(match.group(1))
                losses = dict(record["losses"])
                losses["gradient_norm_all"] = record["gradient_norm_before_clipping"]["all"]["total"]
                return losses
    raise ValueError(f"No step=1 diagnostics in {log_path}")


def main() -> None:
    options = parse_args()
    run_dir = Path(options.run_dir).resolve()
    output_path = Path(options.output).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite {output_path}")
    if options.student_weights == "step_checkpoint" and not options.step_checkpoint:
        raise ValueError("--student-weights step_checkpoint needs --step-checkpoint")
    args = load_training_arguments(run_dir)
    check_supported(args)
    print(f"Training arguments loaded from {run_dir / 'round_01.pt'}", flush=True)
    # MS-Swift builds the model exactly as in training (run_ms_swift_online); its
    # output_dir is the parent of args.output, so point it away from the run folder.
    swift_dir = output_path.parent / f"{output_path.stem}_ms_swift_workdir"
    swift_dir.mkdir(parents=True, exist_ok=False)
    args.output = str(swift_dir / "unused_final_checkpoint.pt")

    def coordinator(stage2_args, *, model_override, processor_override, swift_trainer, swift_template):
        diagnose(options, run_dir, output_path, stage2_args,
                 model_override, processor_override, swift_trainer, swift_template)

    tl.main = coordinator  # OnlineRerankerTrainer.train calls train_loop.main
    tl.run_ms_swift_online(args)


def diagnose(options, run_dir, output_path, args, model, processor, swift_trainer, swift_template) -> None:
    if swift_trainer is None or swift_template is None:
        raise RuntimeError("The diagnostic expects the MS-Swift online backend (as in training).")
    tl.ACTIVE_SWIFT_TEMPLATE = swift_template
    device = torch.device(args.device)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ---- data, exactly as train_loop.main builds it (round 1) ----
    inspector_dir = tl.required_inspector_checkpoint_dir(args.inspector_checkpoint)
    human_root = Path(tl.DEFAULT_EDITREWARD_ROOT).expanduser().resolve()
    human_records = tl.load_human_records(Path(args.human_dataset), human_root, keep_ties=True)
    student_checkpoint = torch.load(args.student_checkpoint, map_location="cpu", weights_only=False, mmap=True)
    human_train, _ = tl.human_split_from_student_train_order(
        human_records, student_checkpoint, inspector_dir
    )
    gemma_by_id = {r.sample_id: r for r in tl.load_gemma_records(Path(args.gemma_dataset))}
    gemma_train = [gemma_by_id[i] for i in student_checkpoint["data"]["train_sample_ids"]]
    if args.gemma_targets == "majority_vote":
        gemma_train = [tl.majority_vote_gemma_record(r) for r in gemma_train]
    gemma_lookup = {r.sample_id: r for r in gemma_train}
    labeled_pairs = [(h, gemma_lookup[h.sample_id]) for h in human_train]
    labeled_sampler = tl.CyclingRecordSampler(labeled_pairs, args.seed + 10_000)
    prompt = student_checkpoint["prompt"]
    del student_checkpoint

    unlabeled_pool = tl.index_unlabeled_pool(Path(args.unlabeled_dataset), human_root)
    inspector_labels = tl.load_inspector_cache(Path(args.inspector_cache), unlabeled_pool.sample_ids)
    screening = torch.load(run_dir / "screening_round_01.pt", map_location="cpu", weights_only=False)
    trained_order_ids = list(screening["selected_sample_ids"])
    del screening
    summary = json.loads((run_dir / "round_01_summary.json").read_text())
    # round_01_summary.json stores trained_sample_ids sorted by pool index; the training
    # order is selected_sample_ids of screening_round_01.pt (one pass: microbatches in order).
    # The order itself is checked through the step-1 Unlabeled loss (--reference-log).
    if (len(trained_order_ids) != len(set(trained_order_ids))
            or set(trained_order_ids) != set(summary["trained_sample_ids"])):
        raise RuntimeError("screening_round_01.pt IDs differ from round_01_summary.json trained_sample_ids")
    index_of = {sample_id: i for i, sample_id in enumerate(unlabeled_pool.sample_ids)}
    pairs_per_step = args.labeled_pairs_per_step
    accumulation = args.gradient_accumulation_steps
    if pairs_per_step != accumulation:
        raise ValueError("Expected labeled_pairs_per_step == gradient_accumulation_steps (1 pair per microbatch).")
    needed = options.steps * accumulation
    step_unlabeled_indices = [index_of[i] for i in trained_order_ids[:needed]]
    unlabeled_records = tl.materialize_unlabeled_round(unlabeled_pool, step_unlabeled_indices)
    print(f"Labeled pairs: {len(labeled_pairs)}; Unlabeled steps replayed: {options.steps} "
          f"({needed} samples, training order from screening_round_01.pt)", flush=True)

    # ---- Student weights and Teacher ----
    student_source = str(args.student_checkpoint)
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location="cpu", weights_only=False)
    teacher_state = {
        # As in train_loop.main with the MS-Swift backend: fp32 Teacher on the GPU.
        name: tl._ema_dtype(tensor.detach()).to(device)
        for name, tensor in teacher_checkpoint["adapter_state_dict"].items()
    }
    del teacher_checkpoint
    teacher_change = None
    if options.student_weights == "step_checkpoint":
        step_checkpoint = torch.load(options.step_checkpoint, map_location="cpu", weights_only=False)
        expected = step_checkpoint["student_adapter_state_dict"]
        tl.load_adapter_state(model, expected)
        actual = tl.get_peft_model_state_dict(model)
        if actual.keys() != expected.keys() or any(
            not torch.equal(actual[n].detach().cpu(), expected[n]) for n in expected
        ):
            raise RuntimeError("Step checkpoint Student weights were not restored exactly.")
        student_source = str(Path(options.step_checkpoint).resolve())
        saved_teacher = step_checkpoint["teacher_adapter_state_dict"]
        teacher_change = max(
            float((saved_teacher[n].float() - teacher_state[n].float().cpu()).abs().max()) for n in teacher_state
        )
        print(f"Student weights from {student_source}; Teacher in it vs Teacher checkpoint: "
              f"max absolute change {teacher_change:.3e}", flush=True)
        del step_checkpoint, expected, actual, saved_teacher

    groups = group_parameters(model)
    group_sizes = {g: int(sum(p.numel() for p in groups[g])) for g in GROUPS}
    print(f"Trainable parameters per group: {group_sizes}", flush=True)
    objective = tl.FactorizedWarmStartLoss()
    # Round 1 scales (train_loop.main initial values; changed only after a loop).
    gemma_replay_scale = mask_replay_scale = set_scale = 1.0
    ramp_steps = args.set_ramp_steps or 1
    if args.set_ramp_steps == 0:
        raise ValueError("set_ramp_steps=0 (20% ramp) is not supported here.")

    reference = reference_losses(Path(options.reference_log)) if options.reference_log else None

    step_reports: List[Dict[str, Any]] = []
    running_sum = {c: {g: torch.zeros(group_sizes[g]) for g in GROUPS} for c in COMPONENTS + ("total",)}
    previous_total: Dict[str, torch.Tensor] | None = None
    failures: List[str] = []
    started_all = time.perf_counter()

    for step in range(1, options.steps + 1):
        step_started = time.perf_counter()
        group_indices = step_unlabeled_indices[(step - 1) * accumulation: step * accumulation]
        group_records = [unlabeled_records[i] for i in group_indices]
        teacher_cache = tl.teacher_cache_for_batch(
            model, processor, group_records, prompt, teacher_state,
            inspector_labels.labels[group_indices], None, args, device,
        )
        lambda_set_now = tl.set_loss_weight(step - 1, args.lambda_set, ramp_steps) * set_scale
        weight = 1.0 / pairs_per_step
        model.train()
        zero_gradients(model)
        buffers = {c: {g: torch.zeros(group_sizes[g]) for g in GROUPS} for c in COMPONENTS}
        verify = step <= options.verify_total_steps
        verify_buffer = {g: torch.zeros(group_sizes[g]) for g in GROUPS} if verify else None
        repeat_buffer = {g: torch.zeros(group_sizes[g]) for g in GROUPS} if verify else None
        scaled_buffer = {g: torch.zeros(group_sizes[g]) for g in GROUPS} if verify else None
        loss_sums = {k: 0.0 for k in ("human", "gemma", "mask", "tie", "winner",
                                      "unlabeled", "unlabeled_inspector_cross_entropy",
                                      "unlabeled_teacher_kl")}
        microbatches: List[Dict[str, Any]] = []

        def backward_components(named_losses, kind, sample_id, tokens):
            record = {"kind": kind, "sample_id": sample_id, **tokens, "norms": {}}
            component_sum = {g: torch.zeros(group_sizes[g]) for g in GROUPS} if verify else None
            for position, (name, loss) in enumerate(named_losses):
                last = position == len(named_losses) - 1
                swift_trainer.accelerator.backward(loss, retain_graph=verify or not last)
                record["norms"][name] = {}
                total_sq = 0.0
                for g in GROUPS:
                    flat = flat_gradient(groups[g])
                    buffers[name][g] += flat
                    if verify:
                        component_sum[g] += flat
                    value = norm(flat)
                    record["norms"][name][g] = value
                    total_sq += value * value
                record["norms"][name]["all"] = math.sqrt(total_sq)
                zero_gradients(model)
            if verify:
                summed_loss = torch.stack([loss for _, loss in named_losses]).sum()
                # Two backwards of the same summed loss: their difference is the
                # numerical noise floor (bf16 rounding, non-deterministic kernels).
                swift_trainer.accelerator.backward(summed_loss, retain_graph=True)
                first = {g: flat_gradient(groups[g]) for g in GROUPS}
                # Trainable tensors that receive no gradient from this microbatch's loss.
                record["tensors_without_gradient"] = [
                    name for name, p in named_trainable(model)
                    if p.grad is None or not bool(torch.any(p.grad != 0))
                ]
                zero_gradients(model)
                swift_trainer.accelerator.backward(summed_loss, retain_graph=True)
                second = {g: flat_gradient(groups[g]) for g in GROUPS}
                zero_gradients(model)
                # Rounding floor of a linear backward in bf16: backward(3 L) / 3 vs backward(L)
                # (3 is not a power of two, so the scaled pass rounds differently).
                swift_trainer.accelerator.backward(summed_loss * 3.0)
                scaled = {g: flat_gradient(groups[g]) / 3.0 for g in GROUPS}
                zero_gradients(model)
                record["verify"] = {}
                for g in GROUPS:
                    verify_buffer[g] += first[g]
                    repeat_buffer[g] += second[g]
                    scaled_buffer[g] += scaled[g]
                    base = norm(first[g])
                    record["verify"][g] = {
                        "sum_of_components_vs_total": norm(component_sum[g] - first[g]) / base if base else 0.0,
                        "repeat_total_vs_total": norm(second[g] - first[g]) / base if base else 0.0,
                        "scaled_total_vs_total": norm(scaled[g] - first[g]) / base if base else 0.0,
                        "cosine_sum_of_components_total": cosine(component_sum[g], first[g]),
                    }
                del first, second, scaled, component_sum
            microbatches.append(record)

        for _ in range(pairs_per_step):
            pairs = labeled_sampler.draw(args.human_batch_size)
            human_batch = [h for h, _ in pairs]
            gemma_batch = [gm for _, gm in pairs]
            inputs, starts = tl.prepare_pair_inputs(processor, human_batch, prompt, device)
            tokens = image_token_count(inputs)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = tl.forward_prepared_pairs(model, inputs, starts)
            labels = torch.tensor([r.label_a for r in human_batch], device=device, dtype=torch.float32)
            loss_human = tl.three_way_cross_entropy(
                tl.overall_three_way_probs(output, None, args.overall_aggregation),
                tl.human_three_way_targets(labels),
            ).mean()
            gemma_losses = tl.gemma_replay_losses(output, gemma_batch, objective, device)
            tie = gemma_losses["tie_loss"]
            winner = gemma_losses["conditional_winner_loss"]
            if not torch.allclose(tie + winner, gemma_losses["preference_loss"], atol=1e-5, rtol=1e-4):
                raise RuntimeError("tie_loss + conditional_winner_loss differs from preference_loss.")
            scale = len(pairs) * weight
            named = [
                ("human", args.lambda_human * loss_human * scale),
                ("gemma_tie", args.lambda_gemma * gemma_replay_scale * tie * scale),
                ("gemma_winner", args.lambda_gemma * gemma_replay_scale * winner * scale),
                ("mask", args.lambda_mask * mask_replay_scale * gemma_losses["relevance_loss"] * scale),
            ]
            loss_sums["human"] += float(loss_human.detach())
            loss_sums["gemma"] += float(gemma_losses["preference_loss"].detach())
            loss_sums["mask"] += float(gemma_losses["relevance_loss"].detach())
            loss_sums["tie"] += float(tie.detach())
            loss_sums["winner"] += float(winner.detach())
            backward_components(named, "labeled", human_batch[0].sample_id, tokens)
            del output, gemma_losses, named, inputs

        for local_index, pool_index in enumerate(group_indices):
            record = unlabeled_records[pool_index]
            inputs, starts = tl.prepare_pair_inputs(processor, [record], prompt, device)
            tokens = image_token_count(inputs)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                output = tl.forward_prepared_pairs(model, inputs, starts)
            common = dict(
                student_output=output, cache=teacher_cache, indices=[local_index],
                aggregator_weights=None, aggregator_beta=None, device=device, soft_targets=None,
                aggregation=args.overall_aggregation, probability=args.overall_probability,
                consistency_metrics=args.consistency_metrics,
                three_way_targets=inspector_labels.three_way[[pool_index]],
            )
            loss_full, stats = tl.inspector_supervision_loss(consistency_weight=args.lambda_consistency, **common)
            loss_inspector, _ = tl.inspector_supervision_loss(consistency_weight=0.0, **common)
            loss_teacher = loss_full - loss_inspector
            scale = lambda_set_now * weight
            named = [
                ("unlabeled_inspector", loss_inspector * scale),
                ("unlabeled_teacher", loss_teacher * scale),
            ]
            loss_sums["unlabeled"] += float(loss_full.detach())
            loss_sums["unlabeled_inspector_cross_entropy"] += stats["inspector_cross_entropy_sum"]
            loss_sums["unlabeled_teacher_kl"] += stats["teacher_consistency_sum"]
            backward_components(named, "unlabeled", record.sample_id, tokens)
            del output, named, inputs
        del teacher_cache

        # ---- per-step report ----
        losses = {k: v / pairs_per_step for k, v in loss_sums.items()}
        total = {g: sum(buffers[c][g] for c in COMPONENTS) for g in GROUPS}
        report: Dict[str, Any] = {"step": step, "losses": losses, "lambda_unlabeled": lambda_set_now,
                                  "groups": {}}
        for g in GROUPS + ("all",):
            vec = {c: (all_group(buffers[c]) if g == "all" else buffers[c][g]) for c in COMPONENTS}
            tot = all_group(total) if g == "all" else total[g]
            component_norms = {c: norm(vec[c]) for c in COMPONENTS}
            sum_sq = sum(v * v for v in component_norms.values())
            report["groups"][g] = {
                "component_norm": component_norms,
                "total_norm": norm(tot),
                # 1 = no cancellation (orthogonal on average); < 1 = components cancel; > 1 = aligned
                "total_squared_over_sum_of_squares": (norm(tot) ** 2 / sum_sq) if sum_sq > 0 else None,
                "cosine": {f"{a}|{b}": cosine(vec[a], vec[b])
                           for i, a in enumerate(COMPONENTS) for b in COMPONENTS[i + 1:]},
                "component_vs_total_cosine": {c: cosine(vec[c], tot) for c in COMPONENTS},
                "labeled_vs_unlabeled_cosine": cosine(
                    sum(vec[c] for c in LABELED_COMPONENTS), sum(vec[c] for c in UNLABELED_COMPONENTS)
                ),
            }
            if previous_total is not None:
                prev = all_group(previous_total) if g == "all" else previous_total[g]
                report["groups"][g]["cosine_with_previous_step_total"] = cosine(tot, prev)
        if verify:
            differences, noise, rounding = {}, {}, {}
            for g in GROUPS:
                reference_norm = norm(verify_buffer[g])
                differences[g] = norm(total[g] - verify_buffer[g]) / reference_norm if reference_norm else 0.0
                noise[g] = norm(repeat_buffer[g] - verify_buffer[g]) / reference_norm if reference_norm else 0.0
                rounding[g] = norm(scaled_buffer[g] - verify_buffer[g]) / reference_norm if reference_norm else 0.0
            report["verify_sum_of_components_relative_error"] = differences
            report["verify_repeat_total_relative_error"] = noise
            report["verify_scaled_total_relative_error"] = rounding
            print(f"step {step} verify: sum of components vs total: "
                  + ", ".join(f"{g}={v:.2e}" for g, v in differences.items())
                  + " | repeated total vs total (noise floor): "
                  + ", ".join(f"{g}={v:.2e}" for g, v in noise.items())
                  + " | backward(3L)/3 vs total (bf16 rounding floor): "
                  + ", ".join(f"{g}={v:.2e}" for g, v in rounding.items()), flush=True)
            # The decomposition is exact up to numerical noise: its error may not exceed
            # three times the larger of the two floors (non-determinism, bf16 rounding) or 1e-3.
            bad = {g: (differences[g], noise[g], rounding[g]) for g in GROUPS
                   if differences[g] > max(1e-3, 3.0 * max(noise[g], rounding[g]))}
            if bad:
                failures.append(f"step {step}: sum of components differs from the total beyond the noise floor "
                                f"(error, repeat floor, rounding floor) per group: {bad}")
        if step == 1 and reference is not None:
            comparison = {k: {"diagnostic": losses[k] if k in losses else report["groups"]["all"]["total_norm"],
                              "training_log": reference[k]}
                          for k in ("human", "gemma", "mask", "tie", "winner",
                                    "unlabeled_inspector_cross_entropy", "gradient_norm_all")}
            report["step1_vs_training_log"] = comparison
            print("step 1 vs training log: " + ", ".join(
                f"{k} {v['diagnostic']:.4f}/{v['training_log']:.4f}" for k, v in comparison.items()), flush=True)
            for k in ("human", "gemma", "mask", "unlabeled_inspector_cross_entropy"):
                if abs(comparison[k]["diagnostic"] - comparison[k]["training_log"]) > 0.05:
                    failures.append(f"Step-1 {k} differs from the training log by more than 0.05: {comparison[k]}")
            ratio = comparison["gradient_norm_all"]["diagnostic"] / comparison["gradient_norm_all"]["training_log"]
            if not 0.75 <= ratio <= 1.25:
                failures.append(f"Step-1 gradient norm differs from the training log by more than 25%: {ratio:.3f}")
        report["microbatches"] = microbatches
        report["seconds"] = time.perf_counter() - step_started
        report["peak_memory_gib"] = (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None
        step_reports.append(report)
        for c in COMPONENTS:
            for g in GROUPS:
                running_sum[c][g] += buffers[c][g]
        for g in GROUPS:
            running_sum["total"][g] += total[g]
        previous_total = total
        a = report["groups"]["all"]
        print(f"step {step}: total norm {a['total_norm']:.2f}; component norms "
              + ", ".join(f"{c}={v:.2f}" for c, v in a["component_norm"].items())
              + f"; total^2/sum of squares {a['total_squared_over_sum_of_squares']:.3f}; "
              f"{report['seconds']:.0f}s, peak {report['peak_memory_gib']:.1f} GiB", flush=True)
        del buffers
        if failures:
            print("Stopping after a failed check: " + " | ".join(failures), flush=True)
            break

    # ---- across steps: signal vs noise ----
    across: Dict[str, Any] = {}
    for c in COMPONENTS + ("total",):
        across[c] = {}
        for g in GROUPS + ("all",):
            mean_vec = (all_group(running_sum[c]) if g == "all" else running_sum[c][g]) / len(step_reports)
            per_step = [
                (r["groups"][g]["total_norm"] if c == "total" else r["groups"][g]["component_norm"][c])
                for r in step_reports
            ]
            mean_norm = float(np.mean(per_step))
            across[c][g] = {
                "norm_of_mean_gradient": norm(mean_vec),
                "mean_of_step_norms": mean_norm,
                # 1 = every step points the same way; ~1/sqrt(steps) = pure noise
                "ratio": norm(mean_vec) / mean_norm if mean_norm else None,
            }
    result = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "student_weights": options.student_weights,
        "student_source": student_source,
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "teacher_change_in_step_checkpoint": teacher_change,
        "steps_replayed": len(step_reports),
        "pure_noise_ratio_reference": 1.0 / math.sqrt(len(step_reports)),
        "trainable_parameters_per_group": group_sizes,
        "components": list(COMPONENTS),
        "weights": {"lambda_human": args.lambda_human, "lambda_gemma": args.lambda_gemma,
                    "lambda_mask": args.lambda_mask, "lambda_set": args.lambda_set,
                    "lambda_consistency": args.lambda_consistency, "per_pair_weight": 1.0 / pairs_per_step},
        "steps": step_reports,
        "across_steps": across,
        "seconds_total": time.perf_counter() - started_all,
        "failed_checks": failures,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=1))
    temporary.replace(output_path)
    print(f"Wrote {output_path}", flush=True)
    if failures:
        raise RuntimeError("Checks failed (results written for inspection): " + " | ".join(failures))


if __name__ == "__main__":
    main()
