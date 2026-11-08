#!/bin/bash
# Stage 2 (train_loop) experiment versions and their Slurm submission.
#
# Usage:
#   bash config_training.sh <version> <full|large> [--dry-run]
#
# A version name is its output folder in output_training/. COMMON_CONFIG holds
# the shared setting (1008 pairs, Inspector checkpoint-63); each version lists
# only what differs from it. Every submission is appended to
# experiment_submissions.log with the job ID, resources, the full exported
# configuration and the md5 of the launcher and train_loop.py at submit time.
# --dry-run prints the sbatch command and the record without submitting.

set -euo pipefail

ROOT=/datastore/truongtq/TrainingRewardModel/TrainingRewardModel
OUT_ROOT=$ROOT/Qwen3-VL-Embedding/output_training
LAUNCHER=$ROOT/run_train_loop.slurm
TRAIN_SCRIPT=$ROOT/Qwen3-VL-Embedding/scripts/train_loop.py
SUBMISSION_LOG=$ROOT/experiment_submissions.log
TIME_LIMIT=48:00:00  # a version may override it with VERSION_TIME_LIMIT

STUDENT_DIR=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_head_fp32
INSPECTOR_CHECKPOINT=$OUT_ROOT/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-63

# Setting 1008 pairs, first Stage 2 configuration (pipeline_process.md 3.7).
# The Student Tie Head is read from the Stage-1 checkpoint; TIE_HEAD_MODE is not set.
COMMON_CONFIG=(
    TRAIN_LOOP_STUDENT_CHECKPOINT=$STUDENT_DIR/student_warmstart.pt
    TRAIN_LOOP_TEACHER_CHECKPOINT=$STUDENT_DIR/teacher_initialized.pt
    TRAIN_LOOP_INSPECTOR_CHECKPOINT=$INSPECTOR_CHECKPOINT
    TRAIN_LOOP_INSPECTOR_CACHE=$INSPECTOR_CHECKPOINT/inspector_pseudo_labels_10k.jsonl
    REFINEMENT_MAX_ROUNDS=1
    REFINEMENT_UNLABELED_PER_ROUND=10000
    REFINEMENT_UNLABELED_SAMPLING=all_once
    REFINEMENT_UNLABELED_OBJECTIVE=inspector_soft_ema_consistency
    REFINEMENT_OVERALL_LOSS=three_way
    REFINEMENT_OVERALL_AGGREGATION=selected_equal
    REFINEMENT_OVERALL_PROBABILITY=linear
    REFINEMENT_CONSISTENCY_METRICS=selected
    REFINEMENT_HUMAN_SPLIT=student_train_order
    REFINEMENT_CHECKPOINT_SELECTION=last
    REFINEMENT_EXCLUDE_PROBE_FROM_TRAINING=false
    REFINEMENT_HUMAN_BATCH_SIZE=1
    REFINEMENT_GEMMA_BATCH_SIZE=1
    REFINEMENT_UNLABELED_BATCH_SIZE=6
    REFINEMENT_GRADIENT_ACCUMULATION_STEPS=2
    REFINEMENT_MIN_SUPERVISED_FRACTION=0.10
    REFINEMENT_LEARNING_RATE=1e-5
    REFINEMENT_EMA_SCHEDULE=cosine
    REFINEMENT_EMA_DECAY_START=0.99
    REFINEMENT_EMA_DECAY_END=0.9999
    REFINEMENT_EVAL_SAVE_STEPS=100
    REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=10
    REFINEMENT_LOGGING_STEPS=10
)

# Stage 2 tuning (Scriptions/stage2_tuning.md section 3, confirmed 2026-11-02): batch 16 =
# 8 labeled + 8 Unlabeled per step; fixed 1008-sample Unlabeled subset (selection seed 42),
# one pass = 126 steps, each labeled pair exactly once (HumanGT + GemmaGT on one forward);
# per-pair weights, all lambdas 1, no lambda_U ramp; Teacher = Student-Stage1, one pass with
# dropout, fixed inside the loop (no per-step EMA); no fixed probe; checkpoint every 21 steps.
# The versions written before 2026-11-02 used the earlier (wrong-batch) values.
TUNING_CONFIG=(
    REFINEMENT_UNLABELED_SUBSET_SIZE=1008
    REFINEMENT_UNLABELED_PER_ROUND=1008
    REFINEMENT_SELECTION_SEED=42
    REFINEMENT_SEED=42
    REFINEMENT_UNLABELED_BATCH_SIZE=1
    REFINEMENT_GRADIENT_ACCUMULATION_STEPS=8
    REFINEMENT_LABELED_SOURCE=merged
    REFINEMENT_LABELED_PAIRS_PER_STEP=8
    REFINEMENT_LOSS_WEIGHTING=per_pair
    REFINEMENT_SET_RAMP_STEPS=1
    REFINEMENT_MC_PASSES=1
    REFINEMENT_TEACHER_DROPOUT=on
    REFINEMENT_TEACHER_UPDATE=after_round
    REFINEMENT_TEACHER_BATCH_SIZE=1
    REFINEMENT_EVAL_BATCH_SIZE=1
    REFINEMENT_EVAL_SAVE_STEPS=21
    REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=10
    REFINEMENT_GEMMA_MONITOR_PAIRS=32
    REFINEMENT_DIAGNOSTIC_PROBE_SIZE=0
    REFINEMENT_LOGGING_STEPS=1
    REFINEMENT_STEP_DIAGNOSTICS=true
)
# Group A (overfit check): 256 Unlabeled x 4 passes, 8 per step (128 steps), Unlabeled CE only,
# no lambda_U ramp, constant learning rate after warmup, checkpoint at the end of every pass.
OVERFIT_CONFIG=(
    REFINEMENT_UNLABELED_SUBSET_SIZE=256
    REFINEMENT_UNLABELED_PASSES=4
    REFINEMENT_UNLABELED_PER_ROUND=1024
    REFINEMENT_GRADIENT_ACCUMULATION_STEPS=8
    REFINEMENT_LABELED_PAIRS_PER_STEP=0
    REFINEMENT_LAMBDA_HUMAN=0
    REFINEMENT_LAMBDA_GEMMA=0
    REFINEMENT_LAMBDA_MASK=0
    REFINEMENT_LAMBDA_CONSISTENCY=0
    REFINEMENT_SET_RAMP_STEPS=1
    REFINEMENT_LR_SCHEDULER_TYPE=constant_with_warmup
    REFINEMENT_EVAL_SAVE_STEPS=32
)

# Claim 2 (loss weights): every run passes over the 1008 labeled pairs exactly once.
# 24 Unlabeled + 12 labeled pairs per step: ceil(2000 / 24) = 84 steps (the last step has 8 Unlabeled),
# 84 x 12 = 1008 labeled pairs. Per-pair weights: one labeled pair 1/12 (was 1/4), one Unlabeled pair
# lambda_U / 12 (was lambda_U / 4); Unlabeled : labeled pair weight stays lambda_U, pairs per step 2:1
# (was 5:1). Checkpoint every 21 steps (21, 42, 63, 84).
LABELED_ONE_PASS_CONFIG=(
    REFINEMENT_GRADIENT_ACCUMULATION_STEPS=24
    REFINEMENT_LABELED_PAIRS_PER_STEP=12
    REFINEMENT_EVAL_SAVE_STEPS=21
)

# tuning <setting>...: TUNING_CONFIG with the given settings replacing entries of the same name.
tuning() {
    declare -A replaced=()
    local entry
    for entry in "$@"; do replaced[${entry%%=*}]=$entry; done
    VERSION_CONFIG=()
    for entry in "${TUNING_CONFIG[@]}"; do
        if [ -n "${replaced[${entry%%=*}]:-}" ]; then
            VERSION_CONFIG+=("${replaced[${entry%%=*}]}")
            unset "replaced[${entry%%=*}]"
        else
            VERSION_CONFIG+=("$entry")
        fi
    done
    for entry in "$@"; do
        # The last setting of a name wins (e.g. OVERFIT_CONFIG then an explicit override).
        if [ -n "${replaced[${entry%%=*}]:-}" ]; then
            VERSION_CONFIG+=("${replaced[${entry%%=*}]}")
            unset "replaced[${entry%%=*}]"
        fi
    done
}

# Claim 2 starting point: Student-Stage1 batch 16 initialized from Inspector checkpoint-63.
CLAIM2_STUDENT_DIR=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63
CLAIM2_STUDENT=(
    TRAIN_LOOP_STUDENT_CHECKPOINT=$CLAIM2_STUDENT_DIR/student_warmstart.pt
    TRAIN_LOOP_TEACHER_CHECKPOINT=$CLAIM2_STUDENT_DIR/teacher_initialized.pt
)

# Default setting of every new Stage 2 tuning experiment (confirmed 2026-11-03):
# TUNING_CONFIG + Student-Stage1 initialized from Inspector checkpoint-63, all lambdas 1,
# LLM and ViT LoRA trained, and the per-sample log (labels of Teacher, Student and
# Inspector for every pair at every step, <output dir>/per_sample_log_<job>.jsonl).
# A new version is: tuning "${DEFAULT_STAGE2[@]}" <one change>.
DEFAULT_STAGE2=(
    "${CLAIM2_STUDENT[@]}"
    REFINEMENT_PER_SAMPLE_LOG=true
)

# Step 1 + step 2 of stage2_tuning.md section 8.1 (2026-11-04): the 1008 Unlabeled samples on
# which the Teacher and the Inspector agree (job 9667) and online augmentation of the Student
# input (Teacher and Inspector keep the original sample).
AGREEMENT_DIR=$OUT_ROOT/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint63
AGREEMENT_SUBSET_ONLINE_AUGMENTATION=(
    REFINEMENT_UNLABELED_SUBSET_FILE=$AGREEMENT_DIR/selected_unlabeled_1008.json
    REFINEMENT_STUDENT_AUGMENTATION=online
    REFINEMENT_AUGMENTATION_SCENARIOS=$AGREEMENT_DIR/augmentation_scenarios_1008.json
)
# Combined strong augmentation of the Student input (2026-11-05): every sample and
# iteration gets one information-loss text operation (drop the request words / mask
# content words) + one noise text operation (typos / unrelated instruction marked as not
# to apply) + a layout change (one distractor image next to the source, or flip) +
# JPEG / blur / downscale / noise on source, A and B (probability 0.7) + A<->B swap.
AGREEMENT_SUBSET_COMBINED_AUGMENTATION=(
    REFINEMENT_UNLABELED_SUBSET_FILE=$AGREEMENT_DIR/selected_unlabeled_1008.json
    REFINEMENT_STUDENT_AUGMENTATION=combined
    REFINEMENT_AUGMENTATION_SCENARIOS=$AGREEMENT_DIR/augmentation_combined_scenarios_1008.json
)
# Hand-written instructions (2026-11-05): the same combined augmentation, except that the
# information-loss text operation is replaced by one hand-written rewrite of the
# instruction (indirect / convoluted / terse, same meaning, quoted text kept); the noise
# text operation, layout, degradation and swap are the same draws as without it.
AGREEMENT_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN=(
    "${AGREEMENT_SUBSET_COMBINED_AUGMENTATION[@]}"
    REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$AGREEMENT_DIR/handwritten_hard_instructions_640.json
)

# Moderate agreement subset (2026-11-05): 1008 Unlabeled samples on which the Teacher and the
# Inspector of the 1008 setting agree, taken only where the largest Inspector probability is
# in [0.6, 0.8) (the first agreement subset is >= 0.85), different instructions; same combined
# strong augmentation, 8 random samples whose source images are kept in augmentation_tmp/.
MODERATE_AGREEMENT_DIR=$OUT_ROOT/teacher_inspector_moderate_agreement_inspector_probability_0.6_0.8_student_initialized_from_inspector_checkpoint63
MODERATE_AGREEMENT_SUBSET_COMBINED_AUGMENTATION=(
    REFINEMENT_UNLABELED_SUBSET_FILE=$MODERATE_AGREEMENT_DIR/selected_unlabeled_1008.json
    REFINEMENT_STUDENT_AUGMENTATION=combined
    REFINEMENT_AUGMENTATION_SCENARIOS=$MODERATE_AGREEMENT_DIR/augmentation_combined_scenarios_1008.json
    REFINEMENT_AUGMENTATION_EXAMPLES=8
)
# Hand-written instructions of the moderate agreement subset (2026-11-05): 283 rewrites from
# the earlier files + 725 new ones; same scenarios file, so the same image draws as without it.
MODERATE_AGREEMENT_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN=(
    "${MODERATE_AGREEMENT_SUBSET_COMBINED_AUGMENTATION[@]}"
    REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$MODERATE_AGREEMENT_DIR/handwritten_hard_instructions_1008.json
)
# Setting 512 pairs (2026-11-05): the 1008-pair pipeline repeated at 512 labeled pairs.
# Inspector = Inspector Baseline checkpoint-32 (32 steps x 16 = first 512 pairs of
# train_order.jsonl) with its own pseudo-label cache; Student-Stage1 = batch 16, 512 GemmaGT
# pairs, initialized from checkpoint-32; 512 Unlabeled = 64 steps, checkpoint every 16 steps.
INSPECTOR_512_CHECKPOINT=$OUT_ROOT/inspector_baseline_overall_humanGT_full_r32_lr5e-5_constant_head_fp32/checkpoint-32
STUDENT_512_DIR=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_512_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint32
STAGE2_512=(
    TRAIN_LOOP_STUDENT_CHECKPOINT=$STUDENT_512_DIR/student_warmstart.pt
    TRAIN_LOOP_TEACHER_CHECKPOINT=$STUDENT_512_DIR/teacher_initialized.pt
    TRAIN_LOOP_INSPECTOR_CHECKPOINT=$INSPECTOR_512_CHECKPOINT
    TRAIN_LOOP_INSPECTOR_CACHE=$INSPECTOR_512_CHECKPOINT/inspector_pseudo_labels_10k.jsonl
    REFINEMENT_UNLABELED_SUBSET_SIZE=512
    REFINEMENT_UNLABELED_PER_ROUND=512
    REFINEMENT_EVAL_SAVE_STEPS=16
    REFINEMENT_PER_SAMPLE_LOG=true
)
# The 512 Unlabeled samples on which the Teacher and the Inspector of the 512 setting agree,
# combined strong augmentation, and 8 random samples whose original and augmented source
# images are kept in <output dir>/augmentation_tmp/.
AGREEMENT_512_DIR=$OUT_ROOT/teacher_inspector_agreement_student_initialized_from_inspector_checkpoint32
AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION=(
    REFINEMENT_UNLABELED_SUBSET_FILE=$AGREEMENT_512_DIR/selected_unlabeled_512.json
    REFINEMENT_STUDENT_AUGMENTATION=combined
    REFINEMENT_AUGMENTATION_SCENARIOS=$AGREEMENT_512_DIR/augmentation_combined_scenarios_512.json
    REFINEMENT_AUGMENTATION_EXAMPLES=8
)
# Hand-written instructions of the 512 agreement subset (2026-11-05): 426 rewrites from the
# 1008 file + 86 new ones; same draws as the combined strong augmentation version otherwise.
AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN=(
    "${AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION[@]}"
    REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$AGREEMENT_512_DIR/handwritten_hard_instructions_512.json
)
# Moderate agreement subset of the 512 setting (2026-11-05): Teacher and Inspector of the 512
# setting agree, largest Inspector probability in [0.45, 0.65) (the 512 agreement subset is
# >= 0.652, so the two subsets do not overlap), different instructions.
MODERATE_AGREEMENT_512_DIR=$OUT_ROOT/teacher_inspector_moderate_agreement_inspector_probability_0.45_0.65_student_initialized_from_inspector_checkpoint32
MODERATE_AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION=(
    REFINEMENT_UNLABELED_SUBSET_FILE=$MODERATE_AGREEMENT_512_DIR/selected_unlabeled_512.json
    REFINEMENT_STUDENT_AUGMENTATION=combined
    REFINEMENT_AUGMENTATION_SCENARIOS=$MODERATE_AGREEMENT_512_DIR/augmentation_combined_scenarios_512.json
    REFINEMENT_AUGMENTATION_EXAMPLES=8
)
MODERATE_AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN=(
    "${MODERATE_AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION[@]}"
    REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$MODERATE_AGREEMENT_512_DIR/handwritten_hard_instructions_512.json
)
# Part-wise augmentation on the 512 moderate agreement subset (2026-11-05): which part of the
# Student input makes it differ from the Teacher; the hand-written combined version is the
# "both" run. choose_part_augmentation of student_augmentation.py; 25 random samples saved
# with every path and instruction in augmentation_tmp/examples.jsonl.
MODERATE_AGREEMENT_512_PART=(
    "${STAGE2_512[@]}"
    REFINEMENT_UNLABELED_SUBSET_FILE=$MODERATE_AGREEMENT_512_DIR/selected_unlabeled_512.json
    REFINEMENT_AUGMENTATION_SCENARIOS=$MODERATE_AGREEMENT_512_DIR/augmentation_combined_scenarios_512.json
    REFINEMENT_AUGMENTATION_EXAMPLES=25
)
PART_TEST_2_STEPS=(REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 REFINEMENT_EVAL_SAVE_STEPS=1)
# Part-wise augmentation on the 1008 moderate agreement subset (2026-11-05): the same as the
# 512 one, on the 1008 setting (Inspector checkpoint-63, largest Inspector probability in [0.6, 0.8)).
MODERATE_AGREEMENT_1008_PART=(
    "${DEFAULT_STAGE2[@]}"
    REFINEMENT_UNLABELED_SUBSET_FILE=$MODERATE_AGREEMENT_DIR/selected_unlabeled_1008.json
    REFINEMENT_AUGMENTATION_SCENARIOS=$MODERATE_AGREEMENT_DIR/augmentation_combined_scenarios_1008.json
    REFINEMENT_AUGMENTATION_EXAMPLES=25
)
# Stage 2 over iterations (2026-11-05): up to 5 iterations; the Teacher is updated by EMA after
# every optimizer step (cosine 0.99 -> 0.9999, as Co-rewarding); after every iteration the
# Teacher scores the samples of that iteration on the original input and only the samples whose
# Teacher label differs from the Inspector label are left out of the next iteration (certainty
# logged, not used to cut); stop early when both mean losses (Unlabeled, HumanGT CE) fall by
# less than 1% from the previous iteration. Every step checkpoint is kept.
ITERATIONS_TEACHER_INSPECTOR_AGREEMENT=(
    REFINEMENT_UNLABELED_SAMPLING=previous_iteration_teacher_inspector_agreement
    REFINEMENT_MAX_ROUNDS=5
    REFINEMENT_TEACHER_UPDATE=optimizer_step
    REFINEMENT_EMA_SCHEDULE=cosine
    REFINEMENT_EMA_DECAY_START=0.99
    REFINEMENT_EMA_DECAY_END=0.9999
    REFINEMENT_LOSS_PLATEAU_TOLERANCE=0.01
)
ITERATIONS_TEST_2_STEPS=("${PART_TEST_2_STEPS[@]}" REFINEMENT_MAX_ROUNDS=2)
# Single-operation augmentation (2026-11-05): 1008 setting (Student-Stage1 initialized from
# Inspector checkpoint-63), 1008 Unlabeled samples on which the Teacher and the Inspector agree,
# largest Inspector probability in [0.5, 0.7), visited in random order (seed 42), different
# instructions (job 10450; scenarios job 10451). One loop (126 steps). Every job applies exactly
# one operation of student_augmentation.SINGLE_OPERATIONS (no A<->B swap); the baseline has no
# augmentation. The per-sample log has, for every Unlabeled sample, the Teacher and Student
# probabilities, confidences, KL and CE overall and for each of the 6 metrics.
PROBABILITY_05_07_DIR=$OUT_ROOT/teacher_inspector_moderate_agreement_inspector_probability_0.5_0.7_student_initialized_from_inspector_checkpoint63
PROBABILITY_05_07_BASELINE=(
    "${DEFAULT_STAGE2[@]}"
    REFINEMENT_UNLABELED_SUBSET_FILE=$PROBABILITY_05_07_DIR/selected_unlabeled_1008.json
)
PROBABILITY_05_07_SINGLE_OPERATION=(
    "${PROBABILITY_05_07_BASELINE[@]}"
    REFINEMENT_STUDENT_AUGMENTATION=single_operation
    REFINEMENT_AUGMENTATION_SCENARIOS=$PROBABILITY_05_07_DIR/augmentation_combined_scenarios_1008.json
    REFINEMENT_AUGMENTATION_EXAMPLES=25
)
SINGLE_OPERATIONS_PATTERN="drop_action|mask_content|typo|distractor_instruction|injection|composite|flip|blur|noise|rotation|black_rectangles"
# Combined operations (2026-11-06): the 4 strongest single operations applied together, every
# combination of 2, 3 and 4 (fixed order composite, blur, noise, mask_content); version suffix =
# the operation names joined by "_", value = joined by "+" (REFINEMENT_AUGMENTATION_OPERATION).
declare -A COMBINED_OPERATIONS=(
    [composite_blur_noise_mask_content]=composite+blur+noise+mask_content
    [composite_blur_noise]=composite+blur+noise
    [composite_blur_mask_content]=composite+blur+mask_content
    [composite_noise_mask_content]=composite+noise+mask_content
    [blur_noise_mask_content]=blur+noise+mask_content
    [composite_blur]=composite+blur
    [composite_noise]=composite+noise
    [composite_mask_content]=composite+mask_content
    [blur_noise]=blur+noise
    [blur_mask_content]=blur+mask_content
    [noise_mask_content]=noise+mask_content
)
# Inspector one-hot (2026-11-06): CE(one-hot of the Inspector argmax, Student overall) instead of
# the soft Inspector label (floor 0 instead of the Inspector entropy); suffix _inspector_one_hot.
INSPECTOR_ONE_HOT=(REFINEMENT_INSPECTOR_TARGET=one_hot)
# Labeled pairs augmented too (2026-11-06): HumanGT + GemmaGT pairs get the same operation(s) and
# draw rule as the Unlabeled samples, labels kept; suffix _labeled_augmented (4-op combination; since
# 2026-11-07 also composite_blur_mask_content and composite_noise_mask_content).
LABELED_AUGMENTED=(REFINEMENT_AUGMENT_LABELED=true)
# Iterations + Teacher relevance KL (2026-11-07): composite+mask_content on the labeled pairs and
# the Unlabeled samples, soft Inspector, up to 5 iterations (iteration 1 = the 1008 samples; later
# iterations keep the samples whose Teacher label equals the Inspector label; early stop when the
# mean Unlabeled and HumanGT losses both fall by less than 1% from the previous iteration).
# Two EMA updates x two Teacher consistency losses = 4 versions:
#   _ema_step_0.99       EMA after every optimizer step, constant decay 0.99
#   _ema_iteration_0.99  Teacher fixed inside an iteration, EMA 0.99 once after it
#   _teacher_kl_relevance_0.5  consistency = 0.5 KL_pref + 0.5 KL_rel (no suffix: KL_pref only)
# 2 checkpoint-<step>.pt per iteration (middle and end), all kept (limit 10 = 2 x 5); round_<N>.pt
# only for iteration 1 (the iterations cannot resume; round_01.pt is read by the gradient diagnosis).
ITERATIONS_COMPOSITE_MASK_CONTENT=(
    "${PROBABILITY_05_07_SINGLE_OPERATION[@]}"
    REFINEMENT_AUGMENTATION_OPERATION=composite+mask_content
    "${LABELED_AUGMENTED[@]}"
    REFINEMENT_UNLABELED_SAMPLING=previous_iteration_teacher_inspector_agreement
    REFINEMENT_MAX_ROUNDS=5
    REFINEMENT_LOSS_PLATEAU_TOLERANCE=0.01
    REFINEMENT_EMA_SCHEDULE=constant
    REFINEMENT_EMA_DECAY=0.99
    REFINEMENT_EVAL_SAVE_STEPS=0
    REFINEMENT_CHECKPOINTS_PER_ITERATION=2
    REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=10
    REFINEMENT_SAVE_ROUND_CHECKPOINTS=first
)

# Only the settings that differ from COMMON_CONFIG.
version_config() {
    case "$1" in
        train_loop_pipeline_1008_all_once_three_way_selected_equal)
            VERSION_CONFIG=() ;;
        train_loop_pipeline_1008_all_once_three_way_selected_equal_tie_coupling_lr1e-3)
            VERSION_CONFIG=(REFINEMENT_TIE_COUPLING_LEARNING_RATE=1e-3) ;;
        train_loop_pipeline_1008_all_once_without_unlabeled_loss_three_way_selected_equal)
            # Control: Unlabeled is still forwarded (same data stream and schedules), loss weight 0.
            VERSION_CONFIG=(REFINEMENT_LAMBDA_SET=0) ;;
        train_loop_pipeline_1008_all_once_three_way_selected_equal_lambda_unlabeled6)
            # Per-pair weight parity of the 3k setting: with 2+2+12 per step, lambda_U=6 makes
            # one Unlabeled pair weigh 6/12 = 1/2, the same as one HumanGT pair (1/2).
            VERSION_CONFIG=(REFINEMENT_LAMBDA_SET=6) ;;
        train_loop_pipeline_1008_all_once_three_way_selected_equal_lambda_unlabeled6_ema_decay_constant0.99)
            # 3k-setting balance: lambda_U=6 (per-pair parity with HumanGT at 2+2+12) and
            # constant EMA decay 0.99 (as train_loop_single_pass_selected_equal) instead of cosine.
            VERSION_CONFIG=(
                REFINEMENT_LAMBDA_SET=6
                REFINEMENT_EMA_SCHEDULE=constant
                REFINEMENT_EMA_DECAY=0.99
            ) ;;
        stage2_tuning_timing_test_normal)
            # Stage 2 tuning, group 0: peak memory and seconds per component on normal (2g.35gb)
            # with one pair per source per microbatch; stopped by the 1-hour time limit.
            VERSION_CONFIG=(
                REFINEMENT_UNLABELED_BATCH_SIZE=1
                REFINEMENT_GRADIENT_ACCUMULATION_STEPS=4
                REFINEMENT_TEACHER_BATCH_SIZE=1
                REFINEMENT_EVAL_BATCH_SIZE=1
                REFINEMENT_EVAL_SAVE_STEPS=0
            )
            VERSION_TIME_LIMIT=01:00:00 ;;
        stage2_tuning_timing_test_normal_subset)
            # Stage 2 tuning, group 0 (second test): 240-sample Unlabeled subset, one pass,
            # 4 x (1 HumanGT + 1 GemmaGT + 1 Unlabeled) per step = 60 steps, Teacher J=4;
            # light start/end evaluation (32 Gemma pairs, 32 probe samples), no step evaluation.
            VERSION_CONFIG=(
                REFINEMENT_UNLABELED_SUBSET_SIZE=240
                REFINEMENT_UNLABELED_PER_ROUND=240
                REFINEMENT_UNLABELED_BATCH_SIZE=1
                REFINEMENT_GRADIENT_ACCUMULATION_STEPS=4
                REFINEMENT_TEACHER_BATCH_SIZE=1
                REFINEMENT_EVAL_BATCH_SIZE=1
                REFINEMENT_EVAL_SAVE_STEPS=0
                REFINEMENT_GEMMA_MONITOR_PAIRS=32
                REFINEMENT_DIAGNOSTIC_PROBE_SIZE=32
            )
            VERSION_TIME_LIMIT=02:00:00 ;;
        # ----- Stage 2 tuning, stage 1 -----
        stage2_tuning_starting_point_seed42) tuning; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_starting_point_seed43) tuning REFINEMENT_SEED=43; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_starting_point_seed44) tuning REFINEMENT_SEED=44; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_lambda_unlabeled0)
            # Control: Unlabeled still forwarded (same data stream and schedules), loss weight 0.
            tuning REFINEMENT_LAMBDA_SET=0; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_overfit_unlabeled_only_lr1e-5)
            tuning "${OVERFIT_CONFIG[@]}"; VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_overfit_unlabeled_only_lr2e-6)
            tuning "${OVERFIT_CONFIG[@]}" REFINEMENT_LEARNING_RATE=2e-6; VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_overfit_unlabeled_with_teacher_kl)
            tuning "${OVERFIT_CONFIG[@]}" REFINEMENT_LAMBDA_CONSISTENCY=1; VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_overfit_full_loss)
            # Overfit check with the labeled losses competing: 4 labeled pairs per step.
            tuning "${OVERFIT_CONFIG[@]}" REFINEMENT_LABELED_PAIRS_PER_STEP=4 REFINEMENT_LAMBDA_HUMAN=1 \
                REFINEMENT_LAMBDA_GEMMA=1 REFINEMENT_LAMBDA_MASK=1; VERSION_TIME_LIMIT=04:00:00 ;;
        # ----- Stage 2 tuning, stage 2 (one change from the starting point) -----
        stage2_tuning_lambda_unlabeled0.25) tuning REFINEMENT_LAMBDA_SET=0.25; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_lambda_unlabeled0.1) tuning REFINEMENT_LAMBDA_SET=0.1; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_lambda_unlabeled_ramp_none)
            # Ramp length 1 step: lambda_U is at its full value from the first step.
            tuning REFINEMENT_SET_RAMP_STEPS=1; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_backbone_lr2e-6)
            tuning REFINEMENT_LEARNING_RATE=2e-6 REFINEMENT_HEAD_LEARNING_RATE=1e-5; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_backbone_lr1e-6)
            tuning REFINEMENT_LEARNING_RATE=1e-6 REFINEMENT_HEAD_LEARNING_RATE=1e-5; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_lambda_consistency0) tuning REFINEMENT_LAMBDA_CONSISTENCY=0; VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_teacher_mc_passes4) tuning REFINEMENT_MC_PASSES=4; VERSION_TIME_LIMIT=09:00:00 ;;
        # ----- Claim 1b (2026-11-02): Stage 2 from each batch-16 Student-Stage1, same seed 42 -----
        stage2_tuning_unlabeled1008_teacher_dropout_without_ema)
            # Student-Stage1 batch 16, fresh LoRA.
            tuning TRAIN_LOOP_STUDENT_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32/student_warmstart.pt \
                TRAIN_LOOP_TEACHER_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32/teacher_initialized.pt
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint63)
            # Student-Stage1 batch 16, initialized from Inspector checkpoint-63 (only change).
            # Also the starting point of claim 2 (loss weights) below.
            tuning TRAIN_LOOP_STUDENT_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/student_warmstart.pt \
                TRAIN_LOOP_TEACHER_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/teacher_initialized.pt
            VERSION_TIME_LIMIT=05:00:00 ;;
        # ----- Claim 2 (2026-11-02): loss weights, one change from the claim-1b version
        # initialized from Inspector checkpoint-63 (all lambdas 1). Every step is clipped, so
        # only weight ratios matter: lambda_H = 1 is the reference. lambda_U scales both the
        # Inspector CE and the Teacher KL; lambda_U = 0 is the control without Unlabeled. -----
        stage2_tuning_unlabeled1008_lambda_unlabeled0|\
        stage2_tuning_unlabeled1008_lambda_unlabeled0.5|\
        stage2_tuning_unlabeled1008_lambda_unlabeled2)
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_LAMBDA_SET=${1##*lambda_unlabeled}; VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_lambda_consistency0|\
        stage2_tuning_unlabeled1008_lambda_consistency2)
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_LAMBDA_CONSISTENCY=${1##*lambda_consistency}; VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_lambda_mask0.5|\
        stage2_tuning_unlabeled1008_lambda_mask2)
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_LAMBDA_MASK=${1##*lambda_mask}; VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_lambda_gemma0.5|\
        stage2_tuning_unlabeled1008_lambda_gemma2)
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_LAMBDA_GEMMA=${1##*lambda_gemma}; VERSION_TIME_LIMIT=05:00:00 ;;
        # ----- ViT/aligner frozen (2026-11-03): one change from the claim-1b version initialized
        # from Inspector checkpoint-63: only the LLM LoRA and metric_heads are trained, as the
        # Inspector; the ViT and aligner LoRA keep their Student-Stage1 values (gradient norm
        # diagnosis: the ViT gives about 97% of the squared gradient norm). -----
        stage2_tuning_unlabeled1008_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint63_vit_aligner_frozen)
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_FREEZE_VISUAL=true; VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint63_vit_aligner_frozen_test_2_steps)
            # Code check only: 16 Unlabeled = 2 steps, a checkpoint at each step (logs the
            # visual LoRA change), 30-minute limit.
            tuning "${CLAIM2_STUDENT[@]}" REFINEMENT_FREEZE_VISUAL=true \
                REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 \
                REFINEMENT_EVAL_SAVE_STEPS=1
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint63_test_2_steps)
            # Code check of the default setting with the per-sample log on GPU: 16 Unlabeled =
            # 2 steps, a checkpoint at each step, 30-minute limit.
            tuning "${DEFAULT_STAGE2[@]}" \
                REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 \
                REFINEMENT_EVAL_SAVE_STEPS=1
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_online_augmentation)
            # Default setting + the Teacher-Inspector agreement subset (1008, job 9667) + online
            # augmentation of the Student input; one pass (126 steps), per-sample log on.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_ONLINE_AUGMENTATION[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_online_augmentation_test_2_steps)
            # Code check on GPU: first 16 IDs of the agreement subset = 2 steps, a checkpoint at
            # each step, 30-minute limit.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_ONLINE_AUGMENTATION[@]}" \
                REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 \
                REFINEMENT_EVAL_SAVE_STEPS=1
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_combined_strong_online_augmentation)
            # Default setting + the agreement subset + combined strong augmentation of the
            # Student input (text + image operations together); one pass, per-sample log on.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_COMBINED_AUGMENTATION[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_combined_strong_online_augmentation_test_2_steps)
            # Code check on GPU: first 16 IDs = 2 steps, a checkpoint at each step, 30 minutes.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_COMBINED_AUGMENTATION[@]}" \
                REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 \
                REFINEMENT_EVAL_SAVE_STEPS=1
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_combined_strong_online_augmentation_handwritten_instructions)
            # Same as the combined strong augmentation version, except the
            # information-loss text operation is replaced by a hand-written rewrite.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_combined_strong_online_augmentation_handwritten_instructions_test_2_steps)
            # Code check on GPU: first 16 IDs = 2 steps, a checkpoint at each step, 30 minutes.
            tuning "${DEFAULT_STAGE2[@]}" "${AGREEMENT_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN[@]}" \
                REFINEMENT_UNLABELED_SUBSET_SIZE=16 REFINEMENT_UNLABELED_PER_ROUND=16 \
                REFINEMENT_EVAL_SAVE_STEPS=1
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_moderate_agreement_combined_strong_online_augmentation)
            # Same as the combined strong augmentation version, except the Unlabeled subset
            # (moderate agreement, largest Inspector probability in [0.6, 0.8)).
            tuning "${DEFAULT_STAGE2[@]}" "${MODERATE_AGREEMENT_SUBSET_COMBINED_AUGMENTATION[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_moderate_agreement_combined_strong_online_augmentation_handwritten_instructions)
            # Moderate agreement subset + combined strong augmentation with the
            # information-loss text operation replaced by a hand-written rewrite.
            tuning "${DEFAULT_STAGE2[@]}" "${MODERATE_AGREEMENT_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        # ----- Setting 512 pairs -----
        stage2_tuning_unlabeled512_teacher_dropout_without_ema_student_initialized_from_inspector_checkpoint32)
            # Default Stage 2 setting at 512 pairs (random 512 Unlabeled, 64 steps).
            tuning "${STAGE2_512[@]}"
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_combined_strong_online_augmentation)
            # 512 setting + the agreement subset of the 512 setting + combined strong augmentation.
            tuning "${STAGE2_512[@]}" "${AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION[@]}"
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_combined_strong_online_augmentation_handwritten_instructions)
            # Same as the 512 combined strong augmentation version, except the
            # information-loss text operation is replaced by a hand-written rewrite.
            tuning "${STAGE2_512[@]}" "${AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN[@]}"
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_combined_strong_online_augmentation)
            # Same as the 512 combined strong augmentation version, except the Unlabeled
            # subset (moderate agreement, largest Inspector probability in [0.45, 0.65)).
            tuning "${STAGE2_512[@]}" "${MODERATE_AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION[@]}"
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_handwritten_instructions_instruction_only)
            # 512 moderate agreement subset, text part of the hand-written combined augmentation only; images unchanged.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=instruction_only REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$MODERATE_AGREEMENT_512_DIR/handwritten_hard_instructions_512.json
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_handwritten_instructions_instruction_only_test_2_steps)
            # 512 moderate agreement subset, text part of the hand-written combined augmentation only; images unchanged. Code check: 16 IDs = 2 steps.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=instruction_only REFINEMENT_HANDWRITTEN_INSTRUCTIONS=$MODERATE_AGREEMENT_512_DIR/handwritten_hard_instructions_512.json "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_source_image_only)
            # 512 moderate agreement subset, only the source image changes (composite / rotation / black rectangles / degradation).
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=source_image_only
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_source_image_only_test_2_steps)
            # 512 moderate agreement subset, only the source image changes (composite / rotation / black rectangles / degradation). Code check: 16 IDs = 2 steps.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=source_image_only "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_instruction_words_dropped_90_source_blacked_80_100)
            # 512 moderate agreement subset, 90% of the instruction words dropped, 80-100% of the source blacked out.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_instruction_words_dropped_90_source_blacked_80_100_test_2_steps)
            # 512 moderate agreement subset, 90% of the instruction words dropped, 80-100% of the source blacked out. Code check: 16 IDs = 2 steps.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_instruction_blank_source_black)
            # 512 moderate agreement subset, instruction " " and the whole source black; A, B unchanged.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=blank
            VERSION_TIME_LIMIT=04:00:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_online_augmentation_instruction_blank_source_black_test_2_steps)
            # Code check: 16 IDs = 2 steps.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=blank "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled512_moderate_agreement_combined_strong_online_augmentation_handwritten_instructions)
            # 512 moderate agreement subset + combined strong augmentation with the
            # information-loss text operation replaced by a hand-written rewrite.
            tuning "${STAGE2_512[@]}" "${MODERATE_AGREEMENT_512_SUBSET_COMBINED_AUGMENTATION_HANDWRITTEN[@]}"
            VERSION_TIME_LIMIT=04:00:00 ;;
        # ----- Extreme augmentation (90% of the instruction words dropped, 80-100% of the source
        # blacked out), moderate agreement subsets -----
        stage2_tuning_unlabeled1008_extreme_augmentation)
            # 1008 moderate agreement subset, one loop, Teacher fixed inside the loop (as the 512 extreme run).
            tuning "${MODERATE_AGREEMENT_1008_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_extreme_augmentation_test_2_steps)
            # Code check: 16 IDs = 2 steps.
            tuning "${MODERATE_AGREEMENT_1008_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled512_extreme_augmentation_iterations_teacher_inspector_agreement)
            # 512 moderate agreement subset, up to 5 iterations (at most 320 steps, 20 step checkpoints).
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme \
                "${ITERATIONS_TEACHER_INSPECTOR_AGREEMENT[@]}" REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=25
            VERSION_TIME_LIMIT=12:00:00 ;;
        stage2_tuning_unlabeled512_extreme_augmentation_iterations_teacher_inspector_agreement_test_2_steps)
            # Code check: 16 IDs = 2 steps per iteration, 2 iterations.
            tuning "${MODERATE_AGREEMENT_512_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme \
                "${ITERATIONS_TEACHER_INSPECTOR_AGREEMENT[@]}" REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=25 "${ITERATIONS_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=01:00:00 ;;
        stage2_tuning_unlabeled1008_extreme_augmentation_iterations_teacher_inspector_agreement)
            # 1008 moderate agreement subset, up to 5 iterations (at most 630 steps, 30 step checkpoints).
            tuning "${MODERATE_AGREEMENT_1008_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme \
                "${ITERATIONS_TEACHER_INSPECTOR_AGREEMENT[@]}" REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=35
            VERSION_TIME_LIMIT=22:00:00 ;;
        stage2_tuning_unlabeled1008_extreme_augmentation_iterations_teacher_inspector_agreement_test_2_steps)
            # Code check: 16 IDs = 2 steps per iteration, 2 iterations.
            tuning "${MODERATE_AGREEMENT_1008_PART[@]}" REFINEMENT_STUDENT_AUGMENTATION=extreme \
                "${ITERATIONS_TEACHER_INSPECTOR_AGREEMENT[@]}" REFINEMENT_EVAL_SAVE_TOTAL_LIMIT=35 "${ITERATIONS_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=01:00:00 ;;
        # ----- Single-operation augmentation, 1008 Unlabeled with largest Inspector probability in [0.5, 0.7) -----
        stage2_tuning_unlabeled1008_baseline)
            # No augmentation: Student and Teacher see the original samples.
            tuning "${PROBABILITY_05_07_BASELINE[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_baseline_test_2_steps)
            # Code check: 16 IDs = 2 steps.
            tuning "${PROBABILITY_05_07_BASELINE[@]}" "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_baseline_inspector_one_hot)
            # Baseline (no augmentation) with the one-hot Inspector target.
            tuning "${PROBABILITY_05_07_BASELINE[@]}" "${INSPECTOR_ONE_HOT[@]}"
            VERSION_TIME_LIMIT=05:00:00 ;;
        stage2_tuning_unlabeled1008_baseline_inspector_one_hot_test_2_steps)
            tuning "${PROBABILITY_05_07_BASELINE[@]}" "${INSPECTOR_ONE_HOT[@]}" "${PART_TEST_2_STEPS[@]}"
            VERSION_TIME_LIMIT=00:30:00 ;;
        stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_*)
            rest=${1#stage2_tuning_unlabeled1008_augmentation_composite_mask_content_labeled_augmented_iterations_}
            test_steps=()
            if [ "${rest%_test_2_steps}" != "$rest" ]; then
                rest=${rest%_test_2_steps}
                # Code check: 16 IDs = 2 steps per iteration, 2 iterations, 1 checkpoint per step.
                test_steps=("${PART_TEST_2_STEPS[@]}" REFINEMENT_MAX_ROUNDS=2 REFINEMENT_EVAL_SAVE_STEPS=0)
            fi
            relevance=()
            if [ "${rest%_teacher_kl_relevance_0.5}" != "$rest" ]; then
                rest=${rest%_teacher_kl_relevance_0.5}
                relevance=(REFINEMENT_TEACHER_KL_RELEVANCE_WEIGHT=0.5)
            fi
            case "$rest" in
                ema_step_0.99) ema=(REFINEMENT_TEACHER_UPDATE=optimizer_step) ;;
                ema_iteration_0.99) ema=(REFINEMENT_TEACHER_UPDATE=after_round) ;;
                *) echo "ERROR: unknown EMA setting '$rest' in $1" >&2; exit 1 ;;
            esac
            tuning "${ITERATIONS_COMPOSITE_MASK_CONTENT[@]}" "${ema[@]}" \
                ${relevance[@]:+"${relevance[@]}"} ${test_steps[@]:+"${test_steps[@]}"}
            if [ ${#test_steps[@]} -gt 0 ]; then VERSION_TIME_LIMIT=01:00:00; else VERSION_TIME_LIMIT=30:00:00; fi ;;
        stage2_tuning_unlabeled1008_augmentation_*)
            operation=${1#stage2_tuning_unlabeled1008_augmentation_}
            test_steps=()
            if [ "${operation%_test_2_steps}" != "$operation" ]; then
                operation=${operation%_test_2_steps}
                test_steps=("${PART_TEST_2_STEPS[@]}")
            fi
            labeled=()
            if [ "${operation%_labeled_augmented}" != "$operation" ]; then
                operation=${operation%_labeled_augmented}
                labeled=("${LABELED_AUGMENTED[@]}")
            fi
            target=()
            if [ "${operation%_inspector_one_hot}" != "$operation" ]; then
                operation=${operation%_inspector_one_hot}
                target=("${INSPECTOR_ONE_HOT[@]}")
            fi
            if [[ "$operation" =~ ^($SINGLE_OPERATIONS_PATTERN)$ ]]; then
                value=$operation
            elif [ -n "${COMBINED_OPERATIONS[$operation]:-}" ]; then
                value=${COMBINED_OPERATIONS[$operation]}
            else
                echo "ERROR: unknown operation or combination '$operation' in $1" >&2
                exit 1
            fi
            if [ ${#target[@]} -gt 0 ] && [ "$operation" != composite_blur_noise_mask_content ]; then
                echo "ERROR: the one-hot Inspector version exists only for the 4-operation combination." >&2
                exit 1
            fi
            if [ ${#labeled[@]} -gt 0 ] && [[ ! "$operation" =~ ^(composite_blur_noise_mask_content|composite_blur_mask_content|composite_noise_mask_content)$ ]]; then
                echo "ERROR: the labeled-augmented version exists only for composite_blur_noise_mask_content, composite_blur_mask_content and composite_noise_mask_content." >&2
                exit 1
            fi
            # One operation or one combination (plus the code check of 2 steps for *_test_2_steps).
            tuning "${PROBABILITY_05_07_SINGLE_OPERATION[@]}" REFINEMENT_AUGMENTATION_OPERATION=$value \
                ${target[@]:+"${target[@]}"} ${labeled[@]:+"${labeled[@]}"} ${test_steps[@]:+"${test_steps[@]}"}
            if [ ${#test_steps[@]} -gt 0 ]; then VERSION_TIME_LIMIT=00:30:00; else VERSION_TIME_LIMIT=05:00:00; fi ;;
        # ----- Claim: Student initialized from the Inspector (one change: the Student-Stage1) -----
        stage2_tuning_starting_point_student_initialized_from_inspector)
            # Starting point with Student-Stage1 = batch 16, LLM LoRA initialized from Inspector checkpoint-63.
            tuning TRAIN_LOOP_STUDENT_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/student_warmstart.pt \
                TRAIN_LOOP_TEACHER_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32_initialized_from_inspector_checkpoint63/teacher_initialized.pt
            VERSION_TIME_LIMIT=06:00:00 ;;
        stage2_tuning_starting_point_student_batch16)
            # Control: Student-Stage1 = batch 16, fresh LoRA (same Stage 1 setting except the initialization).
            tuning TRAIN_LOOP_STUDENT_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32/student_warmstart.pt \
                TRAIN_LOOP_TEACHER_CHECKPOINT=$OUT_ROOT/student_pipeline_overall_finegrained_gemmaGT_1008_r64_lr4e-5_batch16_head_fp32/teacher_initialized.pt
            VERSION_TIME_LIMIT=06:00:00 ;;
        # ----- Claim 2: loss weights, one pass over the 1008 labeled pairs, Student-Stage1 batch 8 -----
        stage2_tuning_labeled_one_pass_starting_point)
            tuning "${LABELED_ONE_PASS_CONFIG[@]}"; VERSION_TIME_LIMIT=07:00:00 ;;
        stage2_tuning_labeled_one_pass_lambda_unlabeled0|stage2_tuning_labeled_one_pass_lambda_unlabeled0.1|\
        stage2_tuning_labeled_one_pass_lambda_unlabeled0.25|stage2_tuning_labeled_one_pass_lambda_unlabeled0.5|\
        stage2_tuning_labeled_one_pass_lambda_unlabeled2)
            # lambda_U = 0: Unlabeled still forwarded (same data stream and schedules), loss weight 0.
            tuning "${LABELED_ONE_PASS_CONFIG[@]}" REFINEMENT_LAMBDA_SET=${1##*lambda_unlabeled}
            VERSION_TIME_LIMIT=07:00:00 ;;
        stage2_tuning_labeled_one_pass_lambda_mask0.5|stage2_tuning_labeled_one_pass_lambda_mask2)
            tuning "${LABELED_ONE_PASS_CONFIG[@]}" REFINEMENT_LAMBDA_MASK=${1##*lambda_mask}
            VERSION_TIME_LIMIT=07:00:00 ;;
        stage2_tuning_labeled_one_pass_lambda_gemma0.5|stage2_tuning_labeled_one_pass_lambda_gemma2)
            tuning "${LABELED_ONE_PASS_CONFIG[@]}" REFINEMENT_LAMBDA_GEMMA=${1##*lambda_gemma}
            VERSION_TIME_LIMIT=07:00:00 ;;
        stage2_tuning_labeled_one_pass_lambda_consistency0|stage2_tuning_labeled_one_pass_lambda_consistency0.5|\
        stage2_tuning_labeled_one_pass_lambda_consistency2)
            tuning "${LABELED_ONE_PASS_CONFIG[@]}" REFINEMENT_LAMBDA_CONSISTENCY=${1##*lambda_consistency}
            VERSION_TIME_LIMIT=07:00:00 ;;
        stage2_tuning_three_rounds_inspector_teacher_rank)
            # Group K: 667 Unlabeled per round by Inspector confidence (easy, medium, hard), 25% replay.
            tuning REFINEMENT_UNLABELED_SAMPLING=inspector_teacher_rank REFINEMENT_MAX_ROUNDS=3 \
                REFINEMENT_UNLABELED_PER_ROUND=667; VERSION_TIME_LIMIT=08:00:00 ;;
        train_loop_pipeline_1008_inspector_teacher_rank_three_rounds_three_way_selected_equal)
            # 3,334 / 4,168 / 4,166 Unlabeled occurrences (25% hard replay from round 2), about 974 steps.
            VERSION_CONFIG=(
                REFINEMENT_UNLABELED_SAMPLING=inspector_teacher_rank
                REFINEMENT_MAX_ROUNDS=3
                REFINEMENT_UNLABELED_PER_ROUND=3334
            ) ;;
        *)
            echo "ERROR: unknown version $1; add it to version_config first." >&2
            exit 1 ;;
    esac
}

partition_args() {
    case "$1" in
        full) PARTITION_ARGS=(--partition=full --qos=mig7-141 --gres=gpu:7g.141gb:1) ;;
        large) PARTITION_ARGS=(--partition=large --qos=mig3-71 --gres=gpu:3g.71gb:1) ;;
        # mig2-35 allows 5 running jobs, 24 CPUs and 192 GB per user: 4 CPUs / 36 GB lets 5 run.
        normal) PARTITION_ARGS=(--partition=normal --qos=mig2-35 --gres=gpu:2g.35gb:1 --cpus-per-task=4 --mem=36G) ;;
        *) echo "ERROR: partition must be full, large or normal." >&2; exit 1 ;;
    esac
}

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
    echo "Usage: bash $0 <version> <full|large|normal> [--dry-run]" >&2
    exit 1
fi
VERSION=$1
DRY_RUN=0
if [ $# -eq 3 ]; then
    [ "$3" = "--dry-run" ] || { echo "ERROR: unknown option $3" >&2; exit 1; }
    DRY_RUN=1
fi
VERSION_TIME_LIMIT=""
version_config "$VERSION"
TIME_LIMIT=${VERSION_TIME_LIMIT:-$TIME_LIMIT}
partition_args "$2"

OUTPUT_DIR=$OUT_ROOT/$VERSION
# A version setting replaces the COMMON_CONFIG entry of the same name in place,
# so every variable is exported exactly once.
declare -A OVERRIDE=()
for entry in ${VERSION_CONFIG[@]:+"${VERSION_CONFIG[@]}"}; do
    OVERRIDE[${entry%%=*}]=$entry
done
CONFIG=(REFINEMENT_OUTPUT_DIR=$OUTPUT_DIR)
for entry in "${COMMON_CONFIG[@]}"; do
    key=${entry%%=*}
    if [ -n "${OVERRIDE[$key]:-}" ]; then
        CONFIG+=("${OVERRIDE[$key]}")
        unset "OVERRIDE[$key]"
    else
        CONFIG+=("$entry")
    fi
done
for entry in ${VERSION_CONFIG[@]:+"${VERSION_CONFIG[@]}"}; do
    [ -n "${OVERRIDE[${entry%%=*}]:-}" ] && CONFIG+=("$entry")
done
if [ "$(printf '%s\n' "${CONFIG[@]%%=*}" | sort | uniq -d)" != "" ]; then
    echo "ERROR: a configuration variable is set twice." >&2
    exit 1
fi
EXPORT=$(IFS=,; echo "ALL,${CONFIG[*]}")
# Optional SUBMIT_DEPENDENCY=<job id>: start only after that job completed successfully.
DEPENDENCY_ARGS=()
[ -n "${SUBMIT_DEPENDENCY:-}" ] && DEPENDENCY_ARGS=(--dependency=afterok:$SUBMIT_DEPENDENCY)
SBATCH=(sbatch --parsable "${PARTITION_ARGS[@]}" ${DEPENDENCY_ARGS[@]:+"${DEPENDENCY_ARGS[@]}"} --time="$TIME_LIMIT" --job-name="$VERSION" --export="$EXPORT" "$LAUNCHER")

if [ "$DRY_RUN" = 0 ]; then
    if [ -e "$OUTPUT_DIR" ]; then
        echo "ERROR: $OUTPUT_DIR already exists; a new run needs a new version name." >&2
        exit 1
    fi
    mkdir "$OUTPUT_DIR"
    cd "$ROOT"
    JOB_ID=$("${SBATCH[@]}")
else
    JOB_ID="dry-run"
fi

RECORD=$(
    echo "===== $(date '+%Y-%m-%d %H:%M:%S %z') | job $JOB_ID | user $(whoami)"
    echo "version:      $VERSION"
    echo "output_dir:   $OUTPUT_DIR"
    echo "resources:    ${PARTITION_ARGS[*]} ${DEPENDENCY_ARGS[*]:-} --time=$TIME_LIMIT"
    echo "launcher:     $LAUNCHER md5=$(md5sum "$LAUNCHER" | cut -d' ' -f1)"
    echo "train_loop:   $TRAIN_SCRIPT md5=$(md5sum "$TRAIN_SCRIPT" | cut -d' ' -f1)"
    echo "config:"
    printf '  %s\n' "${CONFIG[@]}"
)
if [ "$DRY_RUN" = 0 ]; then
    printf '%s\n\n' "$RECORD" >> "$SUBMISSION_LOG"
    echo "Submitted job $JOB_ID; recorded in $SUBMISSION_LOG"
else
    printf '%q ' "${SBATCH[@]}"; echo
fi
echo "$RECORD"
