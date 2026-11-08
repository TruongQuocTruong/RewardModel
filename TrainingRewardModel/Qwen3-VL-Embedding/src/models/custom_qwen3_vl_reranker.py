"""Qwen3-VL six-metric reward model components.

This module defines the shared architecture of the six-metric Student and EMA
Teacher, together with the independent three-way Overall Inspector.  The Inspector
must use a separate model/adapter and must not share trainable parameters with
the Student.  The Inspector reuses the Student Score and Tie Heads with one
overall output (no Relevance Head).  The multi-metric model consumes three representations for every pair:
candidate A, candidate B, and a source-only representation.  It exposes three
trainable sub-heads:

* a shared score head for A and B;
* a swap-invariant tie head; and
* a source-only multi-label relevance head.

The public pairwise forward contract is :class:`MultiMetricRewardOutput`.  It no
longer overloads the causal-language-model ``lm_head`` with a flat 12-value
output.  The original ``lm_head`` remains intact and the reward heads are
registered under ``model.metric_heads`` so they can be saved explicitly by
PEFT in the training integration.
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.utils import ModelOutput
from qwen_vl_utils import process_vision_info

from src.prompt_config import DEFAULT_SYSTEM_MESSAGE, DEFAULT_SYSTEM_PROMPT

logger = logging.getLogger(__name__)


METRIC_NAMES = (
    "aesthetic",
    "fidelity",
    "realism",
    "background_consistency",
    "foreground_consistency",
    "structure_consistency",
)

NUM_METRICS: int = len(METRIC_NAMES)
# ``TOTAL_OUTPUTS`` describes only a legacy [preferences, mask] label width;
# it is not the output width of the structured pairwise model.
TOTAL_OUTPUTS: int = NUM_METRICS * 2

DEFAULT_HEAD_HIDDEN_DIM: int = 512
DEFAULT_TIE_PROJECTION_DIM: int = 256
DEFAULT_HEAD_DROPOUT: float = 0.05
LM_HEAD_INIT_ORTHOGONAL_SCALE: float = 0.35
LM_HEAD_INIT_EPSILON: float = 1e-8

# Tie Head variants (ablation 2026-10-26).  ``independent``: t = sigmoid(MLP).
# ``margin_coupled``: logit(t) = theta0 - kappa * |d| + residual MLP, with
# d = score_A - score_B, so tie pairs also pull |d| towards 0.
# ``margin_coupled_detached``: same, but no gradient flows from t into d.
# ``margin_only``: logit(t) = theta0 - kappa * |d| (Rao-Kupper form, no residual).
TIE_HEAD_MODES = ("independent", "margin_coupled", "margin_coupled_detached", "margin_only")
DEFAULT_TIE_HEAD_MODE = "independent"
TIE_HEAD_MODE_ENV = "TIE_HEAD_MODE"
DEFAULT_TIE_PRIOR: float = 0.2
DEFAULT_TIE_KAPPA: float = 1.0
TIE_COUPLING_KEY = "tie_coupling"


def resolve_tie_head_mode(mode: Optional[str] = None) -> str:
    """Explicit mode, else ``TIE_HEAD_MODE`` from the environment, else independent."""
    value = (mode or os.environ.get(TIE_HEAD_MODE_ENV, "") or DEFAULT_TIE_HEAD_MODE).strip()
    if value not in TIE_HEAD_MODES:
        raise ValueError(f"Unsupported Tie Head mode {value!r}; expected one of {TIE_HEAD_MODES}.")
    return value


def tie_head_mode_from_state_keys(keys: Any) -> str:
    """Structure needed to load saved head weights.

    The detached variant differs only in gradients, and ``margin_only`` keeps
    its never-trained residual at exactly zero, so every coupled variant loads
    and infers identically as ``margin_coupled``.
    """
    return "margin_coupled" if any(TIE_COUPLING_KEY in str(key) for key in keys) else "independent"

METRIC_TO_INDEX: Dict[str, int] = {
    "aesthetic": 0,
    "aesthetic quality": 0,
    "fidelity": 1,
    "realism": 2,
    "background consistency": 3,
    "background_consistency": 3,
    "bc": 3,
    "foreground consistency": 4,
    "foreground_consistency": 4,
    "fc": 4,
    "structure consistency": 5,
    "structure_consistency": 5,
    "sc": 5,
}


@dataclass
class MultiMetricRewardOutput(ModelOutput):
    """Pairwise output contract shared by the six-metric Student and Teacher."""

    scores_A: Optional[torch.FloatTensor] = None
    scores_B: Optional[torch.FloatTensor] = None
    margins: Optional[torch.FloatTensor] = None
    cond_win_prob_A: Optional[torch.FloatTensor] = None
    tie_probs: Optional[torch.FloatTensor] = None
    relevance_probs: Optional[torch.FloatTensor] = None
    three_way_probs: Optional[torch.FloatTensor] = None
    signed_evidence: Optional[torch.FloatTensor] = None


class OverallInspectorHead(nn.Module):
    """Three-way overall head: the Student Score and Tie Heads with one output.

    ``P(A) = (1 - t) w``, ``P(B) = (1 - t)(1 - w)`` and ``P(Tie) = t`` with
    ``w = sigmoid(score_A - score_B)``.  The score projection is seeded exactly
    from ``W_yes - W_no`` and its residual MLP starts at zero, so at
    initialization the margin equals the pretrained reranker margin.  There is
    no Relevance Head.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        tie_projection_dim: int = DEFAULT_TIE_PROJECTION_DIM,
        dropout_p: float = DEFAULT_HEAD_DROPOUT,
        tie_head_mode: str = DEFAULT_TIE_HEAD_MODE,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("hidden_dim", hidden_dim),
            ("tie_projection_dim", tie_projection_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be finite and in [0, 1).")
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tie_projection_dim = tie_projection_dim
        self.dropout_p = float(dropout_p)
        self.score_head = _SharedScoreHead(input_dim, hidden_dim, 1, self.dropout_p)
        self.tie_head = _SymmetricTieHead(
            input_dim, tie_projection_dim, hidden_dim, 1, self.dropout_p
        )
        self.tie_head_mode = resolve_tie_head_mode(tie_head_mode)
        coupling = _build_tie_coupling(self.tie_head, 1, self.tie_head_mode)
        if coupling is not None:
            self.tie_coupling = coupling

    def initialize_from_lm_head_direction(self, direction: torch.Tensor) -> None:
        if direction.ndim != 1 or direction.numel() != self.input_dim:
            raise ValueError(
                f"lm_head direction must have shape [{self.input_dim}], "
                f"got {tuple(direction.shape)}."
            )
        direction_fp32 = direction.detach().float()
        if not torch.isfinite(direction_fp32).all():
            raise ValueError("The lm_head yes/no direction contains non-finite values.")
        if direction_fp32.norm() <= LM_HEAD_INIT_EPSILON:
            raise ValueError("The lm_head yes/no direction has zero norm.")
        projection = self.score_head.pretrained_projection
        with torch.no_grad():
            projection.weight.copy_(
                direction_fp32.to(
                    device=projection.weight.device, dtype=projection.weight.dtype
                ).unsqueeze(0)
            )

    def forward(self, h_A: torch.Tensor, h_B: torch.Tensor) -> MultiMetricRewardOutput:
        for name, value in (("h_A", h_A), ("h_B", h_B)):
            if (
                not isinstance(value, torch.Tensor)
                or value.ndim != 2
                or value.size(1) != self.input_dim
            ):
                actual = (
                    tuple(value.shape)
                    if isinstance(value, torch.Tensor)
                    else type(value).__name__
                )
                raise ValueError(
                    f"{name} must have shape [B, {self.input_dim}], got {actual}."
                )
        if h_A.shape != h_B.shape:
            raise ValueError("h_A and h_B must have the same shape.")
        head_dtype = self.score_head.hidden.weight.dtype
        h_A = h_A.to(dtype=head_dtype)
        h_B = h_B.to(dtype=head_dtype)
        scores_A, scores_B = self.score_head.forward_pair(h_A, h_B)
        scores_A = scores_A.float()
        scores_B = scores_B.float()
        margins = scores_A - scores_B
        if self.tie_head_mode == "margin_only":
            tie_logits = self.tie_coupling(margins, torch.zeros_like(margins))
        else:
            tie_logits = self.tie_head(h_A, h_B).float()
            if self.tie_head_mode != "independent":
                tie_logits = self.tie_coupling(margins, tie_logits)
        cond_win_prob_A = torch.sigmoid(margins)
        tie_probs = torch.sigmoid(tie_logits)
        non_tie_probs = 1.0 - tie_probs
        prob_a = non_tie_probs * cond_win_prob_A
        prob_b = non_tie_probs * (1.0 - cond_win_prob_A)
        return MultiMetricRewardOutput(
            scores_A=scores_A,
            scores_B=scores_B,
            margins=margins,
            cond_win_prob_A=cond_win_prob_A,
            tie_probs=tie_probs,
            three_way_probs=torch.stack((prob_a, prob_b, tie_probs), dim=-1),
            signed_evidence=prob_a - prob_b,
        )


class OverallInspectorLoss(nn.Module):
    """Three-way cross-entropy on human overall votes.

    ``targets`` is ``[B, 3]`` with columns ``(y_A, y_B, y_T)``.  The factorized
    form ``BCE(y_T, t) + (y_A + y_B) BCE(y_A / (y_A + y_B), w)`` equals
    ``-log P(true class)`` for one-hot votes; it is computed from logits for
    numerical stability.
    """

    def per_pair(
        self,
        output: MultiMetricRewardOutput,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        if output.margins is None or output.tie_probs is None:
            raise ValueError("Inspector output must contain margins and tie_probs.")
        margins = output.margins
        if margins.ndim != 2 or margins.size(1) != 1 or margins.size(0) == 0:
            raise ValueError(f"Inspector margins must have shape [B, 1], got {tuple(margins.shape)}.")
        if not isinstance(targets, torch.Tensor) or tuple(targets.shape) != (margins.size(0), 3):
            raise ValueError("Inspector targets must have shape [B, 3].")
        targets = targets.to(device=margins.device, dtype=torch.float32)
        if not torch.isfinite(targets).all() or ((targets < 0.0) | (targets > 1.0)).any():
            raise ValueError("Inspector targets must be probabilities in [0, 1].")
        if not torch.allclose(targets.sum(dim=1), torch.ones_like(targets[:, 0]), atol=1e-6):
            raise ValueError("Inspector targets must sum to 1 for every pair.")
        y_a, y_b, y_t = targets.unbind(dim=1)
        tie_probs = output.tie_probs[:, 0].float()
        tie_logits = torch.logit(tie_probs.clamp(1e-6, 1.0 - 1e-6))
        tie_loss = F.binary_cross_entropy_with_logits(tie_logits, y_t, reduction="none")
        non_tie_mass = y_a + y_b
        conditional_target_a = torch.where(
            non_tie_mass > 0.0,
            y_a / non_tie_mass.clamp_min(1e-12),
            torch.zeros_like(y_a),
        )
        winner_loss = F.binary_cross_entropy_with_logits(
            margins[:, 0].float(), conditional_target_a, reduction="none"
        )
        return tie_loss + non_tie_mass * winner_loss

    def forward(
        self,
        output: MultiMetricRewardOutput,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.per_pair(output, targets).mean()


class FactorizedWarmStartLoss(nn.Module):
    """Gemma warm-start objective for the structured three-head model.

    ``g_A``, ``g_B`` and ``g_T`` are empirical probabilities computed only
    from Gemma runs where the metric is applicable. ``relevance_targets`` is
    the applicable-run fraction and weights the preference term. Consequently
    a non-applicable metric supervises only the Relevance Head; it is not
    incorrectly converted into a semantic tie.
    """

    def __init__(self, relevance_weight: float = 1.0) -> None:
        super().__init__()
        self.relevance_weight = float(relevance_weight)
        if not math.isfinite(self.relevance_weight) or self.relevance_weight < 0.0:
            raise ValueError("relevance_weight must be finite and non-negative.")

    @staticmethod
    def _validate_target(name: str, value: torch.Tensor, shape: tuple[int, int]) -> None:
        if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape:
            actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
            raise ValueError(f"{name} must have shape {shape}, got {actual}.")
        if not value.is_floating_point():
            raise ValueError(f"{name} must be a floating-point tensor.")
        if not torch.isfinite(value).all() or ((value < 0.0) | (value > 1.0)).any():
            raise ValueError(f"{name} must contain finite probabilities in [0, 1].")

    def components(
        self,
        output: MultiMetricRewardOutput,
        g_A: torch.Tensor,
        g_B: torch.Tensor,
        g_T: torch.Tensor,
        relevance_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        required = {
            "cond_win_prob_A": output.cond_win_prob_A,
            "tie_probs": output.tie_probs,
            "relevance_probs": output.relevance_probs,
        }
        if any(value is None for value in required.values()):
            missing = [name for name, value in required.items() if value is None]
            raise ValueError(f"MultiMetricRewardOutput is missing required fields: {missing}.")
        shape = tuple(output.tie_probs.shape)
        if len(shape) != 2 or shape[1] != NUM_METRICS or shape[0] == 0:
            raise ValueError(
                f"Student probabilities must have shape [B, {NUM_METRICS}], got {shape}."
            )
        for name, value in (
            ("g_A", g_A),
            ("g_B", g_B),
            ("g_T", g_T),
            ("relevance_targets", relevance_targets),
        ):
            self._validate_target(name, value, shape)

        device = output.tie_probs.device
        dtype = output.tie_probs.dtype
        g_A = g_A.to(device=device, dtype=dtype)
        g_B = g_B.to(device=device, dtype=dtype)
        g_T = g_T.to(device=device, dtype=dtype)
        relevance_targets = relevance_targets.to(device=device, dtype=dtype)
        vote_sums = g_A + g_B + g_T
        valid_vote_sums = torch.isclose(
            vote_sums, torch.zeros_like(vote_sums), atol=1e-6, rtol=0.0
        ) | torch.isclose(
            vote_sums, torch.ones_like(vote_sums), atol=1e-6, rtol=0.0
        )
        if not valid_vote_sums.all():
            raise ValueError(
                "g_A + g_B + g_T must equal 1 for applicable metrics and 0 "
                "for metrics with no applicable Gemma run."
            )
        if ((vote_sums == 0.0) & (relevance_targets > 0.0)).any():
            raise ValueError(
                "A positive relevance target requires a conditional preference target."
            )

        non_tie_mass = g_A + g_B
        conditional_target_A = torch.where(
            non_tie_mass > 0.0,
            g_A / non_tie_mass.clamp_min(torch.finfo(dtype).eps),
            torch.zeros_like(g_A),
        )
        tie_loss = F.binary_cross_entropy(
            output.tie_probs, g_T, reduction="none"
        )
        winner_loss = F.binary_cross_entropy(
            output.cond_win_prob_A, conditional_target_A, reduction="none"
        )
        applicability_weight = relevance_targets
        weighted_tie_loss = applicability_weight * tie_loss
        weighted_winner_loss = (
            applicability_weight * non_tie_mass * winner_loss
        )
        preference_loss = (weighted_tie_loss + weighted_winner_loss).mean()
        relevance_loss = F.binary_cross_entropy(
            output.relevance_probs, relevance_targets
        )
        total_loss = preference_loss + self.relevance_weight * relevance_loss
        return {
            "loss": total_loss,
            "preference_loss": preference_loss,
            "tie_loss": weighted_tie_loss.mean(),
            "conditional_winner_loss": weighted_winner_loss.mean(),
            "relevance_loss": relevance_loss,
        }

    def forward(
        self,
        output: MultiMetricRewardOutput,
        g_A: torch.Tensor,
        g_B: torch.Tensor,
        g_T: torch.Tensor,
        relevance_targets: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(output, g_A, g_B, g_T, relevance_targets)["loss"]


class _SharedScoreHead(nn.Module):
    """Shared A/B pretrained projection plus residual MLP.

    Sharing the stochastic dropout mask across A and B preserves exact margin
    anti-symmetry in every Monte Carlo forward, not only in evaluation mode.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_metrics: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.pretrained_projection = nn.Linear(input_dim, num_metrics, bias=False)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.residual_output = nn.Linear(hidden_dim, num_metrics)
        # At initialization the head is exactly the projection seeded from the
        # Qwen reranker's yes/no direction.  The MLP learns only a residual.
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    def forward_pair(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        base_a = self.pretrained_projection(h_a)
        base_b = self.pretrained_projection(h_b)
        hidden_a = self.activation(self.hidden(h_a))
        hidden_b = self.activation(self.hidden(h_b))
        if self.training and self.dropout.p > 0.0:
            # Passing ones through nn.Dropout creates one inverted-dropout mask
            # per pair.  Applying that same mask to both candidates makes a
            # stochastic A/B swap exactly equivariant.
            shared_mask = self.dropout(torch.ones_like(hidden_a))
            hidden_a = hidden_a * shared_mask
            hidden_b = hidden_b * shared_mask
        else:
            hidden_a = self.dropout(hidden_a)
            hidden_b = self.dropout(hidden_b)
        return (
            base_a + self.residual_output(hidden_a),
            base_b + self.residual_output(hidden_b),
        )


class _SymmetricTieHead(nn.Module):
    """Swap-invariant tie predictor based on symmetric pair features."""

    def __init__(
        self,
        input_dim: int,
        projection_dim: int,
        hidden_dim: int,
        num_metrics: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        # Projection stays deterministic.  Dropout is applied only after the
        # symmetric representation has been formed.
        self.projection = nn.Linear(input_dim, projection_dim)
        self.mlp = nn.Sequential(
            nn.Linear(2 * projection_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_dim, num_metrics),
        )

    def forward(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        u_a = self.projection(h_a)
        u_b = self.projection(h_b)
        symmetric_features = torch.cat(
            [0.5 * (u_a + u_b), torch.abs(u_a - u_b)],
            dim=-1,
        )
        return self.mlp(symmetric_features)

    def zero_init_output(self) -> None:
        """Start as a zero residual (used by the margin-coupled variants)."""
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)


class _MarginCoupledTie(nn.Module):
    """logit(t) = theta0 - kappa * |d| + residual, kappa = softplus(raw) > 0."""

    def __init__(self, num_outputs: int, prior: float, kappa: float, detach_margin: bool) -> None:
        super().__init__()
        if not 0.0 < prior < 1.0 or kappa <= 0.0:
            raise ValueError("Tie prior must be in (0, 1) and kappa positive.")
        self.detach_margin = bool(detach_margin)
        self.theta0 = nn.Parameter(torch.full((num_outputs,), math.log(prior / (1.0 - prior))))
        # softplus^{-1}(kappa)
        self.kappa_raw = nn.Parameter(torch.full((num_outputs,), math.log(math.expm1(kappa))))

    def forward(self, margins: torch.Tensor, residual_logits: torch.Tensor) -> torch.Tensor:
        if self.detach_margin:
            margins = margins.detach()
        theta0 = self.theta0.to(dtype=residual_logits.dtype)
        kappa = F.softplus(self.kappa_raw).to(dtype=residual_logits.dtype)
        return theta0 - kappa * margins.to(dtype=residual_logits.dtype).abs() + residual_logits


def _build_tie_coupling(
    tie_head: _SymmetricTieHead, num_outputs: int, mode: str
) -> Optional[_MarginCoupledTie]:
    if mode == "independent":
        return None
    tie_head.zero_init_output()
    return _MarginCoupledTie(
        num_outputs,
        DEFAULT_TIE_PRIOR,
        DEFAULT_TIE_KAPPA,
        detach_margin=(mode == "margin_coupled_detached"),
    )


class _RelevanceHead(nn.Module):
    """Source-only pretrained projection plus residual relevance MLP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_metrics: int,
        dropout_p: float,
    ) -> None:
        super().__init__()
        self.pretrained_projection = nn.Linear(input_dim, num_metrics, bias=False)
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(p=dropout_p)
        self.residual_output = nn.Linear(hidden_dim, num_metrics)
        nn.init.zeros_(self.residual_output.weight)
        nn.init.zeros_(self.residual_output.bias)

    def forward(self, h_src: torch.Tensor) -> torch.Tensor:
        residual = self.residual_output(
            self.dropout(self.activation(self.hidden(h_src)))
        )
        return self.pretrained_projection(h_src) + residual


class MultiMetricRewardHeads(nn.Module):
    """Three-head pairwise reward module implementing the output contract."""

    def __init__(
        self,
        input_dim: int,
        *,
        num_metrics: int = NUM_METRICS,
        hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        tie_projection_dim: int = DEFAULT_TIE_PROJECTION_DIM,
        dropout_p: float = DEFAULT_HEAD_DROPOUT,
        tie_head_mode: str = DEFAULT_TIE_HEAD_MODE,
    ) -> None:
        super().__init__()
        for name, value in (
            ("input_dim", input_dim),
            ("num_metrics", num_metrics),
            ("hidden_dim", hidden_dim),
            ("tie_projection_dim", tie_projection_dim),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        if not math.isfinite(dropout_p) or not 0.0 <= dropout_p < 1.0:
            raise ValueError("dropout_p must be finite and in [0, 1).")

        self.input_dim = input_dim
        self.num_metrics = num_metrics
        self.hidden_dim = hidden_dim
        self.tie_projection_dim = tie_projection_dim
        self.dropout_p = float(dropout_p)

        self.score_head = _SharedScoreHead(
            input_dim, hidden_dim, num_metrics, self.dropout_p
        )
        self.tie_head = _SymmetricTieHead(
            input_dim,
            tie_projection_dim,
            hidden_dim,
            num_metrics,
            self.dropout_p,
        )
        self.relevance_head = _RelevanceHead(
            input_dim, hidden_dim, num_metrics, self.dropout_p
        )
        self.tie_head_mode = resolve_tie_head_mode(tie_head_mode)
        coupling = _build_tie_coupling(self.tie_head, num_metrics, self.tie_head_mode)
        if coupling is not None:
            self.tie_coupling = coupling

    @staticmethod
    def _build_lm_seed_rows(
        direction: torch.Tensor,
        num_metrics: int,
    ) -> torch.Tensor:
        """Create distinct metric rows around the pretrained yes/no direction."""
        direction_fp32 = direction.detach().float()
        if direction_fp32.ndim != 1:
            raise ValueError("The lm_head yes/no direction must be rank 1.")
        if not torch.isfinite(direction_fp32).all():
            raise ValueError("The lm_head yes/no direction contains non-finite values.")
        direction_norm = direction_fp32.norm()
        if direction_norm <= LM_HEAD_INIT_EPSILON:
            raise ValueError("The lm_head yes/no direction has zero norm.")

        direction_unit = direction_fp32 / direction_norm
        basis = []
        for _ in range(num_metrics):
            vector = torch.randn_like(direction_fp32)
            vector = vector - torch.dot(vector, direction_unit) * direction_unit
            for previous in basis:
                vector = vector - torch.dot(vector, previous) * previous
            vector_norm = vector.norm()
            if vector_norm <= LM_HEAD_INIT_EPSILON:
                raise RuntimeError("Failed to construct an orthogonal metric initializer.")
            basis.append(vector / vector_norm)
        orthogonal_basis = torch.stack(basis)
        return (
            direction_fp32.unsqueeze(0)
            + LM_HEAD_INIT_ORTHOGONAL_SCALE * direction_norm * orthogonal_basis
        )

    def initialize_from_lm_head_direction(self, direction: torch.Tensor) -> None:
        """Seed Score and Relevance projections from ``W_yes - W_no``.

        Tie prediction has no compatible pretrained yes/no direction because
        it operates on a symmetric pair representation.  Its projection/MLP
        therefore retain their independent neural-network initialization.
        """
        if direction.numel() != self.input_dim:
            raise ValueError(
                f"lm_head direction must contain {self.input_dim} values, "
                f"got {direction.numel()}."
            )
        seed_rows = self._build_lm_seed_rows(direction, self.num_metrics)
        with torch.no_grad():
            for projection in (
                self.score_head.pretrained_projection,
                self.relevance_head.pretrained_projection,
            ):
                projection.weight.copy_(
                    seed_rows.to(
                        device=projection.weight.device,
                        dtype=projection.weight.dtype,
                    )
                )

    def _validate_representations(
        self,
        h_a: torch.Tensor,
        h_b: torch.Tensor,
        h_src: torch.Tensor,
    ) -> None:
        expected_shape = (
            (h_a.size(0), self.input_dim)
            if isinstance(h_a, torch.Tensor) and h_a.ndim == 2
            else None
        )
        for name, value in (("h_A", h_a), ("h_B", h_b), ("h_src", h_src)):
            if not isinstance(value, torch.Tensor) or value.ndim != 2:
                actual = tuple(value.shape) if isinstance(value, torch.Tensor) else type(value).__name__
                raise ValueError(f"{name} must be a rank-2 tensor, got {actual}.")
            if expected_shape is None or tuple(value.shape) != expected_shape:
                raise ValueError(
                    f"h_A, h_B, and h_src must all have shape [B, {self.input_dim}]."
                )
            if not value.is_floating_point():
                raise ValueError(f"{name} must be a floating-point tensor.")
            if value.device != h_a.device:
                raise ValueError("h_A, h_B, and h_src must be on the same device.")

    def compute_head_logits(
        self,
        h_A: torch.Tensor,
        h_B: torch.Tensor,
        h_src: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return score outputs plus raw tie/relevance logits for stable losses."""
        self._validate_representations(h_A, h_B, h_src)
        head_dtype = self.score_head.hidden.weight.dtype
        h_A = h_A.to(dtype=head_dtype)
        h_B = h_B.to(dtype=head_dtype)
        h_src = h_src.to(dtype=head_dtype)
        scores_A, scores_B = self.score_head.forward_pair(h_A, h_B)
        if self.tie_head_mode == "margin_only":
            margins = scores_A.float() - scores_B.float()
            tie_logits = self.tie_coupling(margins, torch.zeros_like(margins))
        else:
            tie_logits = self.tie_head(h_A, h_B)
            if self.tie_head_mode != "independent":
                tie_logits = self.tie_coupling(
                    scores_A.float() - scores_B.float(), tie_logits.float()
                )
        relevance_logits = self.relevance_head(h_src)
        # The external contract is float32 even when the backbone/heads run in
        # bfloat16 under autocast.  Casting preserves autograd.
        return (
            scores_A.float(),
            scores_B.float(),
            tie_logits.float(),
            relevance_logits.float(),
        )

    def forward(
        self,
        h_A: torch.Tensor,
        h_B: torch.Tensor,
        h_src: torch.Tensor,
    ) -> MultiMetricRewardOutput:
        scores_A, scores_B, tie_logits, relevance_logits = self.compute_head_logits(
            h_A, h_B, h_src
        )
        margins = scores_A - scores_B
        cond_win_prob_A = torch.sigmoid(margins)
        tie_probs = torch.sigmoid(tie_logits)
        relevance_probs = torch.sigmoid(relevance_logits)

        non_tie_probs = 1.0 - tie_probs
        prob_a = non_tie_probs * cond_win_prob_A
        prob_b = non_tie_probs * (1.0 - cond_win_prob_A)
        three_way_probs = torch.stack((prob_a, prob_b, tie_probs), dim=-1)
        signed_evidence = prob_a - prob_b

        return MultiMetricRewardOutput(
            scores_A=scores_A,
            scores_B=scores_B,
            margins=margins,
            cond_win_prob_A=cond_win_prob_A,
            tie_probs=tie_probs,
            relevance_probs=relevance_probs,
            three_way_probs=three_way_probs,
            signed_evidence=signed_evidence,
        )


def _resolve_hidden_size(model: Any) -> int:
    old_head = getattr(model, "lm_head", None)
    if isinstance(old_head, nn.Linear):
        return int(old_head.in_features)
    config = getattr(model, "config", None)
    text_config = getattr(config, "text_config", None)
    for candidate in (
        getattr(config, "hidden_size", None),
        getattr(text_config, "hidden_size", None),
    ):
        if isinstance(candidate, int) and candidate > 0:
            return candidate
    raise ValueError("Cannot determine the Qwen backbone hidden size.")


def _resolve_lm_head_direction(
    model: Any,
    tokenizer: Any,
) -> tuple[nn.Linear, torch.Tensor, tuple[int, int]]:
    """Resolve the original reranker ``W_yes - W_no`` initialization vector."""
    lm_head = getattr(model, "lm_head", None)
    if not isinstance(lm_head, nn.Linear):
        raise TypeError(
            "Fresh reward-head installation requires the original linear "
            f"Qwen lm_head, got {type(lm_head).__name__ if lm_head is not None else 'None'}."
        )
    actual_tokenizer = getattr(tokenizer, "tokenizer", tokenizer)
    if actual_tokenizer is None or not hasattr(actual_tokenizer, "convert_tokens_to_ids"):
        raise TypeError("A tokenizer with convert_tokens_to_ids() is required.")
    token_ids = tuple(
        actual_tokenizer.convert_tokens_to_ids(token) for token in ("yes", "no")
    )
    if any(
        not isinstance(token_id, int)
        or not 0 <= token_id < lm_head.out_features
        for token_id in token_ids
    ):
        raise ValueError(
            "yes/no tokens must map into the original Qwen lm_head vocabulary "
            f"(vocab_size={lm_head.out_features}, got ids={token_ids})."
        )
    direction = lm_head.weight[token_ids[0]] - lm_head.weight[token_ids[1]]
    return lm_head, direction, token_ids


def install_multi_metric_heads(
    model: Any,
    tokenizer: Any,
    *,
    num_metrics: int = NUM_METRICS,
    hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
    tie_projection_dim: int = DEFAULT_TIE_PROJECTION_DIM,
    dropout_p: float = DEFAULT_HEAD_DROPOUT,
    tie_head_mode: Optional[str] = None,
) -> MultiMetricRewardHeads:
    """Attach and seed the three reward heads from the Qwen reranker head.

    The vocabulary ``lm_head`` remains intact.  Its ``W_yes - W_no`` direction
    initializes the Score and Relevance base projections; the Tie Head keeps
    an independent initialization because it has no compatible scalar-token
    direction.
    """
    if getattr(model, "overall_inspector_head", None) is not None:
        raise ValueError(
            "Student/Teacher metric heads must not share a model instance with the Overall Inspector."
        )
    tie_head_mode = resolve_tie_head_mode(tie_head_mode)
    existing = getattr(model, "metric_heads", None)
    if existing is not None:
        if not isinstance(existing, MultiMetricRewardHeads):
            raise TypeError("model.metric_heads exists but has an incompatible type.")
        expected = (num_metrics, hidden_dim, tie_projection_dim, float(dropout_p), tie_head_mode)
        actual = (
            existing.num_metrics,
            existing.hidden_dim,
            existing.tie_projection_dim,
            existing.dropout_p,
            existing.tie_head_mode,
        )
        if actual != expected:
            raise ValueError(
                f"Existing metric-head configuration {actual} does not match {expected}."
            )
        return existing

    lm_head, lm_direction, token_ids = _resolve_lm_head_direction(model, tokenizer)
    hidden_size = _resolve_hidden_size(model)
    # Heads keep fp32 parameters: in bf16, AdamW steps of the size of the
    # learning rate round away for weights above ~0.02 (theta0 and kappa never
    # moved). forward() casts the bf16 hidden states to the head dtype.
    heads = MultiMetricRewardHeads(
        hidden_size,
        num_metrics=num_metrics,
        hidden_dim=hidden_dim,
        tie_projection_dim=tie_projection_dim,
        dropout_p=dropout_p,
        tie_head_mode=tie_head_mode,
    ).to(device=lm_head.weight.device, dtype=torch.float32)
    heads.initialize_from_lm_head_direction(lm_direction)
    model.metric_heads = heads

    config = getattr(model, "config", None)
    if config is not None:
        config.reward_model_role = "multi_metric_student_or_ema_teacher"
        config.num_metrics = num_metrics
        config.metric_order = list(METRIC_NAMES)
        config.metric_head_hidden_dim = hidden_dim
        config.tie_projection_dim = tie_projection_dim
        config.metric_head_dropout = float(dropout_p)
        config.metric_head_initialization = (
            f"score_relevance:lm_head_yes_minus_no;tie:{tie_head_mode}"
        )
        config.tie_head_mode = tie_head_mode
        config.metric_head_pretrained_token_ids = {
            "yes": token_ids[0],
            "no": token_ids[1],
        }
    logger.info(
        "Installed three-head reward module: hidden=%d, metrics=%d, "
        "head_hidden=%d, tie_projection=%d, dropout=%.3f, "
        "score/relevance init=W_yes[%d]-W_no[%d], tie head=%s.",
        hidden_size,
        num_metrics,
        hidden_dim,
        tie_projection_dim,
        dropout_p,
        token_ids[0],
        token_ids[1],
        tie_head_mode,
    )
    return heads


def install_overall_inspector_head(
    model: Any,
    tokenizer: Any,
    tie_head_mode: Optional[str] = None,
) -> OverallInspectorHead:
    """Attach the independent three-way Inspector head seeded from ``W_yes-W_no``."""
    if getattr(model, "metric_heads", None) is not None:
        raise ValueError(
            "The Overall Inspector must not share a model instance with Student/Teacher metric heads."
        )
    tie_head_mode = resolve_tie_head_mode(tie_head_mode)
    existing = getattr(model, "overall_inspector_head", None)
    if existing is not None:
        if not isinstance(existing, OverallInspectorHead):
            raise TypeError(
                "model.overall_inspector_head exists but has an incompatible type."
            )
        if existing.tie_head_mode != tie_head_mode:
            raise ValueError(
                f"Existing Inspector Tie Head mode {existing.tie_head_mode} != {tie_head_mode}."
            )
        return existing

    lm_head, lm_direction, token_ids = _resolve_lm_head_direction(model, tokenizer)
    hidden_size = _resolve_hidden_size(model)
    # fp32 parameters, as for the Student heads (bf16 rounds AdamW steps away).
    head = OverallInspectorHead(hidden_size, tie_head_mode=tie_head_mode).to(
        device=lm_head.weight.device,
        dtype=torch.float32,
    )
    head.initialize_from_lm_head_direction(lm_direction)
    model.overall_inspector_head = head

    config = getattr(model, "config", None)
    if config is not None:
        config.reward_model_role = "overall_inspector"
        config.overall_inspector_head_initialization = (
            f"score:lm_head_yes_minus_no;tie:{tie_head_mode}"
        )
        config.tie_head_mode = tie_head_mode
        config.overall_inspector_outputs = ["P(A)", "P(B)", "P(Tie)"]
        config.overall_inspector_head_token_ids = {
            "yes": token_ids[0],
            "no": token_ids[1],
        }
    logger.info(
        "Installed independent three-way Overall Inspector head: hidden=%d, "
        "score init=W_yes[%d]-W_no[%d], tie head=%s.",
        hidden_size,
        token_ids[0],
        token_ids[1],
        tie_head_mode,
    )
    return head


def install_multi_metric_head(
    model: Any,
    tokenizer: Any = None,
    num_metrics: int = NUM_METRICS,
    total_outputs: int = TOTAL_OUTPUTS,
    **kwargs: Any,
) -> MultiMetricRewardHeads:
    """Singular compatibility entry point for installing all metric heads."""
    if total_outputs != TOTAL_OUTPUTS:
        raise ValueError(
            "The new model has a structured pairwise output; total_outputs is "
            f"only a legacy label-width check and must remain {TOTAL_OUTPUTS}."
        )
    return install_multi_metric_heads(
        model,
        tokenizer,
        num_metrics=num_metrics,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Model wrapper
# ---------------------------------------------------------------------------

def _pool_qwen_last_hidden(
    language_model: nn.Module,
    model_inputs: Dict[str, Any],
    attention_mask: Optional[torch.Tensor],
) -> torch.Tensor:
    """Run the adapted Qwen backbone and pool its last valid token."""
    base_lm = (
        language_model.get_base_model()
        if hasattr(language_model, "get_base_model")
        else language_model
    )
    backbone = getattr(base_lm, "model", None)
    if backbone is None:
        raise TypeError("The wrapped Qwen model does not expose a backbone named 'model'.")
    backbone_out = backbone(**model_inputs)
    hidden_sequence = getattr(backbone_out, "last_hidden_state", None)
    if hidden_sequence is None:
        raise TypeError("The Qwen backbone output does not contain last_hidden_state.")
    if attention_mask is None:
        return hidden_sequence[:, -1, :]
    if attention_mask.shape != hidden_sequence.shape[:2]:
        raise ValueError("attention_mask must match the backbone hidden-state dimensions.")
    valid = attention_mask.to(hidden_sequence.device).bool()
    if not valid.any(dim=1).all():
        raise ValueError("Each sequence must contain at least one unmasked token.")
    positions = torch.arange(
        hidden_sequence.size(1), device=hidden_sequence.device
    ).expand_as(valid)
    last_indices = positions.masked_fill(~valid, -1).max(dim=1).values
    batch_indices = torch.arange(hidden_sequence.size(0), device=hidden_sequence.device)
    return hidden_sequence[batch_indices, last_indices]


class CustomQwen3VLForOverallInspector(nn.Module):
    """Independent three-way Qwen3-VL Overall Inspector Model.

    This class owns its own Qwen model instance and three-way head.  It must be
    paired with an Inspector-specific LoRA adapter; it never contains or shares
    Student/Teacher metric heads.
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self._lm: Qwen3VLForConditionalGeneration = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **kwargs,
            )
        )
        if tokenizer is None:
            tokenizer = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side="left",
            ).tokenizer
        install_overall_inspector_head(self._lm, tokenizer)

    @property
    def overall_inspector_head(self) -> OverallInspectorHead:
        return self._lm.overall_inspector_head

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self._lm, "gradient_checkpointing_enable"):
            self._lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self._lm, "gradient_checkpointing_disable"):
            self._lm.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        if hasattr(self._lm, "enable_input_require_grads"):
            self._lm.enable_input_require_grads()
        else:
            def _hook(_module: nn.Module, _inputs: Any, output: torch.Tensor) -> None:
                output.requires_grad_(True)
            self._lm.model.get_input_embeddings().register_forward_hook(_hook)

    def freeze(self) -> "CustomQwen3VLForOverallInspector":
        """Freeze the selected Inspector checkpoint for offline pseudo-labeling."""
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()
        return self

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pair_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MultiMetricRewardOutput:
        model_inputs = {
            key: value
            for key, value in {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                **kwargs,
            }.items()
            if value is not None
        }
        model_inputs.pop("labels", None)
        hidden = _pool_qwen_last_hidden(self._lm, model_inputs, attention_mask)

        if (
            not isinstance(pair_indices, torch.Tensor)
            or pair_indices.ndim != 2
            or pair_indices.size(1) != 2
            or pair_indices.size(0) == 0
        ):
            raise ValueError("pair_indices must have non-empty shape [B, 2].")
        if pair_indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("pair_indices must contain integer row indices.")
        pair_indices = pair_indices.to(device=hidden.device, dtype=torch.long)
        used_rows = pair_indices.flatten().sort().values
        if not torch.equal(used_rows, torch.arange(hidden.size(0), device=hidden.device)):
            raise ValueError(
                "Inspector pair_indices must use every candidate row exactly once."
            )
        rows_a, rows_b = pair_indices.unbind(dim=1)
        return self.overall_inspector_head(hidden[rows_a], hidden[rows_b])


class CustomQwen3VLForMultiMetricReranking(nn.Module):
    """Qwen3-VL backbone with structured pairwise six-metric outputs.

    Parameters
    ----------
    model_name_or_path:
        HuggingFace repo id or local checkpoint path.
    tokenizer:
        Tokenizer used to resolve the original reranker's ``yes`` and ``no``
        vocabulary rows for Score/Relevance initialization.  It is loaded from
        ``model_name_or_path`` when omitted.
    **kwargs:
        Forwarded to ``Qwen3VLForConditionalGeneration.from_pretrained``
        (e.g. ``torch_dtype=torch.bfloat16``).
    """

    def __init__(
        self,
        model_name_or_path: str,
        tokenizer: Optional[Any] = None,
        num_metrics: int = NUM_METRICS,
        head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        tie_projection_dim: int = DEFAULT_TIE_PROJECTION_DIM,
        head_dropout: float = DEFAULT_HEAD_DROPOUT,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        if "num_metrics" in kwargs:  # Backward-compatible constructor alias.
            num_metrics = kwargs.pop("num_metrics")
        kwargs.pop("total_outputs", None)
        kwargs.pop("hidden_size", None)
        kwargs.pop("hidden_dim", None)

        self.num_metrics = num_metrics

        self._lm: Qwen3VLForConditionalGeneration = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                **kwargs,
            )
        )

        if tokenizer is None:
            tokenizer = AutoProcessor.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side="left",
            ).tokenizer

        install_multi_metric_heads(
            self._lm,
            tokenizer,
            num_metrics=self.num_metrics,
            hidden_dim=head_hidden_dim,
            tie_projection_dim=tie_projection_dim,
            dropout_p=head_dropout,
        )

    @property
    def metric_heads(self) -> MultiMetricRewardHeads:
        """Return the single head module registered on the underlying Qwen model."""
        return self._lm.metric_heads

    # ── gradient checkpointing helpers ──────────────────────────────────────

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        if hasattr(self._lm, "gradient_checkpointing_enable"):
            self._lm.gradient_checkpointing_enable(**kwargs)

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self._lm, "gradient_checkpointing_disable"):
            self._lm.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        if hasattr(self._lm, "enable_input_require_grads"):
            self._lm.enable_input_require_grads()
        else:
            def _hook(_m: nn.Module, _inp: Any, out: torch.Tensor) -> None:
                out.requires_grad_(True)
            self._lm.model.get_input_embeddings().register_forward_hook(_hook)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.Tensor] = None,
        image_grid_thw: Optional[torch.Tensor] = None,
        pair_indices: Optional[torch.Tensor] = None,
        source_indices: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> MultiMetricRewardOutput:
        """Return one structured output per explicit A/B/source triplet."""
        model_inputs = {
            k: v
            for k, v in {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                **kwargs,
            }.items()
            if v is not None
        }

        # Remove labels if passed by trainer, since backbone model does not accept labels
        model_inputs.pop("labels", None)

        # Run the adapted backbone only, keeping injected LoRA modules active
        # while avoiding the vocabulary-sized language-model projection.
        h = _pool_qwen_last_hidden(self._lm, model_inputs, attention_mask)

        if pair_indices is None or source_indices is None:
            raise ValueError("pair_indices [B,2] and source_indices [B] are required.")
        if not isinstance(pair_indices, torch.Tensor) or pair_indices.ndim != 2 or pair_indices.size(1) != 2:
            raise ValueError("pair_indices must have shape [B, 2].")
        if not isinstance(source_indices, torch.Tensor) or source_indices.ndim != 1:
            raise ValueError("source_indices must have shape [B].")
        if source_indices.size(0) != pair_indices.size(0) or pair_indices.size(0) == 0:
            raise ValueError("pair_indices and source_indices must describe the same non-empty batch.")
        if pair_indices.dtype not in (torch.int32, torch.int64) or source_indices.dtype not in (
            torch.int32,
            torch.int64,
        ):
            raise ValueError("Pair/source indices must be integer tensors.")

        pair_indices = pair_indices.to(device=h.device, dtype=torch.long)
        source_indices = source_indices.to(device=h.device, dtype=torch.long)
        used_rows = torch.cat((pair_indices.flatten(), source_indices)).sort().values
        expected_rows = torch.arange(h.size(0), device=h.device)
        if not torch.equal(used_rows, expected_rows):
            raise ValueError(
                "Pair/source indices must use every backbone row exactly once, "
                "without duplicates or out-of-range values."
            )

        rows_a, rows_b = pair_indices.unbind(dim=1)
        return self.metric_heads(h[rows_a], h[rows_b], h[source_indices])


# ---------------------------------------------------------------------------
# Inference wrapper
# ---------------------------------------------------------------------------

class CustomQwen3VLReranker:
    """High-level inference wrapper for ``CustomQwen3VLForMultiMetricReranking``.

    This interface is pairwise: every call compares exactly two documents and
    returns one :class:`MultiMetricRewardOutput` represented as Python values.

    Parameters
    ----------
    model_name_or_path:
        Base or fine-tuned checkpoint.
    peft_checkpoint:
        Optional PEFT/LoRA adapter path loaded on top of the base model.
    """

    def __init__(
        self,
        model_name_or_path: str,
        min_pixels: int = 4_096,
        max_pixels: int = 200_704,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        peft_checkpoint: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        self.device = torch.device(device)
        self.system_prompt = system_prompt

        self.processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            padding_side="left",
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        self.model = CustomQwen3VLForMultiMetricReranking(
            model_name_or_path=model_name_or_path,
            tokenizer=self.processor.tokenizer,
            **kwargs,
        ).to(self.device)

        if peft_checkpoint is not None:
            from peft import PeftModel
            # Adapters are trained directly on Qwen3VLForConditionalGeneration,
            # so load directly onto self.model._lm to ensure exact module key matching
            self.model._lm = PeftModel.from_pretrained(
                self.model._lm, peft_checkpoint, is_trainable=False
            )
            logger.info("Loaded PEFT adapter from: %s", peft_checkpoint)

        self.model.eval()

    def _format_pair(
        self,
        source_img: str,
        edit_instruction: str,
        doc_img: str,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        inst = instruction or self.system_prompt

        def _clean_img_path(p: str) -> str:
            p_str = str(p).strip()
            if p_str.startswith("file://"):
                p_str = p_str[7:]
            return f"file://{os.path.abspath(p_str)}"

        return [
            {
                "role": "system",
                "content": [{
                    "type": "text",
                    "text": DEFAULT_SYSTEM_MESSAGE,
                }],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text",  "text": f"<Instruct>: {inst}\n<Query>:"},
                    {"type": "image", "image": _clean_img_path(source_img)},
                    {"type": "text",  "text": f" {edit_instruction}\n<Document>:"},
                    {"type": "image", "image": _clean_img_path(doc_img)},
                ],
            },
        ]

    def _format_source(
        self,
        source_img: str,
        edit_instruction: str,
        instruction: Optional[str] = None,
    ) -> List[Dict]:
        """Format the representation used exclusively by the relevance head."""
        inst = instruction or self.system_prompt
        source_path = str(source_img).strip()
        if source_path.startswith("file://"):
            source_path = source_path[7:]
        return [
            {
                "role": "system",
                "content": [{"type": "text", "text": DEFAULT_SYSTEM_MESSAGE}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"<Instruct>: {inst}\n<Query>:"},
                    {"type": "image", "image": f"file://{os.path.abspath(source_path)}"},
                    {"type": "text", "text": f" {edit_instruction}"},
                ],
            },
        ]

    def _tokenize(self, conversations: List[List[Dict]]) -> Dict[str, torch.Tensor]:
        texts = self.processor.apply_chat_template(
            conversations, tokenize=False, add_generation_prompt=True
        )
        images, videos = process_vision_info(conversations)
        return self.processor(
            text=texts, images=images, videos=videos,
            padding=True, return_tensors="pt",
        )

    @torch.no_grad()
    def process(
        self,
        inputs: Dict,
        active_metrics: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Compare exactly two candidate documents and return one pair output."""
        query     = inputs.get("query", {})
        documents = inputs.get("documents", [])
        instruction = inputs.get("instruction", self.system_prompt)
        if not query or not documents:
            return []
        if len(documents) != 2:
            raise ValueError(
                "The three-head reranker requires exactly two documents: candidate A and B."
            )

        candidate_conversations = [
            self._format_pair(
                source_img=query.get("image", ""),
                edit_instruction=query.get("text", ""),
                doc_img=doc.get("image", ""),
                instruction=instruction,
            )
            for doc in documents
        ]
        source_conversation = self._format_source(
            source_img=query.get("image", ""),
            edit_instruction=query.get("text", ""),
            instruction=instruction,
        )
        conversations = candidate_conversations + [source_conversation]
        tokenized = {k: v.to(self.device) for k, v in self._tokenize(conversations).items()}
        output = self.model(
            **tokenized,
            pair_indices=torch.tensor([[0, 1]], device=self.device, dtype=torch.long),
            source_indices=torch.tensor([2], device=self.device, dtype=torch.long),
        )

        if active_metrics is not None:
            user_report_idx = {
                METRIC_TO_INDEX[m.strip().lower().replace("_", " ")]
                for m in active_metrics
                if m.strip().lower().replace("_", " ") in METRIC_TO_INDEX
            }
        else:
            user_report_idx = None

        relevance = output.relevance_probs[0]
        selected_indices = (
            sorted(user_report_idx)
            if user_report_idx is not None
            else torch.where(relevance >= 0.5)[0].tolist()
        )
        if not selected_indices:
            selected_indices = [int(torch.argmax(relevance).item())]

        def aspect_dict(tensor: torch.Tensor) -> Dict[str, float]:
            values = tensor[0].detach().cpu()
            return {
                name: round(float(values[idx]), 6)
                for idx, name in enumerate(METRIC_NAMES)
            }

        three_way = output.three_way_probs[0].detach().cpu()
        three_way_by_aspect = {
            name: {
                "A": round(float(three_way[idx, 0]), 6),
                "B": round(float(three_way[idx, 1]), 6),
                "Tie": round(float(three_way[idx, 2]), 6),
            }
            for idx, name in enumerate(METRIC_NAMES)
        }
        overall_signed_evidence = output.signed_evidence[0, selected_indices].mean()
        return [{
            "scores_A": aspect_dict(output.scores_A),
            "scores_B": aspect_dict(output.scores_B),
            "margins": aspect_dict(output.margins),
            "cond_win_prob_A": aspect_dict(output.cond_win_prob_A),
            "tie_probs": aspect_dict(output.tie_probs),
            "relevance_probs": aspect_dict(output.relevance_probs),
            "three_way_probs": three_way_by_aspect,
            "signed_evidence": aspect_dict(output.signed_evidence),
            "selected_metrics": [METRIC_NAMES[idx] for idx in selected_indices],
            "overall_signed_evidence": round(float(overall_signed_evidence), 6),
        }]

    @torch.no_grad()
    def process_single(
        self,
        source_image_path: str,
        edit_instruction: str,
        candidate_image_path: str,
        active_metrics: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        raise ValueError(
            "process_single is not defined for the pairwise tie contract. "
            "Call process() with exactly two documents (candidate A and B)."
        )
