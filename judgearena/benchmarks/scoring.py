"""Construction and execution of metrics for annotations and battles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd

from judgearena.benchmarks.elo.scoring import BradleyTerryMetric
from judgearena.benchmarks.meta_eval.scoring import (
    MetaEvalAgreementMetric,
    MetaEvalEloGapMetric,
    MetaEvalRankingMetric,
)
from judgearena.benchmarks.pairwise.scoring.alpaca_eval import (
    AlpacaEvalLengthControlledMetric,
)
from judgearena.benchmarks.pairwise.scoring.arena_hard import (
    ArenaHardV01Metric,
    ArenaHardV20Metric,
)
from judgearena.benchmarks.pairwise.scoring.metrics import (
    AbsoluteScoreMetric,
    LengthControlledWinrateMetric,
    PairwiseWinRateMetric,
)
from judgearena.log import get_logger

logger = get_logger(__name__)


class _Metric(Protocol):
    """The calculation and renderer supplied by every metric implementation."""

    def calculate(
        self, battles: pd.DataFrame, **runtime: object
    ) -> dict[str, object]: ...

    @staticmethod
    def render(result: dict[str, object]) -> str: ...


class MetricRequest(Protocol):
    """The request fields needed to construct and group one metric."""

    metric: str
    breakdown_by: tuple[str, ...]
    parameters: Mapping[str, object]


@dataclass(frozen=True)
class MeanScoreMetric:
    """Average finite scalar annotations on their original scale."""

    def calculate(self, annotations: pd.DataFrame) -> dict[str, object]:
        scores = pd.to_numeric(annotations["score"], errors="raise")
        if not np.isfinite(scores.dropna()).all():
            raise ValueError("Scores must be finite or missing.")
        return {
            "score": float(scores.mean()) if scores.notna().any() else None,
            "num_samples": len(scores),
            "num_missing": int(scores.isna().sum()),
        }

    @staticmethod
    def render(result: dict[str, object]) -> str:
        return f"Mean score: {result['score']} ({result['num_missing']} missing)"


_METRIC_TYPES: dict[str, type[_Metric]] = {
    "mean_score": MeanScoreMetric,
    "pairwise_win_rate": PairwiseWinRateMetric,
    "length_controlled_winrate": LengthControlledWinrateMetric,
    "bradley_terry": BradleyTerryMetric,
    "arena_hard_v01": ArenaHardV01Metric,
    "arena_hard_v20": ArenaHardV20Metric,
    "alpaca_eval_length_controlled": AlpacaEvalLengthControlledMetric,
    "meta_eval_agreement": MetaEvalAgreementMetric,
    "meta_eval_elo_gap": MetaEvalEloGapMetric,
    "meta_eval_ranking": MetaEvalRankingMetric,
    "absolute_score": AbsoluteScoreMetric,
}

ConfiguredMetrics = tuple[tuple[MetricRequest, _Metric], ...]


def available_metrics() -> tuple[str, ...]:
    """Return the stable set of supported metric identifiers."""
    return tuple(sorted(_METRIC_TYPES))


def _metric_type(name: str) -> type[_Metric]:
    try:
        return _METRIC_TYPES[name]
    except KeyError as exc:
        choices = ", ".join(available_metrics())
        raise ValueError(
            f"Unknown metric {name!r}; available metrics: {choices}."
        ) from exc


def build_metric(name: str, parameters: Mapping[str, object] | None = None) -> _Metric:
    """Build one fresh, configured metric instance."""
    try:
        return _metric_type(name)(**dict(parameters or {}))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid parameters for metric {name!r}: {exc}") from exc


def build_metrics(
    requests: Sequence[MetricRequest],
    *,
    parameter_overrides_by_metric: Mapping[str, Mapping[str, object]] | None = None,
) -> ConfiguredMetrics:
    """Build requested metrics in order, warning and skipping failed construction."""
    overrides = parameter_overrides_by_metric or {}
    configured: list[tuple[MetricRequest, _Metric]] = []
    for request in requests:
        parameters = dict(request.parameters)
        parameters.update(overrides.get(request.metric, {}))
        try:
            configured.append((request, build_metric(request.metric, parameters)))
        except Exception as exc:
            logger.warning("Skipping metric %r: %s", request.metric, exc)
    return tuple(configured)


def _group_value(value: object) -> object:
    if pd.isna(value):
        return None
    return value.item() if hasattr(value, "item") else value


def calculate_metrics(
    battles: pd.DataFrame,
    metrics: ConfiguredMetrics,
    *,
    runtime_by_metric: Mapping[str, Mapping[str, object]] | None = None,
) -> dict[str, dict[str, object]]:
    """Run metrics, warning and skipping failed metrics or breakdowns."""
    runtime_by_metric = runtime_by_metric or {}
    calculated: dict[str, dict[str, object]] = {}
    for request, metric in metrics:
        runtime = dict(runtime_by_metric.get(request.metric, {}))
        try:
            values = dict(metric.calculate(battles, **runtime))
        except Exception as exc:
            logger.warning("Skipping metric %r: %s", request.metric, exc)
            continue
        if request.breakdown_by:
            groups: dict[str, list[dict[str, object]]] = {}
            for field in request.breakdown_by:
                try:
                    groups[field] = [
                        {
                            "group": _group_value(key),
                            "values": metric.calculate(group, **runtime),
                        }
                        for key, group in battles.groupby(
                            field, sort=False, dropna=False
                        )
                    ]
                except Exception as exc:
                    logger.warning(
                        "Skipping metric %r breakdown %r: %s",
                        request.metric,
                        field,
                        exc,
                    )
            values["groups"] = groups
        calculated[request.metric] = values
    return calculated


def _indent(text: str, prefix: str = "  ") -> str:
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


def render_metrics(results: Mapping[str, dict[str, object]]) -> str:
    """Render metric results in their configured dictionary order."""
    sections: list[str] = []
    for name, result in results.items():
        try:
            metric_type = _metric_type(name)
            overall = {key: value for key, value in result.items() if key != "groups"}
            sections.append(
                metric_type.render(overall) if overall else f"{name}: unavailable"
            )
            for field, groups in result.get("groups", {}).items():
                for group in groups:
                    rendered = (
                        metric_type.render(group["values"])
                        if group["values"]
                        else f"{name}: unavailable"
                    )
                    sections.append(f"{field}={group['group']}:\n{_indent(rendered)}")
        except Exception as exc:
            logger.warning("Skipping rendering for metric %r: %s", name, exc)
    return "\n\n".join(sections)
