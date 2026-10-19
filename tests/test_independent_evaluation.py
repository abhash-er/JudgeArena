import json
from dataclasses import replace

import pandas as pd
import pytest

from judgearena.benchmarks.runner import run_benchmark
from judgearena.benchmarks.scoring import build_metric
from judgearena.config import RunConfig
from judgearena.datasets.arena_hard import ArenaHardAdapter
from judgearena.datasets.data import CompletionData
from judgearena.prompts.parsing import JUDGE_PARSERS
from judgearena.prompts.registry import resolve_judge_prompt
from judgearena.tasks.registry import get_packaged_task
from judgearena.tasks.schema import TaskSpec


def _independent_task():
    task = get_packaged_task("arena-hard-v0.1")
    spec = task.model_dump()
    spec["protocol"] = {
        "runner": "single_answer",
        "generation": {"mode": "provided_messages"},
        "judge": {"default_prompt_preset": "single-score"},
        "scoring": {"metrics": [{"metric": "mean_score"}]},
    }
    return replace(task, spec=TaskSpec.model_validate(spec))


@pytest.mark.parametrize(
    "value, expected",
    [
        ("1", 1),
        ('"10"', 10),
        ("8.5", 8.5),
        ("0", None),
        ("11", None),
        ("true", None),
        ('"NaN"', None),
    ],
)
def test_single_score_parser_validates_numeric_range(value, expected):
    parsed = resolve_judge_prompt(preset="single-score").parser.parse_result(
        '{"strengths": "Clear", "score": ' + value + "}"
    )
    if expected is None:
        assert parsed is None
    else:
        assert parsed.score == expected
        assert parsed.details["strengths"] == "Clear"


def test_mean_score_preserves_scale_and_counts_missing_results():
    metric = build_metric("mean_score")
    assert metric.calculate(pd.DataFrame({"score": [-2, 4, None]})) == {
        "score": 1.0,
        "num_samples": 3,
        "num_missing": 1,
    }
    assert metric.calculate(pd.DataFrame({"score": [None]}))["score"] is None
    with pytest.raises(ValueError, match="finite"):
        metric.calculate(pd.DataFrame({"score": [float("inf")]}))


@pytest.mark.parametrize("value", ["true", '"NaN"', "Infinity", "null"])
def test_generic_score_parser_rejects_non_numeric_rewards(value):
    from judgearena.prompts.parsing import resolve_judge_parser

    assert (
        resolve_judge_parser("json-score").parse_result('{"score": ' + value + "}")
        is None
    )


def test_independent_evaluation_reuses_answers_and_preserves_results(
    monkeypatch, tmp_path
):
    task = _independent_task()
    data = CompletionData.from_frames(
        pd.DataFrame(
            {
                "instruction_index": ["saved", "missing"],
                "instruction": ["Remember {42}", "Explain it"],
                "messages": [
                    [{"role": "user", "content": "Remember {42}"}],
                    [
                        {"role": "user", "content": "Remember {42}"},
                        {"role": "assistant", "content": "OK"},
                        {"role": "user", "content": "Explain it"},
                    ],
                ],
                "history": ["", "USER: Remember {42}\n\nASSISTANT: OK\n\n"],
                "checklist": ["- Relevant", "- Clear"],
                "category": ["Reasoning", "Reasoning"],
            }
        ),
        pd.DataFrame(
            {
                "instruction_index": ["saved"],
                "model": ["Dummy/generated"],
                "output": ["stored answer"],
            }
        ),
    )
    monkeypatch.setattr(ArenaHardAdapter, "load_data", lambda self: data)

    class Parser:
        name = "test-correctness"
        result_type = dict
        requires_top_logprobs = False

        def parse_result(self, text, **kwargs):
            return json.loads(text)

    monkeypatch.setitem(JUDGE_PARSERS, Parser.name, Parser())
    system, user = tmp_path / "system.txt", tmp_path / "user.txt"
    system.write_text("Use {checklist}")
    user.write_text("{history}\nQ: {user_prompt}\nA: {completion}")
    judge = {
        "model": 'Dummy/{"correct": true, "criteria": {"clarity": "pass"}, "score": 0.8}',
        "prompt": {
            "system_file": str(system),
            "user_file": str(user),
            "parser": Parser.name,
        },
    }

    first = None
    for run in ("live", "replay"):
        cfg = RunConfig(
            task=task.task,
            resolved_task=task,
            model={"name": "Dummy/generated"},
            judge=judge,
            run={
                "store_root": str(tmp_path / "cache"),
                "result_folder": str(tmp_path / run),
                "use_tqdm": False,
                "no_log_file": True,
            },
        )
        annotations = run_benchmark(cfg)
        assert annotations.set_index("instruction_index").completion.to_dict() == {
            "missing": "generated",
            "saved": "stored answer",
        }
        assert annotations.category.tolist() == ["Reasoning", "Reasoning"]
        assert (
            "Remember {42}"
            in annotations.set_index("instruction_index").loc["saved", "judge_input"]
        )
        assert annotations.parsed.iloc[0] == {
            "correct": True,
            "criteria": {"clarity": "pass"},
            "score": 0.8,
        }
        assert annotations.score.tolist() == [0.8, 0.8]
        metadata = json.loads(
            next((tmp_path / run).rglob("run-metadata.v1.json")).read_text()
        )
        if first is None:
            assert metadata["usage"]["by_stage"]["generation"]["requests"] == 1
            assert metadata["usage"]["by_stage"]["judging"]["requests"] == 2
            first = annotations
        else:
            pd.testing.assert_frame_equal(first, annotations)
            assert metadata["usage"]["total"]["requests"] == 0
        report = json.loads(next((tmp_path / run).rglob("results-*.json")).read_text())
        assert report["metrics"]["mean_score"]["score"] == 0.8
        exported = pd.read_csv(next((tmp_path / run).rglob("*-annotations.csv")))
        assert json.loads(exported.parsed.iloc[0]) == annotations.parsed.iloc[0]

    spec = task.model_dump()
    spec["protocol"].update(
        runner="pairwise",
        baseline={"strategy": "runtime_required"},
        judge={"format": "independent_scores", "default_prompt_preset": "single-score"},
        scoring={"metrics": [{"metric": "pairwise_win_rate"}]},
    )
    pairwise_task = replace(task, spec=TaskSpec.model_validate(spec))
    prefs = run_benchmark(
        RunConfig(
            task=task.task,
            resolved_task=pairwise_task,
            model={"name": "Dummy/generated", "baseline": "Dummy/reference"},
            judge={"model": 'Dummy/{"score": 8}'},
            run={
                "result_folder": str(tmp_path / "pairwise"),
                "store_root": str(tmp_path / "cache"),
                "use_tqdm": False,
            },
        )
    )
    assert prefs.tolist() == [0.5, 0.5]


def test_independent_rejects_pairwise_settings():
    task = _independent_task()
    for overrides in (
        {
            "model": {"name": "Dummy/x", "baseline": "Dummy/y"},
            "judge": {"model": "Dummy/x"},
        },
        {
            "model": {"name": "Dummy/x"},
            "judge": {"model": "Dummy/x", "swap_mode": "both"},
        },
    ):
        with pytest.raises(ValueError, match="single.answer|independent"):
            RunConfig(task=task.task, resolved_task=task, **overrides)


@pytest.mark.parametrize("value", ["1e2", "5e-1", "8garbage"])
def test_score_fallback_does_not_truncate_invalid_numbers(value):
    parser = resolve_judge_prompt(preset="single-score").parser
    assert parser.parse_result('```json\n{"score": ' + value + "}\n```") is None


def test_independent_empty_selection_fails_before_model_work(monkeypatch, tmp_path):
    data = CompletionData.from_frames(
        pd.DataFrame({"instruction_index": ["q"], "instruction": ["hi"]})
    )
    monkeypatch.setattr(ArenaHardAdapter, "load_data", lambda self: data)
    task = _independent_task()
    cfg = RunConfig(
        task=task.task,
        resolved_task=task,
        model={"name": "Dummy/x"},
        judge={"model": "Dummy/x"},
        generation={"n_instructions": 0},
        run={"result_folder": str(tmp_path)},
    )
    with pytest.raises(ValueError, match="No instructions"):
        run_benchmark(cfg)
    assert not list(tmp_path.iterdir())
