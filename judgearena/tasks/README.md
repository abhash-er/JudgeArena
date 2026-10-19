# Task definitions

Task YAML describes **what benchmark is being run**. A run config describes
**how one experiment runs it**.

- Keep stable benchmark behavior in the task: pinned data, field mappings,
  baseline policy, judge protocol, and scoring definition.
- Keep experiment choices in `RunConfig`: candidate and judge models, sampling
  parameters, instruction limits, caching, and supported overrides.
- Prefer a new task version when changing data or evaluation behavior. Changing
  only the model or runtime settings does not create a new task.

## Layout

```text
tasks/
├── definitions/       # Packaged task YAML, grouped by benchmark family
├── schema/            # Typed YAML contracts, separated by responsibility
├── registry.py        # Loading, discovery, task lookup, and validation
└── cli.py             # `judgearena tasks ...` commands
```

Executable behavior lives outside the definitions:

- `judgearena/datasets/` downloads and normalizes upstream data.
- `judgearena/benchmarks/` runs the selected evaluation workflow.
- `judgearena/prompts/` owns reusable judge prompt presets.

## Add a task using existing components

Create a public YAML file under `definitions/<family>/`. Files beginning with
`_` are private bases and are not runnable tasks.

```yaml
schema_version: 1
task: example-pairwise
task_version: 1
description: Pairwise evaluation on the Example instruction set.
tags: [pairwise, instruction-following]

dataset:
  adapter: judgearena_tables
  sources:
    examples:
      type: huggingface_dataset
      repo_id: organization/example
      revision: "0123456789abcdef0123456789abcdef01234567"
      allow_patterns:
        - instructions/example-pairwise.csv
        - model_outputs/example-pairwise.csv.zip
  fields:
    id: instruction_index
    instruction: instruction

protocol:
  runner: pairwise
  generation:
    mode: single_turn_chat
  baseline:
    strategy: task_default
    reference_id: example-baseline
  judge:
    default_prompt_preset: default
  scoring:
    metrics:
      - metric: pairwise_win_rate
      - metric: length_controlled_winrate
```

Reuse a private `_base.yaml` with `extends: _base.yaml` when several versions
share meaningful settings. Avoid inheritance merely to save a few lines.

Validate the result before running it:

```bash
judgearena tasks validate example-pairwise
judgearena tasks show example-pairwise --resolved
```

The resolved view includes inherited values and hashes of every YAML resource.

## Add a dataset adapter

An adapter loads data and returns `CompletionData` or `PreferenceData`.
For example, this adapter reads local Parquet files:

```python
import pandas as pd

from judgearena.datasets.data import CompletionData
from judgearena.datasets.registry import DatasetAdapter, register_dataset


@register_dataset("local_questions")
class LocalQuestions(DatasetAdapter):
    def load_data(self) -> CompletionData:
        questions = pd.read_parquet(self.task.spec.dataset.options["path"])
        return CompletionData.from_frames(
            questions,
            instruction_columns={"id": "instruction_index", "prompt": "instruction"},
        )
```

Replace the example task's `dataset` section with:

```yaml
dataset:
  adapter: local_questions
  options:
    path: questions.parquet
```

The adapter decides which `sources`, `fields` and `options` it needs.
Pin data revisions when publishing a reproducible benchmark.

Import the adapter before loading a task that uses it. Registration lasts for
that Python process. To run `my_tasks/example/example-pairwise.yaml`:

```python
from pathlib import Path

from judgearena.benchmarks.runner import run_benchmark
from judgearena.config import RunConfig
from judgearena.tasks import load_tasks

task = load_tasks(Path("my_tasks"))["example-pairwise"]
cfg = RunConfig(
    task=task.task,
    resolved_task=task,
    model={"name": "Dummy/candidate", "baseline": "Dummy/baseline"},
    judge={"model": "Dummy/score_A: 8 score_B: 6"},
)
run_benchmark(cfg)
```

Use Python for external adapters and task roots; the CLI loads installed tasks.

### Dataset contracts

| Workflow | Adapter output |
|---|---|
| Text pairwise | `CompletionData`: indexed `instruction` rows; optional outputs with `instruction_index`, `model`, and `output`. Missing answers are generated. |
| Multi-turn pairwise | `CompletionData` with `user_turns` (a list of user prompts) per instruction. Saved outputs also carry `messages`, a list of role/content dictionaries ending in an assistant answer. |
| Supplied-history generation | `CompletionData` with input `messages` per row; optional `instruction_id` sets cache identity. Each row produces one answer. |
| Anchored Elo | `PreferenceData`: battles with `instruction_index`, `instruction`, `model_a`, `model_b`, `completion_a`, `completion_b`, `pref`, and `source`. Human rows supply the rating anchors. |
| Meta-evaluation | `PreferenceData` with human labels of `0`, `0.5`, or `1` and nonempty model names. `question_id` defaults to `instruction_index` and must be nonmissing and unique after conversion to strings; the runner derives battle IDs from it. |

Instruction IDs must be unique and nonmissing; output IDs must match their types.
`pref=0` favors A, `pref=1` favors B, and `0.5` is a tie. Map source labels with
`PreferenceData.from_battles(..., outcome_preferences={"left": 0, "draw": 0.5, "right": 1})`.
See [data.py](../datasets/data.py) for column mapping and validation.

`from_battles()` derives text outputs. For saved conversations, supply
`model_outputs.messages`. The built-in arena adapter exposes only the first
user/assistant turn; multi-turn evaluation needs complete histories.

Extra columns remain metadata. Use `--run.filters '{"category": ["coding"]}'`
to select any available metadata field. Task variants and runtime filters are
intersected; empty selections raise an error.

## Multi-turn pairwise judging

Use the shared pairwise runner with these protocol settings:

```yaml
generation:
  mode: multi_turn_chat
judge:
  format: standard
  default_prompt_preset: default
  turns: all
```

`turns` accepts `all`, `last`, or numbered turns such as `[1, 3]`. Numbered/all
comparisons require equally many answers from both candidates. `last` compares
complete histories, including histories of unequal length. Generation proceeds
one turn at a time using each candidate's own previous answers.

Select `fastchat-pairwise` to use MT-Bench's renderer with any compatible dataset.
Its one/two-turn prompts require matching user questions across candidates.
Set reference routing in `judge.prompt_options.reference_categories`; see the
[packaged task](definitions/mt_bench/mt-bench.yaml) for the benchmark defaults.
FastChat combines swapped verdicts by agreement; ordinary presets keep per-pass
preferences.

MT-Bench templates accept `{question}`, `{answer_a}` and `{answer_b}` for one
turn, numbered fields such as `{question_1}` and `{answer_a_2}` for two turns,
and `{ref_answer_1}` and `{ref_answer_2}` for references. This format does not
support parsers requiring token logprobs.

## Multiple references

A pairwise task can compare each candidate answer against every listed reference:

```yaml
baseline:
  strategy: all_references
  references: [reference-a, reference-b, reference-c]
```

The runner prepares each candidate answer once, then loads or generates the
references. Comparison rows keep `source_instruction_index` and have unique IDs.
The candidate must differ from the reference models. Change the list in the task
definition; `model.baseline` cannot override it.
`wildbench-reward-mix` uses this policy and averages per-reference WB-Rewards equally.

## Independent evaluation

Use `runner: single_answer` to annotate one answer per instruction. Dataset
adapters return the same `CompletionData` contract: instructions plus optional
saved model outputs. Select the saved model through `model.name`; missing
answers are generated. `provided_messages` uses supplied conversation history.

```yaml
protocol:
  runner: single_answer
  generation:
    mode: provided_messages
  judge:
    default_prompt_preset: wildbench-score
  scoring:
    metrics:
      - metric: wildbench_score
        breakdown_by: [category]
```

The standard renderer supplies `{user_prompt}`, `{completion}`, their `_json`
forms, and dataset metadata such as `{checklist}` or `{history}`. Required
metadata is checked before generation. Each preset pairs its template with an
`OutputParser`: `parse_result()` returns a dataclass, a JSON-compatible value,
or `None` for an unparseable response. Parsed evidence is retained in `parsed`;
a `score` field is also exposed as the `score` column. This supports correctness
and criterion annotations without requiring scalar scores.

Metrics receive these annotations through the same `calculate_metrics` used by
pairwise evaluation. Omit `scoring` or use `metrics: []` to save annotations
without metrics. The runner returns the annotation dataframe. A baseline and
swapped judging are not used.

## Independent scoring

To score each candidate separately and then compare scores, use
`independent_scores`. MT-Bench-101 configures it as follows:

```yaml
limit_by: dialogue_uid
generation:
  mode: provided_messages
judge:
  format: independent_scores
  default_prompt_preset: mt-bench-101
  minimum_score_by: [dialogue_uid, dialogue_id, task, ability, domain]
```

`limit_by` makes `n_instructions` select complete groups. `minimum_score_by`
compares each model's minimum score per group; omit it to compare individual
rows. The `absolute_score` metric accepts `macro_by` to weight groups equally.
MT-Bench-101's adapter supplies the reference histories used by its prompt.

## Prompts and metrics

A prompt preset defines the judge's output format and parser. Metrics consume
annotation or battle dataframes. Select them in task YAML, with optional
parameters and breakdowns:

```yaml
scoring:
  metrics:
    - metric: pairwise_win_rate
      breakdown_by: [category]
```

Each `breakdown_by` field produces a separate breakdown. For example,
`[category, turn]` gives results per category and per turn, not per combination.

Task loading validates metric names and parameters. Calculation failures and
missing breakdown columns produce warnings and are skipped. Meta-evaluation
supports its agreement/ranking/Elo-gap metrics without breakdowns.

Standard pairwise prompt templates can reference adapter metadata columns such
as `{history}` or `{checklist}`. Required fields are checked before generation.
The shared renderer supplies `{user_prompt}`, `{completion_A}`, `{completion_B}`
and their `_json` variants; these fields take precedence over source metadata
and follow the actual answer order.

Independent templates receive `{user_prompt}`, `{completion}`, their `_json`
variants, and adapter metadata. Within a `single_answer` protocol, use:

```yaml
judge:
  default_prompt_preset: single-score
scoring:
  metrics:
    - metric: mean_score
```

`single-score` gives one overall score from 1 to 10. `criteria-score` uses the
same rubric definitions as the pairwise `criteria` preset, selected through
`judge.criteria_name` or `judge.criteria_file`. It retains each criterion's
1–10 score in `parsed.details.criteria` and sets `score` to their equal-weight
mean. Missing or invalid criteria make that annotation unparsed.

`mean_score` averages available scores without normalization and reports sample
and missing counts. For custom prompt files using another numeric scale, choose
the `json-score` parser: it accepts a finite numeric `score` field and preserves
the remaining JSON evidence. Benchmark tasks keep their configured defaults;
these components can be selected in other task definitions.

Use `judge.prompt.system_file` and `judge.prompt.user_file` with a registered
`judge.prompt.parser` for custom prompts. For new implementations, see
[prompt registration](../prompts/registry.py), [parsers](../prompts/parsing.py)
and [metrics](../benchmarks/scoring.py).

Independent scalar parsers return `ParsedScore` or `None`; other independent
annotations may use a custom serializable result. Parsers declare `result_type`,
and presets declare `judging_mode` as `pairwise` or `single_answer`.

## Add a workflow

If a benchmark needs a genuinely different evaluation algorithm:

1. Add its typed protocol under `schema/`.
2. Add that protocol to `ProtocolSpec` in `schema/task.py`.
3. Implement and register its benchmark runner.
4. Keep algorithm-specific fields in that protocol, as Elo does in `EloProtocol`.

Adding a schema field makes it valid in task YAML; it does not automatically
make it a run-config or CLI option. Runtime overrides must be explicitly added
to `RunConfig` and resolved with this precedence:

```text
CLI > run config YAML > task YAML > code default
```

Dataset revisions and evaluation definitions should normally remain
task-owned. Expose overrides only for settings users are expected to change
between experiments.
