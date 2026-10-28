"""Pairwise judge backed by autotrust/JEV-9B, an open distillation of TypeSafe Jev 1.13.

JEV-9B answers typed questions (System One) in one prefill step: the decision
LoRA scores the option tokens, then the head bias and per-kind temperature turn
them into calibrated probabilities (see the model card). This wrapper asks one
Choice question per battle, following the TypeSafe agent skill: the state is a
JSON object with named fields, the question is plain, and the options name the
state fields they compare.

Use it as ``--judge.model JEV9B/<local model dir>`` together with the
``jev-state.txt`` user prompt, whose delimiters let the wrapper recover the
three fields from the rendered judge prompt. The reply is a JSON line with the
full distribution, then the Arena-Hard verdict label of the most likely option.
Arena-Hard reads the label (parser ``arena-hard-verdict``); soft ELO reads the
distribution (parser ``jev-choice``).
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from judgearena.log import get_logger
from judgearena.models import ChatVLLM

logger = get_logger(__name__)

QUESTION = "Which response better fulfils `user_request`?"
# Option text is sent to the model; the verdict label is what JudgeArena parses.
OPTIONS = (
    ("`response_A` is better than `response_B`", "[[A>B]]"),
    ("`response_B` is better than `response_A`", "[[B>A]]"),
    ("`response_A` and `response_B` are equally good", "[[A=B]]"),
)

# Delimiters written by jev-state.txt around the three rendered fields.
_FIELDS = ("user_request", "response_A", "response_B")
_STATE_PATTERN = re.compile(
    r"<<JEV:user_request>>\n(?P<user_request>.*?)\n"
    r"<<JEV:response_A>>\n(?P<response_A>.*?)\n"
    r"<<JEV:response_B>>\n(?P<response_B>.*?)\n<<JEV:end>>",
    re.DOTALL,
)


class ChatJEV9B(ChatVLLM):
    """Offline vLLM engine that answers one Choice question per judge input.

    Subclasses ChatVLLM only so JudgeArena takes its synchronous batch path;
    the parent constructor is not used.
    """

    def __init__(
        self,
        model: str,
        jev_max_model_len: int = 16384,
        jev_tensor_parallel_size: int = 1,
        jev_gpu_memory_utilization: float = 0.9,
        jev_max_response_tokens: int = 6000,
        **_ignored,
    ):
        from vllm import LLM
        from vllm.lora.request import LoRARequest

        root = Path(model)
        head = json.loads((root / "adapter_vllm" / "decision_head.json").read_text())
        temps = json.loads((root / "calibration.json").read_text())["per_kind"]
        start, end = head["slots"]["ranges"]["choice"]
        start = int(start)
        self.option_ids = head["verbalizer_ids"][start : start + len(OPTIONS)]
        self.option_bias = head["bias"][start : start + len(OPTIONS)]
        self.temperature = float(temps["choice"])
        self.max_response_tokens = jev_max_response_tokens

        self.model_path = model
        self._top_logprobs = None
        self.llm = LLM(
            model=model,
            enable_lora=True,
            max_lora_rank=32,
            logprobs_mode="processed_logprobs",
            max_model_len=jev_max_model_len,
            tensor_parallel_size=jev_tensor_parallel_size,
            gpu_memory_utilization=jev_gpu_memory_utilization,
        )
        self.tokenizer = self.llm.get_tokenizer()
        self.lora = LoRARequest("jev-decision", 1, str(root / "adapter_vllm"))
        self.n_truncated = 0

    def set_temperature(self, temperature: float) -> None:
        """Decisions are always read at the calibrated temperature."""

    def _truncate(self, text: str) -> str:
        """Keep head 60% / tail 40% of a response longer than the budget."""
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= self.max_response_tokens:
            return text
        self.n_truncated += 1
        head = int(self.max_response_tokens * 0.6)
        tail = self.max_response_tokens - head
        return (
            self.tokenizer.decode(ids[:head])
            + "\n[...]\n"
            + self.tokenizer.decode(ids[-tail:])
        )

    def _state(self, input_item) -> dict[str, str]:
        user_text = self._to_messages(input_item)[-1]["content"]
        match = _STATE_PATTERN.search(user_text)
        if match is None:
            raise ValueError(
                "JEV9B judge needs the jev-state.txt user prompt "
                "(--judge.prompt.user_file); its delimiters were not found."
            )
        state = {field: match.group(field) for field in _FIELDS}
        for field in ("response_A", "response_B"):
            state[field] = self._truncate(state[field])
        return state

    def _prompt(self, state: dict[str, str]) -> str:
        options = "\n".join(
            f"{'ABCDEFGHIJKLMNOP'[i]}) {text}" for i, (text, _) in enumerate(OPTIONS)
        )
        return (
            f"[kind] choice\n[state] {json.dumps(state, ensure_ascii=False)}\n"
            f"[question] {QUESTION}\n[options]\n{options}\n[decision]:"
        )

    def batch(self, inputs: list, **invoke_kwargs) -> list[str]:
        from vllm import SamplingParams, TokensPrompt

        prompts = []
        n_tokens = []
        for item in inputs:
            ids = self.tokenizer.encode(
                self._prompt(self._state(item)), add_special_tokens=False
            )
            prompts.append(TokensPrompt(prompt_token_ids=ids))
            n_tokens.append(len(ids))
        params = SamplingParams(
            max_tokens=1,
            temperature=1.0,
            allowed_token_ids=list(self.option_ids),
            logprobs=len(self.option_ids),
        )
        outputs = self.llm.generate(prompts, params, lora_request=self.lora)

        replies = []
        for out, length in zip(outputs, n_tokens, strict=True):
            logprobs = out.outputs[0].logprobs[0]
            z = [
                (
                    (logprobs[tok].logprob if tok in logprobs else -1e9)
                    + self.option_bias[i]
                )
                / self.temperature
                for i, tok in enumerate(self.option_ids)
            ]
            top = max(z)
            weights = [math.exp(x - top) for x in z]
            probs = [w / sum(weights) for w in weights]
            best = max(range(len(OPTIONS)), key=probs.__getitem__)
            distribution = {
                label.strip("[]"): round(p, 6)
                for (_, label), p in zip(OPTIONS, probs, strict=True)
            }
            replies.append(
                json.dumps({"probabilities": distribution, "prompt_tokens": length})
                + f"\n{OPTIONS[best][1]}"
            )
        if self.n_truncated:
            logger.info("JEV9B: truncated %d responses so far.", self.n_truncated)
        return replies
