import json
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any, Literal, get_args

import huggingface_hub
from bench import (
    REASONING_LEVELS,
    BenchRequest,
    BenchResponse,
    BenchSampling,
    CooldownResponse,
    GreedySampling,
    ModelConfig,
    ReadyResponse,
    ReasoningEffort,
    Request,
    StochasticSampling,
)
from huggingface_hub.errors import LocalEntryNotFoundError
from mach import await_cooldown
from pydantic import TypeAdapter


class InferenceEngine(ABC):
    @abstractmethod
    def execute(self, request: BenchRequest) -> BenchResponse:
        pass


def get_model_path(model: str, revision: str | None = None) -> str:
    path = Path(model).expanduser()
    if path.is_dir():
        if revision is not None:
            raise ValueError(f"A local model directory has no revision: {model}")
        return str(path)

    cache_dir = Path("~/.cache/huggingface/hub").expanduser()
    try:
        return huggingface_hub.snapshot_download(
            repo_id=model, revision=revision, cache_dir=cache_dir, local_files_only=True
        )
    except LocalEntryNotFoundError:
        return huggingface_hub.snapshot_download(repo_id=model, revision=revision, cache_dir=cache_dir)


def get_model_arguments(config_file: str) -> list[str]:
    config = json.loads(Path(config_file).read_text())
    arguments = ["--config", config_file, "--model", get_model_path(config["model"], config.get("revision"))]
    if config.get("draft_model") is not None:
        arguments += ["--draft-model", get_model_path(config["draft_model"], config.get("draft_revision"))]
    elif config.get("draft_revision") is not None:
        raise ValueError("draft_revision requires draft_model")
    return arguments


def check_specdec(config: ModelConfig, speculative: bool) -> None:
    """`speculative` is whether the engine decodes speculatively as configured."""
    if config.specdec != speculative:
        actual = "decodes" if speculative else "does not decode"
        raise ValueError(f"specdec is {str(config.specdec).lower()}, but the engine {actual} speculatively")


def get_sampling(sampling: BenchSampling, *, supports_repetition: bool = False) -> StochasticSampling:
    if isinstance(sampling, GreedySampling):
        return StochasticSampling(
            temp=0, top_k=1, top_p=None, min_p=None, repetition_penalty=None, suffix_repetition_length=None
        )
    if sampling.temp == 0:
        raise ValueError("sampling.temp cannot be 0; use sampling.greedy: true")
    if sampling.repetition_penalty is not None and sampling.suffix_repetition_length is None:
        raise ValueError("repetition_penalty requires suffix_repetition_length")
    if not supports_repetition and (
        sampling.repetition_penalty is not None or sampling.suffix_repetition_length is not None
    ):
        raise ValueError("This engine does not support repetition_penalty and suffix_repetition_length")
    return sampling


def check_reasoning_effort(effort: ReasoningEffort, render: Callable[[ReasoningEffort], object]) -> None:
    """`render` returns the engine's prompt for an effort, or None where the engine rejects it."""
    if effort == "default":
        return
    prompt = render(effort)
    if effort == "disabled":
        others = [render(other) for other in get_args(ReasoningEffort) if other != effort]
        honored = any(other not in (None, prompt) for other in others)
    else:
        honored = prompt not in [render(level) for level in REASONING_LEVELS if level != effort]
    if prompt is None or not honored:
        raise ValueError(f"The model does not support reasoning_effort {effort}")


def get_template_reasoning(effort: ReasoningEffort) -> dict[str, Any]:
    if effort == "default":
        return {}
    if effort == "disabled":
        return {"enable_thinking": False}
    # Qwen's templates take the level as reasoning_effort, Muse Glimmer's as reasoning_strength.
    return {"enable_thinking": True, "reasoning_effort": effort, "reasoning_strength": effort}


def get_finish_reason(reason: str | None) -> Literal["stop", "length"]:
    if reason in ("stop", "tool_calls"):
        return "stop"
    if reason == "length":
        return "length"
    raise RuntimeError(f"Unexpected finish reason: {reason}")


def get_tokenized_prompt(request: BenchRequest, tokenizer: Any) -> list[int]:
    from jinja2 import TemplateError

    messages = [message.model_dump(mode="json") for message in request.messages]
    options = {"add_generation_prompt": True, **request.model_dump(mode="json", include={"tools", "tool_choice"})}
    effort = request.reasoning_effort

    def render(candidate: ReasoningEffort) -> str | None:
        reasoning = get_template_reasoning(candidate)
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, **options, **reasoning)
        except TemplateError:
            return None

    check_reasoning_effort(effort, render)
    reasoning = get_template_reasoning(effort)
    return tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False, **options, **reasoning)


def run_loop(engine: InferenceEngine) -> None:
    requests = TypeAdapter(Request)
    print(ReadyResponse().model_dump_json(), flush=True)
    for line in sys.stdin:
        if not line.strip():
            continue

        request = requests.validate_json(line)
        if isinstance(request, BenchRequest):
            with redirect_stdout(sys.stderr):
                response = engine.execute(request)
            print(response.model_dump_json(), flush=True)
        else:
            await_cooldown(request.temperature, request.only_if_capped)
            print(CooldownResponse().model_dump_json(), flush=True)
