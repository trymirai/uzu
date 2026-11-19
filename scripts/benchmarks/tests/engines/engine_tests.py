"""Shared behavioral checks and subprocess lifecycle for benchmark engines."""

import json
import math
import subprocess
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, ClassVar, get_args

import pytest
from bench import BenchRequest, BenchResponse, ChatMessage, CooldownRequest, ReasoningEffort

from .engine_process import EngineProcess, run_engine

ROOT = Path(__file__).resolve().parents[2]


def assert_response(response: BenchResponse) -> None:
    assert response.text, "Expected generated text for the nonempty test prompt"
    speculation = {"tokens_per_forward_pass", "forwards_per_second"}
    measurements = response.model_dump(exclude={"type", "text", "finish_reason"} | speculation)
    for name, value in measurements.items():
        assert math.isfinite(value) and value >= 0, f"Invalid {name}: {value}"

    assert response.input_tokens_count > 0
    assert response.duration > 0
    assert response.energy_gpu > 0
    assert response.tokens_per_forward_pass is None or response.tokens_per_forward_pass >= 1
    assert (response.forwards_per_second is None) == (response.tokens_per_forward_pass is None)
    assert response.forwards_per_second is None or response.forwards_per_second > 0
    assert response.time_to_first_token <= response.duration
    assert response.memory_used > 0
    assert response.memory_phys_footprint > 0
    assert response.memory_resident > 0


class EngineTests:
    """Shared process lifecycle and checks inherited by each engine's test class."""

    pytestmark = pytest.mark.integration
    engine_name: ClassVar[str]
    supports_seed: ClassVar[bool] = True
    specdec: ClassVar[bool] = False

    @classmethod
    def command(cls, config: pytest.Config, model: Path) -> list[str]:
        """Builds the engine in the checkout and returns the command that runs it."""
        directory = ROOT / f"engine-{cls.engine_name}"
        subprocess.run([directory / "build.sh"], check=True)
        return [str(directory / "run.sh"), str(model)]

    @classmethod
    @contextmanager
    def launch(
        cls,
        request: pytest.FixtureRequest,
        tmp_path_factory: pytest.TempPathFactory,
        sampling: dict[str, Any] | None,
        *,
        wrong_specdec: bool = False,
    ) -> Iterator[EngineProcess]:
        """Without sampling, the model.json has only the model."""
        if cls.engine_name not in request.config.getoption("engine"):
            pytest.skip(f"Enable {cls.engine_name} tests with --engine {cls.engine_name}")

        timeout = request.config.getoption("engine_timeout")
        if not math.isfinite(timeout) or timeout <= 0:
            raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

        stderr_log = tmp_path_factory.mktemp(f"engine-{cls.engine_name}") / "stderr.log"
        draft = request.config.getoption(f"{cls.engine_name}_draft_model", default=None)
        model = {"model": request.config.getoption(f"{cls.engine_name}_model")}
        if sampling is not None:
            model |= {
                "specdec": (cls.specdec or draft is not None) != wrong_specdec,
                "sampling": sampling,
                "reasoning_efforts": dict.fromkeys(get_args(ReasoningEffort), "default"),
                "model_size": 0,
            }
        if draft:
            model["draft_model"] = draft
        model_file = stderr_log.with_name("model.json")
        model_file.write_text(json.dumps(model))
        command = cls.command(request.config, model_file)
        with run_engine(command, ROOT, stderr_log, timeout, ready=not wrong_specdec) as process:
            yield process

    @pytest.fixture(
        scope="class",
        params=[
            {"greedy": True},
            {
                "temp": 0.7,
                "top_k": 20,
                "top_p": 0.9,
                "min_p": None,
                "repetition_penalty": None,
                "suffix_repetition_length": None,
            },
        ],
        ids=["greedy", "stochastic"],
    )
    @classmethod
    def engine(
        cls, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> Iterator[EngineProcess]:
        with cls.launch(request, tmp_path_factory, request.param) as process:
            yield process

    def test_invalid_request_exits(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        with self.launch(request, tmp_path_factory, {"greedy": True}) as engine:
            engine.reject("not a request")

    def test_request_without_reasoning_effort_exits(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        unspecified = {"type": "BenchRequest", "messages": [{"role": "user", "content": "Say hello."}]}
        with self.launch(request, tmp_path_factory, {"greedy": True}) as engine:
            engine.reject(json.dumps(unspecified))

    def test_model_alone(self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> None:
        prompt = [ChatMessage(role="user", content="Say hello in one short sentence.")]
        with self.launch(request, tmp_path_factory, None) as engine:
            for effort in ("default", *engine.ready.supported_reasoning_efforts):
                assert_response(
                    engine.request(BenchRequest(messages=prompt, max_output_tokens=8, reasoning_effort=effort))
                )

    def test_wrong_specdec_exits(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        with self.launch(request, tmp_path_factory, {"greedy": True}, wrong_specdec=True):
            pass

    def test_unsupported_seed_exits(
        self, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> None:
        if self.supports_seed:
            pytest.skip(f"{self.engine_name} supports seed")
        seeded = BenchRequest(
            messages=[ChatMessage(role="user", content="Say hello.")],
            max_output_tokens=8,
            reasoning_effort="default",
            seed=7,
        )
        with self.launch(request, tmp_path_factory, {"greedy": True}) as engine:
            engine.reject(seeded.model_dump_json(exclude_none=True))

    def test_cooldown(self, engine: EngineProcess) -> None:
        engine.cooldown(CooldownRequest(threshold=100.0, target=100.0))
        # A Mac is never as hot as this threshold, so a target that it never cools to is not waited for.
        engine.cooldown(CooldownRequest(threshold=150.0, target=0.0))

    def test_chat_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(
            messages=[ChatMessage(role="user", content="Say hello in one short sentence.")],
            max_output_tokens=8,
            reasoning_effort="default",
        )
        assert_response(engine.request(request))

    def test_multiple_requests_in_one_process(self, engine: EngineProcess) -> None:
        first = BenchRequest(
            messages=[ChatMessage(role="user", content="Complete this sentence: The sky is")],
            max_output_tokens=8,
            reasoning_effort="default",
        )
        assert_response(engine.request(first))

        second = BenchRequest(
            messages=[ChatMessage(role="user", content="Complete this sentence: The grass is")],
            max_output_tokens=8,
            reasoning_effort="default",
        )
        assert_response(engine.request(second))

    def test_seed(self, engine: EngineProcess) -> None:
        if not self.supports_seed:
            pytest.skip(f"{self.engine_name} does not support seed")
        request = BenchRequest(
            messages=[ChatMessage(role="user", content="Write one sentence about the sea.")],
            max_output_tokens=16,
            reasoning_effort="default",
            seed=7,
        )
        assert engine.request(request).text == engine.request(request).text
