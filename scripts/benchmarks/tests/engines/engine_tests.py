"""Shared behavioral checks and subprocess lifecycle for benchmark engines."""

import math
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import ClassVar

import pytest
from bench import BenchRequest, BenchResponse, BenchSampling, ChatMessage, ChatRole

from .engine_process import EngineProcess, run_engine

ROOT = Path(__file__).resolve().parents[2]


def assert_responses(responses: list[BenchResponse], num_runs: int) -> None:
    assert len(responses) == num_runs
    for response in responses:
        assert response.text, "Expected generated text for the nonempty test prompt"
        for name, value in response.model_dump(exclude={"text"}).items():
            assert math.isfinite(value) and value >= 0, f"Invalid {name}: {value}"

        assert response.duration > 0
        assert response.time_to_first_token <= response.duration
        assert response.memory_phys_footprint > 0
        assert response.memory_resident > 0


class EngineTests:
    """Shared process lifecycle and checks inherited by each engine's test class."""

    pytestmark = pytest.mark.integration
    engine_name: ClassVar[str]
    command: ClassVar[Callable[[pytest.Config], list[str]]]

    @pytest.fixture(scope="class")
    @classmethod
    def engine(
        cls, request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
    ) -> Iterator[EngineProcess]:
        if cls.engine_name not in request.config.getoption("engine"):
            pytest.skip(f"Enable {cls.engine_name} tests with --engine {cls.engine_name}")

        timeout = request.config.getoption("engine_timeout")
        if not math.isfinite(timeout) or timeout <= 0:
            raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

        stderr_log = tmp_path_factory.mktemp(f"engine-{cls.engine_name}") / "stderr.log"
        with run_engine(cls.command(request.config), ROOT, stderr_log, timeout) as process:
            yield process

    def test_text_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(prompt_text="The capital of France is", max_tokens=8)
        assert_responses(engine.request(request), 1)

    def test_chat_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(
            prompt_chat=[ChatMessage(role="user", content="Say hello in one short sentence.")],
            max_tokens=8,
        )
        assert_responses(engine.request(request), 1)

    def test_multiple_runs(self, engine: EngineProcess) -> None:
        request = BenchRequest(prompt_text="Count from one to five:", max_tokens=8, num_runs=2)
        assert_responses(engine.request(request), 2)

    def test_sampling(self, engine: EngineProcess) -> None:
        request = BenchRequest(
            prompt_text="The capital of France is",
            max_tokens=8,
            sampling=BenchSampling(top_k=20, top_p=0.9, temp=0.7),
        )
        assert_responses(engine.request(request), 1)

    def test_multiple_requests_in_one_process(self, engine: EngineProcess) -> None:
        first = BenchRequest(prompt_text="Complete this sentence: The sky is", max_tokens=8, num_runs=1)
        assert_responses(engine.request(first), 1)

        second = BenchRequest(prompt_text="Complete this sentence: The grass is", max_tokens=8, num_runs=2)
        assert_responses(engine.request(second), 2)
