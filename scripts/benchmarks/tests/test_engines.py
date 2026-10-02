"""The same behavioral checks run against every selected engine launcher."""

import math
from collections.abc import Iterator
from pathlib import Path

import pytest
from bench import BenchRequest, BenchResponse, BenchSampling, ChatMessage, ChatRole

from .engines import llamacpp, mlx, mtplx, omlx
from .engines.engine_process import EngineProcess, run_engine

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.integration


@pytest.fixture(scope="class")
def llamacpp_engine(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[EngineProcess]:
    if "llamacpp" not in request.config.getoption("engine"):
        pytest.skip("Enable llama.cpp tests with --engine llamacpp")

    timeout = request.config.getoption("engine_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

    stderr_log = tmp_path_factory.mktemp("engine-llamacpp") / "stderr.log"
    with run_engine(llamacpp.command(request.config), ROOT, stderr_log, timeout) as process:
        yield process


@pytest.fixture(scope="class")
def mlx_engine(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EngineProcess]:
    if "mlx" not in request.config.getoption("engine"):
        pytest.skip("Enable MLX tests with --engine mlx")

    timeout = request.config.getoption("engine_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

    stderr_log = tmp_path_factory.mktemp("engine-mlx") / "stderr.log"
    with run_engine(mlx.command(request.config), ROOT, stderr_log, timeout) as process:
        yield process


@pytest.fixture(scope="class")
def mtplx_engine(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EngineProcess]:
    if "mtplx" not in request.config.getoption("engine"):
        pytest.skip("Enable MTPLX tests with --engine mtplx")

    timeout = request.config.getoption("engine_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

    stderr_log = tmp_path_factory.mktemp("engine-mtplx") / "stderr.log"
    with run_engine(mtplx.command(request.config), ROOT, stderr_log, timeout) as process:
        yield process


@pytest.fixture(scope="class")
def omlx_engine(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[EngineProcess]:
    if "omlx" not in request.config.getoption("engine"):
        pytest.skip("Enable oMLX tests with --engine omlx")

    timeout = request.config.getoption("engine_timeout")
    if not math.isfinite(timeout) or timeout <= 0:
        raise pytest.UsageError("--engine-timeout must be finite and greater than zero")

    stderr_log = tmp_path_factory.mktemp("engine-omlx") / "stderr.log"
    with run_engine(omlx.command(request.config), ROOT, stderr_log, timeout) as process:
        yield process


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
    """Shared tests inherited by explicitly declared engine test classes."""

    def test_text_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(prompt_text="The capital of France is", max_tokens=8)
        assert_responses(engine.request(request), 1)

    def test_chat_generation(self, engine: EngineProcess) -> None:
        request = BenchRequest(
            prompt_chat=[ChatMessage(role=ChatRole.USER, content="Say hello in one short sentence.")],
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


class TestLlamaCpp(EngineTests):
    @pytest.fixture
    def engine(self, llamacpp_engine: EngineProcess) -> EngineProcess:
        return llamacpp_engine


class TestMLX(EngineTests):
    @pytest.fixture
    def engine(self, mlx_engine: EngineProcess) -> EngineProcess:
        return mlx_engine


class TestMTPLX(EngineTests):
    @pytest.fixture
    def engine(self, mtplx_engine: EngineProcess) -> EngineProcess:
        return mtplx_engine


class TestOMLX(EngineTests):
    @pytest.fixture
    def engine(self, omlx_engine: EngineProcess) -> EngineProcess:
        return omlx_engine
