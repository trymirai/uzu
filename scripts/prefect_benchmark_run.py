from __future__ import annotations

import argparse
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


class PoolName(str, Enum):
    MACOS_A18_PRO = "macos-a18-pro-pool"
    MACOS_M1 = "macos-m1-pool"
    MACOS_M2 = "macos-m2-pool"
    MACOS_M2_PRO = "macos-m2-pro-pool"
    MACOS_M4 = "macos-m4-pool"
    MACOS_M4_PRO = "macos-m4-pro-pool"


class FlowRunStateType(str, Enum):
    SCHEDULED = "SCHEDULED"
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CRASHED = "CRASHED"
    CANCELLED = "CANCELLED"
    CANCELLING = "CANCELLING"
    PAUSED = "PAUSED"
    UNKNOWN = "UNKNOWN"

    @property
    def is_terminal(self) -> bool:
        return self in {
            FlowRunStateType.COMPLETED,
            FlowRunStateType.FAILED,
            FlowRunStateType.CRASHED,
            FlowRunStateType.CANCELLED,
        }

    @classmethod
    def from_api_value(cls, raw_value: str | None) -> FlowRunStateType:
        if raw_value is None:
            return cls.UNKNOWN
        try:
            return cls(raw_value)
        except ValueError:
            return cls.UNKNOWN


POOL_DEPLOYMENTS: dict[PoolName, str] = {
    PoolName.MACOS_A18_PRO: "run-benchmark-worker-flow/run-benchmark-a18-pro",
    PoolName.MACOS_M1: "run-benchmark-worker-flow/run-benchmark-m1",
    PoolName.MACOS_M2: "run-benchmark-worker-flow/run-benchmark-m2",
    PoolName.MACOS_M2_PRO: "run-benchmark-worker-flow/run-benchmark-m2-pro",
    PoolName.MACOS_M4: "run-benchmark-worker-flow/run-benchmark-m4",
    PoolName.MACOS_M4_PRO: "run-benchmark-worker-flow/run-benchmark-m4-pro",
}


@dataclass(frozen=True)
class FlowRun:
    id: str
    name: str
    state_type: FlowRunStateType
    state_name: str
    url: str


@dataclass(frozen=True)
class PrefectLog:
    timestamp: str
    level: int
    message: str


@dataclass(frozen=True)
class BenchmarkRunResult:
    deployment_name: str
    pool: PoolName
    flow_run: FlowRun
    logs: tuple[PrefectLog, ...]
    failure_reason: str | None = None
    benchmark_args: tuple[str, ...] = ()
    actor: str | None = None
    triggering_actor: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.failure_reason is None and self.flow_run.state_type == FlowRunStateType.COMPLETED


@dataclass(frozen=True)
class CliConfig:
    api_url: str
    api_key: str
    pool: PoolName
    benchmark_args: tuple[str, ...]
    output_dir: Path
    poll_interval_seconds: int
    timeout_seconds: int
    actor: str | None
    triggering_actor: str | None


class PrefectClientProtocol(Protocol):
    def read_deployment_id(self, deployment_name: str) -> str:
        ...

    def create_flow_run_from_deployment(
        self,
        deployment_id: str,
        benchmark_args: tuple[str, ...],
    ) -> FlowRun:
        ...

    def read_flow_run(self, flow_run_id: str) -> FlowRun:
        ...

    def read_logs_page(self, flow_run_id: str, *, limit: int, offset: int) -> tuple[PrefectLog, ...]:
        ...


class PrefectApiError(RuntimeError):
    pass


@dataclass(frozen=True)
class PrefectApiClient:
    api_url: str
    api_key: str
    ui_base_url: str

    def read_deployment_id(self, deployment_name: str) -> str:
        flow_name, deployment_slug = deployment_name.split("/", maxsplit=1)
        response = self._request_json(
            method="GET",
            path=(
                "/deployments/name/"
                f"{urllib.parse.quote(flow_name, safe='')}/"
                f"{urllib.parse.quote(deployment_slug, safe='')}"
            ),
        )
        deployment_id = response.get("id")
        if not isinstance(deployment_id, str) or not deployment_id:
            raise PrefectApiError(f"deployment lookup returned invalid id for {deployment_name}")
        return deployment_id

    def create_flow_run_from_deployment(
        self,
        deployment_id: str,
        benchmark_args: tuple[str, ...],
    ) -> FlowRun:
        response = self._request_json(
            method="POST",
            path=f"/deployments/{urllib.parse.quote(deployment_id, safe='')}/create_flow_run",
            body={
                "parameters": {
                    "benchmark_args": list(benchmark_args),
                }
            },
        )
        return parse_flow_run(response, self.ui_base_url)

    def read_flow_run(self, flow_run_id: str) -> FlowRun:
        response = self._request_json(
            method="GET",
            path=f"/flow_runs/{urllib.parse.quote(flow_run_id, safe='')}",
        )
        return parse_flow_run(response, self.ui_base_url)

    def read_logs_page(self, flow_run_id: str, *, limit: int, offset: int) -> tuple[PrefectLog, ...]:
        response = self._request_json(
            method="POST",
            path="/logs/filter",
            body={
                "logs": {
                    "flow_run_id": {
                        "any_": [flow_run_id],
                    }
                },
                "limit": limit,
                "offset": offset,
                "sort": "TIMESTAMP_ASC",
            },
        )
        if not isinstance(response, list):
            raise PrefectApiError(f"log query returned invalid payload for flow run {flow_run_id}")
        return tuple(parse_prefect_log(item) for item in response)

    def _request_json(self, method: str, path: str, body: dict[str, Any] | None = None) -> Any:
        encoded_body = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.api_url.rstrip('/')}{path}",
            data=encoded_body,
            method=method,
            headers={
                "Accept": "application/json",
                "Authorization": f"Basic {self.api_key}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise PrefectApiError(
                f"prefect api request failed: method={method} path={path} status={exc.code} body={error_body}"
            ) from exc
        except urllib.error.URLError as exc:
            raise PrefectApiError(f"prefect api request failed: method={method} path={path} reason={exc.reason}") from exc

        if not payload:
            return None
        return json.loads(payload)


def parse_benchmark_args_json(raw_value: str) -> tuple[str, ...]:
    parsed_value = json.loads(raw_value)
    if not isinstance(parsed_value, list):
        raise ValueError("benchmark_args must decode to a JSON array")
    return tuple(str(item) for item in parsed_value)


def deployment_name_for_pool(pool: PoolName) -> str:
    return POOL_DEPLOYMENTS[pool]


def parse_flow_run(payload: dict[str, Any], ui_base_url: str) -> FlowRun:
    flow_run_id = str(payload.get("id", ""))
    flow_run_name = str(payload.get("name", "") or flow_run_id)
    state_payload = payload.get("state")
    state_type = FlowRunStateType.UNKNOWN
    state_name = ""
    if isinstance(state_payload, dict):
        state_type = FlowRunStateType.from_api_value(_read_optional_str(state_payload, "type"))
        state_name = _read_optional_str(state_payload, "name") or state_type.value.title()
    else:
        state_type = FlowRunStateType.from_api_value(_read_optional_str(payload, "state_type"))
        state_name = _read_optional_str(payload, "state_name") or state_type.value.title()
    return FlowRun(
        id=flow_run_id,
        name=flow_run_name,
        state_type=state_type,
        state_name=state_name,
        url=f"{ui_base_url.rstrip('/')}/runs/flow-run/{flow_run_id}",
    )


def parse_prefect_log(payload: dict[str, Any]) -> PrefectLog:
    timestamp = _read_optional_str(payload, "timestamp") or ""
    level = payload.get("level")
    if not isinstance(level, int):
        raise PrefectApiError(f"prefect log payload has invalid level: {payload}")
    message = _read_optional_str(payload, "message") or ""
    return PrefectLog(timestamp=timestamp, level=level, message=message)


def execute_benchmark_run(
    client: PrefectClientProtocol,
    pool: PoolName,
    benchmark_args: tuple[str, ...],
    ui_base_url: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None] = time.sleep,
    actor: str | None = None,
    triggering_actor: str | None = None,
) -> BenchmarkRunResult:
    deployment_name = deployment_name_for_pool(pool)
    deployment_id = client.read_deployment_id(deployment_name)
    created_run = client.create_flow_run_from_deployment(deployment_id, benchmark_args)
    final_run = wait_for_terminal_flow_run(
        client=client,
        initial_run=created_run,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
    )
    logs = read_all_logs(client, created_run.id)
    failure_reason = None
    if not final_run.state_type.is_terminal:
        failure_reason = (
            f"timed out after {timeout_seconds} seconds waiting for terminal state "
            f"(last state: {final_run.state_name})"
        )
    elif final_run.state_type != FlowRunStateType.COMPLETED:
        failure_reason = f"flow run finished in state {final_run.state_name}"
    return BenchmarkRunResult(
        deployment_name=deployment_name,
        pool=pool,
        flow_run=FlowRun(
            id=final_run.id,
            name=final_run.name,
            state_type=final_run.state_type,
            state_name=final_run.state_name,
            url=f"{ui_base_url.rstrip('/')}/runs/flow-run/{final_run.id}",
        ),
        logs=logs,
        failure_reason=failure_reason,
        benchmark_args=benchmark_args,
        actor=actor,
        triggering_actor=triggering_actor,
    )


def wait_for_terminal_flow_run(
    client: PrefectClientProtocol,
    initial_run: FlowRun,
    poll_interval_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None],
) -> FlowRun:
    current_run = initial_run
    deadline = time.monotonic() + timeout_seconds
    while not current_run.state_type.is_terminal and time.monotonic() < deadline:
        if poll_interval_seconds > 0:
            sleep(poll_interval_seconds)
        current_run = client.read_flow_run(current_run.id)
    return current_run


def read_all_logs(client: PrefectClientProtocol, flow_run_id: str) -> tuple[PrefectLog, ...]:
    offset = 0
    page_size = 200
    all_logs: list[PrefectLog] = []
    while True:
        page = client.read_logs_page(flow_run_id, limit=page_size, offset=offset)
        all_logs.extend(page)
        if len(page) < page_size:
            return tuple(all_logs)
        offset += page_size


def write_run_artifacts(output_dir: Path | str, result: BenchmarkRunResult) -> None:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(json.dumps(build_summary_payload(result), indent=2, sort_keys=True))
    (artifact_dir / "summary.md").write_text(build_summary_markdown(result))
    (artifact_dir / "logs.txt").write_text(format_logs(result.flow_run, result.logs))


def build_summary_payload(result: BenchmarkRunResult) -> dict[str, Any]:
    return {
        "actor": result.actor,
        "benchmark_args": list(result.benchmark_args),
        "deployment_name": result.deployment_name,
        "failure_reason": result.failure_reason,
        "flow_run_id": result.flow_run.id,
        "flow_run_name": result.flow_run.name,
        "flow_run_state_name": result.flow_run.state_name,
        "flow_run_state_type": result.flow_run.state_type.value,
        "flow_run_url": result.flow_run.url,
        "pool": result.pool.value,
        "succeeded": result.succeeded,
        "triggering_actor": result.triggering_actor,
    }


def build_summary_markdown(result: BenchmarkRunResult) -> str:
    lines = [
        "## Prefect Benchmark Run",
        "",
        f"- status: `{'succeeded' if result.succeeded else 'failed'}`",
        f"- pool: `{result.pool.value}`",
        f"- deployment: `{result.deployment_name}`",
        f"- flow run: `{result.flow_run.name}` (`{result.flow_run.id}`)",
        f"- state: `{result.flow_run.state_name}`",
        f"- url: {result.flow_run.url}",
        f"- actor: `{result.actor or 'unknown'}`",
        f"- triggering actor: `{result.triggering_actor or result.actor or 'unknown'}`",
        f"- benchmark args: `{json.dumps(list(result.benchmark_args))}`",
    ]
    if result.failure_reason is not None:
        lines.append(f"- failure reason: `{result.failure_reason}`")
    return "\n".join(lines) + "\n"


def format_logs(flow_run: FlowRun, logs: tuple[PrefectLog, ...]) -> str:
    lines = [
        f"{log.timestamp} | {logging.getLevelName(log.level):7s} | Flow run {flow_run.name!r} - {log.message}"
        for log in logs
    ]
    return "\n".join(lines) + ("\n" if lines else "")


def parse_cli_args(argv: list[str] | None = None) -> CliConfig:
    parser = argparse.ArgumentParser(description="Trigger one Prefect benchmark worker deployment and fetch its logs.")
    parser.add_argument("--api-url", required=True, help="Prefect API URL")
    parser.add_argument(
        "--pool",
        required=True,
        choices=[pool.value for pool in PoolName],
        help="Benchmark worker pool to target.",
    )
    parser.add_argument(
        "--benchmark-args",
        required=True,
        help='JSON array of argv passed after "uv run benchmarks".',
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary and log artifacts.")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Polling interval for flow run state.")
    parser.add_argument("--timeout-seconds", type=int, default=21600, help="Overall timeout for waiting on the flow run.")
    parser.add_argument("--actor", default=None, help="GitHub actor that started the workflow.")
    parser.add_argument("--triggering-actor", default=None, help="GitHub actor that triggered this run or re-run.")
    parsed_args = parser.parse_args(argv)
    api_key = os.environ.get("PREFECT_API_KEY")
    if not api_key:
        raise SystemExit("PREFECT_API_KEY environment variable is required")
    api_url = parsed_args.api_url.strip()
    if not api_url:
        raise SystemExit("--api-url must be non-empty")
    return CliConfig(
        api_url=api_url,
        api_key=api_key,
        pool=PoolName(parsed_args.pool),
        benchmark_args=parse_benchmark_args_json(parsed_args.benchmark_args),
        output_dir=Path(parsed_args.output_dir),
        poll_interval_seconds=parsed_args.poll_interval_seconds,
        timeout_seconds=parsed_args.timeout_seconds,
        actor=parsed_args.actor,
        triggering_actor=parsed_args.triggering_actor,
    )


def ui_base_url_from_api_url(api_url: str) -> str:
    normalized_url = api_url.rstrip("/")
    if normalized_url.endswith("/api"):
        return normalized_url[: -len("/api")]
    return normalized_url


def main(argv: list[str] | None = None) -> int:
    config = parse_cli_args(argv)
    client = PrefectApiClient(
        api_url=config.api_url.rstrip("/"),
        api_key=config.api_key,
        ui_base_url=ui_base_url_from_api_url(config.api_url),
    )
    result = execute_benchmark_run(
        client=client,
        pool=config.pool,
        benchmark_args=config.benchmark_args,
        ui_base_url=client.ui_base_url,
        poll_interval_seconds=config.poll_interval_seconds,
        timeout_seconds=config.timeout_seconds,
        actor=config.actor,
        triggering_actor=config.triggering_actor,
    )
    write_run_artifacts(config.output_dir, result)
    print(build_summary_markdown(result))
    return 0 if result.succeeded else 1


def _read_optional_str(payload: dict[str, Any], key: str) -> str | None:
    raw_value = payload.get(key)
    return raw_value if isinstance(raw_value, str) else None


if __name__ == "__main__":
    raise SystemExit(main())
