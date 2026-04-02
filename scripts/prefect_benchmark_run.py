from __future__ import annotations

import argparse
import json
import os
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Protocol

from prefect_flow_run_common import (
    FlowRun,
    FlowRunStateType,
    LogSink,
    PoolName,
    PrefectApiClient as BasePrefectApiClient,
    PrefectLog,
    observe_flow_run,
    parse_flow_run,
    print_live_log_line,
    ui_base_url_from_api_url,
    write_artifacts,
)


POOL_DEPLOYMENTS: dict[PoolName, str] = {
    PoolName.MACOS_A18_PRO: "run-benchmark-worker-flow/run-benchmark-a18-pro",
    PoolName.MACOS_M1: "run-benchmark-worker-flow/run-benchmark-m1",
    PoolName.MACOS_M2: "run-benchmark-worker-flow/run-benchmark-m2",
    PoolName.MACOS_M2_PRO: "run-benchmark-worker-flow/run-benchmark-m2-pro",
    PoolName.MACOS_M4: "run-benchmark-worker-flow/run-benchmark-m4",
    PoolName.MACOS_M4_PRO: "run-benchmark-worker-flow/run-benchmark-m4-pro",
}


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


class PrefectApiClient(BasePrefectApiClient):
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


def parse_benchmark_args_json(raw_value: str) -> tuple[str, ...]:
    parsed_value = json.loads(raw_value)
    if not isinstance(parsed_value, list):
        raise ValueError("benchmark_args must decode to a JSON array")
    return tuple(str(item) for item in parsed_value)


def deployment_name_for_pool(pool: PoolName) -> str:
    return POOL_DEPLOYMENTS[pool]


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
    log_sink: LogSink | None = None,
) -> BenchmarkRunResult:
    deployment_name = deployment_name_for_pool(pool)
    deployment_id = client.read_deployment_id(deployment_name)
    created_run = client.create_flow_run_from_deployment(deployment_id, benchmark_args)
    observed_run = observe_flow_run(
        client=client,
        created_run=created_run,
        deployment_name=deployment_name,
        ui_base_url=ui_base_url,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        log_sink=log_sink,
    )
    return BenchmarkRunResult(
        deployment_name=observed_run.deployment_name,
        pool=pool,
        flow_run=observed_run.flow_run,
        logs=observed_run.logs,
        failure_reason=observed_run.failure_reason,
        benchmark_args=benchmark_args,
        actor=actor,
        triggering_actor=triggering_actor,
    )


def write_run_artifacts(output_dir: Path | str, result: BenchmarkRunResult) -> None:
    write_artifacts(
        output_dir,
        summary_payload=build_summary_payload(result),
        summary_markdown=build_summary_markdown(result),
        flow_run=result.flow_run,
        logs=result.logs,
    )


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
        log_sink=print_live_log_line,
    )
    write_run_artifacts(config.output_dir, result)
    print(build_summary_markdown(result))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
