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
    PoolName.MACOS_A18_PRO: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-a18-pro",
    PoolName.MACOS_M1: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m1",
    PoolName.MACOS_M2: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m2",
    PoolName.MACOS_M2_PRO: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m2-pro",
    PoolName.MACOS_M4: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m4",
    PoolName.MACOS_M4_PRO: "run-uzu-cargo-command-worker-flow/run-uzu-cargo-command-m4-pro",
}


@dataclass(frozen=True)
class UzuCargoRunResult:
    deployment_name: str
    pool: PoolName
    flow_run: FlowRun
    logs: tuple[PrefectLog, ...]
    git_ref: str
    cargo_args: tuple[str, ...]
    failure_reason: str | None = None
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
    cargo_args: tuple[str, ...]
    git_ref: str
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
        git_ref: str,
        cargo_args: tuple[str, ...],
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
        git_ref: str,
        cargo_args: tuple[str, ...],
    ) -> FlowRun:
        response = self._request_json(
            method="POST",
            path=f"/deployments/{urllib.parse.quote(deployment_id, safe='')}/create_flow_run",
            body={
                "parameters": {
                    "git_ref": git_ref,
                    "cargo_args": list(cargo_args),
                }
            },
        )
        return parse_flow_run(response, self.ui_base_url)


def parse_cargo_args_json(raw_value: str) -> tuple[str, ...]:
    parsed_value = json.loads(raw_value)
    if not isinstance(parsed_value, list):
        raise ValueError("cargo_args must decode to a JSON array")
    if not all(isinstance(item, str) for item in parsed_value):
        raise ValueError("cargo_args array must contain only strings")
    return tuple(parsed_value)


def deployment_name_for_pool(pool: PoolName) -> str:
    return POOL_DEPLOYMENTS[pool]


def execute_uzu_cargo_run(
    client: PrefectClientProtocol,
    pool: PoolName,
    git_ref: str,
    cargo_args: tuple[str, ...],
    ui_base_url: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None],
    actor: str | None,
    triggering_actor: str | None,
    log_sink: LogSink | None = None,
) -> UzuCargoRunResult:
    deployment_name = deployment_name_for_pool(pool)
    deployment_id = client.read_deployment_id(deployment_name)
    created_run = client.create_flow_run_from_deployment(deployment_id, git_ref, cargo_args)
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
    return UzuCargoRunResult(
        deployment_name=observed_run.deployment_name,
        pool=pool,
        flow_run=observed_run.flow_run,
        logs=observed_run.logs,
        git_ref=git_ref,
        cargo_args=cargo_args,
        failure_reason=observed_run.failure_reason,
        actor=actor,
        triggering_actor=triggering_actor,
    )


def write_run_artifacts(output_dir: Path | str, result: UzuCargoRunResult) -> None:
    write_artifacts(
        output_dir,
        summary_payload=build_summary_payload(result),
        summary_markdown=build_summary_markdown(result),
        flow_run=result.flow_run,
        logs=result.logs,
    )


def build_summary_payload(result: UzuCargoRunResult) -> dict[str, Any]:
    return {
        "actor": result.actor,
        "cargo_args": list(result.cargo_args),
        "deployment_name": result.deployment_name,
        "failure_reason": result.failure_reason,
        "flow_run_id": result.flow_run.id,
        "flow_run_name": result.flow_run.name,
        "flow_run_state_name": result.flow_run.state_name,
        "flow_run_state_type": result.flow_run.state_type.value,
        "flow_run_url": result.flow_run.url,
        "git_ref": result.git_ref,
        "pool": result.pool.value,
        "succeeded": result.succeeded,
        "triggering_actor": result.triggering_actor,
    }


def build_summary_markdown(result: UzuCargoRunResult) -> str:
    lines = [
        "## Prefect Uzu Cargo Command Run",
        "",
        f"- status: `{'succeeded' if result.succeeded else 'failed'}`",
        f"- pool: `{result.pool.value}`",
        f"- deployment: `{result.deployment_name}`",
        f"- flow run: `{result.flow_run.name}` (`{result.flow_run.id}`)",
        f"- state: `{result.flow_run.state_name}`",
        f"- url: {result.flow_run.url}",
        f"- actor: `{result.actor or 'unknown'}`",
        f"- triggering actor: `{result.triggering_actor or result.actor or 'unknown'}`",
        f"- git_ref: `{result.git_ref}`",
        f"- cargo_args: `{json.dumps(list(result.cargo_args))}`",
    ]
    if result.failure_reason is not None:
        lines.append(f"- failure reason: `{result.failure_reason}`")
    return "\n".join(lines) + "\n"


def parse_cli_args(argv: list[str] | None = None) -> CliConfig:
    parser = argparse.ArgumentParser(
        description="Trigger one Prefect cargo-command worker deployment and fetch its logs."
    )
    parser.add_argument("--api-url", required=True, help="Prefect API URL")
    parser.add_argument(
        "--pool",
        required=True,
        choices=[pool.value for pool in PoolName],
        help="MacOS pool to target.",
    )
    parser.add_argument(
        "--cargo-args",
        required=True,
        help="JSON array of argv passed after 'cargo'.",
    )
    parser.add_argument(
        "--git-ref",
        required=True,
        help="git ref (branch/tag/SHA) to check out in uzu",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for summary and log artifacts.")
    parser.add_argument("--poll-interval-seconds", type=int, default=15, help="Polling interval for flow run state.")
    parser.add_argument("--timeout-seconds", type=int, default=21600, help="Timeout in seconds waiting for final state.")
    parser.add_argument("--actor", default=None, help="GitHub actor that started the workflow.")
    parser.add_argument("--triggering-actor", default=None, help="GitHub actor that triggered this run or re-run.")
    parsed_args = parser.parse_args(argv)
    api_key = os.environ.get("PREFECT_API_KEY")
    if not api_key:
        raise SystemExit("PREFECT_API_KEY environment variable is required")
    api_url = parsed_args.api_url.strip()
    if not api_url:
        raise SystemExit("--api-url must be non-empty")
    git_ref = parsed_args.git_ref.strip()
    if not git_ref:
        raise SystemExit("--git-ref must be non-empty")
    cargo_args = parse_cargo_args_json(parsed_args.cargo_args)
    return CliConfig(
        api_url=api_url.rstrip("/"),
        api_key=api_key,
        pool=PoolName(parsed_args.pool),
        cargo_args=cargo_args,
        git_ref=git_ref,
        output_dir=Path(parsed_args.output_dir),
        poll_interval_seconds=parsed_args.poll_interval_seconds,
        timeout_seconds=parsed_args.timeout_seconds,
        actor=parsed_args.actor,
        triggering_actor=parsed_args.triggering_actor,
    )


def main(argv: list[str] | None = None) -> int:
    config = parse_cli_args(argv)
    client = PrefectApiClient(
        api_url=config.api_url,
        api_key=config.api_key,
        ui_base_url=ui_base_url_from_api_url(config.api_url),
    )
    result = execute_uzu_cargo_run(
        client=client,
        pool=config.pool,
        git_ref=config.git_ref,
        cargo_args=config.cargo_args,
        ui_base_url=client.ui_base_url,
        poll_interval_seconds=config.poll_interval_seconds,
        timeout_seconds=config.timeout_seconds,
        sleep=time.sleep,
        actor=config.actor,
        triggering_actor=config.triggering_actor,
        log_sink=print_live_log_line,
    )
    write_run_artifacts(config.output_dir, result)
    print(build_summary_markdown(result))
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
