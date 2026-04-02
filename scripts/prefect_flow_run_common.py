from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Protocol


type LogSink = Callable[[str], None]


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


PREFECT_LOG_PAGE_SIZE = 200


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
class ObservedFlowRun:
    deployment_name: str
    flow_run: FlowRun
    logs: tuple[PrefectLog, ...]
    failure_reason: str | None = None


class FlowRunObserverClientProtocol(Protocol):
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


def observe_flow_run(
    client: FlowRunObserverClientProtocol,
    created_run: FlowRun,
    deployment_name: str,
    ui_base_url: str,
    poll_interval_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None],
    log_sink: LogSink | None = None,
) -> ObservedFlowRun:
    final_run = wait_for_terminal_flow_run(
        client=client,
        initial_run=created_run,
        poll_interval_seconds=poll_interval_seconds,
        timeout_seconds=timeout_seconds,
        sleep=sleep,
        log_sink=log_sink,
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
    return ObservedFlowRun(
        deployment_name=deployment_name,
        flow_run=FlowRun(
            id=final_run.id,
            name=final_run.name,
            state_type=final_run.state_type,
            state_name=final_run.state_name,
            url=f"{ui_base_url.rstrip('/')}/runs/flow-run/{final_run.id}",
        ),
        logs=logs,
        failure_reason=failure_reason,
    )


def wait_for_terminal_flow_run(
    client: FlowRunObserverClientProtocol,
    initial_run: FlowRun,
    poll_interval_seconds: int,
    timeout_seconds: int,
    sleep: Callable[[float], None],
    log_sink: LogSink | None = None,
) -> FlowRun:
    current_run = initial_run
    next_log_offset = stream_new_logs(client, current_run, 0, log_sink)
    deadline = time.monotonic() + timeout_seconds
    while not current_run.state_type.is_terminal:
        remaining_timeout = deadline - time.monotonic()
        if remaining_timeout <= 0:
            break
        if poll_interval_seconds > 0:
            sleep(min(poll_interval_seconds, remaining_timeout))
            if time.monotonic() >= deadline:
                break
        current_run = client.read_flow_run(current_run.id)
        next_log_offset = stream_new_logs(client, current_run, next_log_offset, log_sink)
    stream_new_logs(client, current_run, next_log_offset, log_sink)
    return current_run


def stream_new_logs(
    client: FlowRunObserverClientProtocol,
    flow_run: FlowRun,
    next_log_offset: int,
    log_sink: LogSink | None,
) -> int:
    if log_sink is None:
        return next_log_offset

    new_logs, updated_log_offset = read_logs_since_offset(
        client=client,
        flow_run_id=flow_run.id,
        offset=next_log_offset,
    )
    for log in new_logs:
        log_sink(format_log_line(flow_run, log))
    return updated_log_offset


def read_logs_since_offset(
    client: FlowRunObserverClientProtocol,
    flow_run_id: str,
    offset: int,
) -> tuple[tuple[PrefectLog, ...], int]:
    next_log_offset = offset
    new_logs: list[PrefectLog] = []
    while True:
        page = client.read_logs_page(flow_run_id, limit=PREFECT_LOG_PAGE_SIZE, offset=next_log_offset)
        new_logs.extend(page)
        next_log_offset += len(page)
        if len(page) < PREFECT_LOG_PAGE_SIZE:
            return tuple(new_logs), next_log_offset


def read_all_logs(client: FlowRunObserverClientProtocol, flow_run_id: str) -> tuple[PrefectLog, ...]:
    offset = 0
    all_logs: list[PrefectLog] = []
    while True:
        page = client.read_logs_page(flow_run_id, limit=PREFECT_LOG_PAGE_SIZE, offset=offset)
        all_logs.extend(page)
        if len(page) < PREFECT_LOG_PAGE_SIZE:
            return tuple(all_logs)
        offset += PREFECT_LOG_PAGE_SIZE


def format_log_line(flow_run: FlowRun, log: PrefectLog) -> str:
    return f"{log.timestamp} | {logging.getLevelName(log.level):7s} | Flow run {flow_run.name!r} - {log.message}"


def format_logs(flow_run: FlowRun, logs: tuple[PrefectLog, ...]) -> str:
    lines = [format_log_line(flow_run, log) for log in logs]
    return "\n".join(lines) + ("\n" if lines else "")


def print_live_log_line(line: str) -> None:
    print(line, flush=True)


def write_artifacts(
    output_dir: Path | str,
    *,
    summary_payload: dict[str, Any],
    summary_markdown: str,
    flow_run: FlowRun,
    logs: tuple[PrefectLog, ...],
) -> None:
    artifact_dir = Path(output_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    (artifact_dir / "summary.json").write_text(json.dumps(summary_payload, indent=2, sort_keys=True))
    (artifact_dir / "summary.md").write_text(summary_markdown)
    (artifact_dir / "logs.txt").write_text(format_logs(flow_run, logs))


def ui_base_url_from_api_url(api_url: str) -> str:
    normalized_url = api_url.rstrip("/")
    if normalized_url.endswith("/api"):
        return normalized_url[: -len("/api")]
    return normalized_url


def _read_optional_str(payload: dict[str, Any], key: str) -> str | None:
    raw_value = payload.get(key)
    return raw_value if isinstance(raw_value, str) else None
