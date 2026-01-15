from __future__ import annotations

import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

from .models import (
    CounterType,
    GpuCounterDefinition,
    GpuPerformanceStateInterval,
    GpuState,
    GpuUtilization,
    KernelDispatch,
    KernelStats,
    PerformanceState,
    ShaderInfo,
    ShaderType,
    TraceMetadata,
)


def _check_output(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True)


def _xctrace_cmd(*args: str) -> list[str]:
    return ["xcrun", "xctrace", *args]


def _build_id_maps(
    root: ET.Element,
) -> tuple[dict[str, ET.Element], dict[str, str]]:
    id_to_element: dict[str, ET.Element] = {}
    id_to_text: dict[str, str] = {}
    for el in root.iter():
        el_id = el.attrib.get("id")
        if el_id is None:
            continue
        id_to_element[el_id] = el
        if el.text is not None:
            id_to_text[el_id] = el.text
    return id_to_element, id_to_text


def _deref(
    el: ET.Element | None, id_to_element: dict[str, ET.Element]
) -> ET.Element | None:
    if el is None:
        return None
    ref = el.attrib.get("ref")
    return id_to_element.get(ref) if ref else el


def _resolve_text(el: ET.Element | None, id_to_text: dict[str, str]) -> str | None:
    if el is None:
        return None
    if el.text is not None:
        return el.text
    ref = el.attrib.get("ref")
    return id_to_text.get(ref) if ref else None


def _toc_root(trace: Path) -> ET.Element:
    xml = _check_output(_xctrace_cmd("export", "--input", str(trace), "--toc"))
    return ET.fromstring(xml)


def _export_table(trace: Path, run_number: int, schema: str) -> ET.Element:
    xpath = f'/trace-toc/run[@number="{run_number}"]/data/table[@schema="{schema}"]'
    xml = _check_output(_xctrace_cmd("export", "--input", str(trace), "--xpath", xpath))
    return ET.fromstring(xml)


def _union_ns(intervals: list[tuple[int, int]]) -> int:
    if not intervals:
        return 0
    sorted_intervals = sorted(intervals)
    total = 0
    cur_s, cur_e = sorted_intervals[0]
    for s, e in sorted_intervals[1:]:
        if s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    total += cur_e - cur_s
    return total


def extract_metadata(trace: Path, run_number: int) -> TraceMetadata:
    toc = _toc_root(trace)
    run = toc.find(f".//run[@number='{run_number}']")
    if run is None:
        raise ValueError(f"Run {run_number} not found in trace TOC")

    device = run.find("./info/target/device")
    device_name = device.attrib.get("name", "") if device is not None else ""
    device_model = device.attrib.get("model", "") if device is not None else ""
    os_version = device.attrib.get("os-version", "") if device is not None else ""

    process = run.find("./info/target/process")
    target_process = process.attrib.get("name", "") if process is not None else ""
    target_pid_str = process.attrib.get("pid", "0") if process is not None else "0"
    target_pid = int(target_pid_str) if target_pid_str else 0

    summary = run.find("./info/summary")
    duration_el = summary.find("./duration") if summary is not None else None
    duration_ns = (
        int(float(duration_el.text) * 1e9)
        if duration_el is not None and duration_el.text
        else 0
    )

    start_date_el = summary.find("./start-date") if summary is not None else None
    start_date = start_date_el.text if start_date_el is not None else ""

    end_date_el = summary.find("./end-date") if summary is not None else None
    end_date = end_date_el.text if end_date_el is not None else ""

    template_el = summary.find("./template-name") if summary is not None else None
    template_name = template_el.text if template_el is not None else ""

    return TraceMetadata(
        device_name=device_name or "",
        device_model=device_model or "",
        os_version=os_version or "",
        target_process=target_process or "",
        target_pid=target_pid,
        duration_ns=duration_ns,
        start_date=start_date or "",
        end_date=end_date or "",
        template_name=template_name or "",
    )


def extract_gpu_utilization(trace: Path, run_number: int) -> GpuUtilization:
    toc = _toc_root(trace)
    run = toc.find(f".//run[@number='{run_number}']")
    if run is None:
        raise ValueError(f"Run {run_number} not found")

    duration_el = run.find("./info/summary/duration")
    run_duration_ns = (
        int(float(duration_el.text) * 1e9)
        if duration_el is not None and duration_el.text
        else 0
    )

    pid_el = run.find("./info/target/process")
    pid = int(pid_el.attrib.get("pid", "0")) if pid_el is not None else 0

    root = _export_table(trace, run_number, "metal-gpu-state-intervals")
    _, id_to_text = _build_id_maps(root)

    active_ns = 0
    idle_ns = 0
    for row in root.findall(".//row"):
        state = _resolve_text(row.find("./gpu-state"), id_to_text)
        dur_s = _resolve_text(row.find("./duration"), id_to_text)
        if state is None or dur_s is None:
            continue
        dur = int(dur_s)
        if state == GpuState.ACTIVE.value:
            active_ns += dur
        elif state == GpuState.IDLE.value:
            idle_ns += dur

    target_active_ns = 0
    window_start_ns = 0
    window_end_ns = 0
    command_buffer_count = 0
    encoder_count = 0

    try:
        encoders = _export_table(trace, run_number, "metal-application-encoders-list")
        gpu_intervals = _export_table(trace, run_number, "metal-gpu-intervals")

        enc_id_to_el, enc_id_to_text = _build_id_maps(encoders)
        command_buffers: set[int] = set()
        for row in encoders.findall(".//row"):
            proc = _deref(row.find("./process"), enc_id_to_el)
            proc_pid_s = _resolve_text(
                proc.find("./pid") if proc is not None else None, enc_id_to_text
            )
            if proc_pid_s is None or int(proc_pid_s) != pid:
                continue
            ids = row.findall("./metal-command-buffer-id")
            if not ids:
                continue
            cb_s = _resolve_text(ids[0], enc_id_to_text)
            if cb_s:
                command_buffers.add(int(cb_s))
            encoder_count += 1

        command_buffer_count = len(command_buffers)

        gpu_id_to_el, gpu_id_to_text = _build_id_maps(gpu_intervals)
        intervals: list[tuple[int, int]] = []
        for row in gpu_intervals.findall(".//row"):
            ids = row.findall("./metal-command-buffer-id")
            if not ids:
                continue
            cb_s = _resolve_text(ids[0], gpu_id_to_text)
            if cb_s is None:
                continue
            cb = int(cb_s)
            if cb == 0 or cb not in command_buffers:
                continue
            start_s = _resolve_text(row.find("./start-time"), gpu_id_to_text)
            dur_s = _resolve_text(row.find("./duration"), gpu_id_to_text)
            if start_s is None or dur_s is None:
                continue
            start_ns = int(start_s)
            dur_ns = int(dur_s)
            if dur_ns <= 0:
                continue
            intervals.append((start_ns, start_ns + dur_ns))

        if intervals:
            window_start_ns = min(start for start, _ in intervals)
            window_end_ns = max(end for _, end in intervals)
            target_active_ns = _union_ns(intervals)
    except subprocess.CalledProcessError:
        pass

    return GpuUtilization(
        run_duration_ns=run_duration_ns,
        global_active_ns=active_ns,
        global_idle_ns=idle_ns,
        global_coverage_ns=active_ns + idle_ns,
        target_active_ns=target_active_ns,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        command_buffer_count=command_buffer_count,
        encoder_count=encoder_count,
    )


def extract_performance_states(
    trace: Path, run_number: int
) -> tuple[GpuPerformanceStateInterval, ...]:
    try:
        root = _export_table(trace, run_number, "gpu-performance-state-intervals")
    except subprocess.CalledProcessError:
        return ()

    _, id_to_text = _build_id_maps(root)
    states: list[GpuPerformanceStateInterval] = []

    for row in root.findall(".//row"):
        start_s = _resolve_text(row.find("./start-time"), id_to_text)
        dur_s = _resolve_text(row.find("./duration"), id_to_text)
        state_s = _resolve_text(row.find("./gpu-performance-state"), id_to_text)
        gpu_s = _resolve_text(row.find("./track-label"), id_to_text)
        induced_s = _resolve_text(row.find("./is-induced"), id_to_text)

        if start_s is None or dur_s is None or state_s is None:
            continue

        try:
            state = PerformanceState(state_s)
        except ValueError:
            state = PerformanceState.UNKNOWN

        states.append(
            GpuPerformanceStateInterval(
                start_ns=int(start_s),
                duration_ns=int(dur_s),
                state=state,
                is_induced=induced_s == "1",
                gpu_name=gpu_s or "",
            )
        )

    return tuple(states)


def extract_counter_definitions(
    trace: Path, run_number: int
) -> tuple[GpuCounterDefinition, ...]:
    try:
        root = _export_table(trace, run_number, "gpu-counter-info")
    except subprocess.CalledProcessError:
        return ()

    _, id_to_text = _build_id_maps(root)
    counters: list[GpuCounterDefinition] = []

    for row in root.findall(".//row"):
        counter_id_s = _resolve_text(row.find("./counter-id"), id_to_text)
        name = _resolve_text(row.find("./name"), id_to_text)
        description = _resolve_text(row.find("./description"), id_to_text)
        max_value_s = _resolve_text(row.find("./max-value"), id_to_text)
        counter_type_s = _resolve_text(row.find("./type"), id_to_text)
        sample_interval_s = _resolve_text(row.find("./sample-interval"), id_to_text)

        if counter_id_s is None or name is None:
            continue

        try:
            counter_type = CounterType(counter_type_s) if counter_type_s else CounterType.VALUE
        except ValueError:
            counter_type = CounterType.VALUE

        counters.append(
            GpuCounterDefinition(
                counter_id=int(counter_id_s),
                name=name,
                description=description or "",
                max_value=int(max_value_s) if max_value_s else 0,
                counter_type=counter_type,
                sample_interval_us=int(sample_interval_s) if sample_interval_s else 0,
            )
        )

    return tuple(counters)


def extract_kernel_dispatches(
    trace: Path, run_number: int
) -> tuple[KernelDispatch, ...]:
    toc = _toc_root(trace)
    pid_el = toc.find(f".//run[@number='{run_number}']/info/target/process")
    pid = int(pid_el.attrib.get("pid", "0")) if pid_el is not None else 0

    try:
        root = _export_table(trace, run_number, "metal-shader-profiler-intervals")
    except subprocess.CalledProcessError:
        return ()

    id_to_el, id_to_text = _build_id_maps(root)
    dispatches: list[KernelDispatch] = []

    for row in root.findall(".//row"):
        proc = _deref(row.find("./process"), id_to_el)
        pid_s = _resolve_text(
            proc.find("./pid") if proc is not None else None, id_to_text
        )
        if pid_s is None or int(pid_s) != pid:
            continue

        name = _resolve_text(row.find("./metal-object-label"), id_to_text)
        start_s = _resolve_text(row.find("./start-time"), id_to_text)
        dur_s = _resolve_text(row.find("./duration"), id_to_text)
        channel = _resolve_text(row.find("./gpu-channel-name"), id_to_text)

        cb_ids = row.findall("./metal-command-buffer-id")
        cb_s = _resolve_text(cb_ids[0], id_to_text) if cb_ids else None

        if name is None or start_s is None or dur_s is None:
            continue

        dur_ns = int(dur_s)
        if dur_ns <= 0:
            continue

        dispatches.append(
            KernelDispatch(
                name=name,
                start_ns=int(start_s),
                duration_ns=dur_ns,
                command_buffer_id=int(cb_s) if cb_s else 0,
                channel=channel or "",
            )
        )

    dispatches.sort(key=lambda d: d.start_ns)
    return tuple(dispatches)


def extract_shaders(trace: Path, run_number: int) -> tuple[ShaderInfo, ...]:
    toc = _toc_root(trace)
    pid_el = toc.find(f".//run[@number='{run_number}']/info/target/process")
    pid = int(pid_el.attrib.get("pid", "0")) if pid_el is not None else 0

    try:
        root = _export_table(trace, run_number, "metal-shader-profiler-shader-list")
    except subprocess.CalledProcessError:
        return ()

    id_to_el, id_to_text = _build_id_maps(root)
    shaders: list[ShaderInfo] = []

    for row in root.findall(".//row"):
        proc = _deref(row.find("./process"), id_to_el)
        pid_s = _resolve_text(
            proc.find("./pid") if proc is not None else None, id_to_text
        )
        if pid_s is None or int(pid_s) != pid:
            continue

        shader_id_s = _resolve_text(row.find("./id"), id_to_text)
        name = _resolve_text(row.find("./name"), id_to_text)
        shader_type_s = _resolve_text(row.find("./shader-type"), id_to_text)
        pc_start_s = _resolve_text(row.find("./pc-start"), id_to_text)
        pc_end_s = _resolve_text(row.find("./pc-end"), id_to_text)

        if shader_id_s is None or name is None:
            continue

        try:
            shader_type = ShaderType(shader_type_s) if shader_type_s else ShaderType.UNKNOWN
        except ValueError:
            shader_type = ShaderType.UNKNOWN

        shaders.append(
            ShaderInfo(
                shader_id=int(shader_id_s),
                name=name,
                shader_type=shader_type,
                pc_start=int(pc_start_s) if pc_start_s else 0,
                pc_end=int(pc_end_s) if pc_end_s else 0,
            )
        )

    return tuple(shaders)


def compute_kernel_summary(
    dispatches: tuple[KernelDispatch, ...]
) -> tuple[KernelStats, ...]:
    stats: dict[str, list[int]] = {}
    for d in dispatches:
        if d.name not in stats:
            stats[d.name] = []
        stats[d.name].append(d.duration_ns)

    result: list[KernelStats] = []
    for name, durations in stats.items():
        result.append(
            KernelStats(
                name=name,
                total_ns=sum(durations),
                count=len(durations),
                min_ns=min(durations),
                max_ns=max(durations),
            )
        )

    result.sort(key=lambda s: s.total_ns, reverse=True)
    return tuple(result)
