#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import save_file

BLOCK_SIZE = 32
LOGIT_PROBE_TOPK = 128
LOGIT_PROBE_HIDDEN_DIM = 1024
LOGIT_PROBE_STEPS = 500


@dataclass(frozen=True)
class LayerFiles:
    rows: int
    cols: int
    pre_mlp_norm_file: str
    mlp_file: str


@dataclass(frozen=True)
class DenseLogitsFiles:
    rows: int
    cols: int
    logits_file: str


@dataclass(frozen=True)
class SparseLogitsFiles:
    rows: int
    cols: int
    topk: int
    indices_file: str
    values_file: str


@dataclass
class LogitProbeData:
    probe: torch.nn.Module
    train_targets: torch.Tensor
    valid_targets: torch.Tensor
    vocab_size: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fit a narrower gated MLP student on dumped prefill traces.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, action="append", type=Path)
    parser.add_argument("--input-trace-dir", action="append", type=Path)
    parser.add_argument("--selection-trace-dir", action="append", type=Path)
    parser.add_argument("--layer", required=True, type=int)
    parser.add_argument("--keep-ratio", type=float)
    parser.add_argument("--selected-blocks")
    parser.add_argument("--steps", default=200, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--weight-decay", default=0.0, type=float)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--selector", default="hidden", choices=["hidden", "output_weighted", "output_geom"])
    parser.add_argument(
        "--loss",
        default="mse",
        choices=["mse", "relative_mse_cosine", "mse_logit_probe", "logit_kl"],
    )
    parser.add_argument("--logit-probe-weight", default=1.0, type=float)
    parser.add_argument("--output-model-dir", type=Path)
    return parser.parse_args()


def select_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_manifest(trace_dir: Path) -> dict[int, LayerFiles]:
    payload = json.loads((trace_dir / "manifest.json").read_text())
    return {
        int(layer["layer_index"]): LayerFiles(
            rows=int(layer["rows"]),
            cols=int(layer["cols"]),
            pre_mlp_norm_file=layer["pre_mlp_norm_file"],
            mlp_file=layer["mlp_file"],
        )
        for layer in payload["layers"]
    }


def load_logits_manifest(trace_dir: Path) -> DenseLogitsFiles | SparseLogitsFiles | None:
    payload = json.loads((trace_dir / "manifest.json").read_text())
    if "logits_indices_file" in payload:
        return SparseLogitsFiles(
            rows=int(payload["logits_rows"]),
            cols=int(payload["logits_cols"]),
            topk=int(payload["logits_topk"]),
            indices_file=payload["logits_indices_file"],
            values_file=payload["logits_values_file"],
        )
    if "logits_file" not in payload:
        return None
    return DenseLogitsFiles(
        rows=int(payload["logits_rows"]),
        cols=int(payload["logits_cols"]),
        logits_file=payload["logits_file"],
    )


def load_trace_matrix(
    path: Path,
    rows: int,
    cols: int,
) -> torch.Tensor:
    values = torch.from_file(str(path), shared=False, size=rows * cols, dtype=torch.float32)
    return values.view(rows, cols).clone()


def load_i32_matrix(
    path: Path,
    rows: int,
    cols: int,
) -> torch.Tensor:
    values = torch.from_file(str(path), shared=False, size=rows * cols, dtype=torch.int32)
    return values.view(rows, cols).clone()


def load_layer_traces(
    trace_dirs: list[Path],
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for trace_dir in trace_dirs:
        layer_files = load_manifest(trace_dir)[layer]
        xs.append(load_trace_matrix(trace_dir / layer_files.pre_mlp_norm_file, layer_files.rows, layer_files.cols))
        ys.append(load_trace_matrix(trace_dir / layer_files.mlp_file, layer_files.rows, layer_files.cols))
    return torch.cat(xs, dim=0), torch.cat(ys, dim=0)


def load_logit_targets(trace_dirs: list[Path]) -> torch.Tensor:
    manifests = [load_logits_manifest(trace_dir) for trace_dir in trace_dirs]
    assert all(manifest is not None for manifest in manifests), "Missing logits trace"
    manifests = [manifest for manifest in manifests if manifest is not None]
    if all(isinstance(manifest, DenseLogitsFiles) for manifest in manifests):
        assert len(manifests) == len(trace_dirs), "Trace and logits manifests must stay aligned"
        logits: list[torch.Tensor] = []
        for trace_dir, logits_files in zip(trace_dirs, manifests):
            assert isinstance(logits_files, DenseLogitsFiles)
            logits.append(load_trace_matrix(trace_dir / logits_files.logits_file, logits_files.rows, logits_files.cols))
        full_logits = torch.cat(logits, dim=0)
        topk = min(LOGIT_PROBE_TOPK, full_logits.shape[1])
        vocab_indices = torch.topk(full_logits, k=topk, dim=1).indices.reshape(-1).unique(sorted=True)
        return full_logits.index_select(1, vocab_indices)

    assert all(isinstance(manifest, SparseLogitsFiles) for manifest in manifests), "Mixed logits formats are unsupported"
    sparse_entries: list[tuple[torch.Tensor, torch.Tensor, int, int]] = []
    vocab_indices: list[torch.Tensor] = []
    for trace_dir in trace_dirs:
        logits_files = load_logits_manifest(trace_dir)
        assert isinstance(logits_files, SparseLogitsFiles)
        indices = load_i32_matrix(trace_dir / logits_files.indices_file, logits_files.rows, logits_files.topk).to(torch.long)
        values = load_trace_matrix(trace_dir / logits_files.values_file, logits_files.rows, logits_files.topk)
        sparse_entries.append((indices, values, logits_files.rows, logits_files.cols))
        vocab_indices.append(indices.reshape(-1))
    union = torch.cat(vocab_indices).unique(sorted=True)
    union_lookup = {int(value): index for index, value in enumerate(union.tolist())}
    targets: list[torch.Tensor] = []
    for indices, values, rows, _ in sparse_entries:
        remapped = torch.tensor([union_lookup[int(value)] for value in indices.reshape(-1)], dtype=torch.long).view(rows, -1)
        target = torch.zeros(rows, union.numel(), dtype=torch.float32)
        target.scatter_(1, remapped, values)
        targets.append(target)
    return torch.cat(targets, dim=0)


def load_teacher_weights(
    model_dir: Path,
    layer: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with safe_open(model_dir / "model.safetensors", framework="pt") as tensors:
        fused = tensors.get_tensor(f"transformer.layers.{layer}.mlp.up_projection.weights").float()
        down = tensors.get_tensor(f"transformer.layers.{layer}.mlp.down_projection.weights").float()
    hidden_dim = down.shape[1]
    up = fused[:hidden_dim]
    gate = fused[hidden_dim:]
    return up, gate, down


def load_all_tensors(model_dir: Path) -> dict[str, torch.Tensor]:
    with safe_open(model_dir / "model.safetensors", framework="pt") as tensors:
        return {key: tensors.get_tensor(key) for key in tensors.keys()}


def split_train_validation(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    split = max(1, int(values.shape[0] * 0.8))
    if split >= values.shape[0]:
        split = values.shape[0] - 1
    return values[:split], values[split:]


def teacher_hidden(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
) -> torch.Tensor:
    return F.silu(x @ gate.t()) * (x @ up.t())


def select_hidden_indices(
    x: torch.Tensor,
    up: torch.Tensor,
    gate: torch.Tensor,
    down: torch.Tensor,
    keep_blocks: int,
    selector: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    hidden = teacher_hidden(x, up, gate)
    hidden_scores = hidden.square().mean(dim=0)
    down_scores = down.square().sum(dim=0)
    if selector == "hidden":
        feature_scores = hidden_scores
    elif selector == "output_weighted":
        feature_scores = hidden_scores * down_scores
    elif selector == "output_geom":
        feature_scores = torch.sqrt(hidden_scores * down_scores)
    else:
        raise AssertionError(f"Unknown selector: {selector}")
    block_scores = feature_scores.view(-1, BLOCK_SIZE).sum(dim=1)
    blocks = torch.topk(block_scores, keep_blocks, largest=True, sorted=False).indices.sort().values
    offsets = torch.arange(BLOCK_SIZE, dtype=torch.long)
    indices = (blocks[:, None] * BLOCK_SIZE + offsets[None, :]).reshape(-1)
    return blocks, indices


class StudentMlp(torch.nn.Module):
    def __init__(
        self,
        up: torch.Tensor,
        gate: torch.Tensor,
        down: torch.Tensor,
    ) -> None:
        super().__init__()
        self.up = torch.nn.Parameter(up.clone())
        self.gate = torch.nn.Parameter(gate.clone())
        self.down = torch.nn.Parameter(down.clone())

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.silu(x @ self.gate.t()) * (x @ self.up.t())
        return hidden @ self.down.t()


def metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, float]:
    error = prediction - target
    mse = error.square().mean().item()
    relative_mse = mse / max(target.square().mean().item(), 1e-12)
    cosine = F.cosine_similarity(prediction, target, dim=1).mean().item()
    return {
        "mse": mse,
        "relative_mse": relative_mse,
        "mean_cosine": cosine,
    }


def relative_mse_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(prediction, target) / target.square().mean().clamp_min(1e-12)


def build_logit_probe(
    train_output: torch.Tensor,
    train_targets: torch.Tensor,
    valid_targets: torch.Tensor,
    device: torch.device,
) -> LogitProbeData:
    probe = torch.nn.Sequential(
        torch.nn.Linear(train_output.shape[1], LOGIT_PROBE_HIDDEN_DIM),
        torch.nn.SiLU(),
        torch.nn.Linear(LOGIT_PROBE_HIDDEN_DIM, train_targets.shape[1]),
    ).to(device)
    optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=0.0)
    train_output = train_output.to(device)
    train_targets = train_targets.to(device)
    for _ in range(LOGIT_PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = relative_mse_loss(probe(train_output), train_targets)
        loss.backward()
        optimizer.step()
    for parameter in probe.parameters():
        parameter.requires_grad_(False)
    return LogitProbeData(
        probe=probe,
        train_targets=train_targets,
        valid_targets=valid_targets.to(device),
        vocab_size=int(train_targets.shape[1]),
    )


def training_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    loss_name: str,
    logit_probe: LogitProbeData | None,
    logit_probe_weight: float,
) -> torch.Tensor:
    if loss_name == "mse":
        return F.mse_loss(prediction, target)
    if loss_name == "relative_mse_cosine":
        error = (prediction - target).square().mean(dim=1)
        scale = target.square().mean(dim=1).clamp_min(1e-12)
        relative_mse = (error / scale).mean()
        cosine = 1.0 - F.cosine_similarity(prediction, target, dim=1).mean()
        return relative_mse + cosine
    if loss_name == "mse_logit_probe":
        assert logit_probe is not None
        hidden_loss = relative_mse_loss(prediction, target)
        logit_loss = relative_mse_loss(logit_probe.probe(prediction), logit_probe.train_targets)
        return hidden_loss + logit_probe_weight * logit_loss
    if loss_name == "logit_kl":
        raise AssertionError("logit_kl requires dedicated loss handling")
    raise AssertionError(f"Unknown loss: {loss_name}")


def logit_kl_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    teacher_logits: torch.Tensor,
    logit_probe: LogitProbeData,
    logit_probe_weight: float,
) -> torch.Tensor:
    hidden_loss = relative_mse_loss(prediction, target)
    student_logits = logit_probe.probe(prediction)
    target_probs = torch.softmax(teacher_logits.to(student_logits.device), dim=1)
    student_log_probs = torch.log_softmax(student_logits, dim=1)
    kl = F.kl_div(student_log_probs, target_probs, reduction="batchmean")
    return hidden_loss + logit_probe_weight * kl


def write_model_copy(
    model_dir: Path,
    output_dir: Path,
    layer: int,
    student: StudentMlp,
    selected_blocks: torch.Tensor,
) -> None:
    source_model_dir = output_dir if (output_dir / "model.safetensors").exists() else model_dir
    tensors = load_all_tensors(source_model_dir)
    up_key = f"transformer.layers.{layer}.mlp.up_projection.weights"
    down_key = f"transformer.layers.{layer}.mlp.down_projection.weights"
    offsets = torch.arange(BLOCK_SIZE, dtype=torch.long)
    selected_rows = (selected_blocks[:, None] * BLOCK_SIZE + offsets[None, :]).reshape(-1)

    fused = tensors[up_key].clone()
    hidden_dim = fused.shape[0] // 2
    fused[selected_rows] = student.up.detach().cpu().to(fused.dtype)
    fused[hidden_dim + selected_rows] = student.gate.detach().cpu().to(fused.dtype)
    tensors[up_key] = fused

    down = tensors[down_key].clone()
    down[:, selected_rows] = student.down.detach().cpu().to(down.dtype)
    tensors[down_key] = down

    output_dir.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(output_dir / "model.safetensors"))
    blocks_path = output_dir / "selected_blocks.json"
    blocks_by_layer = {}
    if blocks_path.exists():
        blocks_by_layer = json.loads(blocks_path.read_text())
    blocks_by_layer[str(layer)] = selected_blocks.tolist()
    blocks_path.write_text(json.dumps(blocks_by_layer, sort_keys=True))
    for source in model_dir.iterdir():
        if source.name in {"model.safetensors", "selected_blocks.json"}:
            continue
        target = output_dir / source.name
        if target.exists() or target.is_symlink():
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target)
            else:
                target.unlink()
        os.symlink(source.resolve(), target)


def main() -> None:
    args = parse_args()
    assert (args.keep_ratio is None) != (args.selected_blocks is None)
    device = select_device(args.device)
    input_trace_dirs = args.trace_dir if args.input_trace_dir is None else args.input_trace_dir
    x, _ = load_layer_traces(input_trace_dirs, args.layer)
    _, y = load_layer_traces(args.trace_dir, args.layer)
    selection_trace_dirs = input_trace_dirs if args.selection_trace_dir is None else args.selection_trace_dir
    selection_x, _ = load_layer_traces(selection_trace_dirs, args.layer)
    up, gate, down = load_teacher_weights(args.model, args.layer)

    if args.selected_blocks is None:
        keep_blocks = max(1, int((down.shape[1] // BLOCK_SIZE) * args.keep_ratio))
        selected_blocks, indices = select_hidden_indices(selection_x, up, gate, down, keep_blocks, args.selector)
    else:
        selected_blocks = torch.tensor(
            [int(value) for value in args.selected_blocks.split(",") if value],
            dtype=torch.long,
        )
        offsets = torch.arange(BLOCK_SIZE, dtype=torch.long)
        indices = (selected_blocks[:, None] * BLOCK_SIZE + offsets[None, :]).reshape(-1)
    base_up = up.index_select(0, indices)
    base_gate = gate.index_select(0, indices)
    base_down = down.index_select(1, indices)

    train_x, valid_x = split_train_validation(x)
    train_y, valid_y = split_train_validation(y)
    logit_probe = None
    if args.loss in {"mse_logit_probe", "logit_kl"}:
        logit_targets = load_logit_targets(args.trace_dir)
        train_targets, valid_targets = split_train_validation(logit_targets)
        logit_probe = build_logit_probe(train_y, train_targets, valid_targets, device)

    baseline_train = (teacher_hidden(train_x, base_up, base_gate) @ base_down.t()).float()
    baseline_valid = (teacher_hidden(valid_x, base_up, base_gate) @ base_down.t()).float()

    student = StudentMlp(base_up.to(device), base_gate.to(device), base_down.to(device))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    valid_x = valid_x.to(device)
    valid_y = valid_y.to(device)

    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = student(train_x)
        if args.loss == "logit_kl":
            assert logit_probe is not None
            loss = logit_kl_loss(
                prediction,
                train_y,
                logit_probe.train_targets,
                logit_probe,
                args.logit_probe_weight,
            )
        else:
            loss = training_loss(prediction, train_y, args.loss, logit_probe, args.logit_probe_weight)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fitted_train = student(train_x)
        fitted_valid = student(valid_x)

    result = {
        "layer": args.layer,
        "keep_ratio": selected_blocks.numel() / (down.shape[1] // BLOCK_SIZE),
        "steps": args.steps,
        "lr": args.lr,
        "hidden_dim": int(down.shape[1]),
        "student_dim": int(indices.numel()),
        "student_blocks": int(selected_blocks.numel()),
        "selected_blocks": selected_blocks.tolist(),
        "selector": args.selector,
        "loss": args.loss,
        "logit_probe_weight": args.logit_probe_weight,
        "input_trace_dirs": [str(path) for path in input_trace_dirs],
        "target_trace_dirs": [str(path) for path in args.trace_dir],
        "selection_trace_dirs": [str(path) for path in selection_trace_dirs],
        "device": str(device),
        "baseline_train": metrics(baseline_train, train_y.cpu()),
        "baseline_valid": metrics(baseline_valid, valid_y.cpu()),
        "fitted_train": metrics(fitted_train.cpu(), train_y.cpu()),
        "fitted_valid": metrics(fitted_valid.cpu(), valid_y.cpu()),
    }
    if logit_probe is not None:
        with torch.no_grad():
            result["fitted_train_logits"] = metrics(logit_probe.probe(fitted_train).cpu(), logit_probe.train_targets.cpu())
            result["fitted_valid_logits"] = metrics(logit_probe.probe(fitted_valid).cpu(), logit_probe.valid_targets.cpu())
            result["logit_probe_vocab_size"] = logit_probe.vocab_size
    if args.output_model_dir is not None:
        write_model_copy(args.model, args.output_model_dir, args.layer, student, selected_blocks)
        result["output_model_dir"] = str(args.output_model_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
