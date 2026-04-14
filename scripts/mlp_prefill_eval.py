#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    identifier: str
    system: str
    user_prefix: str


CASES: tuple[Case, ...] = (
    Case("summary", "Summarize the user's input.", ""),
    Case("bullets", "Summarize the user's input into exactly 6 bullet points.", ""),
    Case("claims", "Extract the 8 most important claims from the user's input as a numbered list.", ""),
    Case("title_abstract", "Write a title and then a 3-sentence abstract for the user's input.", ""),
    Case("short_summary", "Summarize the user's input in at most 120 words.", ""),
)

MLP_ENV_KEYS = [
    "UZU_MLP_BLOCKS_BY_LAYER_JSON",
    "UZU_MLP_STATIC_BLOCKS",
    "UZU_MLP_STATIC_KEEP_RATIO",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline vs reduced-width MLP prefill on perf, KL, and generation.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--variant-model", type=Path)
    parser.add_argument("--variant-blocks-json", type=Path)
    parser.add_argument("--probe-bin", type=Path)
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--runs", default=1, type=int)
    parser.add_argument("--tokens-limit", default=32, type=int)
    parser.add_argument("--variant-env", action="append", default=[])
    return parser.parse_args()


def load_article(model_path: Path) -> tuple[str, str]:
    task = json.loads((model_path / "benchmark_task.json").read_text())
    user_message = next(message["content"] for message in task["messages"] if message["role"] == "user")
    return task["repo_id"], user_message


def build_task(repo_id: str, article: str, case: Case, tokens_limit: int, runs: int) -> dict:
    return {
        "greedy": True,
        "identifier": case.identifier,
        "messages": [
            {"role": "system", "content": case.system},
            {"role": "user", "content": f"{case.user_prefix}{article}"},
        ],
        "number_of_runs": runs,
        "repo_id": repo_id,
        "tokens_limit": tokens_limit,
    }


def first_diff_index(lhs: str, rhs: str) -> int | None:
    for index, (left_char, right_char) in enumerate(zip(lhs, rhs)):
        if left_char != right_char:
            return index
    if len(lhs) != len(rhs):
        return min(len(lhs), len(rhs))
    return None


def softmax(logits: list[float]) -> list[float]:
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def kl_divergence(reference_logits: list[float], variant_logits: list[float]) -> float:
    reference = softmax(reference_logits)
    variant = softmax(variant_logits)
    return sum(
        p * (math.log(p) - math.log(max(q, 1e-30)))
        for p, q in zip(reference, variant)
        if p > 0.0
    )


def build_probe(env: dict[str, str], probe_bin: Path | None, cwd: Path) -> Path:
    if probe_bin is not None:
        return probe_bin
    subprocess.run(
        ["cargo", "build", "-p", "benchmarks", "--bin", "prefill_probe"],
        check=True,
        cwd=cwd,
        env=env,
        stdout=subprocess.DEVNULL,
    )
    return cwd / "target" / "debug" / "prefill_probe"


def run_probe(
    probe_bin: Path,
    model_path: Path,
    task: dict,
    prefill_step_size: int | None,
    env: dict[str, str],
    tmpdir: Path,
) -> dict:
    task_path = tmpdir / f"{task['identifier']}.json"
    out_path = tmpdir / f"{task['identifier']}.result.json"
    task_path.write_text(json.dumps(task))
    command = [str(probe_bin), str(model_path), str(task_path), str(out_path)]
    if prefill_step_size is not None:
        command.extend(["--prefill-step-size", str(prefill_step_size)])
    subprocess.run(command, check=True, env=env, stdout=subprocess.DEVNULL)
    return json.loads(out_path.read_text())


def baseline_env(base_env: dict[str, str]) -> dict[str, str]:
    env = dict(base_env)
    for key in MLP_ENV_KEYS:
        env.pop(key, None)
    return env


def variant_env(base_env: dict[str, str], overrides: list[str]) -> dict[str, str]:
    env = dict(base_env)
    for entry in overrides:
        key, value = entry.split("=", 1)
        env[key] = value
    return env


def load_variant_blocks(path: Path | None) -> str | None:
    if path is None:
        return None
    payload = json.loads(path.read_text())
    return json.dumps(payload, separators=(",", ":"))


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    repo_id, article = load_article(args.model)

    base_env = dict(os.environ)
    probe = build_probe(base_env, args.probe_bin, repo_root)
    baseline_run_env = baseline_env(base_env)
    variant_run_env = variant_env(base_env, args.variant_env)
    variant_model = args.variant_model or args.model
    variant_blocks = load_variant_blocks(args.variant_blocks_json)

    print(
        "case\tkl_prefill\targmax_same\texact\tfirst_diff\t"
        "mem_base\tmem_var\tttft_base\tttft_var\t"
        "prompt_tps_base\tprompt_tps_var\tgen_tps_base\tgen_tps_var",
        flush=True,
    )
    with tempfile.TemporaryDirectory(prefix="mlp_prefill_eval_") as tmp:
        tmpdir = Path(tmp)
        for case in CASES:
            task = build_task(repo_id, article, case, args.tokens_limit, args.runs)
            baseline = run_probe(probe, args.model, task, args.prefill_step_size, baseline_run_env, tmpdir)
            case_variant_env = dict(variant_run_env)
            if variant_blocks is not None:
                case_variant_env["UZU_MLP_BLOCKS_BY_LAYER_JSON"] = variant_blocks
            variant = run_probe(probe, variant_model, task, args.prefill_step_size, case_variant_env, tmpdir)
            diff = first_diff_index(baseline["text"], variant["text"])
            print(
                f"{case.identifier}\t"
                f"{kl_divergence(baseline['prefill_logits'], variant['prefill_logits']):.6f}\t"
                f"{int(baseline['prefill_top_token'] == variant['prefill_top_token'])}\t"
                f"{int(baseline['text'] == variant['text'])}\t"
                f"{'' if diff is None else diff}\t"
                f"{0 if baseline['memory_used'] is None else baseline['memory_used']}\t"
                f"{0 if variant['memory_used'] is None else variant['memory_used']}\t"
                f"{baseline['time_to_first_token']:.4f}\t{variant['time_to_first_token']:.4f}\t"
                f"{baseline['prompt_tokens_per_second']:.4f}\t{variant['prompt_tokens_per_second']:.4f}\t"
                f"{0.0 if baseline['generate_tokens_per_second'] is None else baseline['generate_tokens_per_second']:.4f}\t"
                f"{0.0 if variant['generate_tokens_per_second'] is None else variant['generate_tokens_per_second']:.4f}"
            )


if __name__ == "__main__":
    main()
