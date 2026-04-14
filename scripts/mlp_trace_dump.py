#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import shutil
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump prefill MLP traces (with logits) for benchmark prompts.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--tokens-limit", default=1024, type=int)
    parser.add_argument("--prefill-step-size", type=int)
    parser.add_argument("--backend", default=os.environ.get("UZU_BACKEND", "metal"))
    parser.add_argument("--cases", nargs="*", choices=[case.identifier for case in CASES])
    parser.add_argument("--release", action="store_true")
    parser.add_argument("--cpu-only", action="store_true", help="Run benchmarks without metal backend features.")
    return parser.parse_args()


def load_article(model_dir: Path) -> tuple[str, str]:
    payload = json.loads((model_dir / "benchmark_task.json").read_text())
    article = next(message["content"] for message in payload["messages"] if message["role"] == "user")
    return payload["repo_id"], article


def build_task(repo_id: str, article: str, case: Case, tokens_limit: int) -> dict:
    return {
        "greedy": True,
        "identifier": case.identifier,
        "messages": [
            {"role": "system", "content": case.system},
            {"role": "user", "content": f"{case.user_prefix}{article}"},
        ],
        "number_of_runs": 1,
        "repo_id": repo_id,
        "tokens_limit": tokens_limit,
    }


def run_case(
    repo_root: Path,
    model_dir: Path,
    case_dir: Path,
    task: dict,
    backend: str,
    prefill_step_size: int | None,
    release: bool,
    cpu_only: bool,
) -> None:
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"mlp_trace_dump_{task['identifier']}_") as tmp:
        task_path = Path(tmp) / "task.json"
        task_path.write_text(json.dumps(task))
        command = ["cargo", "run"]
        if release:
            command.append("--release")
        if cpu_only:
            command.append("--no-default-features")
        command.extend([
            "-p",
            "benchmarks",
            "--bin",
            "dump_prefill_mlp_trace",
            "--features",
            "tracing",
        ])
        command.extend([
            "--",
            str(model_dir),
            str(task_path),
            str(case_dir),
        ])
        if prefill_step_size is not None:
            command.extend(["--prefill-step-size", str(prefill_step_size)])
        env = dict(os.environ)
        env["UZU_BACKEND"] = backend
        subprocess.run(command, check=True, cwd=repo_root, env=env)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    repo_id, article = load_article(args.model)
    selected_cases = [case for case in CASES if args.cases is None or case.identifier in args.cases]
    for case in selected_cases:
        task = build_task(repo_id, article, case, args.tokens_limit)
        case_dir = args.output_root / case.identifier
        run_case(
            repo_root=repo_root,
            model_dir=args.model,
            case_dir=case_dir,
            task=task,
            backend=args.backend,
            prefill_step_size=args.prefill_step_size,
            release=args.release,
            cpu_only=args.cpu_only,
        )


if __name__ == "__main__":
    main()
