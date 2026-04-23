"""Aggregate uzu classify outputs (test_data/results/*.out) into a single
results.md with per-example inputs, tokenized outputs paired with predicted
PII labels, and timing stats.

Usage:
  uv run python test_data/build_results_md.py
"""
from __future__ import annotations

import re
from pathlib import Path
from statistics import mean, median

from transformers import AutoTokenizer

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DATASET = HERE / "privacy_filter_dataset.txt"
OUT_MD = HERE / "results.md"

HF_DIR = (
    Path.home()
    / ".cache/huggingface/hub/models--openai--privacy-filter/snapshots"
    / "7ffa9a043d54d1be65afb281eddf0ffbe629385b"
)

STATS_RE = re.compile(
    r"forward:\s*([\d.]+)s\s+post:\s*([\d.]+)s\s+total:\s*([\d.]+)s\s+"
    r"tokens:\s*(\d+)\s+t/s:\s*([\d.]+)"
)
ROW_RE = re.compile(r"^\s*\[\s*(\d+)\]\s+(\S+)\s+([\d.]+)\s*$")


def parse_out(text: str):
    m = STATS_RE.search(text)
    if not m:
        raise ValueError(f"no stats line found in:\n{text}")
    forward, post, total, tokens, tps = m.groups()
    stats = dict(
        forward=float(forward),
        post=float(post),
        total=float(total),
        tokens=int(tokens),
        tps=float(tps),
    )
    rows = []
    for line in text.splitlines():
        mm = ROW_RE.match(line)
        if mm:
            rows.append((int(mm.group(1)), mm.group(2), float(mm.group(3))))
    return stats, rows


def fmt_token(t: str) -> str:
    # HF GPT-OSS tokens often start with 'Ġ' (BPE space marker). Show a
    # human-readable form: leading 'Ġ' -> space, actual ws as escaped.
    out = t.replace("Ġ", "·").replace("Ċ", "\\n").replace("\n", "\\n")
    return out


def main() -> None:
    tok = AutoTokenizer.from_pretrained(str(HF_DIR))
    lines = [
        ln.rstrip("\n") for ln in DATASET.read_text().splitlines() if ln.strip()
    ]
    all_stats = []
    sections = []

    for idx, sentence in enumerate(lines):
        out_file = RESULTS_DIR / f"{idx:02d}.out"
        if not out_file.exists():
            continue
        text = out_file.read_text()
        stats, rows = parse_out(text)
        all_stats.append(stats)

        enc = tok(sentence, add_special_tokens=False)
        ids = enc["input_ids"]
        toks = tok.convert_ids_to_tokens(ids)

        if len(rows) != len(toks):
            # Mismatch — fall back to rows alone.
            toks = ["?"] * len(rows)

        # Build per-token table rows, flagging any non-O predictions.
        table_rows = []
        for row, tkn in zip(rows, toks):
            i, label, conf = row
            flag = "" if label == "O" else " ←"
            table_rows.append(
                f"| {i:>3} | `{fmt_token(tkn)}` | {label} | {conf:.4f} |{flag}"
            )

        # Summary: entity spans (collapse BIOES into spans).
        spans = []
        cur = None
        for (i, label, _), tkn in zip(rows, toks):
            if label == "O":
                if cur is not None:
                    spans.append(cur)
                    cur = None
                continue
            prefix, _, entity = label.partition("-")
            text_piece = tkn.replace("Ġ", " ")
            if prefix in ("B", "S"):
                if cur is not None:
                    spans.append(cur)
                cur = {"entity": entity, "text": text_piece}
            else:  # I / E
                if cur is None:
                    cur = {"entity": entity, "text": text_piece}
                else:
                    cur["text"] += text_piece
            if prefix in ("E", "S"):
                spans.append(cur)
                cur = None
        if cur is not None:
            spans.append(cur)

        entity_summary = (
            ", ".join(f"`{s['text'].strip()}` → **{s['entity']}**" for s in spans)
            if spans
            else "_(no PII detected)_"
        )

        sections.append(
            f"### Example {idx + 1}\n\n"
            f"**Input:** {sentence}\n\n"
            f"**Detected entities:** {entity_summary}\n\n"
            f"**Stats:** `{stats['tokens']}` tokens · "
            f"forward `{stats['forward']*1000:.1f}` ms · "
            f"total `{stats['total']*1000:.1f}` ms · "
            f"`{stats['tps']:.1f}` tok/s\n\n"
            f"<details>\n<summary>Per-token predictions</summary>\n\n"
            f"| idx | token | label | conf |\n"
            f"|---:|:------|:------|-----:|\n"
            + "\n".join(table_rows)
            + "\n\n</details>\n"
        )

    # Aggregate stats.
    forwards = [s["forward"] for s in all_stats]
    totals = [s["total"] for s in all_stats]
    tps_all = [s["tps"] for s in all_stats]
    total_tokens = sum(s["tokens"] for s in all_stats)
    total_forward = sum(forwards)
    amortized_tps = total_tokens / total_forward if total_forward > 0 else 0.0

    header = (
        "# openai/privacy-filter on uzu — evaluation results\n\n"
        "End-to-end smoke test of the ported `openai/privacy-filter` model on the "
        "`uzu` inference engine. Each example was run via `uzu classify` on the "
        "converted `.uzu` bundle (`/tmp/pf_uzu`, exported by `lalamo convert`, "
        "bf16 runtime) on Apple Metal.\n\n"
        "- Model: **openai/privacy-filter** (20-layer MoE, 128 experts × top-4, "
        "GQA 14 heads / 2 kv-groups, YaRN RoPE, sliding window 128, attention "
        "sinks, bidirectional)\n"
        "- Runtime: **uzu** · bf16 activations & weights · Metal backend\n"
        "- Command: `./target/release/cli classify /tmp/pf_uzu --message \"<sentence>\"`\n"
        "- Reproduce: `test_data/run_eval.sh` then "
        "`uv run python test_data/build_results_md.py`\n\n"
        "## Aggregate performance\n\n"
        f"| metric | value |\n|:---|---:|\n"
        f"| examples | **{len(all_stats)}** |\n"
        f"| total tokens classified | **{total_tokens}** |\n"
        f"| mean forward pass | **{mean(forwards)*1000:.1f}** ms |\n"
        f"| median forward pass | **{median(forwards)*1000:.1f}** ms |\n"
        f"| min forward pass | **{min(forwards)*1000:.1f}** ms |\n"
        f"| max forward pass | **{max(forwards)*1000:.1f}** ms |\n"
        f"| mean total (incl. tokenize+post) | **{mean(totals)*1000:.1f}** ms |\n"
        f"| per-example throughput (mean) | **{mean(tps_all):.1f}** tok/s |\n"
        f"| amortized throughput | **{amortized_tps:.1f}** tok/s |\n\n"
        "`forward` is the model forward-pass time reported by uzu (prefill-only "
        "— classifier has no decode phase). `post` is post-processing (softmax + "
        "top-1 over labels). `total` includes tokenization, load-from-pool, etc.\n\n"
        "## Per-example results\n\n"
    )
    OUT_MD.write_text(header + "\n\n".join(sections) + "\n")
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
