# cargo-keisoku

A `cargo` subcommand that measures GPU/SoC power on Apple Silicon. It wraps a cargo command,
samples power (via [`keisoku`](../keisoku)) while it runs, and can fit and apply a simple
energy model for predicting an LLM's power before you load it.

## Install

```bash
cargo install --path crates/cargo-keisoku
```

## Subcommands

### collect — measure any cargo command

```bash
cargo keisoku [--interval-ms 100] [--out report.json] <run|bench|test ...>
```

Runs `cargo <...>` as a child, records a power timeline, and writes a JSON report
(`device`, full `session` timeline, and per-window summaries). The child can annotate the
timeline by printing two line formats on stdout:

- `KEISOKU_MARK <label>` — opens a labeled window
- `KEISOKU_DATA <label> key=value ...` — attaches numbers (e.g. `iters`, `gpu_ns`) to it

Each report `window` then carries avg/peak GPU watts, energy (J), and whatever data the
child attached.

### calibrate — fit the energy model

```bash
cargo keisoku calibrate --report report.json --out coeffs.json
```

Fits the two-term roofline model `energy ≈ a·bytes + b·flops + P_idle·t` (least squares) from
the GEMM windows in a report and writes the coefficients (`a` J/byte, `b` J/FLOP, idle watts,
peak bandwidth/FLOP-rate) for that chip.

### predict — estimate a model's power before loading it

```bash
cargo keisoku predict --model <dir-or-config.json> --coeffs coeffs.json \
  [--prompt-tokens 512] [--tokens 128] [--out prediction.json]
```

Reads the model's `config.json` (no weights loaded), works out per-token bytes/FLOPs, applies
the calibrated coefficients, and prints predicted decode watts, tokens/sec, energy-per-token,
time-to-first-token and total energy.

## End-to-end calibration recipe

```bash
# 1. collect power per GEMM shape (the GEMM bench emits the MARK/DATA lines under this env var)
KEISOKU_MARK=1 cargo keisoku --out gemm.json bench --bench main -p backend-uzu -- Metal/Kernel/Matmul/GEMM
# 2. fit coefficients for this chip
cargo keisoku calibrate --report gemm.json --out coeffs.json
# 3. predict for any model
cargo keisoku predict --model path/to/model --coeffs coeffs.json
```

## Notes

- Real power only on Apple Silicon; elsewhere the report has empty samples.
- The predictor is a **GEMM-dominated estimate** (treats mixers as attention-like, dense MLP,
  ignores GQA and attention's O(n²)); treat outputs as estimates and validate against a real
  calibration run. IOReport power is itself a model-based estimate.
