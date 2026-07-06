#!/usr/bin/env bash
#
# Cross-check the power-consumption benchmark's keisoku reads against
# powermetrics (macOS ground truth). Run with sudo (powermetrics needs root):
#
#   cargo build --release -p power-consumption      # build first, as your user
#   sudo ./scripts/compare_power_consumption.sh [model_dir]
#
# Tunables via env: PREFILL GENERATE REPS PM_INTERVAL GPU_ACTIVE_MW
#
# powermetrics reports CPU/GPU/ANE power on Apple Silicon but NOT DRAM or a
# whole-package figure, so RAM and the SMC "package" columns have no ground-truth
# counterpart here — the meaningful checks are GPU, CPU, and CPU+GPU+ANE.

set -euo pipefail

MODEL="${1:-workspace/models/0.5.12/Llama-3.2-1B-Instruct}"
PREFILL="${PREFILL:-512}"
GENERATE="${GENERATE:-256}"
REPS="${REPS:-10}"
PM_INTERVAL="${PM_INTERVAL:-200}"        # powermetrics sampling interval (ms)
GPU_ACTIVE_MW="${GPU_ACTIVE_MW:-3000}"   # a sample counts as "under load" above this

BIN=./target/release/power-consumption

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo (powermetrics needs root):  sudo $0 [model_dir]" >&2
    exit 1
fi
if [[ ! -x "$BIN" ]]; then
    echo "Missing $BIN — build it first, as your user (not root):" >&2
    echo "  cargo build --release -p power-consumption" >&2
    exit 1
fi

PM_LOG=/tmp/pc_powermetrics.log
PC_CSV=/tmp/pc_bench.csv
trap 'kill "${PM_PID:-}" 2>/dev/null || true' EXIT

echo "model=$MODEL prefill=$PREFILL generate=$GENERATE reps=$REPS pm_interval=${PM_INTERVAL}ms active>=${GPU_ACTIVE_MW}mW"
echo

powermetrics --samplers cpu_power,gpu_power -i "$PM_INTERVAL" >"$PM_LOG" 2>/dev/null &
PM_PID=$!
sleep 1

"$BIN" --local-model-path "$MODEL" --skip-cooldown \
    --prefill "$PREFILL" --generate "$GENERATE" --repetitions "$REPS" \
    --output "$PC_CSV" >/dev/null

kill "$PM_PID" 2>/dev/null || true
wait "$PM_PID" 2>/dev/null || true

echo "--- raw powermetrics power lines (first 8) ---"
grep -E 'Power:' "$PM_LOG" | head -8
echo "--- highest GPU Power samples (confirms the compute phase was captured) ---"
grep 'GPU Power:' "$PM_LOG" | sort -k3 -n | tail -3
echo

# powermetrics: average over active samples (GPU under load). A sample is
# CPU / GPU / ANE lines (no "Combined" on Apple Silicon), so we commit each sample
# at the next CPU line (its boundary) and flush the last at END; CPU+GPU+ANE is summed.
read -r pm_gpu pm_cpu pm_ane pm_comb pm_n < <(awk -v thr="$GPU_ACTIVE_MW" '
    $1=="CPU" && $2=="Power:" {
        if (have && g+0>=thr) { sc+=c; sg+=g; sa+=a; n++ }
        c=$(NF-1); g=0; a=0; have=1
    }
    $1=="GPU" && $2=="Power:" { g=$(NF-1) }
    $1=="ANE" && $2=="Power:" { a=$(NF-1) }
    END {
        if (have && g+0>=thr) { sc+=c; sg+=g; sa+=a; n++ }
        if (n>0) printf "%.2f %.2f %.2f %.2f %d\n", sg/n/1000, sc/n/1000, sa/n/1000, (sg+sc+sa)/n/1000, n
        else print "0 0 0 0 0"
    }' "$PM_LOG")

# our benchmark: mean over ok rows
read -r pc_gpu pc_cpu pc_ane pc_ram pc_total pc_pkg pc_n < <(awk -F, '
    NR>1 && $17=="ok" { t+=$30; p+=$31; c+=$32; g+=$33; a+=$35; r+=$36; n++ }
    END { if (n>0) printf "%.2f %.2f %.2f %.2f %.2f %.2f %d\n", g/n, c/n, a/n, r/n, t/n, p/n, n
          else print "0 0 0 0 0 0 0" }' "$PC_CSV")

pct() { awk -v o="$1" -v g="$2" 'BEGIN { if (g==0) print "n/a"; else printf "%+.1f%%", 100*(o-g)/g }'; }

echo "                     powermetrics(GT)   keisoku(ours)     delta"
printf "  GPU power          %8s W        %8s W     %s\n"   "$pm_gpu"  "$pc_gpu"   "$(pct "$pc_gpu" "$pm_gpu")"
printf "  CPU power          %8s W        %8s W     %s\n"   "$pm_cpu"  "$pc_cpu"   "$(pct "$pc_cpu" "$pm_cpu")"
printf "  ANE power          %8s W        %8s W     %s\n"   "$pm_ane"  "$pc_ane"   "$(pct "$pc_ane" "$pm_ane")"
cg=$(awk -v c="$pc_cpu" -v g="$pc_gpu" -v a="$pc_ane" 'BEGIN{printf "%.2f", c+g+a}')
printf "  CPU+GPU+ANE        %8s W        %8s W     %s\n"   "$pm_comb" "$cg"       "$(pct "$cg" "$pm_comb")"
echo
echo "  keisoku-only (no powermetrics counterpart on Apple Silicon):"
printf "    RAM=%s W  total(rails,incl RAM)=%s W  package(SMC PSTR)=%s W\n" "$pc_ram" "$pc_total" "$pc_pkg"
echo
echo "  samples: powermetrics active=$pm_n, benchmark ok rows=$pc_n"
echo
echo "  raw logs kept for inspection: $PM_LOG , $PC_CSV"
