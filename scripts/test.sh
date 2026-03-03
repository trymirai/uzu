cd /Users/eugenebokhan/Developer/trymirai/performance-benchmarks

# Clean all previous hadamard results
rm -rf workspace/results/hadamard-*

# Run all 6 modes
for mode in full simd_shuffle block32 block64 block128; do
  echo "=== Running $mode ==="
  UZU_FWHT_MODE=$mode uv run benchmarks run "hadamard-$mode" int4-hadamard-compatible --uzu-branch hadamard
done
echo "=== Running none ==="
uv run benchmarks run hadamard-none int4-hadamard-compatible --uzu-branch hadamard

# Analyze: extract mean gen tok/s and prompt tok/s for all modes
echo ""
echo "=== RESULTS ==="
python3 << 'PYEOF'
import json, os, glob

base = "workspace/results"
modes = ["hadamard-none", "hadamard-full", "hadamard-simd_shuffle", "hadamard-block32", "hadamard-block64", "hadamard-block128"]
mode_labels = ["none", "full", "simd_shuffle", "block32", "block64", "block128"]

all_models = set()
data = {}
for mode, label in zip(modes, mode_labels):
    uzu_dirs = glob.glob(os.path.join(base, mode, "*/uzu"))
    if not uzu_dirs:
        continue
    for filepath in sorted(glob.glob(os.path.join(uzu_dirs[0], "*.json"))):
        model = os.path.basename(filepath).replace(".json", "")
        runs = json.load(open(filepath))
        gen = [r["generate_tokens_per_second"] for r in runs if r.get("generate_tokens_per_second")]
        prompt = [r["prompt_tokens_per_second"] for r in runs]
        all_models.add(model)
        data[(label, model)] = {
            "gen": sum(gen)/len(gen) if gen else 0,
            "prompt": sum(prompt)/len(prompt) if prompt else 0,
            "runs": len(runs),
        }

models = sorted(all_models)
header = f"{'Model':<30s}" + "".join(f"{l:>14s}" for l in mode_labels)

print("\nGeneration tokens/s:")
print(header)
for model in models:
    row = f"{model:<30s}"
    for label in mode_labels:
        val = data.get((label, model), {}).get("gen", 0)
        row += f"{val:>14.1f}" if val else f"{'--':>14s}"
    print(row)

print("\nPrompt tokens/s:")
print(header)
for model in models:
    row = f"{model:<30s}"
    for label in mode_labels:
        val = data.get((label, model), {}).get("prompt", 0)
        row += f"{val:>14.1f}" if val else f"{'--':>14s}"
    print(row)
PYEOF