# Power consumption benchmark

`uzu-tools power-consumption` measures per-model power and energy across a prefill/generate sweep on macOS.

## Modes

### Registry (default)

Uses the live Mirai registry. Configs are downloaded through `Storage`; safetensors headers are range-fetched and cached beside each config.

```bash
uzu-tools power-consumption \
  --output power_consumption.csv \
  --model-id mirai:llama-3-8b-instruct \
  --storage ~/power-cache
```

- Omit `--model-id` to benchmark every downloadable chat model.
- Repeat `--model-id` to select multiple registry models by exact `model.identifier`.
- `--storage` is optional. When set, registry artifacts are written under that directory so they can be replayed in local mode.
- When omitted, registry mode uses a temporary cache at `$TMPDIR/uzu-power-consumption`.

### Local

Replays compatible artifacts from a storage directory. No registry access, `UzuEngine`, or network I/O.

```bash
uzu-tools power-consumption \
  --source local \
  --storage ~/power-cache \
  --model-id llama/llama-3-8b-instruct/v1.0
```

- `--storage` is required.
- Every compatible model under the storage tree is discovered when no `--model-id` is given.
- Repeat `--model-id` to select artifacts by exact storage-relative ID (path under `models/`).

## Storage layout

`--storage` is the `StorageConfig::base_path`. Model artifacts live at:

```text
<storage>/.cache/mirai/models/
  <reference-name>/<cache-identifier>/<checkpoint-version>/
    config.json
    model.header.safetensors
```

Registry mode writes `config.json` through `Storage` and saves the HTTP range-fetched safetensors header as `model.header.safetensors`.

Local mode accepts:

- `model.header.safetensors` (preferred; header-only file), or
- `model.safetensors` (full weights file; only the header is read)

Both work with random-weight loading because tensor payloads are synthesized deterministically.

## Artifact requirements

- `config.json` must be the converted Uzu `LanguageModelConfig`, not a Hugging Face config.
- The safetensors header must contain the exact tensor keys, dtypes, shapes, byte-correct offsets, and `__metadata__` weight specs required by that config.
- No tensor payload is required in `model.header.safetensors`.

## Common options

```text
--source registry|local       # default: registry
--storage <DIR>               # optional cache base for registry (default: temp); required for local
--model-id <ID>               # repeatable exact model ID selector
--prefill 128,512,2048        # prefill token counts
--generate 32,128             # decode token counts
--iterations 6                # measured iterations per prefill/generate pair
```

## Registry-to-local replay

Run once against the registry into a dedicated cache directory:

```bash
uzu-tools power-consumption \
  --storage ~/power-cache \
  --model-id mirai:tiny \
  --output registry.csv
```

Replay the same artifacts offline using the storage-relative IDs printed during discovery:

```bash
uzu-tools power-consumption \
  --source local \
  --storage ~/power-cache \
  --model-id tiny/tiny-model/v1.0 \
  --output local.csv
```

The CSV includes a `source` column (`registry` or `local`) for each row.

## Power-user workflow

1. Create a directory tree matching the layout above.
2. Place a valid Uzu `config.json` and safetensors header (or full `model.safetensors`) in each model directory.
3. Run with `--source local --storage <DIR>`.
