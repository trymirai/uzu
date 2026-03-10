# xctrace Command Quick Reference

## `xctrace record`

Record an Instruments trace.

**Options:**

| Flag | Help | Type | Default |
|------|------|------|---------|
| `--template` | Template name or path | text | (required) |
| `--output` | Output .trace path | path | auto-generated |
| `--device` | Device name or UDID | text | local |
| `--time-limit` | Max recording time (e.g., 5s, 1m) | duration | unlimited |
| `--all-processes` | Trace all processes | flag | |
| `--attach` | Attach to PID or process name | text | |
| `--launch` | Launch and trace command (must be last) | command | |
| `--target-stdin` | Redirect stdin (- or file) | text | |
| `--target-stdout` | Redirect stdout (- or file) | text | |
| `--env` | Set env var for launched process (VAR=value) | text | |
| `--package` | Install Instruments Package for duration | file | |
| `--no-prompt` | Skip confirmation prompts | flag | |
| `--notify-tracing-started` | Send Darwin notification when started | text | |
| `--append-run` | Append run to existing trace | flag | |
| `--window` | Windowed (ring buffer) mode | duration | |

## `xctrace export`

Export data from a .trace file to XML.

**Options:**

| Flag | Help | Type | Default |
|------|------|------|---------|
| `--input` | Input .trace file | file | (required) |
| `--output` | Output file (default: stdout) | path | stdout |
| `--toc` | Export table of contents | flag | |
| `--xpath` | XPath query to select export entities | expression | |
| `--har` | Export as HTTP Archive (if HTTP Traffic present) | flag | |

### XPath patterns for Metal data

```bash
# Table of contents (always start here)
--toc

# Metal driver events (GPU work intervals, wire memory)
--xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# GPU hardware counters
--xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-intervals"]'

# Metal GPU execution intervals
--xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-gpu-intervals"]'

# CPU time profile
--xpath '/trace-toc/run[@number="1"]/data/table[@schema="time-profile"]'

# All tables in first run
--xpath '/trace-toc/run[@number="1"]/data/table'

# Process info
--xpath '/trace-toc/run[@number="1"]/processes'

# Specific process
--xpath '/trace-toc/run[@number="1"]/processes/process[@name="MyApp"]'
```

**NOTE**: Available schemas vary by template, Xcode version, and GPU. Always check `--toc` first.

## `xctrace import`

Import supported file formats into .trace.

**Options:**

| Flag | Help | Type | Default |
|------|------|------|---------|
| `--input` | Input file to import | file | (required) |
| `--output` | Output .trace path | path | auto-generated |
| `--template` | Template for import | text | |
| `--package` | Instruments Package for import | file | |

## `xctrace remodel`

Remodel trace using installed packages.

**Options:**

| Flag | Help | Type | Default |
|------|------|------|---------|
| `--input` | Input .trace file | file | (required) |
| `--output` | Output .trace path | path | |
| `--package` | Instruments Package | file | |

## `xctrace symbolicate`

Symbolicate a trace file.

**Options:**

| Flag | Help | Type | Default |
|------|------|------|---------|
| `--input` | Input .trace file | file | (required) |
| `--output` | Output .trace path | path | |
| `--dsym` | Debug symbols file | file | |

## `xctrace list`

List available resources.

**Subcommands:**

| Subcommand | Description |
|-----------|-------------|
| `devices` | List available devices |
| `templates` | List available recording templates |
| `instruments` | List available instruments |

## Metal Shader Toolchain Commands

### `xcrun metal`

Metal shader compiler.

| Flag | Help |
|------|------|
| `-c` | Compile to .air (intermediate representation) |
| `-o` | Output file path |
| `-sdk macosx` | Target macOS SDK (also: iphoneos, appletvos) |
| `-target air64-apple-macos14` | Target specific platform/version |
| `-std=metal3.0` | Metal language standard version |
| `-gline-tables-only` | Generate line-level debug info |
| `-Weverything` | Enable all warnings |
| `-Werror` | Treat warnings as errors |
| `-I <path>` | Add include search path |
| `-D <macro>` | Define preprocessor macro |

### `xcrun metal-ar`

Metal archive tool (create static libraries of .air files).

```bash
xcrun -sdk macosx metal-ar rcs output.metalar input1.air input2.air
```

### `xcrun metallib`

Metal linker (create .metallib from .air or .metalar).

```bash
xcrun -sdk macosx metallib input.metalar -o output.metallib
```

### `xcrun metal-arch`

List available Metal GPU architecture targets.

### `metal-shaderconverter`

Convert HLSL (via DXIL) to Metal IR.

```bash
metal-shaderconverter input.dxil -o output.metallib
```

## Metal Environment Variables

### Debugging

| Variable | Values | Effect |
|----------|--------|--------|
| `MTL_DEBUG_LAYER` | 1 | Enable API validation layer |
| `MTL_SHADER_VALIDATION` | 1 | Enable GPU shader validation |
| `METAL_CAPTURE_ENABLED` | 1 | Allow programmatic .gputrace capture |
| `METAL_DEVICE_WRAPPER_TYPE` | 1 | Enable Metal device wrapper (extended validation) |

### Performance HUD

| Variable | Values | Effect |
|----------|--------|--------|
| `MTL_HUD_ENABLED` | 1 | Show performance overlay |
| `MTL_HUD_LOGGING_ENABLED` | 1 | Log HUD metrics to system log |

### MoltenVK (Vulkan apps)

| Variable | Values | Effect |
|----------|--------|--------|
| `MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE` | 1=device, 2=first frame | Auto-capture scope |
| `MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE` | path | .gputrace output path |
| `MVK_CONFIG_LOG_LEVEL` | 0-4 | MoltenVK logging verbosity |

## macOS Log Commands for Metal

```bash
# Stream Metal errors live
log stream --predicate 'subsystem == "com.apple.Metal"' --level error

# Stream all Metal messages
log stream --predicate 'subsystem == "com.apple.Metal"'

# Stream HUD performance data
log stream --predicate 'subsystem == "com.apple.Metal" AND category == "HUD"'

# Show recent Metal errors
log show --predicate 'subsystem == "com.apple.Metal"' --last 5m --level error

# Show recent Metal messages (all levels)
log show --predicate 'subsystem == "com.apple.Metal"' --last 5m

# Export to file
log show --predicate 'subsystem == "com.apple.Metal"' --last 10m > metal_log.txt
```
