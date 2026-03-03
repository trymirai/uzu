---
name: metal-gpu-debug
description: >
  Metal GPU debugging and profiling via xctrace and Apple's Metal toolchain. Use this skill when the user mentions:
  Metal debugging, .gputrace files, Metal shaders, Metal pipeline state, Metal performance,
  Metal System Trace, Apple GPU profiling, Metal validation, shader compilation errors,
  MSL (Metal Shading Language), .metallib files, .air files, Metal render pass,
  Metal command buffer, Metal encoder, MTLDevice, MTLTexture, Metal Performance HUD,
  GPU counters on Apple Silicon, Metal frame capture, Metal API validation,
  Metal shader validation, xcrun metal, xctrace, Instruments GPU trace,
  MoltenVK debugging, Vulkan on macOS, Apple Silicon GPU, Metal 3, Metal 4,
  Metal compute kernel, MetalFX, Metal Performance Shaders, Metal ray tracing,
  "profile my Metal app", "why is my shader slow", "capture a Metal frame",
  "Metal validation error", "debug Metal on Mac", "GPU trace on macOS".
  DO NOT use for: RenderDoc, Vulkan on Windows/Linux, D3D debugging, PIX,
  Nsight, CSS rendering, React rendering, server-side rendering, HTML layout,
  browser DevTools, web performance.
---

# Metal GPU Debugging & Profiling Skill

## Overview

This skill enables Metal GPU profiling, debugging, and shader analysis on macOS using Apple's command-line toolchain. The primary tools are:

- **`gpu_trace`** — Unified CLI for recording, analyzing, and comparing Metal GPU traces (in `tools/gpu_trace/`, run via `uv run gpu_trace`). Subcommands:
  - `gpu_trace run` — Record a Metal System Trace while running a command
  - `gpu_trace analyze` — Analyze an existing `.trace` bundle with per-kernel timing & HW counters
  - `gpu_trace compare` — Compare two traces side-by-side
  - `gpu_trace capture` — Inspect `.gputrace` captures (list resources, read buffers/textures)
  - `gpu_trace view` — View a previously exported JSON trace
- **`xctrace`** — CLI for Instruments (used internally by `gpu_trace`; can also be used directly)
- **`xcrun metal`** — Metal shader compiler toolchain (compile, archive, link)
- **Metal environment variables** — validation layers, Performance HUD, programmatic capture
- **`log`** — macOS unified logging for Metal HUD and validation output

### Prerequisites

Before any Metal debugging, verify the environment:

```bash
# Check Xcode is installed (full Xcode, not just Command Line Tools)
xcode-select -p
# Expected: /Applications/Xcode.app/Contents/Developer

# Check xctrace is available
xcrun xctrace version

# Check Metal compiler
xcrun -sdk macosx metal --version

# List available GPU devices (requires a Swift/ObjC helper or Python)
system_profiler SPDisplaysDataType | grep -A5 "Chipset\|Metal"
```

**IMPORTANT**: `xctrace` and Metal tools require **full Xcode**, not just Command Line Tools.

## 1. Automated Session Lifecycle

Every Metal debugging session follows a strict pipeline: **Doctor → Record → Export → Parse → Analyze → Report → Cleanup**. Claude should execute this pipeline autonomously, not just suggest commands.

### Step 0: Doctor — Verify environment

**Run this FIRST before any debugging session.** If any check fails, stop and tell the user what's missing.

```bash
# Create working directories
mkdir -p ./traces/analysis

# Verify toolchain
xcode-select -p                          # Must show Xcode.app path, NOT CommandLineTools
xcrun xctrace version                    # Must succeed
xcrun -sdk macosx metal --version        # Must succeed
system_profiler SPDisplaysDataType | grep -i "metal\|chipset"  # GPU info

# For iOS: check connected devices
xcrun xctrace list devices 2>/dev/null   # List all available targets

# Check gpu_trace is available
cd tools/gpu_trace && uv run gpu_trace --help >/dev/null 2>&1 && echo "gpu_trace OK"
```

**If `xcode-select -p` returns `/Library/Developer/CommandLineTools`**, tell the user:
```
Xcode Command Line Tools is installed but full Xcode is required.
Install from: https://developer.apple.com/xcode/
Then run: sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
```

### Step 1: Record — Capture a Metal System Trace

**Preferred: Use `gpu_trace run`** (handles recording + export + analysis automatically):

```bash
# MODE A: Launch and profile (most common)
cd tools/gpu_trace && uv run gpu_trace run -- /path/to/app [args...]

# MODE A with hardware counters (ALU, memory, cache)
cd tools/gpu_trace && uv run gpu_trace run --gpu-counters -- /path/to/app

# MODE A with time limit
cd tools/gpu_trace && uv run gpu_trace run --time-limit 10s -- /path/to/app

# MODE B: Attach to running process
cd tools/gpu_trace && uv run gpu_trace run --attach <PID> --time-limit 10s

# MODE A with verbose output (all HW counters per kernel)
cd tools/gpu_trace && uv run gpu_trace run -v --gpu-counters -- /path/to/app
```

**Fallback: Use `xctrace` directly** (for advanced options like validation env vars, iOS):

```bash
# MODE C: With validation enabled (for debugging errors)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MTL_DEBUG_LAYER=1 \
  --env MTL_SHADER_VALIDATION=1 \
  --time-limit 10s \
  --no-prompt \
  --output ./traces/capture.trace \
  --launch -- /path/to/app

# MODE D: iOS device (over USB)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --device '<DEVICE_NAME_OR_UDID>' \
  --attach <APP_NAME> \
  --time-limit 10s \
  --no-prompt \
  --output ./traces/capture.trace
```

**Decision guide:**
- User says "profile my app" → `gpu_trace run`
- User says "it's already running" → `gpu_trace run --attach <PID>`
- User says "I'm getting errors" or "something is wrong" → `xctrace` with validation env vars
- User mentions iPhone/iPad → `xctrace` with `--device`

### Steps 2–3: Export & Parse — Analyze the trace

**Preferred: Use `gpu_trace analyze`** (handles export, parsing, and analysis in one step):

```bash
# Analyze an existing trace (recorded with gpu_trace run or xctrace)
cd tools/gpu_trace && uv run gpu_trace analyze /path/to/capture.trace

# Verbose output (all HW counters per kernel)
cd tools/gpu_trace && uv run gpu_trace analyze /path/to/capture.trace -v

# Export to JSON for further processing
cd tools/gpu_trace && uv run gpu_trace analyze /path/to/capture.trace --json output.json

# JSON to stdout (for piping)
cd tools/gpu_trace && uv run gpu_trace analyze /path/to/capture.trace --json -
```

`gpu_trace analyze` automatically:
- Exports all relevant schemas in parallel (GPU state, encoders, intervals, counters, shaders)
- Resolves xctrace's XML reference/ID structure
- Computes GPU utilization with overlapping interval union
- Aggregates per-kernel timing statistics
- Correlates hardware counters with kernel dispatches (O(D log S) bisect)
- Detects bottlenecks (ALU, buffer, cache, MMU limiters)

**Fallback: Use `xctrace export` directly** (for schemas not covered by `gpu_trace`):

```bash
# Get table of contents
xcrun xctrace export --input ./traces/capture.trace --toc

# Export a specific schema
xcrun xctrace export --input ./traces/capture.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'
```

**Schema availability varies** by Xcode version, template, and GPU. `gpu_trace analyze` handles missing schemas gracefully.

### Step 4: Analyze — Interpret the data

After parsing, Claude should analyze the results and look for:

**Performance profiling:**
- GPU busy time per frame (>16ms = below 60fps, >8ms = below 120fps)
- Large wire memory events (excessive per-frame resource allocation)
- Gaps between GPU submissions (CPU-bound)
- Long shader execution intervals (complex shaders)
- Imbalanced encoder durations (one pass dominating)

**Validation errors:**
- Pattern-match stderr and log output for Metal error codes
- Classify errors: API misuse vs shader bug vs resource issue
- Map errors to specific command encoders via labels

**Shader issues:**
- Compile-time warnings/errors from `xcrun metal`
- Runtime validation errors from `MTL_SHADER_VALIDATION`

### Step 5: Report — Present findings to user

Structure the report as:
1. **Environment**: GPU model, macOS version, device (Mac/iOS)
2. **Summary**: Total events, recording duration, overall health
3. **Key findings**: Sorted by severity (errors → warnings → info)
4. **Specific data**: Relevant numbers, event counts, durations
5. **Recommendations**: Concrete next steps

### Step 6: Cleanup

```bash
# Remove large trace files when analysis is complete
# (only if user doesn't need the raw trace)
rm -rf ./traces/capture.trace

# Keep analysis outputs for reference
ls -la ./traces/analysis/
```

### Complete automated workflow example

This is what Claude should execute end-to-end when a user says "profile my Metal app":

```bash
# Single command: records, exports (parallel), analyzes, and prints results
cd tools/gpu_trace && uv run gpu_trace run --gpu-counters -- /path/to/app

# Or if the trace was already recorded separately:
cd tools/gpu_trace && uv run gpu_trace analyze /path/to/capture.trace -v
```

`gpu_trace run` handles the entire pipeline: Doctor → Record → Export (parallel) → Parse → Analyze → Report. JSON is automatically saved alongside the trace.

For **before/after comparison**:

```bash
cd tools/gpu_trace
uv run gpu_trace run -o before.trace -- /path/to/app_before
# ... make code changes ...
uv run gpu_trace run -o after.trace -- /path/to/app_after
uv run gpu_trace compare before.trace after.trace --label1 before --label2 after
```

### Parallel workflows: Validation + Profiling + Logs

Claude can run multiple debugging streams simultaneously for maximum signal:

```bash
# Record trace with all validation layers AND HUD logging in one shot
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MTL_DEBUG_LAYER=1 \
  --env MTL_SHADER_VALIDATION=1 \
  --env MTL_HUD_ENABLED=1 \
  --env MTL_HUD_LOGGING_ENABLED=1 \
  --time-limit 10s \
  --no-prompt \
  --output ./traces/full_debug.trace \
  --launch -- /path/to/app 2> ./traces/analysis/stderr.log &

TRACE_PID=$!

# Simultaneously capture Metal logs from unified log
log stream --predicate 'subsystem == "com.apple.Metal"' \
  --timeout 15 > ./traces/analysis/metal_log.txt 2>/dev/null &

LOG_PID=$!

# Wait for trace to finish
wait $TRACE_PID

# Give log stream a moment then stop it
sleep 2
kill $LOG_PID 2>/dev/null

# Now analyze ALL data sources:
echo "=== Validation Errors (stderr) ==="
grep -i "error\|warning\|invalid\|fault" ./traces/analysis/stderr.log || echo "None"

echo "=== Metal Log Entries ==="
wc -l < ./traces/analysis/metal_log.txt
grep -i "error" ./traces/analysis/metal_log.txt || echo "No errors in log"

echo "=== Trace Data ==="
xcrun xctrace export --input ./traces/full_debug.trace --toc
```

### Session state awareness

Unlike RenderDoc's daemon model (open/close), xctrace sessions are **fire-and-forget**: each `record` command runs to completion, and the resulting `.trace` file is a self-contained snapshot. This means:

- **No session to manage** — no open/close, no daemon, no leaked processes
- **Multiple traces can coexist** — name them meaningfully (before.trace, after.trace)
- **Traces are immutable** — once recorded, they don't change
- **Export is idempotent** — re-export the same trace as many times as needed
- **Cleanup is just file deletion** — `rm` the .trace bundle when done

## 2. iOS Device Support

`xctrace` can profile Metal apps on physical iOS/iPadOS devices over USB or Wi-Fi.

### List connected devices

```bash
xcrun xctrace list devices
# Shows: Mac, connected iPhones/iPads, simulators
```

### Record on iOS device

```bash
# By device name
xcrun xctrace record \
  --template 'Metal System Trace' \
  --device 'iPhone 15 Pro' \
  --attach MyApp \
  --time-limit 10s \
  --output ios_trace.trace

# By UDID (more reliable)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --device 00008110-XXXXXXXXXXXX \
  --attach MyApp \
  --time-limit 10s \
  --output ios_trace.trace

# Launch app on device (must be installed)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --device 'iPhone 15 Pro' \
  --time-limit 10s \
  --output ios_trace.trace \
  --launch -- com.yourcompany.yourapp
```

### iOS-specific notes

- Device must be **unlocked** and **trusted** by the Mac
- App must be installed on device (use Xcode or `ios-deploy`)
- `--launch` uses **bundle identifier** (not path) on iOS
- `--attach` uses **process name** or PID
- Export/parse works identically — the `.trace` format is the same
- Shader compilation for iOS uses `-sdk iphoneos`:
  ```bash
  xcrun -sdk iphoneos metal -c Shader.metal -o Shader.air
  ```
- `.gputrace` capture on iOS requires Xcode attached to the device

## 3. Metal System Trace — Record & Export

Metal System Trace is the primary tool for GPU profiling. It captures CPU/GPU timeline, driver events, shader execution, and hardware counters.

### Record a trace

```bash
# Profile a running app by name
xcrun xctrace record \
  --template 'Metal System Trace' \
  --attach <PID_OR_NAME> \
  --time-limit 5s \
  --output trace.trace

# Launch and profile
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 5s \
  --output trace.trace \
  --launch -- /path/to/your/app [args...]

# With environment variables (e.g., enable validation)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MTL_DEBUG_LAYER=1 \
  --env MTL_SHADER_VALIDATION=1 \
  --time-limit 5s \
  --output trace.trace \
  --launch -- /path/to/your/app
```

Key options:
- `--time-limit 5s` — auto-stop after duration (supports ms, s, m, h)
- `--attach PID` — attach to running process
- `--all-processes` — trace all Metal apps system-wide
- `--env VAR=value` — set environment variables for launched process
- `--target-stdout -` — redirect app stdout to terminal
- `--no-prompt` — skip prompts (useful in scripts)

### Other useful templates

```bash
# List all available templates
xcrun xctrace list templates
# Key Metal-relevant templates:
#   - Metal System Trace    (GPU timeline, driver events, counters)
#   - Game Performance       (Metal + display + thermal)
#   - Counters               (hardware performance counters)
#   - GPU                    (GPU-focused template, if available)
```

### Export trace data as XML

```bash
# See what's in the trace (table of contents)
xcrun xctrace export --input trace.trace --toc

# Export Metal driver events (GPU work intervals, wire memory, etc.)
xcrun xctrace export --input trace.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# Export to file instead of stdout
xcrun xctrace export --input trace.trace \
  --output metal_events.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# Export GPU counter data
xcrun xctrace export --input trace.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-intervals"]'
```

### Common Metal table schemas

| Schema | Contains |
|--------|----------|
| `metal-driver-event-intervals` | Metal driver events (GPU work, wire memory, resource events) |
| `gpu-counter-intervals` | Hardware GPU performance counters |
| `metal-gpu-intervals` | GPU execution intervals per encoder |
| `time-profile` | CPU time profiling samples |

**TIP**: Always run `--toc` first to see available schemas — they vary by template, Xcode version, and GPU.

### Parse exported XML

The XML uses a reference system to avoid duplication. Nodes with `id` attributes are originals; nodes with `ref` attributes point back to them.

```bash
# Use gpu_trace to parse all schemas automatically
cd tools/gpu_trace && uv run gpu_trace analyze trace.trace -v

# Or export to JSON for programmatic access
cd tools/gpu_trace && uv run gpu_trace analyze trace.trace --json output.json
```

## 4. Metal Validation Layers

Runtime error detection for Metal API misuse and shader bugs.

### Enable via environment variables

```bash
# API Validation — catches Metal API misuse
export MTL_DEBUG_LAYER=1

# Shader Validation — instruments shaders to detect GPU-side errors
export MTL_SHADER_VALIDATION=1

# Combined: launch app with both
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 /path/to/your/app
```

### Enable via xctrace

```bash
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MTL_DEBUG_LAYER=1 \
  --env MTL_SHADER_VALIDATION=1 \
  --time-limit 10s \
  --output validated.trace \
  --launch -- /path/to/your/app
```

### Read validation errors

Validation errors appear in stderr and in macOS unified log:

```bash
# Stream Metal validation errors live
log stream --predicate 'subsystem == "com.apple.Metal"' --level error

# Search recent logs
log show --predicate 'subsystem == "com.apple.Metal"' --last 5m
```

### Programmatic validation log access

If you're writing Metal code, `commandBuffer.logs` provides structured error info after completion — encoder label, debug location (file + line), and error classification.

## 5. Metal Performance HUD

Real-time overlay showing FPS, frame time, GPU time, memory usage.

### Enable

```bash
# Per-process via environment variable
MTL_HUD_ENABLED=1 /path/to/your/app

# System-wide (all Metal apps)
/bin/launchctl setenv MTL_HUD_ENABLED 1
# Disable:
/bin/launchctl unsetenv MTL_HUD_ENABLED

# Enable HUD data logging to syslog
MTL_HUD_ENABLED=1 MTL_HUD_LOGGING_ENABLED=1 /path/to/your/app
```

### Parse HUD log data

When `MTL_HUD_LOGGING_ENABLED=1`, metrics are logged to the system log:

```bash
# Stream HUD metrics
log stream --predicate 'subsystem == "com.apple.Metal" AND category == "HUD"'

# Export recent HUD data
log show --predicate 'subsystem == "com.apple.Metal" AND category == "HUD"' --last 1m
```

HUD metrics include: FPS, present interval (frame time), GPU time, process memory, GPU memory, display refresh rate, direct vs composited rendering path.

## 6. Programmatic Frame Capture (.gputrace)

Capture Metal frames to `.gputrace` files without Xcode attached, then inspect buffer/texture data from CLI.

### Via environment variables (MoltenVK / any Metal app)

```bash
# For Vulkan apps via MoltenVK:
export METAL_CAPTURE_ENABLED=1
export MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=2          # 1=device lifecycle, 2=first frame
export MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=/tmp/capture.gputrace
/path/to/vulkan/app

# For native Metal apps (requires Info.plist MetalCaptureEnabled=true
# or METAL_CAPTURE_ENABLED=1 environment variable):
METAL_CAPTURE_ENABLED=1 /path/to/metal/app
```

### Via MTLCaptureManager (in-app)

For apps you control, add capture support using MTLCaptureManager. See `capture_frame.swift` for a complete example. The key pattern for adding capture to an existing app:

```swift
let captureManager = MTLCaptureManager.shared()
if captureManager.supportsDestination(.gpuTraceDocument) {
    let descriptor = MTLCaptureDescriptor()
    descriptor.captureObject = device
    descriptor.destination = .gpuTraceDocument
    descriptor.outputURL = URL(fileURLWithPath: "./capture.gputrace")
    try captureManager.startCapture(with: descriptor)

    // ... encode and submit Metal work ...

    captureManager.stopCapture()
}
```

Run with: `METAL_CAPTURE_ENABLED=1 ./your_app`

### Open .gputrace in Xcode

```bash
open /tmp/capture.gputrace
# Opens in Xcode's Metal Debugger with full inspection capabilities:
# - Draw call list and stepping
# - Pipeline state at each draw
# - Texture/buffer inspection
# - Shader debugging
# - Performance profiling
# - Dependency viewer
```

### Autonomous Metal Debugging Workflow

When debugging a Metal rendering issue, Claude executes this workflow autonomously. It maximizes signal by gathering data from multiple sources in parallel, then branches based on what data is actually available.

#### Step 1: Gather all signal sources (run in parallel)

Launch these data-gathering steps simultaneously — they are independent:

```bash
# A. Screenshot the app's rendered output
./build_and_run.sh --screenshot        # preferred: in-app framebuffer readback
# OR: screencapture -w -x output.png   # fallback: macOS window capture

# B. Capture .gputrace (programmatic)
METAL_CAPTURE_ENABLED=1 ./your_app     # produces capture.gputrace

# C. Compile shaders with maximum warnings
xcrun -sdk macosx metal -c -Weverything Shaders.metal -o /dev/null 2>&1

# D. Read source code — Claude reads .metal files and Swift renderer code directly
```

#### Step 2: Analyze the capture

```bash
# 2a. List all resources and shader functions
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace

# 2b. Check what data files exist in the capture
ls capture.gputrace/MTLBuffer-* 2>/dev/null && echo "BUFFER DATA AVAILABLE" || echo "NO BUFFER FILES"
ls capture.gputrace/MTLTexture-* 2>/dev/null && echo "TEXTURE DATA AVAILABLE" || echo "NO TEXTURE FILES"
```

**Branch on buffer availability:**

- **MTLBuffer files exist** (Xcode-initiated captures): Parse buffer data directly
  ```bash
  cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --buffer "Vertex" --layout float4 --index 0-10
  cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --dump-all
  ```
- **No MTLBuffer files** (typical for programmatic captures): Fall back to source code analysis
  - Read the Swift/ObjC code that creates and fills buffers
  - Read the `.metal` shader code that consumes the buffers
  - The capture metadata still tells you what resources exist and what shaders run — use this as an inventory to guide source code reading
- **MTLTexture files exist**: Read raw texture data for render target analysis

#### Step 3: Diagnose from all available sources

Combine signal from every source gathered in Steps 1–2. Each source catches different bug categories:

| Source | What it reveals | Bug categories |
|--------|----------------|----------------|
| **Screenshot** | Visual output | Wrong colors, flipped geometry, missing faces, transparency issues, blank screen |
| **Source code** (.metal + Swift) | Logic and data flow | Wrong vertex data, shader math errors, incorrect pipeline config, buffer layout mismatches |
| **Capture metadata** | Resource inventory | Confirms what buffers/textures exist, shader function names, resource labels |
| **Shader compilation warnings** | Static analysis | Unused variables, implicit conversions, potential precision issues |
| **Validation errors** (if enabled) | API correctness | Mismatched formats, missing bindings, out-of-bounds access, shader faults |

**Decision logic when buffer data is unavailable:**
- If no `MTLBuffer` files → Claude reads source code to understand vertex data, uniform values, and buffer layouts
- If no `MTLTexture` files → Claude uses screenshot for visual inspection of rendered output
- Programmatic captures always provide: resource labels, shader function names, texture snapshots (render targets)

#### Step 4: Fix and verify

```bash
# 1. Apply fixes to source code (.metal shaders, Swift renderer)

# 2. Rebuild and screenshot
./build_and_run.sh --screenshot

# 3. Compare before/after visually (Claude reads output.png)

# 4. If still wrong, repeat from Step 1
```

The loop continues until the screenshot shows correct output or the user is satisfied.

### Label Injection

Claude should ensure every Metal object has a `.label` set in source code. Labels are preserved in `.gputrace` captures and enable resource identification from CLI.

```swift
// Add .label to every MTLBuffer, MTLTexture, MTLCommandBuffer, MTLCommandEncoder
vertexBuffer.label = "Vertex Buffer"
colorBuffer.label = "Color Buffer"
uniformBuffer.label = "Uniform Buffer"
commandBuffer.label = "Frame \(frameNumber)"
encoder.label = "Main Render Pass"
```

**Rules:**
- Use descriptive names with spaces (e.g., "Particle Buffer" not "particleBuf")
- Labels with spaces are reliably distinguishable from binary data in the `.gputrace`
- Never remove existing labels — only add missing ones
- Add labels to every object that lacks one

### Reference: gpu_trace capture usage

```bash
# List all captured resources with their labels
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace

# Read specific buffer by label (partial match works)
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --buffer "Color Output" --layout float4 --index 100

# Read compound struct (e.g., Particle = position + velocity + color)
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --buffer "Particle" --layout "float4,float4,float4" --index 0-10

# Dump summary statistics for all buffers
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --dump-all

# Output as JSON
cd tools/gpu_trace && uv run gpu_trace capture capture.gputrace --buffer "Color Output" --layout float4 --index 100 --json
```

#### Supported layout types

| Layout | Bytes | Example |
|--------|-------|---------|
| `float` | 4 | Scalar energy, distance, time |
| `float2` | 8 | UV coordinates |
| `float3` | 12 | Position, normal (without padding) |
| `float4` | 16 | RGBA color, position+mass, SIMD4 |
| `uint32` | 4 | Index, count |
| `int32` | 4 | Signed integer |
| `half4` | 8 | Half-precision RGBA |
| `float4,float4,float4` | 48 | Compound struct (e.g., Particle) |

### Reference: .gputrace internal structure

| File | Format | Contents |
|------|--------|----------|
| `MTLBuffer-{id}-{snap}` | Raw binary (IEEE 754 LE) | GPU buffer memory snapshot |
| `MTLTexture-{id}-{snap}` | Raw binary | GPU texture memory snapshot |
| `metadata` | Binary plist | Session UUID, API, device info |
| `device-resources-*` | MTSP binary | Resource registry with labels, shader names |
| `store0` | zlib-compressed MTSP | Encoded command buffer data |
| `capture` / `unsorted-capture` | MTSP binary | Metal API call sequences |
| `index` | Custom (`xdic` magic) | File table / hash table |

### Reference: CLI capabilities by capture type

| Capability | CLI (programmatic capture) | CLI (Xcode-initiated capture) | Xcode GUI |
|-----------|---------------------------|------------------------------|-----------|
| List resources with labels | ✅ | ✅ | ✅ |
| List shader function names | ✅ | ✅ | ✅ |
| Read texture/render target data | ✅ (MTLTexture files present) | ✅ | ✅ |
| Read buffer contents at index | ❌ (no MTLBuffer files) | ✅ | ✅ |
| Buffer statistics (min/max/mean) | ❌ | ✅ | ❌ (manual) |
| Draw call stepping | ❌ | ❌ | ✅ |
| Pixel history | ❌ | ❌ | ✅ |
| Shader step-through | ❌ | ❌ | ✅ |
| Pipeline state inspection | ❌ | ❌ | ✅ |
| Dependency viewer | ❌ | ❌ | ✅ |
| Command buffer replay | ❌ | ❌ | ✅ |

### Reference: Screenshot capture techniques

#### Method 1: In-app auto-screenshot (Preferred)

The most reliable approach is to add a `--screenshot` mode to the app itself. This renders a few frames, reads back the framebuffer texture, and saves it as a PNG — no external tools required.

**Swift code to add to your Metal app:**

```swift
// Add to your renderer class:
static func saveTexture(_ texture: MTLTexture, to path: String) {
    let w = texture.width, h = texture.height
    let bytesPerRow = w * 4
    var pixels = [UInt8](repeating: 0, count: h * bytesPerRow)
    texture.getBytes(&pixels, bytesPerRow: bytesPerRow,
                     from: MTLRegion(origin: MTLOrigin(), size: MTLSize(width: w, height: h, depth: 1)),
                     mipmapLevel: 0)
    // BGRA → RGBA, force alpha to 255 for screenshot visibility
    for i in stride(from: 0, to: pixels.count, by: 4) {
        let tmp = pixels[i]
        pixels[i] = pixels[i + 2]
        pixels[i + 2] = tmp
        pixels[i + 3] = 255
    }
    let rep = NSBitmapImageRep(bitmapDataPlanes: nil, pixelsWide: w, pixelsHigh: h,
                                bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true,
                                isPlanar: false, colorSpaceName: .deviceRGB,
                                bytesPerRow: bytesPerRow, bitsPerPixel: 32)!
    memcpy(rep.bitmapData!, &pixels, pixels.count)
    let data = rep.representation(using: .png, properties: [:])!
    try! data.write(to: URL(fileURLWithPath: path))
}
```

**Usage in the draw loop** (save after a few frames, then exit):

```swift
// In draw(in:) after cb.present(drawable) and cb.commit():
if frameCount == 3 && screenshotMode {
    cb.waitUntilCompleted()
    Self.saveTexture(drawable.texture, to: "./output.png")
    print("Screenshot saved: ./output.png")
    DispatchQueue.main.async { NSApp.terminate(nil) }
}
```

**Important**: Set `metalView.framebufferOnly = false` before rendering to allow texture readback.

#### Method 2: macOS screencapture (Fallback)

If the app doesn't support `--screenshot`, use macOS `screencapture` to capture the window:

```bash
# Launch the app in background
./your_app &
APP_PID=$!
sleep 1

# Capture the app's window (-w = frontmost window, -x = no sound)
screencapture -w -x /tmp/screenshot.png

# Stop the app
kill $APP_PID 2>/dev/null
```

## 7. Metal Shader Compilation

Compile, validate, and inspect Metal shaders from the command line.

### Compile shaders

```bash
# Compile .metal to intermediate representation (.air)
xcrun -sdk macosx metal -c MyShader.metal -o MyShader.air

# With warnings and debug info
xcrun -sdk macosx metal -c -gline-tables-only -Weverything MyShader.metal -o MyShader.air

# Archive
xcrun -sdk macosx metal-ar rcs MyShader.metalar MyShader.air

# Link into a Metal library (.metallib)
xcrun -sdk macosx metallib MyShader.metalar -o MyShader.metallib

# One-step compile + link
xcrun -sdk macosx metal MyShader.metal -o MyShader.metallib
```

### Cross-compile for specific targets

```bash
# List available GPU architectures
xcrun metal-arch

# Target specific platform
xcrun -sdk iphoneos metal -c Shader.metal -o Shader.air
xcrun -sdk macosx metal -c -target air64-apple-macos14 Shader.metal -o Shader.air
```

### HLSL to Metal (via shader converter)

```bash
# Convert HLSL → DXIL → Metal IR
dxc shaders.hlsl -T vs_6_0 -E MainVS -Fo vertex.dxil
metal-shaderconverter vertex.dxil -o vertex.metallib
```

### Validate shaders without running

```bash
# Compile with all warnings — catches issues at build time
xcrun -sdk macosx metal -c -Weverything -Werror MyShader.metal -o /dev/null

# Check Metal version compatibility
xcrun -sdk macosx metal -c -std=metal3.0 MyShader.metal -o /dev/null
```

## 8. GPU Device Information

```bash
# System-level GPU info
system_profiler SPDisplaysDataType

# Metal feature set / GPU family (grep for relevant info)
system_profiler SPDisplaysDataType | grep -E "Chipset|Metal|VRAM|GPU"
```

## 9. Debugging Recipes

### Recipe: Profile GPU performance

```bash
# 1. Record a Metal System Trace
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 10s \
  --output perf.trace \
  --launch -- /path/to/app

# 2. See what data is available
xcrun xctrace export --input perf.trace --toc

# 3. Export GPU driver events
xcrun xctrace export --input perf.trace \
  --output gpu_events.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# 4. Export GPU counters (if available)
xcrun xctrace export --input perf.trace \
  --output gpu_counters.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="gpu-counter-intervals"]'

# 5. Open in Instruments for visual analysis
open perf.trace
```

### Recipe: Find shader compilation errors

```bash
# 1. Compile with maximum warnings
xcrun -sdk macosx metal -c -Weverything -Werror MyShader.metal -o /dev/null 2>&1

# 2. If it compiles, check at runtime with validation
MTL_SHADER_VALIDATION=1 /path/to/app 2>&1 | tee shader_errors.log

# 3. Check logs for shader validation errors
log show --predicate 'subsystem == "com.apple.Metal"' --last 5m --level error
```

### Recipe: Detect Metal API misuse

```bash
# 1. Run with API validation
MTL_DEBUG_LAYER=1 /path/to/app 2>&1 | tee api_errors.log

# 2. Record with validation for post-mortem analysis
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MTL_DEBUG_LAYER=1 \
  --time-limit 10s \
  --output validated.trace \
  --launch -- /path/to/app

# 3. Check for errors
grep -i "error\|warning\|invalid\|violation" api_errors.log
```

### Recipe: Monitor real-time performance

```bash
# 1. Enable HUD with logging
MTL_HUD_ENABLED=1 MTL_HUD_LOGGING_ENABLED=1 /path/to/app &
APP_PID=$!

# 2. Stream performance data
log stream --predicate 'subsystem == "com.apple.Metal" AND category == "HUD"' &

# 3. When done, stop
kill $APP_PID
```

### Recipe: Capture a frame for Xcode debugging

```bash
# For Vulkan apps (via MoltenVK)
METAL_CAPTURE_ENABLED=1 \
MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=2 \
MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=/tmp/frame.gputrace \
/path/to/vulkan/app

# Open in Xcode Metal Debugger
open /tmp/frame.gputrace
```

### Recipe: Debug Metal rendering issues (visual + data)

Use the **Autonomous Metal Debugging Workflow** in Section 6. It covers screenshot capture, .gputrace analysis, source code reading, shader compilation, and the fix/verify loop — with fallbacks when buffer data is unavailable.

### Recipe: Compare performance before/after a change

```bash
# 1. Record baseline
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 10s \
  --output before.trace \
  --launch -- /path/to/app

# 2. Make your change, rebuild

# 3. Record after
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 10s \
  --output after.trace \
  --launch -- /path/to/app

# 4. Export both and compare
xcrun xctrace export --input before.trace --output before.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'
xcrun xctrace export --input after.trace --output after.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# 5. Diff the XML or parse with Python for numerical comparison
```

## 10. Output Size Management

Metal System Traces can be very large. Follow these rules:

1. **Use `--time-limit`**: Always limit recording duration (5-10s is usually enough)
2. **Export specific schemas**: Use `--xpath` to extract only the data you need
3. **Use `--toc` first**: Understand what's in the trace before bulk-exporting
4. **Pipe through filters**: Use `grep`, `xmllint`, or Python to extract specific data
5. **Clean up traces**: `.trace` bundles can be 100MB+; delete when done

## 11. Limitations vs RenderDoc

| Capability | Metal Skill | RenderDoc Skill |
|-----------|-------------|-----------------|
| Frame capture | ✅ .gputrace | ✅ .rdc (full CLI) |
| GPU profiling | ✅ xctrace + export | ⚠️ GPU counters if available |
| Shader compilation | ✅ xcrun metal | N/A (different workflow) |
| Validation errors | ✅ env vars + log stream | ✅ API validation layer |
| Performance HUD | ✅ MTL_HUD_ENABLED | N/A |
| Buffer inspection (CLI) | ⚠️ Xcode-initiated captures only | ✅ rdc buffer |
| Buffer statistics | ⚠️ Xcode-initiated captures only | ❌ |
| Draw call inspection | ❌ Xcode only | ✅ rdc draws/pipeline |
| Pixel history | ❌ Xcode only | ✅ rdc pixel |
| Shader step-through | ❌ Xcode only | ✅ rdc debug pixel |
| Render target export | ❌ Xcode only | ✅ rdc rt → PNG |
| Texture readback (CLI) | ⚠️ MTLTexture files (if present) | ✅ Full rdc-cli API |
| Visual output inspection | ✅ in-app saveTexture or screencapture | ❌ (headless only) |

**Key difference**: Metal debugging is split between CLI (profiling/validation/buffer inspection) and GUI (Xcode for draw calls, pixel history, shader debugging). The `gpu_trace capture` command partially bridges this gap by enabling CLI buffer/texture data inspection from `.gputrace` captures. RenderDoc still provides more complete CLI access.

## Command Reference

For the complete xctrace command reference, see [references/xctrace-quick-ref.md](references/xctrace-quick-ref.md).

For extended debugging recipes, see [references/debugging-recipes.md](references/debugging-recipes.md).
