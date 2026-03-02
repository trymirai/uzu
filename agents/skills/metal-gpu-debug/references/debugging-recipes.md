# Metal GPU Debugging Recipes

Extended debugging workflows with expected outputs and step-by-step procedures.

## Recipe 1: GPU Performance Profiling

**Symptoms**: Low FPS, high frame time, stuttering, thermal throttling.

### Step-by-step

```bash
# 1. Record and analyze in one step (with hardware counters)
cd tools/gpu_trace && uv run gpu_trace run --gpu-counters --time-limit 10s -- /path/to/app

# Or analyze a pre-recorded trace:
cd tools/gpu_trace && uv run gpu_trace analyze perf.trace -v

# Export to JSON for custom analysis:
cd tools/gpu_trace && uv run gpu_trace analyze perf.trace --json perf.json

# Open in Instruments for visual timeline
open perf.trace
```

`gpu_trace` automatically exports all schemas in parallel, parses the XML, computes per-kernel timing, correlates hardware counters, and detects bottlenecks.

**Performance red flags:**
- GPU busy time > 16ms per frame (60fps target) or > 8ms (120fps target)
- Wire memory events with large sizes (indicates heavy resource allocation per frame)
- Shader execution intervals dominating GPU timeline (complex shaders)
- Gaps between GPU submissions (CPU bottleneck)

---

## Recipe 2: Shader Compilation Errors

**Symptoms**: App crashes on launch, black screen, missing effects.

### Step-by-step

```bash
# 1. Compile shader with maximum diagnostics
xcrun -sdk macosx metal -c \
  -gline-tables-only \
  -Weverything \
  -Werror \
  MyShader.metal -o /dev/null 2>&1
```

**Expected error output:**
```
MyShader.metal:42:15: error: use of undeclared identifier 'baseColor'
    float4 c = baseColor * albedo;
               ^
MyShader.metal:18:3: warning: unused variable 'temp' [-Wunused-variable]
  float temp = 0.0;
  ^
```

```bash
# 2. If compile succeeds but runtime fails, check Metal version
xcrun -sdk macosx metal -c -std=metal3.0 MyShader.metal -o /dev/null 2>&1

# 3. Check for runtime shader errors
MTL_SHADER_VALIDATION=1 /path/to/app 2>&1 | head -100

# 4. Stream shader validation errors
log stream --predicate 'subsystem == "com.apple.Metal"' --level error &
MTL_SHADER_VALIDATION=1 /path/to/app

# 5. Build the full pipeline: .metal → .air → .metalar → .metallib
xcrun -sdk macosx metal -c -gline-tables-only MyShader.metal -o MyShader.air
xcrun -sdk macosx metal-ar rcs MyShader.metalar MyShader.air
xcrun -sdk macosx metallib MyShader.metalar -o MyShader.metallib
echo "Shader library built successfully: MyShader.metallib"
```

**Common shader errors:**
| Error | Cause | Fix |
|-------|-------|-----|
| undeclared identifier | Missing variable/function | Check spelling, includes, scope |
| no matching function | Wrong argument types | Verify types match function signature |
| cannot convert | Type mismatch | Use explicit casts (float4, half4, etc.) |
| address space mismatch | Wrong buffer qualifier | Check device/constant/threadgroup qualifiers |
| exceeds max total threads | Threadgroup size too large | Reduce threadgroup dimensions |

---

## Recipe 3: Metal API Validation Errors

**Symptoms**: Crashes, undefined behavior, incorrect rendering.

### Step-by-step

```bash
# 1. Run with full API validation
MTL_DEBUG_LAYER=1 /path/to/app 2>&1 | tee validation.log

# 2. Check for errors
grep -i "error\|warning\|invalid\|violation\|failed" validation.log

# 3. For deeper validation, add shader validation too
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 /path/to/app 2>&1 | tee full_validation.log

# 4. Stream structured Metal log data
log stream --predicate 'subsystem == "com.apple.Metal"' --level error
```

**Expected validation error format:**
```
-[MTLDebugRenderCommandEncoder setVertexBuffer:offset:atIndex:]:
  Execution of the command buffer was aborted due to an error during execution.
  Caused GPU Address Fault (0x00000001)
```

**Common validation issues:**
| Error Pattern | Meaning | Fix |
|--------------|---------|-----|
| GPU Address Fault | Out-of-bounds buffer access | Check buffer sizes, offsets |
| Shader Validation Error | GPU-side out-of-bounds or race | Check array indices, threadgroup sync |
| Invalid Resource | Using deleted/nil resource | Check resource lifetimes |
| Incompatible Pixel Format | Format mismatch | Verify texture formats match pipeline |
| Exceeded max | Resource limit hit | Reduce resource usage |

---

## Recipe 4: Real-Time Performance Monitoring

**Symptoms**: Need ongoing performance data during development.

### Step-by-step

```bash
# 1. Launch app with HUD and logging
MTL_HUD_ENABLED=1 MTL_HUD_LOGGING_ENABLED=1 /path/to/app &
APP_PID=$!

# 2. Collect HUD data for analysis
log show --predicate 'subsystem == "com.apple.Metal" AND category == "HUD"' \
  --last 30s > hud_data.log

# 3. Parse HUD log for key metrics
python3 << 'EOF'
import re

with open('hud_data.log') as f:
    for line in f:
        # Extract FPS, GPU time, memory from HUD log entries
        if 'FPS' in line or 'GPU' in line or 'Memory' in line:
            print(line.strip())
EOF

# 4. When done
kill $APP_PID
```

**HUD overlay shows:**
- FPS (frames per second)
- Present interval / frame time (ms)
- GPU time (ms)
- Process memory (MB)
- GPU memory (MB)
- Display refresh rate
- Rendering path (direct vs composited)

---

## Recipe 5: Frame Capture for Xcode Debugging

**Symptoms**: Visual bugs that need draw-call-level inspection.

### For native Metal apps

```bash
# 1. Ensure capture is enabled (one of):
#    a) Set METAL_CAPTURE_ENABLED=1 environment variable
#    b) Add MetalCaptureEnabled=true to Info.plist
#    c) Use MTLCaptureManager in code

# 2. For MoltenVK / Vulkan apps:
METAL_CAPTURE_ENABLED=1 \
MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=2 \
MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=/tmp/capture.gputrace \
/path/to/vulkan/app

# 3. Wait for capture to complete (app will likely freeze briefly)
echo "Capture saved to /tmp/capture.gputrace"

# 4. Open in Xcode
open /tmp/capture.gputrace
```

**In Xcode Metal Debugger you can:**
- Step through draw calls
- Inspect pipeline state at any draw
- View bound textures and buffers
- Debug shaders line-by-line
- Check pixel history
- View GPU timeline
- Inspect acceleration structures (ray tracing)
- Analyze memory usage

---

## Recipe 6: Compare Performance Across Changes

**Symptoms**: Need to verify a change improved or didn't regress performance.

### Step-by-step

```bash
# 1. Record baseline
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 10s \
  --no-prompt \
  --output baseline.trace \
  --launch -- /path/to/app_before

# 2. Record after changes
xcrun xctrace record \
  --template 'Metal System Trace' \
  --time-limit 10s \
  --no-prompt \
  --output changed.trace \
  --launch -- /path/to/app_after

# 3. Export both
for trace in baseline changed; do
  xcrun xctrace export \
    --input ${trace}.trace \
    --output ${trace}_events.xml \
    --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'
done

# 4. Compare with Python
python3 << 'EOF'
import xml.etree.ElementTree as ET

def count_events(xmlfile):
    tree = ET.parse(xmlfile)
    rows = tree.getroot().findall('.//row')
    return len(rows)

baseline = count_events('baseline_events.xml')
changed = count_events('changed_events.xml')

print(f"Baseline: {baseline} events")
print(f"Changed:  {changed} events")
print(f"Delta:    {changed - baseline:+d} events ({(changed/baseline - 1)*100:+.1f}%)")
EOF
```

---

## Recipe 7: Debug Vulkan App on macOS via MoltenVK

**Symptoms**: Vulkan app has issues only on macOS.

### Step-by-step

```bash
# 1. Enable MoltenVK logging
export MVK_CONFIG_LOG_LEVEL=3  # 0=off, 1=error, 2=warning, 3=info, 4=debug

# 2. Run with Metal validation
MTL_DEBUG_LAYER=1 \
MTL_SHADER_VALIDATION=1 \
MVK_CONFIG_LOG_LEVEL=3 \
/path/to/vulkan/app 2>&1 | tee moltenvk.log

# 3. Capture a frame for Xcode debugging
METAL_CAPTURE_ENABLED=1 \
MVK_CONFIG_AUTO_GPU_CAPTURE_SCOPE=2 \
MVK_CONFIG_AUTO_GPU_CAPTURE_OUTPUT_FILE=/tmp/vk_capture.gputrace \
/path/to/vulkan/app

# 4. Profile with xctrace
xcrun xctrace record \
  --template 'Metal System Trace' \
  --env MVK_CONFIG_LOG_LEVEL=3 \
  --time-limit 10s \
  --output vk_trace.trace \
  --launch -- /path/to/vulkan/app

# 5. Open trace or capture
open /tmp/vk_capture.gputrace  # For state debugging
open vk_trace.trace            # For profiling
```

---

## Recipe 8: Profile an iOS Device

**Symptoms**: Need to profile Metal performance on a physical iPhone/iPad.

### Step-by-step

```bash
# 1. List connected devices
xcrun xctrace list devices
# Look for your device name and UDID

# 2. Record (app must be running or installed on device)
xcrun xctrace record \
  --template 'Metal System Trace' \
  --device 'iPhone 15 Pro' \
  --attach MyApp \
  --time-limit 10s \
  --output ios_profile.trace

# 3. Export and analyze — same as macOS
xcrun xctrace export --input ios_profile.trace --toc

xcrun xctrace export --input ios_profile.trace \
  --output ios_gpu_events.xml \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]'

# 4. Parse with gpu_trace
cd tools/gpu_trace && uv run gpu_trace analyze ios_profile.trace -v

# 5. Or open in Instruments
open ios_profile.trace
```

**iOS-specific tips:**
- Device must be unlocked and trusted
- Use `--attach` with process name for already-running apps
- Use `--launch com.bundle.id` to launch and trace
- Apple Silicon iPhones (A14+) have rich GPU counters
- Thermal throttling is more common on iOS — check for performance drops over time

---

## Recipe 9: System-Wide GPU Monitoring

**Symptoms**: Need to identify which process is consuming GPU resources.

### Step-by-step

```bash
# 1. Trace all processes
xcrun xctrace record \
  --template 'Metal System Trace' \
  --all-processes \
  --time-limit 10s \
  --output system.trace

# 2. Check which processes are in the trace
xcrun xctrace export --input system.trace \
  --xpath '/trace-toc/run[@number="1"]/processes'

# 3. Export GPU events for a specific process
# (after identifying process name from step 2)
xcrun xctrace export --input system.trace \
  --xpath '/trace-toc/run[@number="1"]/data/table[@schema="metal-driver-event-intervals"]' \
  --output all_gpu.xml

# 4. Open in Instruments for per-process breakdown
open system.trace
```
