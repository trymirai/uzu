#include "bench_memory.h"
#include "memory_counters.h"

// Keep Mach's packed IPC structs out of Zig's C-header translator.
int bench_memory_collect(bench_memory_snapshot* snapshot) {
    memory_counters_t counters;
    const kern_return_t result = get_memory_counters(&counters, false);
    if (result != KERN_SUCCESS) {
        return result;
    }
    snapshot->phys_footprint = counters.phys_footprint;
    snapshot->resident_size = counters.resident_size;
    snapshot->graphics_total = counters.graphics_total;
    return 0;
}
