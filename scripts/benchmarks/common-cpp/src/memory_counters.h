#ifndef __benchmarks_memory_counters_h__
#define __benchmarks_memory_counters_h__

#include <mach/mach.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define KERN_RETURN_COUNTERS_NULL (KERN_RETURN_MAX + 1)
#define KERN_RETURN_GRAPHICS_UNAVAILABLE (KERN_RETURN_MAX + 2)

typedef struct {
    int32_t pid;
    uint64_t phys_footprint;
    uint64_t resident_size;
    uint64_t resident_size_peak;
    uint64_t device;
    uint64_t device_peak;
    uint64_t internal;
    uint64_t compressed;
    uint64_t graphics_footprint;
    uint64_t graphics_footprint_compressed;
    uint64_t graphics_nofootprint;
    uint64_t graphics_nofootprint_compressed;
    uint64_t graphics_total;
    uint64_t malloc_allocated;
    uint64_t malloc_in_use;
    uint64_t malloc_max_in_use;
} memory_counters_t;

kern_return_t get_memory_counters(
    memory_counters_t* counters,
    bool with_malloc_zone_stats
);

const char* memory_counters_error_string(kern_return_t result);

#ifdef __cplusplus
}
#endif

#endif  // __benchmarks_memory_counters_h__
