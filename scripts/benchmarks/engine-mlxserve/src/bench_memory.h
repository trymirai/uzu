#ifndef BENCH_MLXSERVE_MEMORY_H
#define BENCH_MLXSERVE_MEMORY_H

#include <stdint.h>

typedef struct {
    uint64_t phys_footprint;
    uint64_t resident_size;
    uint64_t graphics_total;
} bench_memory_snapshot;

int bench_memory_collect(bench_memory_snapshot* snapshot);

#endif
