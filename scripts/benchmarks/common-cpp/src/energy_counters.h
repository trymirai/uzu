#ifndef __benchmarks_energy_counters_h__
#define __benchmarks_energy_counters_h__

#include <mach/kern_return.h>

#ifdef __cplusplus
extern "C" {
#endif

/** Cumulative GPU energy in joules. */
kern_return_t get_gpu_energy(double* joules);

#ifdef __cplusplus
}
#endif

#endif  // __benchmarks_energy_counters_h__
