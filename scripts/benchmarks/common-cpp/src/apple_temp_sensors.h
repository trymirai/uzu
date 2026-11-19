#ifndef __benchmarks_apple_temp_sensors_h__
#define __benchmarks_apple_temp_sensors_h__

#include <mach/kern_return.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double cpu_avg;
    // NaN while the GPU is powered down, when its sensors report no temperature.
    double gpu_avg;
} apple_temp_sensors_t;

kern_return_t get_temp_sensors(apple_temp_sensors_t* out);
/**
 * If the CPU or the GPU average temperature is at or above `threshold`, returns once both are below `target`.
 */
kern_return_t await_cooldown(
    double threshold,
    double target
);

#ifdef __cplusplus
}
#endif

#endif  // __benchmarks_apple_temp_sensors_h__
