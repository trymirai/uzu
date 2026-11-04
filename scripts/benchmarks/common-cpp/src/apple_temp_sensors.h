#ifndef __benchmarks_apple_temp_sensors_h__
#define __benchmarks_apple_temp_sensors_h__

#include <mach/kern_return.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double cpu_avg;
    double gpu_avg;
} apple_temp_sensors_t;

kern_return_t get_temp_sensors(apple_temp_sensors_t* out);
/**
 * Returns once the CPU and GPU average temperatures are below `temperature`. With `only_if_capped`, it waits only
 * if the thermal manager capped the GPU since the previous such call, of which the first has nothing to compare
 * with and waits.
 */
kern_return_t await_cooldown(
    double temperature,
    bool only_if_capped
);

#ifdef __cplusplus
}
#endif

#endif  // __benchmarks_apple_temp_sensors_h__
