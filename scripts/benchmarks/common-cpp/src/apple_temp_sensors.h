#ifndef __benchmarks_apple_temp_sensors_h__
#define __benchmarks_apple_temp_sensors_h__

#include <mach/kern_return.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    double cpu_avg;
    double gpu_avg;
} apple_temp_sensors_t;

kern_return_t get_temp_sensors(apple_temp_sensors_t* out);

#ifdef __cplusplus
}
#endif

#endif  // __benchmarks_apple_temp_sensors_h__
