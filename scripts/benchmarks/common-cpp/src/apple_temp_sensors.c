#include "apple_temp_sensors.h"

#include <IOKit/IOKitLib.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/sysctl.h>

#if !defined(__APPLE__) || !defined(__arm64__)
#error This implementation targets macOS on Apple Silicon.
#endif

enum { MAX_SENSORS = 32 };

// clang-format off
static const char* const CPU_M1[] = {
    "Tp09", "Tp0T", "Tp01", "Tp05", "Tp0D", "Tp0H", "Tp0L", "Tp0P", "Tp0X", "Tp0b", NULL
};

static const char* const GPU_M1[] = {
    "Tg05", "Tg0D", "Tg0L", "Tg0T", NULL
};

static const char* const CPU_M2[] = {
    "Tp1h", "Tp1t", "Tp1p", "Tp1l", "Tp01", "Tp05", "Tp09", "Tp0D", "Tp0X", "Tp0b", "Tp0f", "Tp0j", NULL
};

static const char* const GPU_M2[] = {
    "Tg0f", "Tg0j", NULL
};

static const char* const CPU_M3[] = {
    "Te05", "Te0L", "Te0P", "Te0S", "Tf04", "Tf09", "Tf0A", "Tf0B", "Tf0D", "Tf0E", "Tf44", "Tf49", "Tf4A", "Tf4B", "Tf4D", "Tf4E", NULL
};

static const char* const GPU_M3[] = {
    "Tf14", "Tf18", "Tf19", "Tf1A", "Tf24", "Tf28", "Tf29", "Tf2A", NULL
};

static const char* const CPU_M4[] = {
    "Te05", "Te0S", "Te09", "Te0H", "Tp01", "Tp05", "Tp09", "Tp0D", "Tp0V", "Tp0Y", "Tp0b", "Tp0e", NULL
};

static const char* const GPU_M4[] = {
    "Tg0G", "Tg0H", "Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k", NULL
};

static const char* const GPU_M4_PRO_MAX[] = {
    "Tg1U", "Tg1k", "Tg0K", "Tg0L", "Tg0d", "Tg0e", "Tg0j", "Tg0k", NULL
};

static const char* const CPU_M5[] = {
    "Tp00", "Tp04", "Tp08", "Tp0C", "Tp0G", "Tp0K", "Tp0O", "Tp0R", "Tp0U", "Tp0X", "Tp0a", "Tp0d", "Tp0g", "Tp0j", "Tp0m", "Tp0p", "Tp0u", "Tp0y", NULL
};

static const char* const GPU_M5[] = {
    "Tg0U", "Tg0X", "Tg0d", "Tg0g", "Tg0j", "Tg1Y", "Tg1c", "Tg1g", NULL
};

static const char* const CPU_M6[] = {
    "Te07", "Te08", "Te09", "Te0c", "Te0e", "Te0f", "Te0g", "Te0h", "Te0i", "Tp05", "Tp07", "Tp09", "Tp0A", "Tp0E", "Tp0G", "Tp0I", "Tp0L", "Tp0b", "Tp0d", "Tp0g", "Tp0j", "Tp0m", "Tp0o", "Tp0r", "Tp0t", "Tp0u", "Tp0w", "Tp0x", "Tp0y", "Tp0z", NULL
};

static const char* const GPU_M6[] = {
    "Tg01", "Tg07", "Tg08", "Tg09", "Tg0a", "Tg0b", "Tg0c", "Tg0d", "Tg10", "Tg17", "Tg1a", "Tg1b", "Tg1c", "Tg1d", "Tg1e", "Tg1f", "Tg1g", "Tg1h", NULL
};
// clang-format on

// Fixed AppleSMC user-client ABI. Keep natural alignment, not packed.
typedef struct {
    uint32_t key;
    uint8_t version[6];
    uint8_t version_padding[2];
    uint16_t limit_version, limit_length;
    uint32_t cpu_limit, gpu_limit, memory_limit;
    uint32_t size, type;
    uint8_t attributes;
    uint8_t key_info_padding[3];
    uint8_t result, status, command;
    uint32_t index;
    uint8_t bytes[32];
} smc_packet_t;

typedef struct {
    uint32_t key;
    uint32_t size;
    uint32_t type;
} sensor_t;

typedef struct {
    io_connect_t connection;
    sensor_t cpu[MAX_SENSORS], gpu[MAX_SENSORS];
    unsigned cpu_count, gpu_count;
} apple_temps_t;

static uint32_t fourcc(const char* s) {
    uint32_t s0 = (uint32_t)(uint8_t)s[0] << 24;
    uint32_t s1 = (uint32_t)(uint8_t)s[1] << 16;
    uint32_t s2 = (uint32_t)(uint8_t)s[2] << 8;
    uint8_t s3 = (uint8_t)s[3];
    return s0 | s1 | s2 | s3;
}

static kern_return_t exchange(
    io_connect_t connection,
    const smc_packet_t* in,
    smc_packet_t* out
) {
    memset(out, 0, sizeof(*out));
    size_t size = sizeof(*out);
    kern_return_t error = IOConnectCallStructMethod(connection, 2, in, sizeof(*in), out, &size);
    if (error != KERN_SUCCESS) {
        return error;
    }
    if (size != sizeof(*out)) {
        return kIOReturnUnderrun;
    }
    if (out->result != 0) {
        return kIOReturnError;
    }
    return KERN_SUCCESS;
}

static unsigned discover(
    io_connect_t c,
    const char* const* keys,
    sensor_t* dest
) {
    size_t count = 0;
    for (; *keys && count < MAX_SENSORS; ++keys) {
        smc_packet_t query = {.key = fourcc(*keys), .command = 9}, reply;
        if (exchange(c, &query, &reply) != KERN_SUCCESS) {
            continue;
        }
        if (!((reply.type == fourcc("flt ") && reply.size == 4) || (reply.type == fourcc("sp78") && reply.size == 2))) {
            continue;
        }
        dest[count++] = (sensor_t){query.key, reply.size, reply.type};
    }
    return count;
}

static double average(
    io_connect_t c,
    const sensor_t* s,
    unsigned count
) {
    double sum = 0;
    size_t valid = 0;
    for (size_t i = 0; i < count; ++i) {
        smc_packet_t query = {
            .key = s[i].key,
            .size = s[i].size,
            .command = 5,
        };
        smc_packet_t reply;
        if (exchange(c, &query, &reply) != KERN_SUCCESS) {
            continue;
        }

        double value;
        if (s[i].type == fourcc("flt ")) {
            float f;
            memcpy(&f, reply.bytes, sizeof(f)); /* native little-endian float */
            value = f;
        } else {
            unsigned raw = ((unsigned)reply.bytes[0] << 8) | reply.bytes[1];
            int signed_raw = raw >= 0x8000 ? (int)raw - 65536 : (int)raw;
            value = signed_raw / 256.0; /* big-endian signed 8.8 fixed point */
        }

        // Application plausibility policy: reject zero/error sentinels and outliers.
        // Adjust these bounds if your measurement conditions differ.
        if (!isfinite(value) || value <= 0 || value >= 150) {
            continue;
        }

        sum += value;
        ++valid;
    }

    return valid ? (sum / valid) : NAN;
}

static kern_return_t apple_temps_close(apple_temps_t* ctx);

static kern_return_t apple_temps_open(apple_temps_t** out) {
    if (!out) {
        return kIOReturnBadArgument;
    }

    *out = NULL;
    char model[128] = {0};
    size_t size = sizeof(model) - 1;
    if (sysctlbyname("machdep.cpu.brand_string", model, &size, NULL, 0) != 0) {
        return kIOReturnError;
    }

    unsigned generation = 0;
    int end = 0;
    if (sscanf(model, "Apple M%u%n", &generation, &end) != 1 || (model[end] && model[end] != ' ')) {
        return kIOReturnUnsupported;
    }

    const char* const* cpu = NULL;
    const char* const* gpu = NULL;
    switch (generation) {
        case 1:
            cpu = CPU_M1;
            gpu = GPU_M1;
            break;
        case 2:
            cpu = CPU_M2;
            gpu = GPU_M2;
            break;
        case 3:
            cpu = CPU_M3;
            gpu = GPU_M3;
            break;
        case 4:
            cpu = CPU_M4;
            gpu = model[end] ? GPU_M4_PRO_MAX : GPU_M4;
            break;
        case 5:
            cpu = CPU_M5;
            gpu = GPU_M5;
            break;
        case 6:
            cpu = CPU_M6;
            gpu = GPU_M6;
            break;
        default:
            return kIOReturnUnsupported;
    }

    apple_temps_t* ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        return kIOReturnNoMemory;
    }

    CFMutableDictionaryRef match = IOServiceMatching("AppleSMC");
    if (!match) {
        free(ctx);
        return kIOReturnNoMemory;
    }

    io_service_t service = IOServiceGetMatchingService(kIOMainPortDefault, match);
    if (!service) {
        free(ctx);
        return kIOReturnNotFound;
    }

    kern_return_t error = IOServiceOpen(service, mach_task_self(), 0, &ctx->connection);
    IOObjectRelease(service);
    if (error != KERN_SUCCESS) {
        free(ctx);
        return error;
    }

    ctx->cpu_count = discover(ctx->connection, cpu, ctx->cpu);
    ctx->gpu_count = discover(ctx->connection, gpu, ctx->gpu);
    if (!ctx->cpu_count && !ctx->gpu_count) {
        apple_temps_close(ctx);
        return kIOReturnNotFound;
    }

    *out = ctx;
    return KERN_SUCCESS;
}

static kern_return_t apple_temps_close(apple_temps_t* ctx) {
    kern_return_t err = IOServiceClose(ctx->connection);
    free(ctx);
    return err;
}

static kern_return_t apple_temps_read(
    apple_temps_t* ctx,
    apple_temp_sensors_t* out
) {
    if (!out) {
        return kIOReturnBadArgument;
    }
    *out = (apple_temp_sensors_t){
        .cpu_avg = NAN,
        .gpu_avg = NAN,
    };

    if (!ctx) {
        return kIOReturnBadArgument;
    }

    out->cpu_avg = average(ctx->connection, ctx->cpu, ctx->cpu_count);
    out->gpu_avg = average(ctx->connection, ctx->gpu, ctx->gpu_count);
    return isfinite(out->cpu_avg) && isfinite(out->gpu_avg) ? KERN_SUCCESS : kIOReturnNotReady;
}

kern_return_t get_temp_sensors(apple_temp_sensors_t* out) {
    if (!out) {
        return kIOReturnBadArgument;
    }
    *out = (apple_temp_sensors_t){.cpu_avg = NAN, .gpu_avg = NAN};

    apple_temps_t* ctx = NULL;
    kern_return_t err = apple_temps_open(&ctx);
    if (err != KERN_SUCCESS) {
        return err;
    }

    err = apple_temps_read(ctx, out);
    kern_return_t close_err = apple_temps_close(ctx);

    return err != KERN_SUCCESS ? err : close_err;
}
