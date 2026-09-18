#include <Python.h>
#include <mach/mach.h>
#include <mach/mach_error.h>
#include <malloc/malloc.h>
#include <stdint.h>
#include <unistd.h>

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

// The caller owns the output buffer; this function never allocates or retains it.
kern_return_t get_memory_counters(memory_counters_t* counters, bool with_malloc_zone_stats) {
  if (counters == NULL) {
    return KERN_RETURN_COUNTERS_NULL;
  }

  task_vm_info_data_t memory_info = {0};
  mach_msg_type_number_t info_count = TASK_VM_INFO_COUNT;
  kern_return_t result = task_info(mach_task_self(), TASK_VM_INFO, (task_info_t)&memory_info, &info_count);
  if (result != KERN_SUCCESS) {
    return result;
  }
  if (info_count < TASK_VM_INFO_REV3_COUNT) {
    return KERN_RETURN_GRAPHICS_UNAVAILABLE;
  }

  uint64_t graphics_total =
      memory_info.ledger_tag_graphics_footprint + memory_info.ledger_tag_graphics_footprint_compressed +
      memory_info.ledger_tag_graphics_nofootprint + memory_info.ledger_tag_graphics_nofootprint_compressed;

  counters->pid = getpid();
  counters->phys_footprint = memory_info.phys_footprint;
  counters->resident_size = memory_info.resident_size;
  counters->resident_size_peak = memory_info.resident_size_peak;
  counters->device = memory_info.device;
  counters->device_peak = memory_info.device_peak;
  counters->internal = memory_info.internal;
  counters->compressed = memory_info.compressed;
  counters->graphics_footprint = memory_info.ledger_tag_graphics_footprint;
  counters->graphics_footprint_compressed = memory_info.ledger_tag_graphics_footprint_compressed;
  counters->graphics_nofootprint = memory_info.ledger_tag_graphics_nofootprint;
  counters->graphics_nofootprint_compressed = memory_info.ledger_tag_graphics_nofootprint_compressed;
  counters->graphics_total = graphics_total;

  if (with_malloc_zone_stats) {
    malloc_statistics_t malloc_stats = {};
    malloc_zone_statistics(NULL, &malloc_stats);
    counters->malloc_allocated = (uint64_t)malloc_stats.size_allocated;
    counters->malloc_in_use = (uint64_t)malloc_stats.size_in_use;
    counters->malloc_max_in_use = (uint64_t)malloc_stats.max_size_in_use;
  }

  return KERN_SUCCESS;
}

const char* memory_counters_error_string(kern_return_t result) {
  switch (result) {
  case KERN_RETURN_COUNTERS_NULL:
    return "Counters pointer cannot be NULL.";
  case KERN_RETURN_GRAPHICS_UNAVAILABLE:
    return "Graphics memory accounting is unavailable.";
  default:
    return mach_error_string(result);
  }
}

static struct PyModuleDef module = {
    .m_base = PyModuleDef_HEAD_INIT,
    .m_name = "mach._mach",
    .m_doc = "Native macOS Mach memory collection.",
    .m_size = 0,
};

PyMODINIT_FUNC PyInit__mach(void) { return PyModule_Create(&module); }
