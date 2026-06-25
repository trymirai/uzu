#include <TargetConditionals.h>
#include <stddef.h>

void uzu_metal_io_simulator_stubs_anchor(void) {}

#if TARGET_OS_SIMULATOR

typedef void* MTLIOCompressionContext;
typedef long MTLIOCompressionMethod;
typedef long MTLIOCompressionStatus;

const void* MTLIOErrorDomain = 0;

size_t MTLIOCompressionContextDefaultChunkSize(void) { return 0; }

MTLIOCompressionContext MTLIOCreateCompressionContext(const char* path, MTLIOCompressionMethod type, size_t chunkSize) {
  (void)path;
  (void)type;
  (void)chunkSize;
  return NULL;
}

void MTLIOCompressionContextAppendData(MTLIOCompressionContext context, const void* data, size_t size) {
  (void)context;
  (void)data;
  (void)size;
}

MTLIOCompressionStatus MTLIOFlushAndDestroyCompressionContext(MTLIOCompressionContext context) {
  (void)context;
  return 1;
}

#endif
