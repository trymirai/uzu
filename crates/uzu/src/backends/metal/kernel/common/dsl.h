#pragma once

#include <metal_stdlib>
using namespace metal;

#if defined(__cpp_if_constexpr)
#define IF_CONSTEXPR(cond) if constexpr (cond)
#else
#define IF_CONSTEXPR(cond) if (cond)
#endif

#ifdef DSL_ANALYZE
#define DSL_META(...) [[clang::annotate("", __VA_ARGS__)]]
#else
#define DSL_META(...)
#endif

#define DSL_STR(X) #X
#define DSL_XSTR(X) DSL_STR(X)

#define VARIANTS(TYPENAME, ...)                                                \
  DSL_META("dsl.variants", #TYPENAME, #__VA_ARGS__)
#define CONSTRAINT(C) DSL_META("dsl.constraint", #C)
#define PUBLIC DSL_META("dsl.public")
#define KERNEL(NAME) DSL_META("dsl.kernel") void NAME

#define SPECIALIZE DSL_META("dsl.specialize")
#define OPTIONAL(EXPR) DSL_META("dsl.optional", DSL_XSTR(EXPR))

#define AXIS(TDS, TPG) DSL_META("dsl.axis", DSL_XSTR(TDS), DSL_XSTR(TPG))
#define GROUPS(EXPR) DSL_META("dsl.groups", DSL_XSTR(EXPR))
#define THREADS(EXPR) DSL_META("dsl.threads", DSL_XSTR(EXPR))

#define generateKernel(                                                        \
    max_threads,                                                               \
    functionName,                                                              \
    scalarType,                                                                \
    outerArgs,                                                                 \
    innerArgs                                                                  \
)                                                                              \
  [[max_total_threads_per_threadgroup(max_threads)]]                           \
  kernel void functionName##_##scalarType outerArgs {                          \
    functionName innerArgs;                                                    \
  }

#define generateKernels(max_threads, functionName)                             \
  generateKernel(                                                              \
      max_threads,                                                             \
      functionName,                                                            \
      float,                                                                   \
      outerArguments(float),                                                   \
      innerArguments                                                           \
  );                                                                           \
  generateKernel(                                                              \
      max_threads,                                                             \
      functionName,                                                            \
      bfloat,                                                                  \
      outerArguments(bfloat),                                                  \
      innerArguments                                                           \
  );                                                                           \
  generateKernel(                                                              \
      max_threads,                                                             \
      functionName,                                                            \
      half,                                                                    \
      outerArguments(half),                                                    \
      innerArguments                                                           \
  );
