#pragma once

#include <metal_stdlib>

using namespace metal;

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
