#pragma once

#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

using os_log = metal::os_log;

#else

struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};

#endif