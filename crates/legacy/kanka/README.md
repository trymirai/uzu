# kanka

Internal helper crate (甘果, "sweet fruit"): generic `dlsym`-resolved bindings to private
Apple framework functions, with obfuscated symbol names. This is the low-level plumbing
[`keisoku`](../keisoku) uses to reach the IOReport/SMC power and telemetry APIs that have no
public headers.

It contains no power logic itself — just two macros:

- `ffi_table!` — declares a struct of `dlsym`-resolved C function pointers from a named
  framework, with a `OnceLock`-cached `get()` (symbol and framework names are obfuscated at
  compile time via `obfstr`).
- `opaque_cf_type!` — declares an opaque CoreFoundation handle type usable with `CFRetained`.

## Platform

Apple only (`target_vendor = "apple"`). On other platforms the macros expand to nothing
usable and the crate is effectively empty.

You normally don't depend on this directly — use [`keisoku`](../keisoku).
