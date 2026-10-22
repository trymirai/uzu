# Benchmark instructions

## Code Review Rules

### Benchmark model parity

These files define the same benchmark request and response models. Paths are
relative to this directory:

- `common-cpp/src/bench.hpp`
- `common-py/src/bench.py`
- `engine-mlxserve/src/bench.zig`
- `engine-uzu/src/bench.rs`

When changing or reviewing any of these files, read all four, including files
absent from the diff. Keep `ChatMessage`, `BenchSampling`, `BenchRequest`,
`BenchResponse`, and any new shared models consistent across the implementations.

- Flag additions, removals, or renames of shared models or fields that are not
  reflected in the other implementations.
- Compare equivalent field types, nested models, collection element types,
  optionality, defaults, and serialized field names. Account for each language's
  native types rather than requiring identical syntax.
- Check changes to numeric ranges, enum constraints, and omission versus JSON
  `null` for incompatible behavior across implementations.
- Helper methods, properties such as Python's `BenchRequest.prompt`, derives,
  and language-specific implementation details do not need counterparts unless
  they change the serialized model or accepted inputs.
- Report mismatches introduced or worsened by the reviewed change, identifying
  the model, field, and counterpart files that need updating. Do not report an
  existing difference as a new regression.
