---
name: trace-model
description: Validate all intermediate calculations during inference are correct using source of truth values
---

Using traces.safetensors to validate all intermediate calculations after each layer are correct.

## Instructions

1. Setup model you want to test in ./crates/uzu/tests/tracer_test.rs or ./crates/uzu/tests/mod.rs

2. Check the model folder to ensure traces.safetensors exists (it should be automatically downloaded alongside the model)

3. Run
   
   ```
   cargo test --release --package uzu --test tracer_test -- test_tracer --exact --nocapture
   ```

4. Check results to ensure number of violations are less than the limit