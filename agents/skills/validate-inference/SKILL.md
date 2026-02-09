---
name: validate-inference
description: Validate inference generates correct text
---

Using tests to validate inference is correct and performance has not degraded.

## Instructions

1. Setup model you want to test in ./crates/uzu/tests/text_session_test.rs or ./crates/uzu/tests/mod.rs

2. Remember that some models can be too large to fit in device RAM. Do not use these models and do not attempt to download them.

3. Run
   
   ```
   cargo test --release --package uzu --test text_session_test -- test_text_session_base --exact --nocapture
   ```

4. Check generated text is correct and readable and stats (processed_tokens_per_second for prefill and tokens_per_second for generate)