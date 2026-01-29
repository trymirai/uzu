---
name: benchmark-model
description: Benchmark inference performance for a specific model
---

Using CLI to perform benchmark.

## Instructions

1. Choose model you want to use from ./models/{ENGINE_VERSION}/{MODEL_NAME} or download it

2. Ensure benchmark_task.json exists in the model folder (it is automatically generated after model download)

3. Run
   ```
   cargo run --release -p cli -- bench ./models/{ENGINE_VERSION}/{MODEL_NAME} ./models/{ENGINE_VERSION}/{MODEL_NAME}/benchmark_task.json {RESULT_PATH}
   ```

4. Check result json to check memory_used, time_to_first_token, prompt_tokens_per_second and generate_tokens_per_second. Optionally check text is readable.
