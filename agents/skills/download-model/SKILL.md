---
name: download-model
description: Download model to test inference engine
---

Using ./tools/helpers Python package to download model

## Instructions

1. Go to helpers folder:
   ```
   cd ./tools/helpers
   ```

2. Run:
   ```
   uv run main.py download-model {REPO_ID}
   ```

3. Downloaded model will be in ./models/{ENGINE_VERSION}/{MODEL_NAME} folder
