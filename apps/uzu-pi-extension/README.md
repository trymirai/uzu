# Uzu for Pi

Run local language models in Pi with uzu engine.

Requires Apple Silicon macOS, Pi 1.1 or later, and Node.js 22.19.0 or newer.

## Install

```sh
pi install npm:@trymirai/uzu-pi-extension
```

## Use

Open `/model` and select a model under the `uzu` provider. The picker shows download
status in the model details. Qwen3.8 27B Mirai Medium Quantization is recommended.

Uzu manages downloaded models in its own cache. To use manually converted models, set
`LOCAL_PATH` to a model directory or a directory containing models before starting
Pi.
