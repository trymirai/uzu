---
name: architecture
description: Description of the main concepts of the inference engine
---

Read this carefully before doing any changes:

## Architecture

1. uzu runs models converted via https://github.com/trymirai/lalamo toolkit in unified format.

2. Session. High-level object used to perform requests to the model. It supports multiple modes (ignore previous context, use static context to perform requests, and dynamically aggregate context). It creates a generator object which implements the decoding loop, manually calls prefill and generate steps, and collects performance stats. Decoding can be a fully GPU-driven pipeline or can use CPU sync after each token when needed (speculative and constrained decoding).

3. Generator. Object that implements forward pass execution for prefill and decode stages. It also implements async decoding loop logic. It creates all necessary buffers, loads weights into memory, prepares masks, speculative suffixes, and grammar masks if needed. For sync loop it also implements preencode logic.

4. Encodables. High-level objects that implement kernel selection. Each encodable has an encode method and depends on the corresponding config object from the model's config.json.

5. Kernels. Backend-specific shaders that use DSL from ./crates/uzu/build to automatically generate Rust interfaces. Each kernel should have a corresponding unit test that validates correctness.

## What is important?

1. Performance is everything. All changes should be as efficient as possible in terms of prefill/decode speed and memory usage.

2. Keep in mind that this is an SDK that will be used by a wide range of developers, so excellent developer experience is crucial.

3. You can explore various docs here: https://docs.trymirai.com