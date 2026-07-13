# Asynchronous Staging

`AsyncStager` owns the shared mechanics for disk-to-GPU-visible staging:

1. Reserve a bounded slot and assign a monotonic generation.
2. Enqueue a request for the worker.
3. Load into the slot and signal `ready[generation]`.
4. Encode a GPU wait immediately before the lookup.
5. Signal `consumed[generation]` after the lookup.
6. Complete the command buffer and release the slot.

`RowRing` is the decode adapter. It copies a sampled token into a mailbox and
the loader resolves that mailbox after the GPU sample event. `PrefillRing` is
the host-known batch adapter. It submits expanded row batches and supplies the
fixed index buffer used by the embedding lookup.

The stager does not know about PLE, vocabularies, transformer layers, or file
offsets. The loader owns file access and row layout. This leaves the same core
usable for future MoE expert or dense weight requests.

The checked-in implementation remains explicitly ticket-completed rather than
recycling slots from `Drop`. This keeps I/O errors visible and avoids hidden
fallible waits during stream destruction.

## Validation

Characterization tests run before the refactor: `8 passed`. After consolidation,
the shared stager tests cover ordering, generations, consumed waits, short
reads, zero-fill, cancellation, and worker shutdown: `4 passed`.

The full backend library suite passes with `518 passed, 10 ignored`.

On the local Apple Silicon Gemma4 run, the refactor compared with the checked-in
implementation using the same 10-run task:

- Decode offload throughput: `95.63 -> 95.69` tokens/s.
- Decode offload TTFT: `63.77 -> 63.80` ms.
- Prefill offload TTFT: `136.25 -> 136.59` ms.
- Prefill prompt throughput: `7577.26 -> 7549.14` tokens/s.
- Exact output matched in every comparison.
- Resident PLE memory reduction remained about `4.66 GB`.

The prefill comparison is within the measurement gate; the small prompt-throughput
change is below 0.4% against the checked-in implementation.
