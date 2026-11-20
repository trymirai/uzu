# Benchmarks status

Updated 2026-11-20. Dates are when something was asked for or last touched.

## Asked for, not done

1. iPhone benchmarks: prod measures two iPhones, `benchman` none. One of the most important gaps.
2. `release_summary.py` has no engine versions and no measurement date, which the website shows (11-18).
3. Result keys per engine, so that a change to uzu doesn't re-measure the other engines (11-17).
4. A failed task not ending its flow run (11-11). Its failure is stored now and the next command measures the rest, but the flow run still ends.
5. A pre-release benchmark from a branch other than `main` on GitHub (11-17).
6. `!bench`: the race between the comment and the checkout, and a pull request with nothing to compare (11-16).
7. Competition benchmarks apart from the uzu release benchmark: how they are triggered (11-17).

## Deferred on purpose

- Model handoff between flow runs, so consecutive runs on a Mac reuse a loaded model. Designed as a handoff directory.
- Energy of the whole machine without the display, not only the GPU.
- Cooldown: 80 °C to 70 °C is provisional, with a TODO in `benchman/src/engine.py`.
- tqLLM: prod added it on 11-10 and it is broken upstream.
- Autoregressive twins of speculative models.

## Waiting for a decision

- Ctrl-C: whether it should detach instead of cancel, with a TODO in `benchman/src/run.py`.
- `-rr` over tasks that a flow run is measuring: a flow run left with none stays queued until a Mac starts it, and then ends at once. Whether to cancel it instead, at least while it has not started (11-19).
- `--cancel` deletes the record, so a cancelled task shows as `Missing`. Marking the record as given up instead would let it say `Cancelled` (11-20).
- A run shows one device, though its tasks may have been measured on several Macs of a `bench-<chip>` queue. Whether differing devices are an error (11-20).
- Without models, every option but `--input` is ignored, also `--output`. Whether that is an error, and whether `-i tag:A -o tag:B` copies the tag (11-20).
- Flow run bundles are stored at the bucket's root, which the expiry rule can't cover. Storing them under `benchman/v1/store/` would (11-20).
- Repetition penalty of the six small Qwen3.5 models: without it MLX and TensorFold loop to the token limit.
- Engine error output in a failed fleet run: it is in the flow run's Prefect log now (11-20), but `benchman` itself still only says the exit code.
- TensorFold's memory with the 27B model, to finish `--filter-ram` for it.
- Moving the C++ half of `common-cpp` into `engine-llamacpp`, its only user.

## Waiting on upstream

- Prefect: `--quiet` for the `uv pip freeze` of a bundle, which `benchman/src/prefect_bundle_quiet_hack.py` adds until then (11-18).

## Waiting on infra

- The bucket's expiry rule (14 days for `store/`, 30 for `tasks/`, tags kept) is an uncommitted edit in `../services` (11-18). It names `benchman/v1/`, so a version bump needs it changed.
- The pool still uses the `experimental` bundle steps (11-10).

## Not verified

- Task records: a request failing in the middle of a flow run (11-18).
- Tags: a release-sized one, and re-measuring into a tag with `-rr` (11-18).
- Results being shown as each task ends; today they are read when the flow runs have ended (11-18).
- Both benchmark workflows, their secrets, and submitting from a Linux runner.
- The iOS app against the engines' new ready message and Splash 1.7.0.
