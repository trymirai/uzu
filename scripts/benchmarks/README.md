# Inference benchmarks

`benchman` benchmarks this checkout's inference engines on this Mac or, through Prefect, on the Mac fleet. It builds each engine, runs datasets against each model, and prints a table, with one run per host, engine and model.

```bash
uv run --project benchman benchman -m LFM2.5-350M-M
uv run --project benchman benchman -h macs-fleet/bench-m5max -m Qwen3.8-27B-M LFM2.5-1.2B-Instruct-M
uv run --project benchman benchman -h macs-fleet/bench-m5{pro,max} -e uzu mlx -m LFM2.5-350M-M -d mt-bench gsm8k
```

## Selecting what to run

`--host`, `--engine`, `--model` and `--dataset` each take several values, separated by spaces or commas, or given again. Braces expand as in a shell: `bench-m{4,5}max`.

**Hosts** (`-h`, default `local`):

- `local`: this Mac, as `local-nix` where there is Nix and as `local-direct` elsewhere.
- `local-nix`: this Mac, building the engine with Nix from the same source a worker would get.
- `local-direct`: this Mac, building the engine in the checkout with its `build.sh` and running it with its `run.sh`.
- `POOL/QUEUE`: a queue of a work pool of the fleet. The run is a Prefect flow run; the source is shipped there and the worker builds the engine with Nix.
- `POOL/PATTERN`, like `macs-fleet/bench-*`: every matching queue that an online worker serves.
- `!HOST` leaves out what it names or matches: `-h 'macs-fleet/bench-*' '!macs-fleet/bench-m2'`.

What you name has to work, and what a pattern matches is left out where it can't:

- A named host has to run every named engine, and a named model has to be one that every named engine can load; otherwise the command or the run fails.
- A host or engine that a pattern matched is skipped where the engine doesn't run on the host's chip, an engine that a pattern matched is skipped for a named model its catalog lacks, and a model pattern covers what each engine's catalog has of it.
- Nothing else is left out unless you ask: with `!`, `--filter-ram`, or `spec:` and `!spec:`.

Splash is the one engine with a chip requirement: an M3 or newer.

**Engines** (`-e`, default `uzu`): `uzu`, `mlx`, `llamacpp`, `mlxserve`, `mtplx`, `omlx`, `splash`, `magnitude`, `tensorfold`, `kapitsa`. `-e '*'` is all of them, and `!ENGINE` leaves out what it names or matches.

**Models** (`-m`): what the engine can load, with `trymirai/` implied.

- A key of the engine's `engine-<engine>/models.json` runs as that entry says: these are Uzu's model names, set up to compare alike across engines.
- Any other name goes to the engine as it is, and the engine's own defaults apply: a Hugging Face repository for most engines, for Uzu a registry model newer than the catalog or a directory on the host.
- A model with `*` in it is every model of the engine's catalog that the pattern matches, and `!MODEL` leaves out what it names or matches: `-m '*' '!LFM*'`.

**Datasets** (`-d`): `[spec:|!spec:][!]NAME[/KEY][^REPEATS]`.

- `NAME` is `oneoff`, `mirai-mixture-fast`, `mirai-mixture-slow`, `mt-bench`, `humaneval`, `math500`, `gsm8k`, `aime25` or `mbpp`; `NAME/KEY` is one prompt of it. Both take `*`: `'oneoff/*-summary'`.
- `^REPEATS` runs each prompt that many times.
- `spec:` keeps the selection to models that decode speculatively, `!spec:` to those that don't.
- `!` after that leaves prompts out: `'!spec:!oneoff/ai-*'` leaves them out for the models that don't decode speculatively.
- The default is `spec:mirai-mixture-fast !spec:oneoff/london-summary`.
- `-d oneoff/london-summary^3 spec:mirai-mixture-slow` is London three times for every model, and the slow mixture too for the speculative ones.

Datasets come from Hugging Face with your own login, which the gated `mirai-mixture` ones need. A request without its own output limit may generate 4,096 tokens, and an engine that doesn't answer a request in 30 minutes fails its run.

`--seed N` seeds the sampling; without it, runs are unseeded.

`--filter-ram` leaves out the models that don't fit a host. What an engine holds besides the model is a rough guess, and so is Metal's working set: two thirds of the memory up to 24 GiB, and above that the memory less a quarter of it, at most 16 GiB:

- Magnitude and TensorFold refuse a model themselves, so their own limits are used. Magnitude: the model and a third of it again within the working set less a reserve (a tenth of it, at least 2 GiB). TensorFold: the model and 6.5 GiB for its whole-context caches within the working set, at most 70% of the memory.
- The other engines: the model and a fifth of it again within the memory less 4 GiB for the system.

A queue's memory is that of its smallest online worker.

## Results

The table has one row per model and host, and a line in it for each engine and label, with means over the tasks:

| Column | |
| --- | --- |
| Decode | output tokens per second after the first |
| Accepted | output tokens per decoding pass, for speculative decoding (Magnitude doesn't report it) |
| Forwards | decoding passes per second, for speculative decoding; unlike Decode, it doesn't vary with what the sampling happens to accept |
| Prefill | input tokens per second |
| First token | seconds until the first token |
| Memory | the peak of what each engine reports as the memory a request took |

The best value of a row is underlined. `benchman` exits with 0 once everything was measured, 1 if a run failed, and 75 while results are missing.

## Shell completion

The dev shell puts the zsh and fish completions where shells look for them (`share/zsh/site-functions`, `share/fish/vendor_completions.d` in `XDG_DATA_DIRS`). With direnv 2.38.1 or newer, fish picks them up on entering the repository and drops them on leaving. zsh reads its completion directories only when it starts, so there it needs something that follows `XDG_DATA_DIRS`, like [zsh-completion-sync](https://github.com/BronzeDeer/zsh-completion-sync). `benchman --show-completion zsh` (or `fish`) prints the same script to load by hand.

Tab then completes options, engines, the models of the engines chosen so far, datasets, `oneoff` prompts, fleet queues and `tag:` names, also after `!`, `spec:` and a comma. Queues and tags are asked from Prefect on each Tab, and left out if that fails or takes more than 5 seconds. Completing changes nothing on disk or on the fleet.

## What is compared

**Models.** A model name is Uzu's. Every other engine runs its own counterpart, listed in `engine-<engine>/model_mapping.json`: the checkpoint the [published benchmarks](https://trymirai.com/models) used where there is one, else the same base model, 4-bit for `M` and 8-bit for `L` where available. These are not equivalent quantizations. A pattern skips the engines whose catalog has no match.

**Speculative decoding** is a property of the engine's entry, not of the model: the same model can decode speculatively with one engine and not with another. It decides which `spec:` datasets a run gets.

**Models outside the catalogs** run as the engine would run them by itself, so they don't compare alike across engines:

- Sampling is the engine's own default for the model.
- Reasoning efforts are mapped to those the engine says the model honors, by the rule below.
- Whether `spec:` or `!spec:` datasets apply is known only once the engine has loaded the model. Both are submitted and the run does those that apply; later invocations know which.
- `--filter-ram` fails for them: their size is not known beforehand.

An entry in `models.json` can also state only part of this for a model, and the engine decides the rest.

**Sampling** is the same for every engine: the model's Uzu generation settings, without the parameters that an engine having the model can't apply. No model keeps a repetition penalty for that reason.

| Engine | `temp`, `top_k`, `top_p` | `min_p` | repetition penalty |
| --- | --- | --- | --- |
| Uzu, MLX, llama.cpp | yes | yes | yes |
| oMLX, TensorFold, Magnitude | yes | yes | no |
| MTPLX, mlx-serve | yes | no | no |
| Splash, Kapitsa | yes, with `top_k` from 1 to 32 | no | no |

**Seeds** are from 0 to 2³² − 1. llama.cpp and Kapitsa reject 2³² − 1, and MTPLX 2³¹ and above.

**Reasoning.** A dataset's requests ask for an effort (`disabled`, `default`, `low`, `medium`, `high`, `xhigh`), and each model gets the nearest one it has:

- `default` is the effort that the model's default amounts to: `xhigh` for Qwen3.8, `high` for Muse Glimmer.
- A level the model lacks is the next one up, else its highest: `high` is `xhigh` for Qwen3.8. Qwen3.5 and Qwen3.6 have no levels and reason as they do by default.
- `disabled` on a model that can't stop reasoning is its lowest level, and on one that never reasons it is `default`.

**A run** is one engine process: it loads the model, answers one warmup request, then the tasks. Nothing is cached between requests. Before each measured request, a Mac whose CPU or GPU is at 80 °C or above cools to 70 °C. A request that fails ends the run.

**Memory** is each engine's own account, since no single counter is fair to all of them:

| Engine | Memory |
| --- | --- |
| Uzu | peak Metal allocation since the process started |
| MLX, MTPLX, oMLX, mlx-serve | MLX's peak active memory during the request, without its buffer cache |
| llama.cpp | size of the target and draft models plus their contexts' state |
| Splash | what it holds in Metal after the request, counting only the KV pages the request fills |
| TensorFold | its allocator's peak since the server started, which includes the cache it reserves for its whole context |
| Kapitsa | its Metal allocation after the request, which keeps the KV room of the longest request so far |
| Magnitude | what its own ledger says the loaded model holds: weights, context, compute buffers |

## Fleet runs

The fleet stores what it measures, addressed by what was asked: the source archive, the host, the engine, the model's settings and the request with its repeats. Any change of the checkout is another source, with results of its own. What `benchman` does about a task depends on what exists:

| Option | Stored | A flow run is measuring it | Neither |
| --- | --- | --- | --- |
| none | shown | waited for | measured |
| `-w`, `--wait` | shown | waited for | shown as missing, or as failed, crashed or cancelled like the last flow run that was to measure it |
| `-n`, `--no-run` | shown | shown with the flow run's state | as with `--wait` |
| `-c`, `--cancel` | shown | given up | as with `--wait` |
| `-rr`, `--rerun` | measured again and replaced | taken over and measured again | measured |

So an interrupted run continues from where it stopped, and the same command from another terminal joins a run instead of repeating it, also when both start at once.

All of this is per task, so commands may overlap in part. A command waits for its own tasks, whichever flow runs measure them, and not for the rest of those flow runs. A flow run that `-rr` takes tasks from goes on with its others, and whoever waited for those tasks gets the new measurement. `-c` gives up the tasks it selects, whoever started them: their flow runs go on with their other tasks, and are cancelled if they have none.

A failure is stored too, and then shown with its error instead of being measured again; `-rr` measures it again. The failure of a request is its task's, and the flow run's other tasks are left for the next command. The failure of the build or of loading the model is every task's.

`--background` exits once the flow runs are submitted. While `benchman` waits in a terminal for runs that are all on the fleet, Ctrl-] leaves them running. Both print `--source NAME`, a short name of the source: with it, the same command is about those runs from any checkout, also after an edit or from someone else's, to show them (`-n`), wait for them, tag them or give them up (`-c`). It finds them as long as the models' catalog entries and the datasets are the same there. Ctrl-C gives up the tasks of the flow runs that this `benchman` started, and not of those it only waited for.

An incompatible change of `benchman` starts over: it doesn't use what other versions stored or tagged.

Local runs happen in `benchman` itself, one after another, and involve no Prefect: they store nothing, always measure, can't be left in the background, and work without the `prefect` dependency group (`uv run --no-group prefect`).

## Saving and comparing

`--output FILE...` saves the runs as JSON, and `--input FILE...` shows saved runs too; without models, only those:

```bash
uv run --project benchman benchman -m Qwen3.8-27B-M -o main.json      # on main
uv run --project benchman benchman -m Qwen3.8-27B-M -i main.json -o branch.json  # on a branch
uv run --project benchman benchman -i main.json branch.json@after
```

Runs are compared by host, engine, model and task name: next to each value is how it differs from the first input's, over the tasks that both have. Saved runs are labelled with their file's name, or with the label after `@`; what is measured now, with the name of the first `--output`, or `now`.

Instead of a file, `-o tag:POOL/NAME` tags the runs in the work pool's storage, replacing what the name held: `-o tag:macs-fleet/0.6.18` on a release, `-i tag:macs-fleet/0.6.18` to compare with it from anywhere. A tag exists at once and doesn't need the results: it gets its own copy of each as it is stored, also from flow runs that another command started, so it works with `--background`. Its copies stay as they are when a task is measured again, unless that command names the tag too. A tag is of runs on its work pool's hosts.

A saved file or tag is loaded only by the version of `benchman` that saved it.

## Merge benchmark

[merge_bench.sh](merge_bench.sh) shows what a merge changes. It checks out the commit where this checkout forked from a destination revision, has this checkout's `benchman` measure there what its arguments select, and then measures the same here and compares:

```bash
./merge_bench.sh main
./merge_bench.sh main -h macs-fleet/bench-m5max -e uzu mlx -m Qwen3.5-2B-M
```

Without arguments it measures Uzu with four models on the fleet's M5 Max and M4 Max, twice each and unseeded. A commit that was measured before, as the fork point usually was when it was merged, is not measured again. It measures nothing, and says so, if the fork point's `benchman` is of another version.

The [merge benchmark workflow](../../.github/workflows/merge-bench.yml) runs it for each commit of `main` against its parent. On a pull request whose branch is in this repository, someone with write access can comment `!bench`, or `!bench` followed by the script's arguments, as in `!bench -h macs-fleet/bench-m{4,5}max -e uzu mlx -m Qwen3.5-2B-M`, to compare the branch against where it forked from the branch it merges into. The comparison goes to the workflow run's summary and `merge-bench` artifact, and into a comment on the pull request: the one commented on, or the one that the commit of `main` merged. For a commit of `main` it also goes to the benchmark channel on Discord. These are tables that [results_table.py](results_table.py) makes of the saved runs. A run that failed or is missing is a row that says so, and an error before any result is replied as text.

## Release benchmark

A release is a selection and a tag. This tags Uzu's release benchmark, submits what is missing of it and returns; the second command shows the tag, exiting with 75 until it has everything:

```bash
uv run --project benchman benchman -h 'macs-fleet/bench-*' -e uzu -m '*' -d oneoff/london-summary^3 spec:mirai-mixture-slow -s 0 --filter-ram -o tag:macs-fleet/0.6.18 -b
uv run --project benchman benchman -i tag:macs-fleet/0.6.18
```

The [release benchmark workflow](../../.github/workflows/release-bench.yml) does that when a GitHub release is published, with the release's tag as the name. Started by hand on a branch, it takes the selection and the name as inputs: by default the same with the fast mixture, under the commit's eight-character hash. With other engines after `-e` it benchmarks the competition. It doesn't wait for the fleet. Its summary and `release-bench` artifact have the tag's results so far, as a table from [results_table.py](results_table.py).

[release_summary.py](release_summary.py) turns a saved release benchmark into what the website shows, one entry per run:

```bash
uv run --project benchman python release_summary.py tag:macs-fleet/0.6.18 summary.json
```

It is where the rules of a release are: every run has to have the London summary three times with seed 0, and a model that decodes speculatively all 155 tasks of the slow mixture, or it fails. Input speed and memory are the London summary's; output speed is the mean over the mixture with speculative decoding, and the London summary's without.

Both workflows take the variable `PREFECT_API_URL` and the secret `PREFECT_API_KEY` from the `prefect-benchmarks` environment, and the repository's secret `HF_TOKEN`, which has to have access to the mixture.
