#!/usr/bin/env bash
# Usage: merge_bench.sh DESTINATION [BENCHMAN ARGUMENT...]
# Benchmarks this checkout against its fork point from DESTINATION.
# The runs are left in the directory MERGE_BENCH_RESULTS, if set, as benchman saves them: base.json and head.json.
set -euo pipefail

destination="$1"
shift
if (($# == 0)); then
    # Eight of the fast mixture's tasks, from its shortest context to its longest.
    set -- -h macs-fleet/bench-m{4,5}max -m Qwen3.8-27B-M Qwen3.5-9B-Dash \
        -d 'mirai-mixture-fast/{Random fiction,HLE math,MT-Bench-101,TerminalBench 2.1,CoSER,AppWorld,DeepSWE,UTILMEM}:*^2'
elif printf '%s\n' "$@" | grep -qxE -- '-o|--outputs?'; then
    echo "Error: the arguments can't give outputs, which are the runs that this saves to compare" >&2
    exit 2
elif ! printf '%s\n' "$@" | grep -qxE -- '-h|--hosts?'; then
    # benchman's own default is this machine, which is not what a merge is compared on.
    echo "Error: the arguments have to give the hosts, as in -h macs-fleet/bench-m5max" >&2
    exit 2
fi

root="$(git rev-parse --show-toplevel)"
benchman=(uv run --quiet --project "$root/scripts/benchmarks/benchman" --frozen -- benchman)
version=scripts/benchmarks/benchman/src/version.py
fork_point="$(git merge-base "$destination" HEAD)"
base="$(git rev-parse --short "$fork_point")"

ours="$(grep '^VERSION = ' "$root/$version")"
theirs="$(git show "$fork_point:$version" 2>/dev/null | grep '^VERSION = ' || true)"
if [[ "$theirs" != "$ours" ]]; then
    there="${theirs:+version ${theirs#VERSION = }}"
    echo "Not measured: benchman is version ${ours#VERSION = } here and ${there:-without a version} at the fork point $base."
    exit 0
fi

directory="$(mktemp -d)"
trap 'git worktree remove --force "$directory/source" 2>/dev/null; rm -rf "$directory"' EXIT
mkdir -p "${MERGE_BENCH_RESULTS:-$directory}"
results="$(realpath "${MERGE_BENCH_RESULTS:-$directory}")"
git worktree add --quiet --detach "$directory/source" "$fork_point"
# A run of the fork point that failed (1) or is missing (75) doesn't keep this checkout from being measured.
status=0
(cd "$directory/source" && "${benchman[@]}" "$@" -o "$results/base.json") >&2 || status=$?
[[ "$status" == 0 || "$status" == 1 || "$status" == 75 ]] || exit "$status"
"${benchman[@]}" "$@" -i "$results/base.json" -o "$results/head.json"
