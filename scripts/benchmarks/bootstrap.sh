#!/usr/bin/env bash

benchmark_source() {
    local engine="$1" destination="$2" repository revision current expected ref
    IFS=$'\t' read -r repository revision < <(python3 - "$engine" "$(dirname -- "${BASH_SOURCE[0]}")/sources.json" <<'PY'
import json, sys
source = json.load(open(sys.argv[2]))[sys.argv[1]]
print(f"https://github.com/{source['owner']}/{source['repo']}.git\t{source['rev']}")
PY
    )
    if [[ ! -d "$destination/.git" ]]; then
        git init "$destination" >&2
        git -C "$destination" remote add origin "$repository"
    fi
    current="$(git -C "$destination" rev-parse --verify HEAD 2>/dev/null || true)"
    expected="$(git -C "$destination" rev-parse --verify "$revision^{commit}" 2>/dev/null || true)"
    if [[ -n "$current" && "$current" == "$expected" ]]; then
        return
    fi
    if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
        echo "Cannot update $destination: it contains local changes" >&2
        return 1
    fi
    ref="$revision"
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || ref="refs/tags/$revision:refs/tags/$revision"
    git -C "$destination" fetch --depth 1 origin "$ref" >&2
    git -C "$destination" checkout --detach FETCH_HEAD >&2
}

benchmark_patches() {
    local source_dir="$1" patch_dir="$2" patch
    for patch in "$patch_dir"/*.patch; do
        if ! git -C "$source_dir" apply --reverse --check "$patch" 2>/dev/null; then
            git -C "$source_dir" apply --check "$patch"
            git -C "$source_dir" apply "$patch"
        fi
    done
}

benchmark_stage() {
    local destination="$1" source
    shift
    mkdir -p "$destination"
    for source in "$@"; do
        if ! cmp -s "$source" "$destination/$(basename "$source")"; then
            cp "$source" "$destination/"
        fi
    done
}
