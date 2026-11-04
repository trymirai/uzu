benchmark_source() {
    local engine="$1" destination="$2" patches=() patch repository revision ref stamp
    shift 2
    for patch in "$@"; do
        [[ "${patch##*/}" == nix-* ]] || patches+=("$patch")
    done
    IFS=$'\t' read -r repository revision < <(python3 - "$engine" "$(dirname -- "${BASH_SOURCE[0]}")/sources.json" <<'PY'
import json, sys
source = json.load(open(sys.argv[2]))[sys.argv[1]]
print(f"https://github.com/{source['owner']}/{source['repo']}.git\t{source['rev']}")
PY
    )
    stamp="$revision $(cat ${patches[@]+"${patches[@]}"} /dev/null | shasum -a 256)"
    if [[ "$(cat "$destination/.git/benchmark-stamp" 2>/dev/null)" == "$stamp" ]]; then
        return
    fi
    if [[ ! -d "$destination/.git" ]]; then
        git init "$destination" >&2
        git -C "$destination" remote add origin "$repository"
    fi
    ref="$revision"
    [[ "$revision" =~ ^[0-9a-f]{40}$ ]] || ref="refs/tags/$revision:refs/tags/$revision"
    git -C "$destination" fetch --depth 1 origin "$ref" >&2
    git -C "$destination" checkout --force --detach FETCH_HEAD >&2
    git -C "$destination" clean -fdq
    for patch in ${patches[@]+"${patches[@]}"}; do
        git -C "$destination" apply "$patch"
    done
    echo "$stamp" > "$destination/.git/benchmark-stamp"
}
