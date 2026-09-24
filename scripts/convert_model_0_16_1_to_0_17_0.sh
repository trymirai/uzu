#!/usr/bin/env bash
set -euo pipefail

if (( $# < 1 || $# > 2 )); then
  printf 'Usage: %s INPUT_MODEL_DIR [OUTPUT_MODEL_DIR]\nOne argument updates in place and needs free space for a full model copy. Set PYTHON to a Python with NumPy if needed.\n' "$0" >&2
  exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
input=$1
python_bin=${PYTHON:-python3}

if [[ ! -d $input || -L $input ]]; then
  printf 'Input must be a model directory, not a symlink: %s\n' "$input" >&2
  exit 2
fi

input=$(cd -- "$input" && pwd -P)
cleanup() {
  if [[ -e $temporary ]]; then
    rm -rf -- "$temporary"
  fi
}

if (( $# == 2 )); then
  output=$2
  if [[ -e $output || ! -d $(dirname -- "$output") ]]; then
    printf 'Output must be a new directory under an existing parent: %s\n' "$output" >&2
    exit 2
  fi
  temporary=$(mktemp -d "${output}.0.17.0.XXXXXX")
  rmdir -- "$temporary"
  trap cleanup EXIT
  "$python_bin" "$script_dir/convert_model_0_16_1_to_0_17_0.py" "$input" "$temporary"
  mv -- "$temporary" "$output"
  exit
fi

temporary=$(mktemp -d "${input}.0.17.0.XXXXXX")
rmdir -- "$temporary"
backup=$(mktemp -d "${input}.0.16.1.backup.XXXXXX")
rmdir -- "$backup"
trap cleanup EXIT

"$python_bin" "$script_dir/convert_model_0_16_1_to_0_17_0.py" "$input" "$temporary"
mv -- "$input" "$backup"
if ! mv -- "$temporary" "$input"; then
  mv -- "$backup" "$input"
  exit 1
fi
rm -rf -- "$backup"
