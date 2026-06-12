#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

PACKAGE="backend-uzu"
CRATE_DIR="crates/backend-uzu"

normalize_paths() {
  perl -MCwd=abs_path -ne 'chomp; my $path = abs_path($_); print "$path\n" if defined $path'
}

relative_paths() {
  ROOT="${ROOT_DIR}/" perl -ne 'chomp; s/^\Q$ENV{"ROOT"}\E//; print "$_\n"'
}

fmt_status=0
fmt_output="$(cargo fmt -p "$PACKAGE" --check -- --verbose 2>&1)" || fmt_status=$?

all_files="$(find "$CRATE_DIR" -type f -name '*.rs' -print | normalize_paths | sort -u)"
discovered_files="$(
  printf '%s\n' "$fmt_output" |
    sed -n 's/^Formatting //p' |
    normalize_paths |
    sort -u
)"

missing_files="$(comm -23 <(printf '%s\n' "$all_files") <(printf '%s\n' "$discovered_files"))"

if [[ -n "$missing_files" ]]; then
  all_count="$(printf '%s\n' "$all_files" | sed '/^$/d' | wc -l | tr -d ' ')"
  discovered_count="$(printf '%s\n' "$discovered_files" | sed '/^$/d' | wc -l | tr -d ' ')"
  missing_count="$(printf '%s\n' "$missing_files" | sed '/^$/d' | wc -l | tr -d ' ')"

  {
    printf 'cargo fmt -p %s did not discover every Rust file under %s.\n' "$PACKAGE" "$CRATE_DIR"
    printf 'Discovered %s of %s files; missing %s:\n' "$discovered_count" "$all_count" "$missing_count"
    printf '%s\n' "$missing_files" | relative_paths
  } >&2
  exit 1
fi

if ((fmt_status != 0)); then
  printf '%s\n' "$fmt_output" >&2
  exit "$fmt_status"
fi

all_count="$(printf '%s\n' "$all_files" | sed '/^$/d' | wc -l | tr -d ' ')"
printf 'cargo fmt -p %s discovers all %s Rust files under %s.\n' "$PACKAGE" "$all_count" "$CRATE_DIR"
