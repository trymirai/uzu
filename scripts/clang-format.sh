#!/usr/bin/env bash
set -euo pipefail

# Format C/C++/Metal headers and sources according to .clang-format at repo root.
# Usage:
#   ./scripts/clang-format.sh
#   ./scripts/clang-format.sh --check   # only check (no changes), non-zero exit on diffs
#
# Excludes:
#   - target/ (Rust build)
#   - external/ (third-party deps)
#   - .git/

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
cd "$ROOT_DIR"

# Resolve clang-format from the selected toolchain.
if ! CLANG_FORMAT="$(xcrun --find clang-format)"; then
  echo "Error: clang-format not found in the selected toolchain." >&2
  exit 127
fi

MODE="fix"
if [[ "${1:-}" == "--check" ]]; then
  MODE="check"
fi

# Files to consider:
#   - tracked files from git
#   - untracked files that are not ignored
FILES=()
while IFS= read -r -d '' file; do
  FILES+=("$file")
done < <(git ls-files -z --cached --others --exclude-standard -- \
  '*.h' '*.hpp' '*.hh' \
  '*.c' '*.cc' '*.cpp' \
  '*.metal')

if [[ ${#FILES[@]} -eq 0 ]]; then
  echo "No files to format."
  exit 0
fi

if [[ "$MODE" == "check" ]]; then
  # Show diffs without writing
  FAILED=0
  for f in "${FILES[@]}"; do
    if ! diff -u "$f" <("$CLANG_FORMAT" "$f"); then
      FAILED=1
    fi
  done
  if [[ $FAILED -ne 0 ]]; then
    echo "\nFormatting check failed. Run ./scripts/clang-format.sh to fix." >&2
    exit 1
  fi
else
  # In-place rewrite
  printf '%s\0' "${FILES[@]}" | xargs -0 -n 50 "$CLANG_FORMAT" -i
  echo "Formatted ${#FILES[@]} file(s)."
fi
