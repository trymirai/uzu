#!/usr/bin/env bash
# sync-into-repo.sh
# Usage:
#   sync-into-repo.sh /path/to/repo RELATIVE/PATH/INSIDE/REPO /path/to/external/folder

set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <repository_path> <relative_target_path_inside_repo_or_.> <external_source_folder>" >&2
  exit 1
fi

repository_path="$1"
relative_target_path="$2"
external_source_folder="$3"

# Validate repository path
if [[ ! -d "$repository_path" ]]; then
  echo "Error: repository path not found: $repository_path" >&2
  exit 1
fi
repository_absolute_path="$(cd "$repository_path" && pwd)"

# Validate source folder
if [[ ! -d "$external_source_folder" ]]; then
  echo "Error: external source folder not found: $external_source_folder" >&2
  exit 1
fi
external_source_absolute_path="$(cd "$external_source_folder" && pwd)"

# Verify it's a Git repository
if [[ ! -d "$repository_absolute_path/.git" ]]; then
  echo "Error: $repository_absolute_path is not a Git repository (missing .git directory)." >&2
  exit 1
fi

# Ensure the target path exists inside the repository
mkdir -p "$repository_absolute_path/$relative_target_path"
target_absolute_path="$(cd "$repository_absolute_path/$relative_target_path" && pwd)"

# Confirm that the target is inside the repository
case "$target_absolute_path" in
  "$repository_absolute_path" | "$repository_absolute_path"/*) ;;  # OK
  *)
    echo "Error: target path escapes the repository: $target_absolute_path" >&2
    exit 1
    ;;
esac

# Rsync all contents (including hidden files), deleting removed ones,
# but never touching any .git directory anywhere.
rsync -a --delete \
  --exclude '.git' --exclude '.git/*' --exclude '**/.git' --exclude '**/.git/*' \
  "$external_source_absolute_path"/ "$target_absolute_path"/

# Stage all resulting changes (entire repo) for commit
git -C "$repository_absolute_path" add -A

echo "✅ Synced $external_source_absolute_path → $target_absolute_path and staged all changes in $repository_absolute_path."