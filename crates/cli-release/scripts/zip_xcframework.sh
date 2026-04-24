#!/usr/bin/env bash
set -euo pipefail

FRAMEWORK_PATH="$1"
DESTINATION_PATH="$2"
VERSION="$3"

ZIP_NAME="${VERSION}.zip"

cd $DESTINATION_PATH
mkdir $VERSION
rsync -a "$FRAMEWORK_PATH" "$VERSION/"
zip -r "$ZIP_NAME" "$VERSION"
rm -rf "$VERSION"

CHECKSUM=$(swift package compute-checksum "$ZIP_NAME")
echo $CHECKSUM