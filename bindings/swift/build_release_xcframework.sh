#!/usr/bin/env bash
set -euo pipefail

# build_release_xcframework.sh
# Builds the Uzu Swift package in Release configuration for macOS,
# iOS device and iOS simulator slices and combines them into a single
# universal `uzu.xcframework` ready for distribution (e.g. publishing
# to a private or public SwiftPM registry).
#
# The script relies on `update_uzu.sh` (which itself wraps `cargo swift`)
# for per-platform builds and on `xcodebuild -create-xcframework` to
# merge the individual static libraries.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SOURCE_PKG_DIR="$SCRIPT_DIR"  # root of Swift package (update_uzu.sh writes here)

# Default destination equals the source package location but can be overridden
DEST_PKG_DIR="$SOURCE_PKG_DIR"

# Parse CLI options
while [[ $# -gt 0 ]]; do
  case "$1" in
    --destination|-d)
      DEST_PKG_DIR="$2"; shift 2;;
    *)
      echo "Unknown option: $1" >&2; exit 1;;
  esac
done

UNIVERSAL_XCFRAMEWORK="$SOURCE_PKG_DIR/uzu.xcframework"

BUILD_ROOT="$SCRIPT_DIR/.release_build"
SLICES_DIR="$BUILD_ROOT/slices"

rm -rf "$BUILD_ROOT"
mkdir -p "$SLICES_DIR"

# 1. Build each platform slice in Release mode
# Allow overriding platforms via env var UZU_SWIFT_PLATFORMS (e.g., "macos" for local)
PLATFORMS=${UZU_SWIFT_PLATFORMS:-"ios ios-sim macos"}
for platform in $PLATFORMS; do
  echo "\n=== Building Uzu for $platform (Release) ==="
  bash "$SCRIPT_DIR/update_uzu.sh" --platform "$platform" --configuration Release
  # Preserve the generated xcframework for later merging
  cp -R "$UNIVERSAL_XCFRAMEWORK" "$SLICES_DIR/${platform}.xcframework"
done

# 2. Combine slices into a single universal xcframework
echo "\n=== Creating universal xcframework ==="
rm -rf "$UNIVERSAL_XCFRAMEWORK"

LIB_ARGS=()
for platform in $PLATFORMS; do
  SLICE="$SLICES_DIR/${platform}.xcframework"
  LIB_PATH="$(find "$SLICE" -name 'libuzu.a' | head -n 1)"
  HEADERS_PATH="$(dirname "$LIB_PATH")/Headers"
  LIB_ARGS+=("-library" "$LIB_PATH" "-headers" "$HEADERS_PATH")
done

xcodebuild -create-xcframework "${LIB_ARGS[@]}" -output "$UNIVERSAL_XCFRAMEWORK" | cat

# Sign the universal xcframework
echo "\n=== Signing universal xcframework ==="
# Try to sign with developer certificate, fall back to ad-hoc signing
if security find-identity -v -p codesigning | grep -q "Apple Development"; then
  SIGNING_IDENTITY=$(security find-identity -v -p codesigning | grep "Apple Development" | head -n 1 | awk '{print $2}')
  echo "Signing with identity: $SIGNING_IDENTITY"
  codesign --force --sign "$SIGNING_IDENTITY" --timestamp "$UNIVERSAL_XCFRAMEWORK"
else
  echo "No developer certificate found, applying ad-hoc signing"
  codesign --force --sign - "$UNIVERSAL_XCFRAMEWORK"
fi

# 3. Copy the finished Swift package to destination if it differs from source
if [[ "$DEST_PKG_DIR" != "$SOURCE_PKG_DIR" ]]; then
  echo "\n=== Copying Swift package to $DEST_PKG_DIR ==="
  rm -rf "$DEST_PKG_DIR"
  mkdir -p "$(dirname "$DEST_PKG_DIR")"
  rsync -a --delete "$SOURCE_PKG_DIR/" "$DEST_PKG_DIR/"
fi

echo "\n✅ Release Swift package ready at $DEST_PKG_DIR"
