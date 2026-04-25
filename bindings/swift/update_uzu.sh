#!/usr/bin/env bash
set -euo pipefail

# update_uzu.sh
# Regenerates the Uzu Swift package (binary target + generated bindings)
# for the requested Apple platform/configuration using cargo-swift.
# This script is invoked from Xcode scheme pre-actions so it MUST stay fast:
# Xcode passes the effective PLATFORM name (ios, iphonesimulator, macosx, …)
# and CONFIGURATION (Debug / Release).
#
# Requires: `cargo swift` to be installed (`cargo install cargo-swift`).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
RUST_CRATE_DIR="$WORKSPACE_ROOT/crates/uzu"

# Destination Swift package (checked in to the repo and referenced by Xcode)
DEST_PKG_DIR="$SCRIPT_DIR"
DEST_SRC_DIR="$DEST_PKG_DIR/Sources"

PLATFORM="${PLATFORM:-ios}"
CONFIGURATION="${CONFIGURATION:-Debug}"

# Capture Xcode-provided deployment targets before we unset them later.
# These come from Project.swift deploymentTargets and are the source of truth.
XCODE_IPHONEOS_DEPLOYMENT_TARGET="${IPHONEOS_DEPLOYMENT_TARGET:-26.0}"
XCODE_MACOSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-26.0}"

# Ensure Rust, Homebrew, etc. binaries are on PATH when called from Xcode (which resets PATH)
if [ -d "/opt/homebrew/bin" ]; then
  export PATH="/opt/homebrew/bin:$PATH"
fi
if [ -d "/usr/local/bin" ]; then
  export PATH="/usr/local/bin:$PATH"
fi
if [ -d "$HOME/.cargo/bin" ]; then
  export PATH="$HOME/.cargo/bin:$PATH"
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --platform requires a value"
        exit 1
      fi
      PLATFORM="$2"
      shift 2
      ;;
    --configuration)
      if [[ -z "${2:-}" ]]; then
        echo "Error: --configuration requires a value"
        exit 1
      fi
      CONFIGURATION="$2"
      shift 2
      ;;
    *)
      shift
      ;;
  esac
done

# Normalise PLATFORM value (xcode passes like -iphonesimulator etc.)
PLATFORM="$(echo "$PLATFORM" | tr '[:upper:]' '[:lower:]')"
[[ "$PLATFORM" == "iphonesimulator" ]] && PLATFORM="ios-sim"
[[ "$PLATFORM" == "macosx" ]] && PLATFORM="macos"
[[ "$PLATFORM" == "iphoneos" ]] && PLATFORM="ios"

if [[ -z "$PLATFORM" ]]; then
  # Fallback to PLATFORM_NAME provided by Xcode (e.g. macosx, iphoneos, iphonesimulator)
  PLATFORM="$(echo "${PLATFORM_NAME:-}" | tr '[:upper:]' '[:lower:]')"
  [[ "$PLATFORM" == "macosx" ]] && PLATFORM="macos"
  [[ "$PLATFORM" == "iphoneos" ]] && PLATFORM="ios"
  [[ "$PLATFORM" == "iphonesimulator" ]] && PLATFORM="ios-sim"
fi

# Validate platform value
case "$PLATFORM" in
  ios|ios-sim|macos) ;;
  *)
    echo "Error: Invalid platform '$PLATFORM'"
    echo "Valid platforms: ios, ios-sim, macos"
    exit 1
    ;;
esac

echo "↪︎ Rebuilding Uzu Swift package for $PLATFORM ($CONFIGURATION)"

#####################################
# 1. Generate Swift package with cargo-swift
#####################################
# Map platform to cargo-swift target triple list (cargo-swift supports multi-target but
# we only build the slice we currently need to speed things up).
case "$PLATFORM" in
  ios)      CS_TARGET="aarch64-apple-ios";;
  ios-sim)  CS_TARGET="aarch64-apple-ios-sim";;
  macos)    CS_TARGET="aarch64-apple-darwin";;
  *) echo "Unsupported platform $PLATFORM"; exit 1;;
esac

CS_RELEASE_FLAG=""
if [[ "$CONFIGURATION" == "Release" ]]; then
  CS_RELEASE_FLAG="--release"
fi

# Determine swift platform flag
case "$PLATFORM" in
  ios)      CS_SWIFT_PLATFORM="ios";;
  ios-sim)  CS_SWIFT_PLATFORM="ios";;
  macos)    CS_SWIFT_PLATFORM="macos";;
  *) CS_SWIFT_PLATFORM="ios";;
esac

# Run cargo-swift inside crate directory

pushd "$RUST_CRATE_DIR" >/dev/null

# Xcode exports deployment targets for ALL platforms, which can confuse C/C++ builds.
# Unset ALL deployment targets first, then set only the one we need.
unset MACOSX_DEPLOYMENT_TARGET
unset IPHONEOS_DEPLOYMENT_TARGET
unset XROS_DEPLOYMENT_TARGET
unset TVOS_DEPLOYMENT_TARGET
unset WATCHOS_DEPLOYMENT_TARGET
unset DRIVERKIT_DEPLOYMENT_TARGET

# Xcode sets SDKROOT to the iOS SDK during iOS builds. That confuses the host-side
# build of procedural-macro crates (which must link against the macOS SDK).
# For iOS builds we unset it; for macOS builds we point it explicitly to the macOS SDK
# so Rust's linker can find system libs like libobjc.
case "$PLATFORM" in
  ios|ios-sim)
    unset SDKROOT
    ;;
  macos)
    export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
    # Help rustc locate system libraries when using -nodefaultlibs by passing the sysroot
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-isysroot -C link-arg=$SDKROOT"
    ;;
esac

unset SWIFT_DEBUG_INFORMATION_FORMAT
unset SWIFT_DEBUG_INFORMATION_VERSION

#####################################
# Set explicit Apple deployment targets for native objects
# Values come from Xcode (Project.swift is source of truth)
# Must be done AFTER unsetting conflicting environment variables above
#
# Note: We do NOT set CFLAGS/CXXFLAGS globally as they affect host builds
# (build scripts, proc-macros) and cause conflicts with cross-compilation.
# cargo-swift handles target-specific flags internally.
#####################################
case "$PLATFORM" in
  macos)
    export MACOSX_DEPLOYMENT_TARGET="$XCODE_MACOSX_DEPLOYMENT_TARGET"
    ;;
  ios)
    export IPHONEOS_DEPLOYMENT_TARGET="$XCODE_IPHONEOS_DEPLOYMENT_TARGET"
    ;;
  ios-sim)
    export IPHONEOS_DEPLOYMENT_TARGET="$XCODE_IPHONEOS_DEPLOYMENT_TARGET"
    ;;
esac

cargo swift package \
  --name uzu \
  --xcframework-name uzu \
  --target "$CS_TARGET" \
  -p "$CS_SWIFT_PLATFORM" \
  --features "bindings-uniffi" \
  $CS_RELEASE_FLAG \
  -y >/dev/null

popd >/dev/null

# The generated package lives at $RUST_CRATE_DIR/uzu
PKG_OUT_DIR="$RUST_CRATE_DIR/uzu"

#####################################
# 2. Copy artefacts into the repository's Swift package wrapper
#####################################

# Ensure wrapper package skeleton exists (Package.swift, Sources folder, etc.)
mkdir -p "$DEST_PKG_DIR"
mkdir -p "$DEST_SRC_DIR/Uzu"

# Package.swift manifest generation has been moved to setup.sh

# Copy generated FFI Swift sources
SRC_GEN_DIR="$PKG_OUT_DIR/Sources"
if [[ -d "$SRC_GEN_DIR" ]]; then
  echo "• Updating generated swift sources"
  rm -rf "$DEST_SRC_DIR/Uzu/Generated"
  mkdir -p "$DEST_SRC_DIR/Uzu/Generated"
  rsync -a --delete "$SRC_GEN_DIR/" "$DEST_SRC_DIR/Uzu/Generated/"
fi

# Copy xcframework
XC_PATH="$PKG_OUT_DIR/uzu.xcframework"
if [[ -d "$XC_PATH" ]]; then
  echo "• Copying xcframework to package root"
  rm -rf "$DEST_PKG_DIR/uzu.xcframework"
  cp -R "$XC_PATH" "$DEST_PKG_DIR/uzu.xcframework"

  # Flatten headers: cargo-swift nests them under Headers/<name>/ but SPM expects
  # the modulemap directly inside Headers/
  for slice_dir in "$DEST_PKG_DIR/uzu.xcframework"/*/; do
    headers_dir="${slice_dir}Headers"
    [[ -d "$headers_dir" ]] || continue
    for nested_modulemap in "$headers_dir"/*/module.modulemap; do
      [[ -f "$nested_modulemap" ]] || continue
      nested_dir="$(dirname "$nested_modulemap")"
      mv "$nested_dir"/* "$headers_dir/"
      rmdir "$nested_dir"
    done
  done
  
  # Sign the xcframework with the current development team
  # For Release builds, this ensures the framework is properly signed for distribution
  if [[ -n "${EXPANDED_CODE_SIGN_IDENTITY:-}" && "$EXPANDED_CODE_SIGN_IDENTITY" != "-" ]]; then
    echo "• Signing xcframework with identity: $EXPANDED_CODE_SIGN_IDENTITY"
    codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" \
      --timestamp \
      --preserve-metadata=identifier,entitlements,flags \
      "$DEST_PKG_DIR/uzu.xcframework" 2>/dev/null || true
  elif [[ "$CONFIGURATION" == "Release" ]]; then
    # For Release builds outside of Xcode, try to use ad-hoc signing
    echo "• Applying ad-hoc signing to xcframework"
    codesign --force --sign - "$DEST_PKG_DIR/uzu.xcframework" 2>/dev/null || true
  fi
fi

# Clean up generated Swift package directory created by cargo-swift
if [[ -d "$PKG_OUT_DIR" ]]; then
  rm -rf "$PKG_OUT_DIR"
fi

# Also remove crate-local generated directory if present
if [[ -d "$RUST_CRATE_DIR/generated" ]]; then
  rm -rf "$RUST_CRATE_DIR/generated"
fi

echo "✓ Uzu package refreshed"
