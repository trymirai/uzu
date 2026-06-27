#!/usr/bin/env bash
# Build a macOS `Mirai.app` bundle (icon + Info.plist + ad-hoc signature) and a
# drag-to-Applications `Mirai.dmg` installer.
#
#   ./bundle.sh            # release build (default)
#   PROFILE=debug ./bundle.sh   # fast debug build, for verifying the bundle
#
# Output: app/target/Mirai.app and app/target/Mirai.dmg
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)" # mirai-app/
APP_WS="$(cd "$SCRIPT_DIR/.." && pwd)"                     # app/ (cargo workspace)
PROFILE="${PROFILE:-release}"

if [[ "$PROFILE" == "release" ]]; then
	CARGO_FLAG="--release"
	TARGET_SUBDIR="release"
else
	CARGO_FLAG=""
	TARGET_SUBDIR="debug"
fi

VERSION="$(grep -m1 '^version' "$SCRIPT_DIR/Cargo.toml" | sed -E 's/.*"(.*)".*/\1/')"
BIN="$APP_WS/target/$TARGET_SUBDIR/mirai-app"
APP="$APP_WS/target/Mirai.app"
DMG="$APP_WS/target/Mirai.dmg"

echo "==> Building mirai-app ($PROFILE, v$VERSION)"
# shellcheck disable=SC2086 # $CARGO_FLAG is intentionally word-split (empty = no flag)
(cd "$APP_WS" && cargo build -p mirai-app $CARGO_FLAG)

echo "==> Assembling $APP"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"
cp "$BIN" "$APP/Contents/MacOS/mirai-app"
chmod +x "$APP/Contents/MacOS/mirai-app"
cp "$SCRIPT_DIR/resources/Mirai.icns" "$APP/Contents/Resources/Mirai.icns"
sed "s/__VERSION__/$VERSION/g" "$SCRIPT_DIR/resources/Info.plist" >"$APP/Contents/Info.plist"

# Sign with $MACOS_SIGNING_IDENTITY (a Developer ID) if set, else ad-hoc. The
# entitlements are applied either way; hardened runtime (--options runtime,
# required for notarization) is only added for a real identity, mirroring Zed's
# script/bundle-mac. Distribution still needs notarytool + stapler afterwards.
IDENTITY="${MACOS_SIGNING_IDENTITY:--}"
ENTITLEMENTS="$SCRIPT_DIR/resources/Mirai.entitlements"
if [[ "$IDENTITY" == "-" ]]; then
	echo "==> Ad-hoc code-signing (set MACOS_SIGNING_IDENTITY for distribution)"
	codesign --force --deep --entitlements "$ENTITLEMENTS" --sign - "$APP"
else
	echo "==> Code-signing + hardened runtime as '$IDENTITY'"
	codesign --force --deep --timestamp --options runtime \
		--entitlements "$ENTITLEMENTS" --sign "$IDENTITY" "$APP"
fi

echo "==> Building $DMG"
STAGE="$(mktemp -d)"
cp -R "$APP" "$STAGE/Mirai.app"
ln -s /Applications "$STAGE/Applications"
rm -f "$DMG"
hdiutil create -volname "Mirai" -srcfolder "$STAGE" -ov -format UDZO "$DMG" >/dev/null
rm -rf "$STAGE"

echo "==> Done:"
echo "    $APP"
echo "    $DMG"
