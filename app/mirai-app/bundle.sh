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

echo "==> Ad-hoc code-signing (replace with a Developer ID for distribution)"
codesign --force --deep --sign - "$APP"

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
