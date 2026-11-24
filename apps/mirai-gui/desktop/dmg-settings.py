import os

# Headless DMG layout for CI, where Tauri's Finder-driven styling is skipped
# (tauri-apps/tauri#1731). dmgbuild writes the .DS_Store directly, no Finder.
# Values mirror bundle.macOS.dmg in tauri.conf.json; keep them in sync.

app_path = os.environ["DMG_APP_PATH"]
app_name = os.path.basename(app_path)

files = [app_path]
symlinks = {"Applications": "/Applications"}

# Read-only compressed UDIF; required for notarization/stapling (a read-write
# UDRW image is the classic "unsupported format" notary rejection).
format = "UDZO"
background = os.environ["DMG_BACKGROUND"]
window_rect = ((100, 100), (660, 408))
icon_size = 128
text_size = 13
default_view = "icon-view"
show_icon_preview = False

icon_locations = {
    app_name: (180, 184),
    "Applications": (480, 184),
}
