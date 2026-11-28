"""Give the DMG background a bookmark made by macOS itself.

dmgbuild writes its own approximation of the background bookmark (pBBk).
Finder on macOS 26 honours it only in Light Mode; in Dark Mode the window
shows no picture. A bookmark from NSURL works in both appearances, so this
mounts the finished image, asks the system for one, writes it into .DS_Store
and compresses the image again.

Usage (macOS, in the dmgbuild venv): python dmg-background-bookmark.py <image.dmg>
"""

import os
import plistlib
import subprocess
import sys
import tempfile

from ds_store import DSStore

BOOKMARK_JXA = """
ObjC.import('Foundation');
function run(argv) {
  const data = $.NSURL.fileURLWithPath(argv[0])
    .bookmarkDataWithOptionsIncludingResourceValuesForKeysRelativeToURLError(0, $(), $(), null);
  if (data.isNil()) throw new Error('Cannot bookmark ' + argv[0]);
  data.writeToFileAtomically(argv[1], true);
}
"""


class RawBookmark:
    # ds_store serialises pBBk through to_bytes(); pass the system's bytes through untouched.
    def __init__(self, data):
        self.data = data

    def to_bytes(self):
        return self.data


def run(*cmd):
    return subprocess.run(cmd, check=True, capture_output=True).stdout


def main(dmg):
    with tempfile.TemporaryDirectory() as tmp:
        rw = os.path.join(tmp, "rw.dmg")
        run("hdiutil", "convert", dmg, "-format", "UDRW", "-o", rw, "-quiet")
        attached = plistlib.loads(run("hdiutil", "attach", rw, "-nobrowse", "-noautoopen", "-plist"))
        mount = next(e["mount-point"] for e in attached["system-entities"] if "mount-point" in e)
        try:
            bookmark = os.path.join(tmp, "background.bookmark")
            run("osascript", "-l", "JavaScript", "-e", BOOKMARK_JXA,
                os.path.join(mount, ".background.tiff"), bookmark)
            with open(bookmark, "rb") as f:
                data = f.read()
            with DSStore.open(os.path.join(mount, ".DS_Store"), "r+") as store:
                store["."]["pBBk"] = RawBookmark(data)
        finally:
            run("hdiutil", "detach", mount, "-quiet")
        os.remove(dmg)
        run("hdiutil", "convert", rw, "-format", "UDZO", "-imagekey", "zlib-level=9", "-o", dmg, "-quiet")


if __name__ == "__main__":
    main(sys.argv[1])
