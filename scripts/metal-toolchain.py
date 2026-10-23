#!/usr/bin/env python3
"""Discover Apple Metal toolchains from Apple's MobileAsset catalog.

    metal-toolchain.py list         id / size / name table of available builds
    metal-toolchain.py get <build>  {"url", "aea_key", "sha256"} JSON for one
"""

# https://theapplewiki.com/wiki/Simulator_Runtime_MobileAssets

import base64
import http.client
import json
import plistlib
import ssl
import sys
import urllib.request


def cmd_list(argv):
    if argv:
        sys.exit(__doc__)
    with urllib.request.urlopen("https://devimages-cdn.apple.com/downloads/xcode/simulators/index2.dvtdownloadableindex", timeout=60) as response:
        index = plistlib.loads(response.read())
    entries, seen = [], set()
    for entry in index["otherDownloadables"]:
        build = entry["assetBuildUpdate"]
        if entry.get("assetType") == "metalToolchain" and build and build not in seen:
            seen.add(build)
            entries.append(entry)
    rows = [
        (entry["assetBuildUpdate"], f"{entry['fileSizeBytes'] / 2**20:.0f} MB", entry["name"])
        for entry in entries
    ]
    w_id = max(len(row[0]) for row in rows) + 3
    w_size = max(len(row[1]) for row in rows) + 3
    for build, size, name in rows:
        print(f"{build:{w_id}}{size:{w_size}}{name}")


def cmd_get(argv):
    if len(argv) != 1:
        sys.exit(__doc__)
    build = argv[0]
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    conn = http.client.HTTPSConnection("gdmf.apple.com", context=ctx, timeout=60)
    conn.request(
        "POST",
        "/v2/assets",
        body=json.dumps({
            "ClientVersion": 2,
            "AssetType": "com.apple.MobileAsset.MetalToolchain",
            "AssetAudience": "02d8e57e-dd1c-4090-aa50-b4ed2aef0062",
            "RequestedBuild": build,
        }),
        headers={"Content-Type": "application/json"},
    )
    response = conn.getresponse()
    if response.status != 200:
        sys.exit(f"error: GDMF asset query for build {build!r} returned HTTP {response.status}")
    _header, payload, _signature = response.read().decode().strip().split(".")
    payload += "=" * (-len(payload) % 4)
    assets = json.loads(base64.urlsafe_b64decode(payload)).get("Assets", [])
    if len(assets) != 1:
        sys.exit(f"error: expected 1 MetalToolchain asset for build {build!r}, got {len(assets)}")
    asset = assets[0]
    print(json.dumps({
        "url": asset["__BaseURL"] + asset["__RelativePath"],
        "aea_key": asset["ArchiveDecryptionKey"],
        "sha256": base64.b64decode(asset["_Measurement-SHA256"]).hex(),
    }, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("list", "get"):
        sys.exit(__doc__)
    {"list": cmd_list, "get": cmd_get}[sys.argv[1]](sys.argv[2:])
