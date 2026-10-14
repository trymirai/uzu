/* global process */
import { homedir } from "node:os";
import { join } from "node:path";

export const slangVersion = "2026.19";
export const slangHost = `${process.platform}-${process.arch}`;
export const slangDirectory = join(homedir(), ".cache", "uzu", "slang", slangVersion, slangHost);
export const slangExecutable = join(slangDirectory, "bin", process.platform === "win32" ? "slangc.exe" : "slangc");

// Digests published with the official v2026.19 GitHub release. The compiler is
// pinned because target intrinsic emission is part of the shader binding ABI.
export const slangPackages = {
  "darwin-arm64": ["macos-aarch64", "fa46822c92d81404a7951de3746093985c1e6f520f4b533e2aa3bea576f0f2f0"],
  "darwin-x64": ["macos-x86_64", "490b6447e3ea6c8fa9dfc45b1dc227db7cf80a60981c4390ed0fdc8ddbedee11"],
  "linux-arm64": ["linux-aarch64", "eeefa4e52ca322cf3c037ee0d273b0ee68f4161fa422eda1bb704126073b69b4"],
  "linux-x64": ["linux-x86_64", "bd6cfc47b7353b2cffa36a866b1584749f76f4b2dbebf6dda26bc3fb40dc3c0e"],
  "win32-x64": ["windows-x86_64", "fa7629c9ce59c87c821614ae4b52e52f785d4c659a5a06a89fa9ec86d615eeee"],
};
