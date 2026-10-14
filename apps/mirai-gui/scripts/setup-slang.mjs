/* global process, console, fetch */
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { createWriteStream } from "node:fs";
import { mkdir, mkdtemp, rename, rm } from "node:fs/promises";
import { dirname, join } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { slangDirectory, slangExecutable, slangHost, slangPackages, slangVersion } from "./slang-toolchain.mjs";

const installed = spawnSync(slangExecutable, ["-version"], { encoding: "utf8" });
if (installed.status === 0 && `${installed.stdout}${installed.stderr}`.trim() === slangVersion) {
  console.log(`Slang ${slangVersion} is ready: ${slangExecutable}`);
} else {
  const pkg = slangPackages[slangHost];
  if (!pkg) throw new Error(`No pinned Slang package for ${slangHost}. Install Slang ${slangVersion} and set SLANGC.`);
  const [platform, expectedHash] = pkg;
  const url = `https://github.com/shader-slang/slang/releases/download/v${slangVersion}/slang-${slangVersion}-${platform}.tar.gz`;
  await mkdir(dirname(slangDirectory), { recursive: true });
  const staging = await mkdtemp(join(dirname(slangDirectory), ".install-"));
  try {
    console.log(`Downloading Slang ${slangVersion} from its official release…`);
    const response = await fetch(url);
    if (!response.ok || !response.body) throw new Error(`Slang download failed: HTTP ${response.status}`);
    const hash = createHash("sha256");
    const archive = join(staging, "slang.tar.gz");
    await pipeline(
      Readable.fromWeb(response.body),
      new Transform({
        transform(chunk, _encoding, callback) {
          hash.update(chunk);
          callback(null, chunk);
        },
      }),
      createWriteStream(archive),
    );
    if (hash.digest("hex") !== expectedHash) throw new Error("Slang archive SHA-256 does not match the pinned release");
    const contents = join(staging, "contents");
    await mkdir(contents);
    const extract = spawnSync("tar", ["-xzf", archive, "-C", contents], { stdio: "inherit" });
    if (extract.status !== 0) throw new Error("Could not extract Slang (tar is required)");
    const executable = join(contents, "bin", process.platform === "win32" ? "slangc.exe" : "slangc");
    const version = spawnSync(executable, ["-version"], { encoding: "utf8" });
    if (version.status !== 0 || `${version.stdout}${version.stderr}`.trim() !== slangVersion) {
      throw new Error(`Downloaded compiler did not report Slang ${slangVersion}`);
    }
    // Replace only our own invalid compiler cache, after verifying the new copy.
    await rm(slangDirectory, { recursive: true, force: true });
    await rename(contents, slangDirectory);
    console.log(`Slang ${slangVersion} installed: ${slangExecutable}`);
  } finally {
    await rm(staging, { recursive: true, force: true });
  }
}
