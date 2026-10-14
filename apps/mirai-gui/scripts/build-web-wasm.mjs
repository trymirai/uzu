/* global process, console */
import { spawnSync } from "node:child_process";
import { mkdir } from "node:fs/promises";
import { fileURLToPath, URL } from "node:url";
import { resolve } from "node:path";
import { existsSync } from "node:fs";
import { slangExecutable } from "./slang-toolchain.mjs";

const repository = fileURLToPath(new URL("../../../", import.meta.url));
const output = fileURLToPath(new URL("../client/public/wasm/", import.meta.url));
if (!process.env.SLANGC && !process.env.SLANG_DIR && existsSync(slangExecutable)) process.env.SLANGC = slangExecutable;

function run(command, args) {
  const result = spawnSync(command, args, { cwd: repository, stdio: "inherit", env: process.env });
  if (result.error) {
    console.error(result.error.message);
    if (command === "wasm-bindgen")
      console.error(
        "Install the CLI version matching Cargo.lock: cargo install wasm-bindgen-cli --locked --version <version>",
      );
    process.exit(1);
  }
  if (result.status !== 0) process.exit(result.status ?? 1);
}

await mkdir(output, { recursive: true });
run("cargo", ["build", "-p", "uzu-web", "--release", "--target", "wasm32-unknown-unknown"]);
run("wasm-bindgen", [
  resolve(repository, process.env.CARGO_TARGET_DIR || "target", "wasm32-unknown-unknown/release/uzu_web.wasm"),
  "--target",
  "web",
  "--out-dir",
  output,
  "--out-name",
  "uzu_web",
]);
