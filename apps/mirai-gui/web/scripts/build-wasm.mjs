import { spawnSync } from "node:child_process";
import { fileURLToPath, URL } from "node:url";
import { exit } from "node:process";

const result = spawnSync("cargo", ["tools", "build", "wasm"], {
  cwd: fileURLToPath(new URL("../../../../", import.meta.url)),
  stdio: "inherit",
});
if (result.error) throw result.error;
exit(result.status ?? 1);
