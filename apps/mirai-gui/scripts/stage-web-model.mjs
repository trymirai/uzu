/* global process, console */
// Optional local fixtures for the engine's numerical/parity harness. These files
// are never registered in the product catalog or needed by the browser app.
import { copyFile, mkdir, stat } from "node:fs/promises";
import { fileURLToPath, URL } from "node:url";
import { resolve } from "node:path";
import { parseArgs } from "node:util";

const { values, positionals } = parseArgs({
  options: { model: { type: "string", default: "lfm2.5-230m" } },
  allowPositionals: true,
});
const source = positionals[0];
if (!source) {
  console.error("Usage: pnpm web:fixture /path/to/model [--model lfm2.5-230m|qwen3.5-0.8b]");
  process.exit(1);
}
if (!["lfm2.5-230m", "qwen3.5-0.8b"].includes(values.model)) throw new Error(`Unknown test fixture: ${values.model}`);
const destination = resolve(fileURLToPath(new URL("../.web-test-models/", import.meta.url)), values.model);
await mkdir(destination, { recursive: true });
for (const name of ["config.json", "tokenizer.json", "model.safetensors"]) {
  const sourceFile = resolve(source, name);
  const size = (await stat(sourceFile)).size;
  await copyFile(sourceFile, resolve(destination, name));
  console.log(`Staged test fixture ${name} (${size} bytes)`);
}
console.log("Test files are ignored by Git. The browser app uses the live model catalog and remote downloads.");
