import { readFile, writeFile, copyFile } from "node:fs/promises";

const readJson = async (path) =>
  JSON.parse(await readFile(new URL(path, import.meta.url), "utf8"));
const source = await readJson("../package.json");
const uzu = await readJson(
  "../../../crates/legacy/uzu/bindings/typescript/package.json",
);
if (source.version !== uzu.version) {
  throw new Error("Extension and Uzu versions differ; run cargo tools sync");
}
const { scripts, devDependencies, publishConfig, packageManager, ...manifest } =
  source;
manifest.dependencies = { "@trymirai/uzu": uzu.version };
manifest.pi = { extensions: ["./index.js"] };
manifest.files = ["*.js", "*.d.ts", "README.md", "LICENSE"];
await writeFile(
  new URL("../dist/package.json", import.meta.url),
  `${JSON.stringify(manifest, null, 2)}\n`,
);
await copyFile(
  new URL("../README.md", import.meta.url),
  new URL("../dist/README.md", import.meta.url),
);
await copyFile(
  new URL("../../../LICENSE", import.meta.url),
  new URL("../dist/LICENSE", import.meta.url),
);
