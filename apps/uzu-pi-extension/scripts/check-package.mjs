import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { execFileSync } from "node:child_process";

const root = new URL("../", import.meta.url);
const manifest = JSON.parse(
  await readFile(new URL("dist/package.json", root), "utf8"),
);
assert.equal(manifest.dependencies["@trymirai/uzu"], manifest.version);
assert.deepEqual(manifest.pi, { extensions: ["./index.js"] });
assert.equal(manifest.devDependencies, undefined);
assert.equal(manifest.publishConfig, undefined);
assert.equal(JSON.stringify(manifest).includes("link:"), false);
const archive = new URL(
  `dist/trymirai-uzu-pi-extension-${manifest.version}.tgz`,
  root,
);
const files = execFileSync("tar", ["-tzf", archive.pathname], {
  encoding: "utf8",
})
  .trim()
  .split("\n");
assert.ok(files.includes("package/index.js"));
assert.ok(files.includes("package/LICENSE"));
assert.ok(
  files.every((file) =>
    /^package\/(?:[^/]+\.(?:js|ts)|README\.md|LICENSE|package\.json)$/.test(
      file,
    ),
  ),
  files.join("\n"),
);
const packed = JSON.parse(
  execFileSync("tar", ["-xOf", archive.pathname, "package/package.json"], {
    encoding: "utf8",
  }),
);
assert.deepEqual(packed.dependencies, manifest.dependencies);
console.log(
  `Verified ${manifest.name}@${manifest.version} with @trymirai/uzu@${manifest.version}`,
);
