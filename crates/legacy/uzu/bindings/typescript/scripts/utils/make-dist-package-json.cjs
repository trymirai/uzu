const pkgJson = require(process.env['PKG_JSON_PATH'] || '../../package.json');

function processExportMap(m) {
    for (const key in m) {
        const value = m[key];
        if (typeof value === 'string') m[key] = value.replace(/^\.\/dist\//, './');
        else processExportMap(value);
    }
}
processExportMap(pkgJson.exports);

for (const key of ['types', 'main', 'module']) {
    if (typeof pkgJson[key] === 'string') pkgJson[key] = pkgJson[key].replace(/^(\.\/)?dist\//, './');
}

// Each target's binary is published as its own package (see `napi create-npm-dirs`), named the way
// the loader requires it. Package managers install the one whose `os` and `cpu` match.
const { parseTriple } = require('@napi-rs/cli');
pkgJson.optionalDependencies = Object.fromEntries(
    pkgJson.napi.targets.map((target) => [
        `${pkgJson.name}-${parseTriple(target).platformArchABI}`,
        pkgJson.version,
    ]),
);

// Browsers get the same API from the wasm package through the `browser` export condition. It's an
// optional peer, so that Node installs don't download it.
const wasmPackageName = require('../../../wasm/package.json').name;
pkgJson.peerDependencies = { [wasmPackageName]: pkgJson.version };
pkgJson.peerDependenciesMeta = { [wasmPackageName]: { optional: true } };

delete pkgJson.devDependencies;
delete pkgJson.scripts.prepack;
delete pkgJson.scripts.prepublishOnly;
delete pkgJson.scripts.prepare;

console.log(JSON.stringify(pkgJson, null, 2));
