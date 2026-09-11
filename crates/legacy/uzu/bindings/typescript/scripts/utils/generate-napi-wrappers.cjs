const fs = require('fs');
const path = require('path');

function unique(items) {
    return [...new Set(items)];
}

function extractRuntimeExportsFromCommonJs(commonJsPath) {
    const content = fs.readFileSync(commonJsPath, 'utf8');
    const exportRegex = /^\s*module\.exports\.([A-Za-z_$][\w$]*)\s*=/gm;
    const exports = [];
    let match;

    while ((match = exportRegex.exec(content)) !== null) {
        exports.push(match[1]);
    }

    return unique(exports);
}

function generateEsmWrapper(runtimeExports, outputPath) {
    const namedExports = runtimeExports
        .map((runtimeExport) => `export const ${runtimeExport} = cjs.${runtimeExport};`)
        .join('\n');
    const content = `// Auto-generated ESM adapter for the NAPI-RS CommonJS wrapper.
// Keep platform and architecture resolution in index.js.

import cjs from './index.js';

export default cjs;

${namedExports}
`;

    fs.writeFileSync(outputPath, content);
}

function generateEsmDeclarations(commonJsDeclarationsPath, outputPath) {
    fs.copyFileSync(commonJsDeclarationsPath, outputPath);
}

function main() {
    const napiDir = path.join(__dirname, '../../src/napi');
    const commonJsPath = path.join(napiDir, 'index.js');
    const commonJsDeclarationsPath = path.join(napiDir, 'index.d.ts');
    const esmPath = path.join(napiDir, 'index.mjs');
    const esmDeclarationsPath = path.join(napiDir, 'index.d.mts');

    if (!fs.existsSync(commonJsPath)) {
        console.error('NAPI file not found:', commonJsPath);
        process.exit(1);
    }

    if (!fs.existsSync(commonJsDeclarationsPath)) {
        console.error('NAPI declarations file not found:', commonJsDeclarationsPath);
        process.exit(1);
    }

    const runtimeExports = extractRuntimeExportsFromCommonJs(commonJsPath);
    if (runtimeExports.length === 0) {
        console.error('No runtime exports found in:', commonJsPath);
        process.exit(1);
    }

    generateEsmWrapper(runtimeExports, esmPath);
    generateEsmDeclarations(commonJsDeclarationsPath, esmDeclarationsPath);

    console.log(`Generated ${esmPath} with ${runtimeExports.length} named exports`);
    console.log(`Generated ${esmDeclarationsPath}`);
}

if (require.main === module) {
    main();
}

module.exports = {
    extractRuntimeExportsFromCommonJs,
    generateEsmDeclarations,
    generateEsmWrapper,
};
