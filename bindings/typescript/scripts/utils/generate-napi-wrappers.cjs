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

function generateEsmWrapper(exports, outputPath) {
    const namedExports = exports
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

function main() {
    const napiDir = path.join(__dirname, '../../src/napi');
    const commonJsPath = path.join(napiDir, 'index.js');
    const esmPath = path.join(napiDir, 'index.mjs');

    if (!fs.existsSync(commonJsPath)) {
        console.error('NAPI file not found:', commonJsPath);
        process.exit(1);
    }

    const runtimeExports = extractRuntimeExportsFromCommonJs(commonJsPath);
    if (runtimeExports.length === 0) {
        console.error('No runtime exports found in:', commonJsPath);
        process.exit(1);
    }

    generateEsmWrapper(runtimeExports, esmPath);

    console.log(`Generated ${esmPath} with ${runtimeExports.length} named exports`);
}

if (require.main === module) {
    main();
}

module.exports = {
    extractRuntimeExportsFromCommonJs,
    generateEsmWrapper,
};
