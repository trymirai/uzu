const fs = require('fs');
const path = require('path');

function extractExportsFromDts(dtsPath) {
    const content = fs.readFileSync(dtsPath, 'utf8');

    // Extract class exports
    const classMatches = content.match(/export declare class (\w+)/g) || [];
    const classes = classMatches.map((match) => match.replace('export declare class ', ''));

    // Extract enum exports
    const enumMatches = content.match(/export declare const enum (\w+)/g) || [];
    const enums = enumMatches.map((match) => match.replace('export declare const enum ', ''));

    // Extract function exports
    const functionMatches = content.match(/export declare function (\w+)/g) || [];
    const functions = functionMatches.map((match) => match.replace('export declare function ', ''));

    // Extract interface exports (including multi-line)
    const interfaceRegex = /export interface (\w+)/g;
    const interfaces = [];
    let interfaceMatch;
    while ((interfaceMatch = interfaceRegex.exec(content)) !== null) {
        interfaces.push(interfaceMatch[1]);
    }

    // Extract type alias exports (including multi-line union types)
    const typeRegex = /export type (\w+)\s*=/g;
    const types = [];
    let typeMatch;
    while ((typeMatch = typeRegex.exec(content)) !== null) {
        types.push(typeMatch[1]);
    }

    // Note: Interfaces and type aliases are TypeScript constructs and won't be available at runtime
    // from the native module. Only classes, enums, and functions are actually exported from .node files.
    // However, we'll track them for completeness and potential future use.

    const runtimeExports = [...classes, ...enums, ...functions];
    const typeOnlyExports = [...interfaces, ...types];

    return {
        runtime: runtimeExports,
        typeOnly: typeOnlyExports,
        all: [...runtimeExports, ...typeOnlyExports],
    };
}

function generateCommonJsWrapper(exports, outputPath) {
    const template = `// Auto-generated NAPI wrapper for uzu native module
// Currently supports only Apple Silicon (arm64) on macOS
// Other platforms may be added in future releases

const { createRequire } = require('module');

const require_ = createRequire(__filename);

// Check for Apple Silicon support only
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error(\`This package only supports Apple Silicon (arm64) on macOS. Current platform: \${process.platform}-\${process.arch}\`);
}

let nativeBinding = null;

try {
  nativeBinding = require_('./uzu.node');
} catch (e) {
  throw new Error(\`Failed to load native binding: \${e.message}\`);
}

// Re-export all runtime exports
const {
${exports.map((exp) => `  ${exp}`).join(',\n')}
} = nativeBinding;

module.exports = {
${exports.map((exp) => `  ${exp}`).join(',\n')}
};`;

    fs.writeFileSync(outputPath, template);
}

function generateEsmWrapper(exports, outputPath) {
    const template = `// Auto-generated NAPI wrapper for uzu native module
// Currently supports only Apple Silicon (arm64) on macOS
// Other platforms may be added in future releases

import { createRequire } from 'module';

const require = createRequire(import.meta.url);

// Check for Apple Silicon support only
if (process.platform !== 'darwin' || process.arch !== 'arm64') {
  throw new Error(\`This package only supports Apple Silicon (arm64) on macOS. Current platform: \${process.platform}-\${process.arch}\`);
}

let nativeBinding = null;

try {
  nativeBinding = require('./uzu.node');
} catch (e) {
  throw new Error(\`Failed to load native binding: \${e.message}\`);
}

// Re-export all runtime exports as named exports  
export const {
${exports.map((exp) => `  ${exp}`).join(',\n')}
} = nativeBinding;

// Also provide default export
export default nativeBinding;`;

    fs.writeFileSync(outputPath, template);
}

function main() {
    const srcDir = path.join(__dirname, '../../src');
    const napiDir = path.join(srcDir, 'napi');
    const dtsPath = path.join(napiDir, 'uzu.d.ts');

    if (!fs.existsSync(dtsPath)) {
        console.error('TypeScript definitions file not found:', dtsPath);
        process.exit(1);
    }

    const exportInfo = extractExportsFromDts(dtsPath);
    console.log('Found runtime exports:', exportInfo.runtime);
    console.log('Found type-only exports:', exportInfo.typeOnly);

    const cjsPath = path.join(napiDir, 'uzu.js');
    const esmPath = path.join(napiDir, 'uzu.mjs');

    // Use all exports (runtime + type-only) for the wrappers
    generateCommonJsWrapper(exportInfo.all, cjsPath);
    generateEsmWrapper(exportInfo.all, esmPath);

    console.log('Generated:', cjsPath);
    console.log('Generated:', esmPath);
}

if (require.main === module) {
    main();
}

module.exports = {
    extractExportsFromDts,
    generateCommonJsWrapper,
    generateEsmWrapper,
};
