// Checks that @trymirai/uzu-wasm exposes the same TypeScript API as @trymirai/uzu (napi),
// so code written against one package compiles against the other.
//
// The napi package is the reference: wasm may have extras (`free`, `initSync`, ...), but must
// not lack or change anything napi has. For every napi export, this reports whether wasm is
// missing it, lacks members, or declares an incompatible type. Incompatible types are found by
// generating type-level assertions and type-checking them in memory.
//
// Types are compared after `Normalize`, which replaces string enums with their string values
// (TypeScript treats const enums nominally, so identical enums from two packages never match)
// and drops the members wasm-bindgen adds to every class.
//
// Only the declarations are compared. Runtime behavior that the .d.ts doesn't describe
// (what errors are thrown, whether `init()` is required) needs a runtime test.
//
// Usage: node scripts/utils/check-api.cjs [--verbose]
// Exits with 0 when the APIs match, 1 when they differ, 2 when the check can't run.

const fs = require('fs');
const path = require('path');
const ts = require('typescript');

const verbose = process.argv.includes('--verbose');

const packageDirectory = path.resolve(__dirname, '../..');
const napiEntry = path.resolve(packageDirectory, '../typescript/src/index.ts');
const wasmEntry = path.resolve(packageDirectory, 'src/index.ts');
const checkFile = path.join(packageDirectory, '__api_check__.ts');

const compilerOptions = {
    target: ts.ScriptTarget.ES2022,
    lib: ['lib.esnext.d.ts', 'lib.dom.d.ts'],
    module: ts.ModuleKind.ESNext,
    moduleResolution: ts.ModuleResolutionKind.Bundler,
    strict: true,
    skipLibCheck: true,
    noEmit: true,
    // Both packages take zod as a dependency; an app has one copy, but each package's node_modules
    // may hold a different version, whose types wouldn't match.
    baseUrl: packageDirectory,
    paths: { zod: ['node_modules/zod'] },
};

// Members wasm-bindgen adds to every class.
const WASM_BINDGEN_EXTRAS = new Set(['free', '[Symbol.dispose]']);

const UNRESOLVED_MODULE_CODES = new Set([2307, 2792]);

const HELPERS = `
type Assignable<Expected, Actual extends Expected> = Actual;
type WasmBindgenExtras = 'free' | typeof Symbol.dispose;
type Opaque = ArrayBufferView | ArrayBufferLike | Date | RegExp | Error;
type Normalize<T> = T extends string
    ? \`\${T}\`
    : T extends void
      ? undefined
      : T extends number | boolean | bigint | symbol | null | Opaque
        ? T
        : T extends (...args: infer A) => infer R
          ? (...args: NormalizeTuple<A>) => Normalize<R>
          : T extends Promise<infer U>
            ? Promise<Normalize<U>>
            : T extends ReadonlyArray<infer U>
              ? Normalize<U>[]
              : T extends object
                ? { [K in keyof T as K extends WasmBindgenExtras ? never : K]: Normalize<T[K]> }
                : T;
type NormalizeTuple<T> = { [K in keyof T]: Normalize<T[K]> };
type Statics<T> = Omit<T, 'prototype'>;
type Constructor<T> = T extends new (...args: infer A) => infer R ? (...args: NormalizeTuple<A>) => Normalize<R> : never;
type EnumValues<T> = { [K in keyof T]: T[K] extends string ? \`\${T[K]}\` : T[K] };
type Common<Expected, Actual> = Pick<Expected, keyof Expected & keyof Actual>;
`;

function main() {
    for (const entry of [napiEntry, wasmEntry]) {
        if (!fs.existsSync(entry)) {
            fail(`Missing ${path.relative(process.cwd(), entry)}`);
        }
    }

    const program = ts.createProgram([napiEntry, wasmEntry], compilerOptions);
    const checker = program.getTypeChecker();
    const napiExports = exportsOf(program, napiEntry);
    const wasmExports = exportsOf(program, wasmEntry);

    const missing = [];
    const typeOnly = [];
    const findings = new Map();
    const assertions = [];
    const addFinding = (name, label, message) => {
        const entries = findings.get(name) ?? [];
        entries.push({ label, message });
        findings.set(name, entries);
    };

    for (const [name, symbol] of napiExports) {
        const wasmSymbol = wasmExports.get(name);
        if (!wasmSymbol) {
            missing.push(name);
            continue;
        }
        if (hasValue(symbol) && !hasValue(wasmSymbol)) {
            typeOnly.push(name);
        }
        // napi exposes a Rust enum with data as a union of one class per variant. Comparing such
        // a union structurally with anything else only yields `{}` noise, so compare the shape.
        const napiVariants = variantClasses(checker, symbol);
        if (napiVariants) {
            const wasmVariants = variantClasses(checker, wasmSymbol);
            if (!wasmVariants) {
                const wasmShape = checker.typeToString(
                    checker.getDeclaredTypeOfSymbol(wasmSymbol),
                    undefined,
                    ts.TypeFormatFlags.InTypeAlias,
                );
                addFinding(
                    name,
                    'shape',
                    `napi is a union of classes (${napiVariants.join(' | ')}), wasm is ${truncate(wasmShape, 160)}`,
                );
                continue;
            }
            const missingVariants = napiVariants.filter((variant) => !wasmVariants.includes(variant));
            if (missingVariants.length > 0) {
                addFinding(name, 'missing variants', missingVariants.join(', '));
            }
        }
        for (const { label, names } of missingMembers(checker, symbol, wasmSymbol)) {
            addFinding(name, label, names.join(', '));
        }
        if (isClass(symbol) && hasPublicConstructor(symbol) && hasValue(wasmSymbol) && !hasPublicConstructor(wasmSymbol)) {
            addFinding(name, 'constructor', 'wasm has no public constructor');
        }
        for (const [label, code] of assertionsFor(name, symbol, wasmSymbol)) {
            assertions.push({ name, label, code });
        }
    }

    const diagnostics = typeCheck(assertions);
    for (const { assertion, diagnostic } of diagnostics.assertionErrors) {
        addFinding(assertion.name, assertion.label, describe(diagnostic));
    }
    if (diagnostics.setupErrors.length > 0) {
        for (const diagnostic of diagnostics.setupErrors) {
            console.error(formatDiagnostic(diagnostic));
        }
        fail(
            'The check itself failed to type-check. If a module is missing, run `pnpm install` in both ' +
                'bindings/typescript and bindings/wasm.',
        );
    }

    const sortedFindings = new Map([...findings].sort(([left], [right]) => compareNames(left, right)));
    report(napiExports.size, missing, typeOnly, sortedFindings);
    process.exit(missing.length + typeOnly.length + findings.size === 0 ? 0 : 1);
}

function exportsOf(program, entry) {
    const checker = program.getTypeChecker();
    const moduleSymbol = checker.getSymbolAtLocation(program.getSourceFile(entry));
    const exports = new Map();
    for (const symbol of checker.getExportsOfModule(moduleSymbol)) {
        const resolved = symbol.flags & ts.SymbolFlags.Alias ? checker.getAliasedSymbol(symbol) : symbol;
        exports.set(symbol.name, resolved);
    }
    return new Map([...exports].sort(([left], [right]) => compareNames(left, right)));
}

function compareNames(left, right) {
    return left.localeCompare(right, 'en', { sensitivity: 'base' }) || left.localeCompare(right);
}

function hasValue(symbol) {
    return (symbol.flags & ts.SymbolFlags.Value) !== 0;
}

function hasType(symbol) {
    return (symbol.flags & ts.SymbolFlags.Type) !== 0;
}

function isClass(symbol) {
    return (symbol.flags & ts.SymbolFlags.Class) !== 0;
}

function isEnum(symbol) {
    return (symbol.flags & ts.SymbolFlags.Enum) !== 0;
}

function isPublic(declaration) {
    return !(ts.getCombinedModifierFlags(declaration) & (ts.ModifierFlags.Private | ts.ModifierFlags.Protected));
}

function hasPublicConstructor(symbol) {
    return (symbol.declarations ?? []).some(
        (declaration) =>
            ts.isClassDeclaration(declaration) &&
            declaration.members.some((member) => ts.isConstructorDeclaration(member) && isPublic(member)),
    );
}

// Returns the variant class names when `symbol` is a type alias for a class or a union of classes.
function variantClasses(checker, symbol) {
    if (!(symbol.flags & ts.SymbolFlags.TypeAlias)) {
        return undefined;
    }
    const type = checker.getDeclaredTypeOfSymbol(symbol);
    const names = (type.isUnion() ? type.types : [type]).map((member) =>
        member.symbol && member.symbol.flags & ts.SymbolFlags.Class ? member.symbol.name : undefined,
    );
    return names.every(Boolean) ? names : undefined;
}

function typeArguments(symbol) {
    const declaration = symbol.declarations?.find((candidate) => candidate.typeParameters);
    const count = declaration?.typeParameters?.length ?? 0;
    return count === 0 ? '' : `<${Array(count).fill('any').join(', ')}>`;
}

function memberNames(checker, type, { skip = [] } = {}) {
    const names = new Set();
    for (const property of checker.getPropertiesOfType(type)) {
        if (property.declarations?.some((declaration) => !isPublic(declaration))) {
            continue;
        }
        const name = checker.symbolToString(property);
        if (!skip.includes(name)) {
            names.add(name);
        }
    }
    return names;
}

function missingMembers(checker, symbol, wasmSymbol) {
    const result = [];
    const compare = (label, napiType, wasmType, options) => {
        const wasmNames = memberNames(checker, wasmType, options);
        const names = [...memberNames(checker, napiType, options)].filter((name) => !wasmNames.has(name));
        if (names.length > 0) {
            result.push({ label, names });
        }
    };

    const instanceLabel = isEnum(symbol) ? undefined : 'missing members';
    if (instanceLabel && hasType(wasmSymbol) && !(symbol.flags & ts.SymbolFlags.TypeAlias)) {
        compare(instanceLabel, checker.getDeclaredTypeOfSymbol(symbol), checker.getDeclaredTypeOfSymbol(wasmSymbol));
    }
    if ((isClass(symbol) || isEnum(symbol)) && hasValue(wasmSymbol)) {
        compare(
            isEnum(symbol) ? 'missing enum members' : 'missing static members',
            checker.getTypeOfSymbol(symbol),
            checker.getTypeOfSymbol(wasmSymbol),
            { skip: ['prototype'] },
        );
    }
    return result;
}

// Yields [label, type expression] pairs; each expression fails to type-check on a mismatch.
function* assertionsFor(name, symbol, wasmSymbol) {
    const napiType = `N.${name}${typeArguments(symbol)}`;
    const wasmType = `W.${name}${typeArguments(wasmSymbol)}`;
    const napiValue = `typeof N.${name}`;
    const wasmValue = `typeof W.${name}`;

    if (isClass(symbol)) {
        if (hasType(wasmSymbol)) {
            yield ['members', `Assignable<Normalize<Common<${napiType}, ${wasmType}>>, Normalize<${wasmType}>>`];
        }
        if (hasValue(wasmSymbol)) {
            const napiStatics = `Statics<${napiValue}>`;
            const wasmStatics = `Statics<${wasmValue}>`;
            yield [
                'static members',
                `Assignable<Normalize<Common<${napiStatics}, ${wasmStatics}>>, Normalize<${wasmStatics}>>`,
            ];
            if (hasPublicConstructor(symbol) && hasPublicConstructor(wasmSymbol)) {
                yield ['constructor', `Assignable<Constructor<${napiValue}>, Constructor<${wasmValue}>>`];
            }
        }
        return;
    }

    if (isEnum(symbol)) {
        if (hasValue(wasmSymbol)) {
            const napiMembers = `EnumValues<${napiValue}>`;
            const wasmMembers = `EnumValues<${wasmValue}>`;
            yield ['enum members', `Assignable<Common<${napiMembers}, ${wasmMembers}>, ${wasmMembers}>`];
        }
        if (hasType(wasmSymbol)) {
            yield ['values wasm lacks', `Assignable<Normalize<${wasmType}>, Normalize<${napiType}>>`];
            yield ['values napi lacks', `Assignable<Normalize<${napiType}>, Normalize<${wasmType}>>`];
        }
        return;
    }

    if (symbol.flags & (ts.SymbolFlags.Interface | ts.SymbolFlags.TypeAlias)) {
        if (hasType(wasmSymbol)) {
            yield ['napi shape → wasm type', `Assignable<Normalize<${wasmType}>, Normalize<${napiType}>>`];
            yield ['wasm shape → napi type', `Assignable<Normalize<${napiType}>, Normalize<${wasmType}>>`];
        }
        return;
    }

    if (symbol.flags & (ts.SymbolFlags.Function | ts.SymbolFlags.Variable) && hasValue(wasmSymbol)) {
        yield ['signature', `Assignable<Normalize<${napiValue}>, Normalize<${wasmValue}>>`];
    }
}

function typeCheck(assertions) {
    const lines = [
        `import type * as N from './${relativeImport(napiEntry)}';`,
        `import type * as W from './${relativeImport(wasmEntry)}';`,
        ...HELPERS.split('\n'),
    ];
    const assertionByLine = new Map();
    assertions.forEach((assertion, index) => {
        lines.push(`type _${index} = ${assertion.code};`);
        assertionByLine.set(lines.length, assertion);
    });
    const source = lines.join('\n');

    const host = ts.createCompilerHost(compilerOptions);
    const getSourceFile = host.getSourceFile.bind(host);
    host.getSourceFile = (fileName, languageVersion, ...rest) =>
        fileName === checkFile
            ? ts.createSourceFile(fileName, source, languageVersion, true)
            : getSourceFile(fileName, languageVersion, ...rest);
    const fileExists = host.fileExists.bind(host);
    host.fileExists = (fileName) => fileName === checkFile || fileExists(fileName);
    const readFile = host.readFile.bind(host);
    host.readFile = (fileName) => (fileName === checkFile ? source : readFile(fileName));

    const program = ts.createProgram([checkFile], compilerOptions, host);
    const assertionErrors = [];
    const setupErrors = [];
    for (const diagnostic of ts.getPreEmitDiagnostics(program)) {
        if (diagnostic.file?.fileName !== checkFile) {
            // Errors inside the packages' own sources are for their own lint to catch (tool.ts,
            // for example, relies on Node's timers), but a module that doesn't resolve turns
            // types into `any` and would make assertions pass silently.
            if (UNRESOLVED_MODULE_CODES.has(diagnostic.code)) {
                setupErrors.push(diagnostic);
            }
            continue;
        }
        const line = diagnostic.file.getLineAndCharacterOfPosition(diagnostic.start).line + 1;
        const assertion = assertionByLine.get(line);
        if (assertion) {
            assertionErrors.push({ assertion, diagnostic });
        } else {
            setupErrors.push(diagnostic);
        }
    }
    return { assertionErrors, setupErrors };
}

function relativeImport(file) {
    return path.relative(packageDirectory, file).replace(/\\/g, '/').replace(/\.ts$/, '');
}

// Condenses TypeScript's elaboration chain into "at <member path>: <innermost reason>". The
// headline ("does not satisfy the constraint") only restates the assertion, so it is dropped.
function describe(diagnostic) {
    const lines = ts
        .flattenDiagnosticMessageText(diagnostic.messageText, '\n')
        .split('\n')
        .map((line) => line.trim().replace(/import\("[^"]+"\)\./g, ''));
    const chain = lines.length > 1 ? lines.slice(1) : lines;
    if (verbose) {
        return chain.join('\n');
    }

    const segments = [];
    for (const line of chain) {
        const match =
            line.match(/^Types of property '(.+?)' are incompatible/) ??
            line.match(/^The types of '(.+?)' are incompatible/) ??
            line.match(/^The types returned by '(.+?)' are incompatible/);
        if (match) {
            segments.push(match[1]);
            continue;
        }
        const parameter = line.match(/^Types of parameters '(.+?)' and '(.+?)' are incompatible/);
        if (parameter) {
            segments.push(`(${parameter[1]})`);
        }
    }
    const reason = truncate(chain[chain.length - 1], 240);
    const location = segments.join('.').replace(/\.\(/g, '(');
    return location ? `at ${location}: ${reason}` : reason;
}

function truncate(text, length) {
    return text.length <= length ? text : `${text.slice(0, length - 1)}…`;
}

function formatDiagnostic(diagnostic) {
    return ts.formatDiagnostic(diagnostic, {
        getCanonicalFileName: (fileName) => fileName,
        getCurrentDirectory: () => process.cwd(),
        getNewLine: () => '\n',
    });
}

function report(total, missing, typeOnly, findings) {
    const matching = total - new Set([...missing, ...typeOnly, ...findings.keys()]).size;
    console.log(`API check: @trymirai/uzu-wasm against @trymirai/uzu (${total} exports, ${matching} match)\n`);

    if (missing.length > 0) {
        console.log(`Missing from wasm (${missing.length}):`);
        console.log(wrap(missing.join(', '), '  '));
        console.log();
    }

    if (typeOnly.length > 0) {
        console.log(`Type only in wasm, no runtime value (${typeOnly.length}):`);
        console.log(wrap(typeOnly.join(', '), '  '));
        console.log();
    }

    if (findings.size > 0) {
        console.log(`Different (${findings.size}):`);
        for (const [name, entries] of findings) {
            console.log(`  ${name}`);
            for (const { label, message } of entries) {
                const [first, ...rest] = message.split('\n');
                console.log(`    ${label}: ${first}`);
                for (const line of rest) {
                    console.log(`      ${line}`);
                }
            }
        }
        console.log();
    }

    if (missing.length + typeOnly.length + findings.size === 0) {
        console.log('The APIs match.');
    }
}

function wrap(text, indent, width = 100) {
    const lines = [];
    let line = '';
    for (const word of text.split(' ')) {
        if (line && indent.length + line.length + word.length + 1 > width) {
            lines.push(indent + line);
            line = word;
        } else {
            line = line ? `${line} ${word}` : word;
        }
    }
    if (line) {
        lines.push(indent + line);
    }
    return lines.join('\n');
}

function fail(message) {
    console.error(message);
    process.exit(2);
}

main();
