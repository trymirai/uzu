#!/usr/bin/env python3
"""Regenerate PUBLIC Slang kernel stubs from current Metal declarations.

Run with `direnv exec . python3 autogen.py` after a Metal build has generated
its shared type headers. Requires the Metal compiler; no cached build metadata
is used. Kernel bodies intentionally remain unimplemented.
"""

import json
from pathlib import Path
import re
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parent
METAL = ROOT / 'crates/uzu-engine/src/backends/metal/kernel'
SLANG = ROOT / 'crates/uzu-engine/src/backends/slang'


def annotations(node):
    def strings(child):
        if child['kind'] == 'StringLiteral':
            yield json.loads(child['value'])
        for nested in child.get('inner', []):
            yield from strings(nested)
    return [list(strings(child)) for child in node.get('inner', [])
            if child['kind'] == 'AnnotateAttr']


def generate(path, cache):
    source = path.read_text()
    result = subprocess.run([
        'xcrun', '-sdk', 'macosx', 'metal', '-x', 'metal', '-std=metal4.0',
        '-mmacosx-version-min=26.4', f'-fmodules-cache-path={cache}',
        '-I', str(METAL / 'generated'), '-DDSL_ANALYZE', str(path),
        '-fsyntax-only', '-Xclang', '-ast-dump=json',
    ], check=True, capture_output=True, text=True)
    ast = json.loads(result.stdout)
    imports = set()
    defines = {}
    visited = set()

    def read_constants(header):
        header = header.resolve()
        if header in visited:
            return
        visited.add(header)
        text = header.read_text()
        for include in re.findall(r'^\s*#include "([^"\n]+)"', text, re.M):
            dependency = header.parent / include
            if not dependency.exists():
                dependency = METAL / 'generated' / include
            if dependency.exists():
                read_constants(dependency)
        defines.update(re.findall(r'^\s*#define[ \t]+(\w+)[ \t]+([^\n]+)', text, re.M))
        defines.update(re.findall(r'(?:static constant constexpr|UZU_CONST)\s+\w+\s+(\w+)\s*=\s*([^;]+);', text))

    read_constants(path)

    def expression(value):
        # Dispatch annotations contain source expressions, not expanded macros.
        for _ in range(len(defines) + 1):
            expanded = re.sub(r'\b\w+\b', lambda m: defines.get(m[0], m[0]), value)
            if expanded == value:
                return re.sub(r'\b(\d+)[uU]\b', r'\1', value)
            value = expanded
        raise ValueError(f'{path}: recursive macro in {value}')

    def slang_type(ty, generics=()):
        if ty in generics:
            return ty
        if match := re.fullmatch(r'(char|uchar|short|ushort|int|uint|float|half)([234])', ty):
            return f'vector<{slang_type(match[1])}, {match[2]}>'
        if ty.startswith('uzu::'):
            _, module, name = ty.split('::')
            imports.add(f'import generated.{module};')
            return name
        return {
            'float': 'float', 'float4': 'float4', 'half': 'half',
            # Fixed BF16 buffers retain their raw 16-bit storage; no BF16 arithmetic.
            'bfloat': 'uint16_t', 'bfloat16_t': 'uint16_t',
            'bool': 'bool', 'char': 'int8_t', 'signed char': 'int8_t',
            'short': 'int16_t', 'int8_t': 'int8_t', 'uchar': 'uint8_t', 'unsigned char': 'uint8_t',
            'uint8_t': 'uint8_t', 'ushort': 'uint16_t', 'uint16_t': 'uint16_t',
            'int': 'int', 'int32_t': 'int', 'uint': 'uint', 'uint32_t': 'uint',
            'unsigned int': 'uint', 'uint64_t': 'uint64_t',
            'unsigned long': 'uint64_t', '_atomic<uint>': 'uint',
            'ThreadContext': 'ThreadContext',
        }[ty]

    kernels = []
    for node in ast['inner']:
        template = []
        if node['kind'] == 'FunctionTemplateDecl':
            template = [n for n in node['inner'] if n['kind'] in
                        ('TemplateTypeParmDecl', 'NonTypeTemplateParmDecl')]
            node = next(n for n in node['inner'] if n['kind'] == 'FunctionDecl')
        if node['kind'] != 'FunctionDecl':
            continue
        attrs = annotations(node)
        if ['dsl.public'] not in attrs or ['dsl.kernel'] not in attrs:
            continue
        parameters = [n for n in node['inner'] if n['kind'] == 'ParmVarDecl']
        generic_names = {n['name'] for n in template}
        generics, lines, arguments = [], [], []
        variants = {a[1]: a[2] for a in attrs if a[0] == 'dsl.variants'}
        for param in template:
            name = param['name']
            is_type = param['kind'] == 'TemplateTypeParmDecl'
            constraint = '__BuiltinFloatingPointType' if is_type else slang_type(param['type']['qualType'])
            generics.append(f'{name}: {constraint}' if is_type else f'let {name}: {constraint}')
            values = [v.strip() for v in variants[name].split(',')]
            if is_type:
                # WebGPU has no BF16 support. An empty variant list keeps the
                # public signature while its constructor remains unsupported.
                values = [slang_type(v) for v in values if v not in ('bfloat', 'bfloat16_t')]
            lines.append(f'[[Variants({json.dumps(name)}, {json.dumps(", ".join(values))})]]')
        for param in parameters:
            name = param['name']
            ty = param['type'].get('desugaredQualType', param['type']['qualType'])
            if 'threadgroup' in ty.split():
                # Scratch storage belongs to the implementation, not the public API.
                continue
            attrs = annotations(param)
            attr = attrs[0] if attrs else []
            if attr and attr[0] in ('dsl.axis', 'dsl.groups', 'dsl.threads'):
                annotation = attr[0].removeprefix('dsl.').capitalize()
                values = ', '.join(json.dumps(expression(v)) for v in attr[1:])
                arguments.append(f'  [[{annotation}({values})]] uint {name},')
                continue
            element = re.sub(r'\b(const|device|constant|threadgroup)\b|[&*()]|\[[^]]*\]', '', ty).strip()
            element = ' '.join(element.split())
            element = slang_type(element, generic_names)
            begin, end = param['range']['begin'], param['range']['end']
            begin, end = begin.get('spellingLoc', begin), end.get('spellingLoc', end)
            declaration = source[begin['offset']:end['offset'] + end['tokLen']]
            array = re.search(r'\[([^]]+)\]', declaration)
            if 'device' in ty.split():
                access = 'StructuredBuffer' if 'const' in ty.split() else 'RWStructuredBuffer'
                ty = f'{access}<{element}>'
            elif 'constant' in ty.split() and '*' in ty:
                ty = f'{element}[{expression(array[1])}]' if array else f'[[Constant]] StructuredBuffer<{element}>'
            else:
                ty = element
            prefix = ''
            if attr:
                if attr[0] == 'dsl.optional':
                    prefix = f'[[Optional({json.dumps(expression(attr[1]))})]] '
                elif attr[0] in ('dsl.specialize', 'dsl.specialize_if'):
                    prefix = '[[Specialize]] '
                    if attr[0] == 'dsl.specialize_if':
                        prefix += f'[[Optional({json.dumps(expression(attr[1]))})]] '
                else:
                    raise ValueError(f'{path}: unsupported annotation {attr}')
            arguments.append(f'  {prefix}{ty} {name},')
        for attr in annotations(node):
            if attr[0] == 'dsl.constraint':
                lines.append(f'[[Constraint({json.dumps(expression(attr[1]))})]]')
        generic = f'<{", ".join(generics)}>' if generics else ''
        lines.extend([f'[[Public]] [[Kernel]] func {node["name"]}{generic}(', *arguments,
                      ') {', '  // TODO', '}'])
        kernels.append('\n'.join(lines))
    if not kernels:
        return None
    return '\n'.join([
        '// Generated by autogen.py. Kernel bodies are unimplemented.',
        'import definitions;', *sorted(imports), '', '\n\n'.join(kernels), '',
    ])


def main():
    generated = {}
    with tempfile.TemporaryDirectory(prefix='uzu-slang-autogen-') as cache:
        for path in sorted(METAL.rglob('*.metal')):
            if not re.search(r'\bPUBLIC\b', path.read_text()):
                continue
            try:
                content = generate(path, cache)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f'{path}:\n{error.stderr}') from error
            if content:
                generated[SLANG / path.relative_to(METAL).with_suffix('.slang')] = content
    # Compute everything first so an analyzer failure never leaves partial output.
    for path, content in generated.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists() or path.read_text() != content:
            path.write_text(content)
    for path in SLANG.rglob('*.slang'):
        relative = path.relative_to(SLANG)
        if relative.parts[0] in ('definitions', 'generated') or relative == Path('definitions.slang'):
            continue
        if path not in generated and path.read_text().startswith('// Generated by autogen.py.'):
            path.unlink()
    print(f'Generated {sum(s.count("[[Public]]") for s in generated.values())} public kernel stubs in {len(generated)} files.')


if __name__ == '__main__':
    main()
