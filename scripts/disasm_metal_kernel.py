#!/usr/bin/env python3
"""Disassembles a Metal kernel as compiled for the M1 GPU, with the instruction timings of Mesa's AGX cycle model.

    scripts/disasm_metal_kernel.py 'UnifiedSampling<T=bfloat, is_stochastic=true, has_bitmask=false, ...>'

A kernel is named with its template parameters and its specialized arguments. The build script compiles it to
a metallib, metal-tt lowers that for the G13 GPU, and https://github.com/dougallj/applegpu disassembles the result.
"""

import argparse
import difflib
import io
import json
import os
import re
import signal
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GENERATED = ROOT / "crates/uzu-engine/src/backends/metal/kernel/generated"
CACHE = Path(os.environ.get("CARGO_TARGET_DIR", ROOT / "target")) / "disasm-metal-kernel"

APPLEGPU_COMMIT = "28dd80468d4a6c94fa4dac34ed488196f5fedc5f"
APPLEGPU_URL = f"https://github.com/dougallj/applegpu/archive/{APPLEGPU_COMMIT}.tar.gz"
MESA_COMMIT = "d9c0971e50754f629f512b846eac482bd7265f21"
MESA_URL = f"https://gitlab.freedesktop.org/mesa/mesa/-/raw/{MESA_COMMIT}/src/asahi/compiler/agx_performance.c"

# Function constant types, as the pipelines script that metal-tt reads names them.
CONSTANT_TYPES = {
    "bool": "Bool",
    "char": "Char",
    "int8_t": "Char",
    "uchar": "UChar",
    "uint8_t": "UChar",
    "short": "Short",
    "int16_t": "Short",
    "ushort": "UShort",
    "uint16_t": "UShort",
    "int": "Int",
    "int32_t": "Int",
    "uint": "UInt",
    "uint32_t": "UInt",
    "unsigned int": "UInt",
    "float": "Float",
}

# applegpu instructions and the Mesa opcodes whose timing describes them.
MESA_OPCODES = {
    "fmadd32": "FFMA",
    "fadd32": "FADD",
    "fmul32": "FMUL",
    "fmadd16": "HFMA",
    "fadd16": "HADD",
    "fmul16": "HMUL",
    "mov_imm": "MOV_IMM",
    "bitop": "BITOP",
    "icmpsel": "ICMPSEL",
    "fcmpsel": "FCMPSEL",
    "iadd": "IADD",
    "get_sr": "GET_SR",
    "imadd": "IMAD",
    "bfi": "BFI",
    "bfeil": "BFEIL",
    "extr": "EXTR",
    "asr": "ASR",
    "floor": "FLOOR",
    "sin_pt_1": "SIN_PT_1",
    "sin_pt_2": "SIN_PT_2",
    "log2": "LOG2",
    "exp2": "EXP2",
    "rcp": "RCP",
    "rsqrt": "RSQRT",
    "rsqrt_special": "SRSQRT",
    "simd_shuffle": "SHUFFLE",
    "quad_shuffle": "QUAD_SHUFFLE",
    "icmp_ballot": "ICMP_BALLOT",
    "fcmp_ballot": "FCMP_BALLOT",
    "icmp_quad_ballot": "ICMP_QUAD_BALLOT",
    "fcmp_quad_ballot": "FCMP_QUAD_BALLOT",
}
REDUCTION = r"(and|or|xor|iadd|fadd|fmul|min\.\w|max\.\w)"
MESA_OPCODE_PATTERNS = {
    rf"simd_prefix_{REDUCTION}": "SIMD_PREFIX",
    rf"quad_prefix_{REDUCTION}": "QUAD_PREFIX",
    rf"simd_{REDUCTION}": "SIMD_REDUCE",
    rf"quad_{REDUCTION}": "QUAD_REDUCE",
}
# Shuffle modes Mesa has no entry for; they are given the timing of the plain shuffle and marked with ~.
ASSUMED_OPCODE_PATTERNS = {r"simd_shuffle_\w+": "SHUFFLE", r"quad_shuffle_\w+": "QUAD_SHUFFLE"}


@dataclass
class Specialization:
    name: str
    index: int
    slot_type: str
    value_type: str
    optional: bool = False


@dataclass
class Variant:
    entry: str
    library: Path
    template: dict


@dataclass
class Kernel:
    name: str
    specializations: list = field(default_factory=list)
    variants: list = field(default_factory=list)

    def signature(self, template, arguments={}):
        """One variant as it is named on the command line, with the types of the arguments still to be given."""
        parameters = [f"{name}={short(value)}" for name, value in template.items()]
        for specialization in self.specializations:
            unset = f"<{short(specialization.value_type)}{'?' if specialization.optional else ''}>"
            parameters.append(f"{specialization.name}={arguments.get(specialization.name, unset)}")
        return f"{self.name}<{', '.join(parameters)}>" if parameters else self.name


@dataclass
class Instruction:
    offset: int
    code: bytes
    name: str
    text: str


def short(name):
    return name.rsplit("::", 1)[-1].strip()


def build_output():
    """Runs the uzu-engine build script, which compiles the kernels, and returns its output directory."""
    cargo = subprocess.Popen(
        ["cargo", "check", "--package", "uzu-engine", "--message-format=json"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        for line in cargo.stdout:
            message = json.loads(line)
            if message.get("reason") == "build-script-executed" and "/uzu-engine#" in message["package_id"]:
                return Path(message["out_dir"]) / "metal"
        sys.exit("cargo did not get to run the uzu-engine build script")
    finally:
        # Once the kernels are built, checking the crate that embeds them is not needed.
        if cargo.poll() is None:
            os.killpg(cargo.pid, signal.SIGTERM)
        cargo.wait()


def wrappers(output):
    """The wrapper sources of this build, found through the libraries that the generated Rust embeds.

    The directory can also hold wrappers of an earlier build: when the number of shards of a kernel changes, the old
    files stay next to the new ones.
    """
    for bindings in sorted(output.rglob("*.rs")):
        for library in re.findall(r'include_bytes!\(\s*"([^"]+\.metallib)"', bindings.read_text(encoding="utf-8")):
            yield Path(library).with_suffix(".metal")


def find_kernel(output, name):
    """Reads what the build script generated for a kernel: its entry points and its function constants."""
    marker = f"void _D{len(name)}{name}"
    kernel = Kernel(name)
    for wrapper in wrappers(output):
        text = wrapper.read_text(encoding="utf-8")
        if marker not in text:
            continue
        defines = {}
        for line in text.splitlines():
            if match := re.match(r"#define (\w+) (.+)", line):
                defines[match[1]] = match[2]
            elif match := re.match(r"#undef (\w+)", line):
                defines.pop(match[1], None)
            elif match := re.match(rf"\[\[kernel.*?\]\] void (_D{len(name)}{name}(?:S\d+V\w+)?)\(", line):
                template = {key: value for key, value in defines.items() if not value.startswith("__dsl_")}
                kernel.variants.append(Variant(match[1], wrapper.with_suffix(".metallib"), template))
        if not kernel.specializations:
            kernel.specializations = specializations(text, name)
    return kernel


def specializations(wrapper, kernel):
    slot = re.compile(rf"constant (.+?) __dsl_specialize_{kernel}_(\w+) \[\[function_constant\((\d+)\)\]\];")
    typed = re.compile(rf"constant (.+?) __dsl_typed_{kernel}_(\w+) = (\w+)")
    found = {m[2]: Specialization(m[2], int(m[3]), m[1], m[1]) for m in slot.finditer(wrapper)}
    for value_type, name, initializer in typed.findall(wrapper):
        found[name].value_type = value_type
        found[name].optional = initializer == "is_function_constant_defined"
    return sorted(found.values(), key=lambda specialization: specialization.index)


def all_kernels(output):
    names = set()
    for wrapper in wrappers(output):
        for length, rest in re.findall(r"\[\[kernel.*?\]\] void _D(\d+)(\w+)\(", wrapper.read_text(encoding="utf-8")):
            names.add(rest[: int(length)])
    return names


def named_constants():
    """The values of the generated enums and option sets, by type name."""
    constants = {}
    for header in GENERATED.glob("*.h"):
        text = header.read_text(encoding="utf-8")
        for name, body in re.findall(r"enum class (\w+) : \w+ \{(.*?)\};", text, re.S):
            constants[name] = {variant: int(value) for variant, value in re.findall(r"(\w+) = (\d+),", body)}
        for name, body in re.findall(r"struct (\w+) \{(.*?)\n\};", text, re.S):
            flags = re.findall(r"static constant constexpr \w+ (\w+) = ([\d <|()]+);", body)
            constants[name] = {flag: eval(value, {"__builtins__": {}}) for flag, value in flags}
    return constants


def constant_value(specialization, text):
    """A specialized argument as written on the command line, as the value of its function constant."""
    kind = CONSTANT_TYPES.get(specialization.slot_type)
    if kind is None:
        raise ValueError(f"{specialization.name}: unsupported type {specialization.slot_type}")
    if kind == "Bool":
        if text not in ("true", "false"):
            raise ValueError(f"{specialization.name} is a bool, got {text!r}")
        return kind, text == "true"
    if kind == "Float":
        return kind, float(text)
    names = named_constants().get(short(specialization.value_type), {})
    value = 0
    for term in text.split("|"):
        term = short(term)
        if term not in names and not re.fullmatch(r"-?(0x[0-9a-fA-F]+|\d+)", term):
            known = f"; it can be {', '.join(names)}" if names else ""
            raise ValueError(f"{specialization.name}: cannot read {term!r} as {specialization.value_type}{known}")
        value |= names[term] if term in names else int(term, 0)
    return kind, value


def parse(text):
    """Splits Name<parameter=value, ...> into the name and its arguments."""
    match = re.fullmatch(r"\s*(\w+)\s*(?:<(.*)>)?\s*", text)
    if not match:
        raise ValueError(f"cannot read {text!r} as Name<parameter=value, ...>")
    arguments = {}
    for part in filter(None, (match[2] or "").split(",")):
        argument = re.fullmatch(r"\s*(\w+)\s*=\s*(.+?)\s*", part)
        if not argument:
            raise ValueError(f"cannot read {part.strip()!r} as parameter=value")
        arguments[argument[1]] = argument[2]
    return match[1], arguments


def select(kernel, arguments):
    """Picks the variant and the function constants that the given arguments describe."""
    special = {specialization.name: specialization for specialization in kernel.specializations}
    template_names = list(kernel.variants[0].template)
    for name in arguments:
        if name not in special and name not in template_names:
            signature = kernel.signature(kernel.variants[0].template)
            raise ValueError(f"{kernel.name} has no parameter {name}; it is {signature}")

    template = {name: short(value) for name, value in arguments.items() if name in template_names}
    variants = [
        variant
        for variant in kernel.variants
        if all(short(variant.template[name]) == value for name, value in template.items())
    ]
    missing = [s.name for s in kernel.specializations if not s.optional and s.name not in arguments]
    if len(variants) != 1 or missing:
        problem = "name one variant and every specialized argument" if variants else "no variant matches, there are"
        candidates = variants or kernel.variants
        shown = [kernel.signature(variant.template, arguments) for variant in candidates[:16]]
        if len(candidates) > len(shown):
            shown.append(f"... and {len(candidates) - len(shown)} more")
        raise ValueError("\n  ".join([f"{kernel.name}: {problem}:", *shown]))

    constants = []
    for specialization in kernel.specializations:
        if specialization.name in arguments:
            constants.append((specialization, *constant_value(specialization, arguments[specialization.name])))
    return variants[0], constants


def compile_for_gpu(variant, constants, architecture, directory):
    """Lowers one entry point to GPU machine code and returns the binary archive metal-tt writes."""
    function = f"alias:library#{variant.entry}"
    libraries = {"paths": [{"label": "library", "path": str(variant.library)}]}
    if constants:
        libraries["specialized_functions"] = [
            {
                "label": "specialized",
                "function": function,
                "specialized_name": "kernel",
                "constant_values": [
                    {
                        "id_type": "FunctionConstantIndex",
                        "id": {"data": specialization.index},
                        "value_type": f"Constant{kind}",
                        "value": {"data": value},
                    }
                    for specialization, kind, value in constants
                ],
            }
        ]
        function = "alias:specialized#kernel"
    script = directory / "pipelines.mtlp-json"
    script.write_text(
        json.dumps({"libraries": libraries, "pipelines": {"compute_pipelines": [{"compute_function": function}]}})
    )
    archive = directory / "archive.bin"
    subprocess.run(["xcrun", "metal-tt", "-arch", architecture, "-o", archive, script], check=True)
    return archive


def fetch(url):
    with urllib.request.urlopen(url) as response:
        return response.read()


def applegpu_checkout():
    """The pinned applegpu sources with their archive extractor built, downloaded on first use."""
    checkout = CACHE / f"applegpu-{APPLEGPU_COMMIT}"
    if not checkout.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=CACHE) as directory:
            with tarfile.open(fileobj=io.BytesIO(fetch(APPLEGPU_URL))) as archive:
                archive.extractall(directory, filter="data")
            sources = Path(directory) / f"applegpu-{APPLEGPU_COMMIT}"
            compiler = ["xcrun", "clang++", "-std=c++11", "-O2"]
            subprocess.run(
                [*compiler, "metal-archive-extractor.cpp", "-o", "metal-archive-extractor"],
                cwd=sources / "compiler_explorer_tools",
                check=True,
            )
            sources.rename(checkout)
    return checkout


def timings():
    """Mesa's table of execution unit, latency and throughput per opcode."""
    table = CACHE / f"agx_performance-{MESA_COMMIT}.c"
    if not table.exists():
        CACHE.mkdir(parents=True, exist_ok=True)
        table.write_bytes(fetch(MESA_URL))
    entry = re.compile(r"\[AGX_OPCODE_(\w+)\]\s*=\s*\{\s*(\w+),\s*(\d+),\s*(\d+)\s*\}")
    return {m[1]: (m[2], int(m[3]), int(m[4])) for m in entry.finditer(table.read_text(encoding="utf-8"))}


def timing(instruction, table):
    """The unit, latency and throughput of an instruction, and whether they are borrowed from a related opcode."""
    if opcode := MESA_OPCODES.get(instruction.name):
        return (*table[opcode], False)
    for patterns, assumed in ((MESA_OPCODE_PATTERNS, False), (ASSUMED_OPCODE_PATTERNS, True)):
        for pattern, opcode in patterns.items():
            if re.fullmatch(pattern, instruction.name):
                return (*table[opcode], assumed)
    return None


def shaders(archive, extractor):
    """The machine code of every function in a binary archive, the main one last."""
    compute = subprocess.run([extractor, "--extract-compute", "-", archive], check=True, capture_output=True).stdout
    listing = subprocess.run([extractor, "--list-shaders", "-"], input=compute, check=True, capture_output=True)
    names = {"_agc.main.constant_program": "prolog", "_agc.main": "main"}
    found = []
    for offset, name in (line.split(" ", 1) for line in listing.stdout.decode().splitlines() if line.strip()):
        extract = [extractor, "--extract-named-shader", name, "--output", "-", "-"]
        code = subprocess.run(extract, input=compute, check=True, capture_output=True).stdout
        while code[-2:] == b"\x08\x00":  # the compiler pads with traps
            code = code[:-2]
        if any(code):
            found.append((name, code, int(offset, 0)))
    found.sort(key=lambda shader: list(names).index(shader[0]) + 1 if shader[0] in names else 0)
    # Addresses only matter when there are functions for the main shader to call.
    called = any(name not in names for name, _, _ in found)
    return [(names.get(name, name), code, offset if called else 0) for name, code, offset in found]


def disassemble(code, base, applegpu):
    position = 0
    while position < len(code):
        number = applegpu.opcode_to_number(code[position:])
        if number & 0xFFFF == 0:  # padding
            position += 2
            continue
        descriptor = next((d for d in applegpu.instruction_descriptors if d.matches(number)), None)
        if descriptor is None:
            yield Instruction(base + position, code[position : position + 2], "", "<disassembly failed>")
            position += 2
            continue
        size = descriptor.decode_size(number)
        text = str(descriptor.disassemble(number, pc=base + position))
        yield Instruction(base + position, code[position : position + size], descriptor.name, text)
        if descriptor.name == "stop":
            return
        position += size


def report(name, instructions, table):
    width = max(len(instruction.text) for instruction in instructions)
    print(f"{name}:")
    print(f"{'':>6}  {'':20} {'':{width}}  unit  lat  tp")
    for instruction in instructions:
        line = f"{instruction.offset:>6x}: {instruction.code.hex():20} {instruction.text:{width}}"
        if found := timing(instruction, table):
            unit, latency, throughput, assumed = found
            line += f"  {unit:4}  {latency:>3}  {throughput:>2}{' ~' if assumed else ''}"
        print(line.rstrip())
    print()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kernel", help="Name, or Name<parameter=value, ...> with template and specialized arguments")
    parser.add_argument("--arch", default="applegpu_g13g", help="metal-tt architecture (default: %(default)s, the M1)")
    options = parser.parse_args()

    try:
        name, arguments = parse(options.kernel)
    except ValueError as error:
        sys.exit(str(error))

    output = build_output()
    kernel = find_kernel(output, name)
    if not kernel.variants:
        close = difflib.get_close_matches(name, all_kernels(output), n=5, cutoff=0.4)
        sys.exit(f"no kernel named {name}" + (f"; did you mean {', '.join(close)}?" if close else ""))
    try:
        variant, constants = select(kernel, arguments)
    except ValueError as error:
        sys.exit(str(error))

    checkout = applegpu_checkout()
    sys.path.insert(0, str(checkout))
    import applegpu

    table = timings()
    with tempfile.TemporaryDirectory(prefix="uzu-disasm-") as directory:
        try:
            archive = compile_for_gpu(variant, constants, options.arch, Path(directory))
        except subprocess.CalledProcessError as error:
            sys.exit(f"metal-tt failed with status {error.returncode}")
        extractor = checkout / "compiler_explorer_tools/metal-archive-extractor"
        found = [(name, list(disassemble(code, base, applegpu))) for name, code, base in shaders(archive, extractor)]

    print(f"{kernel.signature(variant.template, arguments)} for {options.arch}\n")
    for shader, instructions in found:
        report(shader, instructions, table)


if __name__ == "__main__":
    main()
