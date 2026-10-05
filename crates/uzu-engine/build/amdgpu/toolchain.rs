use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
    process::{Command, Stdio},
};

use anyhow::{Context, bail};
use serde::Deserialize;

use super::ast::{MetalAstKind, MetalAstNode, MetalKernelInfo};

/// Directory with `clang` and `ld.lld` that support the AMDGPU target.
const ENV_LLVM: &str = "UZU_AMDGPU_LLVM";
/// Comma separated processor names; the first one is compiled.
const ENV_TARGETS: &str = "UZU_AMDGPU_TARGETS";
const DEFAULT_TARGET: &str = "gfx1150";

#[derive(Debug)]
pub struct AmdgpuToolchain {
    clang: PathBuf,
    lld: PathBuf,
    target: String,
    compat_directory: PathBuf,
    generated_directory: PathBuf,
    cache_key: [u8; blake3::OUT_LEN],
}

fn executable(
    directory: &Path,
    name: &str,
) -> PathBuf {
    directory.join(if cfg!(windows) {
        format!("{name}.exe")
    } else {
        name.to_string()
    })
}

fn supports_amdgpu(clang: &Path) -> bool {
    Command::new(clang)
        .arg("-print-targets")
        .output()
        .map(|output| output.status.success() && String::from_utf8_lossy(&output.stdout).contains("amdgcn"))
        .unwrap_or(false)
}

/// clang 22 stores the generic address of a function parameter captured by reference into the
/// lambda's private (32-bit) reference field. The store runs past the closure, and SROA drops the
/// code that depends on it (AncestorAttention lost its whole KV loop). clang 23 is correct.
const MIN_CLANG_MAJOR: u32 = 23;

fn clang_major_version(clang: &Path) -> Option<u32> {
    let output = Command::new(clang).arg("--version").output().ok()?;
    let text = String::from_utf8_lossy(&output.stdout);
    let version = text.split("clang version ").nth(1)?;
    version.split(|c: char| !c.is_ascii_digit()).next()?.parse().ok()
}

fn find_llvm_directory() -> anyhow::Result<PathBuf> {
    println!("cargo::rerun-if-env-changed={ENV_LLVM}");
    println!("cargo::rerun-if-env-changed=HIP_PATH");
    println!("cargo::rerun-if-env-changed=ROCM_PATH");

    let mut candidates = Vec::new();
    if let Some(directory) = env::var_os(ENV_LLVM) {
        candidates.push(PathBuf::from(directory));
    }
    if let Some(hip_path) = env::var_os("HIP_PATH") {
        candidates.push(PathBuf::from(hip_path).join("bin"));
    }
    if let Some(rocm_path) = env::var_os("ROCM_PATH") {
        candidates.push(PathBuf::from(rocm_path).join("llvm").join("bin"));
    }
    if let Some(path) = env::var_os("PATH") {
        candidates.extend(env::split_paths(&path));
    }

    let mut too_old = Vec::new();
    for directory in candidates {
        let clang = executable(&directory, "clang");
        if !(clang.is_file() && executable(&directory, "ld.lld").is_file() && supports_amdgpu(&clang)) {
            continue;
        }
        match clang_major_version(&clang) {
            Some(major) if major >= MIN_CLANG_MAJOR => return Ok(directory),
            major => too_old.push(format!("{} ({major:?})", clang.display())),
        }
    }

    let too_old = if too_old.is_empty() {
        String::new()
    } else {
        format!(" (too old: {})", too_old.join(", "))
    };
    bail!(
        "no clang >= {MIN_CLANG_MAJOR} with the AMDGPU target found{too_old}; set {ENV_LLVM} to a directory with \
         clang and ld.lld (the official LLVM Windows release is built without AMDGPU; conda-forge clang works)"
    )
}

/// Short name -> `uzu::<file>::<Name>` for every gpu_types enum, option set and struct.
pub type GpuTypeNames = HashMap<String, String>;

/// Rewrites clang's C++ for OpenCL spelling of a type into the MSL spelling that `ast.rs` classifies:
/// address spaces become `device` / `constant` / `threadgroup` (`constant` implies `const`, as Metal
/// prints it), implicit `__private` / `__generic` qualifiers disappear, and gpu_types names that
/// clang prints unqualified get their `uzu::` path.
fn normalize_type(
    type_text: &str,
    gpu_type_names: &GpuTypeNames,
) -> String {
    let mut output = String::with_capacity(type_text.len());
    let mut chars = type_text.char_indices().peekable();
    let mut previous_is_scope = false;

    while let Some((start, character)) = chars.next() {
        if character.is_ascii_alphabetic() || character == '_' {
            let mut end = start + character.len_utf8();
            while let Some(&(index, next)) = chars.peek() {
                if next.is_ascii_alphanumeric() || next == '_' {
                    end = index + next.len_utf8();
                    chars.next();
                } else {
                    break;
                }
            }
            let identifier = &type_text[start..end];
            match identifier {
                "__global" => output.push_str("device"),
                "__constant" => output.push_str("constant"),
                "__local" => output.push_str("threadgroup"),
                "__private" | "__generic" => {},
                _ => match (!previous_is_scope).then(|| gpu_type_names.get(identifier)).flatten() {
                    Some(path) => output.push_str(path),
                    None => output.push_str(identifier),
                },
            }
            previous_is_scope = false;
        } else {
            previous_is_scope = character == ':';
            output.push(character);
        }
    }

    // Separate `*` / `&` from what used to follow them (`&__private` -> `&`) and collapse spaces.
    let normalized = output.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.starts_with("constant ") {
        format!("const {normalized}")
    } else {
        normalized
    }
}

/// End (index of the closing quote) of the JSON string literal starting at `start`.
fn string_end(
    json: &[u8],
    start: usize,
) -> Option<usize> {
    let mut index = start + 1;
    while index < json.len() {
        match json[index] {
            b'\\' => index += 2,
            b'"' => return Some(index),
            _ => index += 1,
        }
    }
    None
}

/// Normalizes the values of `qualType` / `desugaredQualType` keys in the JSON text, keeping the
/// key order intact (clang_ast expects `kind` right after `id`, so no round trip through a map).
fn normalize_types(
    json: &[u8],
    gpu_type_names: &GpuTypeNames,
) -> anyhow::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(json.len() + json.len() / 16);
    let mut index = 0;
    while index < json.len() {
        if json[index] != b'"' {
            output.push(json[index]);
            index += 1;
            continue;
        }
        let end = string_end(json, index).context("unterminated string in ast dump")?;
        let literal = &json[index..=end];
        output.extend_from_slice(literal);
        index = end + 1;
        if literal != b"\"qualType\"" && literal != b"\"desugaredQualType\"" {
            continue;
        }
        while index < json.len() && (json[index].is_ascii_whitespace() || json[index] == b':') {
            output.push(json[index]);
            index += 1;
        }
        if index < json.len() && json[index] == b'"' {
            let value_end = string_end(json, index).context("unterminated string in ast dump")?;
            let value: String = serde_json::from_slice(&json[index..=value_end]).context("bad type string")?;
            output.extend_from_slice(&serde_json::to_vec(&normalize_type(&value, gpu_type_names))?);
            index = value_end + 1;
        }
    }
    Ok(output)
}

/// Address spaces are macros here (`device` -> `__global`), so a parameter that starts with one has
/// its spelling location inside the compatibility header. `ast.rs` slices the kernel source by
/// spelling offsets; point them at the expansion site in the kernel source instead.
fn use_expansion_locations(root: &mut MetalAstNode) {
    let mut stack = vec![root];
    while let Some(node) = stack.pop() {
        if let MetalAstKind::ParmVarDecl {
            range,
            ..
        } = &mut node.kind
        {
            for location in [&mut range.begin, &mut range.end] {
                if let Some(expansion) = location.expansion_loc.clone() {
                    location.spelling_loc = Some(expansion);
                }
            }
        }
        stack.extend(node.inner.iter_mut());
    }
}

impl AmdgpuToolchain {
    pub fn new(
        compat_directory: PathBuf,
        generated_directory: PathBuf,
    ) -> anyhow::Result<Self> {
        let llvm_directory = find_llvm_directory()?;
        let clang = executable(&llvm_directory, "clang");
        let lld = executable(&llvm_directory, "ld.lld");

        println!("cargo::rerun-if-env-changed={ENV_TARGETS}");
        let target = env::var(ENV_TARGETS)
            .ok()
            .and_then(|targets| targets.split(',').map(str::trim).find(|t| !t.is_empty()).map(String::from))
            .unwrap_or_else(|| DEFAULT_TARGET.to_string());

        let version = Command::new(&clang).arg("--version").output().context("cannot execute clang --version")?;

        let cache_key = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(&version.stdout);
            hasher.update(b"\0");
            hasher.update(target.as_bytes());
            hasher.update(b"\0");
            for argument in Self::common_arguments_static() {
                hasher.update(argument.as_bytes());
                hasher.update(b"\0");
            }
            hasher.finalize().into()
        };

        Ok(Self {
            clang,
            lld,
            target,
            compat_directory,
            generated_directory,
            cache_key,
        })
    }

    pub fn cache_key(&self) -> &[u8; blake3::OUT_LEN] {
        &self.cache_key
    }

    fn common_arguments_static() -> [&'static str; 17] {
        [
            "-x",
            "clcpp",
            "-cl-std=clc++2021",
            "--target=amdgcn-amd-amdhsa",
            "-nogpulib",
            // OpenCL's default header declares builtins (exp, max, atomics, ...) that would win overload
            // resolution over metal:: and leave unresolved device-library calls.
            "-cl-no-stdinc",
            // MSL has no double: double literals are float.
            "-cl-single-precision-constant",
            "-Wno-unknown-pragmas",
            "-Wno-unknown-attributes",
            "-Wno-c++20-designator",
            "-Wno-pass-failed",
            // Inline everything, as HIP does. MSL kernels lean on large generic lambdas
            // (SimdgroupMmaCore::run_with_loader / kernel_invoke) that the default thresholds leave as calls;
            // their by-reference captures then live in scratch (a GEMM shard: 155 calls and ~18k scratch
            // accesses before, none and 576 after).
            "-mllvm",
            "-amdgpu-early-inline-all=true",
            "-mllvm",
            "-inline-threshold=100000",
            "-mllvm",
            "-amdgpu-inline-max-bb=100000",
        ]
    }

    fn command(&self) -> Command {
        let mut command = Command::new(&self.clang);
        command.args(Self::common_arguments_static());
        command.arg(format!("-mcpu={}", self.target));
        command.arg("-I").arg(&self.compat_directory);
        command.arg("-I").arg(&self.generated_directory);
        command
    }

    pub fn analyze(
        &self,
        path: &Path,
        gpu_type_names: &GpuTypeNames,
    ) -> anyhow::Result<Vec<MetalKernelInfo>> {
        let mut command = self.command();
        command
            .arg("-DDSL_ANALYZE")
            .arg(path)
            .arg("-fsyntax-only")
            .args(["-Xclang", "-ast-dump=json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let output = command.output().context("cannot execute clang analyzer")?;
        if !output.status.success() {
            bail!("clang analyzer failed: {}", String::from_utf8_lossy(&output.stderr));
        }

        let normalized = normalize_types(&output.stdout, gpu_type_names)?;
        drop(output);
        let mut deserializer = serde_json::Deserializer::from_slice(&normalized);
        deserializer.disable_recursion_limit();
        let mut ast_root = MetalAstNode::deserialize(&mut deserializer).context("cannot deserialize ast dump")?;
        use_expansion_locations(&mut ast_root);

        if !matches!(&ast_root.kind, MetalAstKind::TranslationUnitDecl) {
            bail!("unexpected kind of ast root: {:?}", ast_root.kind);
        }

        let source_contents = fs::read_to_string(path).context("cannot read source file")?;

        ast_root
            .inner
            .into_iter()
            .filter_map(|node| MetalKernelInfo::from_ast_node_and_source(node, &source_contents).transpose())
            .collect::<anyhow::Result<Vec<_>>>()
            .context("cannot parse kernel infos from AST")
    }

    /// Compiles `wrapper_path` (with the kernel source force-included) into a code object.
    /// Returns compiler warnings, if any.
    pub fn compile(
        &self,
        source: &Path,
        wrapper_path: &Path,
        object_path: &Path,
        code_object_path: &Path,
    ) -> anyhow::Result<Option<String>> {
        let mut command = self.command();
        command.arg("-O3").arg("-include").arg(source).arg("-c").arg(wrapper_path);
        self.compile_and_link(command, object_path, code_object_path)
    }

    /// Compiles a self-contained AMD-only kernel source (`kernel/native`) into a code object.
    pub fn compile_native(
        &self,
        source: &Path,
        object_path: &Path,
        code_object_path: &Path,
    ) -> anyhow::Result<Option<String>> {
        let mut command = self.command();
        command.arg("-O3").arg("-c").arg(source);
        self.compile_and_link(command, object_path, code_object_path)
    }

    fn compile_and_link(
        &self,
        mut command: Command,
        object_path: &Path,
        code_object_path: &Path,
    ) -> anyhow::Result<Option<String>> {
        command.arg("-o").arg(object_path).stderr(Stdio::piped());

        let output = command.output().context("cannot execute clang")?;
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        if !output.status.success() {
            bail!("clang failed: {stderr}");
        }

        let link = Command::new(&self.lld)
            .arg("-shared")
            .arg(object_path)
            .arg("-o")
            .arg(code_object_path)
            .output()
            .context("cannot execute ld.lld")?;
        if !link.status.success() {
            bail!("ld.lld failed: {}", String::from_utf8_lossy(&link.stderr));
        }

        Ok((!stderr.trim().is_empty()).then_some(stderr))
    }
}
