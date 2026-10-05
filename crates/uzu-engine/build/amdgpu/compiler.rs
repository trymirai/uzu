use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;
use quote::{format_ident, quote};
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use super::{
    ast::MetalKernelInfo,
    bindgen::{bindgen, bindgen_global, bindgen_stub},
    native,
    sharding::shard_index,
    toolchain::{AmdgpuToolchain, GpuTypeNames},
    wrapper::{VariantWrapper, kernel_wrappers},
};
use crate::{
    common::{
        caching,
        codegen::write_tokens,
        compiler::Compiler,
        enum_paths::EnumPaths,
        gpu_types::{GpuType, GpuTypes},
        identifiers::KernelPath,
        kernel::Kernel,
    },
    debug_log,
};

const VARIANTS_PER_SHARD: usize = 24;
/// Sources left out of the build, with the reason.
const DEFERRED_FILES: &[(&str, &str)] = &[(
    "attention/gemm_grouped/attention_gemm_grouped",
    "MXU-only (MetalPerformancePrimitives matmul2d); the AMD attention policy does not use it",
)];
const MAX_SHARDS: usize = 64;
const ENV_JOBS: &str = "UZU_AMDGPU_JOBS";

#[derive(Serialize, Deserialize)]
struct Cached {
    cache_key: [u8; blake3::OUT_LEN],
    public_kernels: Box<[Kernel]>,
    has_bindings: bool,
    status: String,
}

struct FileOutcome {
    kernel_path: KernelPath,
    public_kernels: Box<[Kernel]>,
    bindings_file: Option<PathBuf>,
    status: String,
}

pub struct AmdgpuCompiler {
    source_directory: PathBuf,
    native_directory: PathBuf,
    output_directory: PathBuf,
    toolchain: AmdgpuToolchain,
    cache_key: [u8; blake3::OUT_LEN],
    reference: HashMap<KernelPath, Box<[Kernel]>>,
    jobs: usize,
}

fn hash_directory(
    hasher: &mut blake3::Hasher,
    directory: &Path,
) -> anyhow::Result<()> {
    let files = WalkDir::new(directory)
        .into_iter()
        .collect::<Result<Vec<_>, _>>()
        .with_context(|| format!("cannot walk {}", directory.display()))?
        .into_iter()
        .filter(|entry| entry.file_type().is_file())
        .map(|entry| entry.into_path())
        .sorted();
    for file in files {
        hasher.update(file.strip_prefix(directory).unwrap_or(&file).to_string_lossy().as_bytes());
        hasher.update(&fs::read(&file).with_context(|| format!("cannot read {}", file.display()))?);
    }
    Ok(())
}

fn warn(message: &str) {
    for line in message.lines().filter(|line| !line.trim().is_empty()) {
        println!("cargo::warning={line}");
    }
}

/// The first `count` error lines of a compiler failure, or its first lines if none says "error".
fn first_lines(
    text: &str,
    count: usize,
) -> String {
    let errors = text.lines().filter(|line| line.contains("error:")).take(count).join("\n");
    if errors.is_empty() {
        text.lines().take(count).join("\n")
    } else {
        errors
    }
}

impl AmdgpuCompiler {
    pub fn new(reference: Option<HashMap<KernelPath, Box<[Kernel]>>>) -> anyhow::Result<Self> {
        let manifest_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
        let source_directory = manifest_directory.join("src/backends/metal/kernel");
        let compat_directory = manifest_directory.join("src/backends/amdgpu/kernel/compat");
        let native_directory = manifest_directory.join("src/backends/amdgpu/kernel/native");
        println!("cargo::rerun-if-changed={}", source_directory.display());
        println!("cargo::rerun-if-changed={}", compat_directory.display());
        println!("cargo::rerun-if-changed={}", native_directory.display());

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("amdgpu");
        fs::create_dir_all(&output_directory)
            .with_context(|| format!("cannot create {}", output_directory.display()))?;

        let toolchain = AmdgpuToolchain::new(compat_directory.clone(), source_directory.join("generated"))
            .context("cannot create AMDGPU toolchain")?;

        // Shared part of every file's cache key; each file adds its own source and transitive includes
        // (`file_cache_key`), so editing a header rebuilds only the files that include it.
        let cache_key = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(caching::build_system_hash().context("cannot get build system hash")?.as_bytes());
            hasher.update(toolchain.cache_key());
            hash_directory(&mut hasher, &compat_directory)?;
            hasher.finalize().into()
        };

        println!("cargo::rerun-if-env-changed={ENV_JOBS}");
        let jobs = env::var(ENV_JOBS)
            .ok()
            .and_then(|jobs| jobs.parse().ok())
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| (n.get() / 2).max(1)).unwrap_or(1));

        Ok(Self {
            source_directory,
            native_directory,
            output_directory,
            toolchain,
            cache_key,
            reference: reference.unwrap_or_default(),
            jobs,
        })
    }

    fn write_stubs(
        &self,
        bindings_file: &Path,
        kernels: &[Kernel],
    ) -> anyhow::Result<()> {
        let stubs = kernels.iter().map(bindgen_stub).collect::<anyhow::Result<Vec<_>>>()?;
        write_tokens(quote! { #(#stubs)* }, bindings_file).context("cannot write stub bindings")
    }

    fn compile_shards(
        &self,
        source_path: &Path,
        output_base_path: &Path,
        variants: &[&VariantWrapper],
    ) -> (Vec<Option<PathBuf>>, Vec<String>) {
        let num_shards = variants.len().div_ceil(VARIANTS_PER_SHARD).clamp(1, MAX_SHARDS);
        let mut sources = vec![String::new(); num_shards];
        for variant in variants {
            sources[shard_index(&variant.name, num_shards)].push_str(&variant.source);
        }

        let results: Vec<(Option<PathBuf>, Option<String>)> = sources
            .into_par_iter()
            .enumerate()
            .map(|(shard, source)| {
                let wrapper_path = output_base_path.with_extension(format!("shard{shard}.clcpp"));
                let object_path = output_base_path.with_extension(format!("shard{shard}.o"));
                let code_object_path = output_base_path.with_extension(format!("shard{shard}.hsaco"));
                if source.is_empty() {
                    return (None, None);
                }
                if let Err(error) = fs::write(&wrapper_path, &source) {
                    return (None, Some(format!("cannot write {}: {error}", wrapper_path.display())));
                }
                match self.toolchain.compile(source_path, &wrapper_path, &object_path, &code_object_path) {
                    Ok(_warnings) => (Some(code_object_path), None),
                    Err(error) => (None, Some(format!("shard {shard}: {}", first_lines(&format!("{error:#}"), 6)))),
                }
            })
            .collect();

        let (code_objects, errors): (Vec<_>, Vec<_>) = results.into_iter().unzip();
        (code_objects, errors.into_iter().flatten().collect())
    }

    /// The shared key plus the source and every header it includes with `#include "..."`, resolved like
    /// clang: next to the including file first, then in the generated-types directory.
    fn file_cache_key(
        &self,
        source_path: &Path,
    ) -> anyhow::Result<[u8; blake3::OUT_LEN]> {
        let generated_directory = self.source_directory.join("generated");
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.cache_key);
        let mut seen = std::collections::BTreeSet::new();
        let mut pending = vec![source_path.to_path_buf()];
        while let Some(path) = pending.pop() {
            let Ok(path) = path.canonicalize() else {
                continue;
            };
            if !seen.insert(path.clone()) {
                continue;
            }
            let contents = fs::read(&path).with_context(|| format!("cannot read {}", path.display()))?;
            let directory = path.parent().map(Path::to_path_buf).unwrap_or_default();
            for line in String::from_utf8_lossy(&contents).lines() {
                let Some(rest) = line.trim_start().strip_prefix("#include") else {
                    continue;
                };
                let Some(include) = rest.trim().strip_prefix('"').and_then(|rest| rest.split('"').next()) else {
                    continue;
                };
                for candidate in [directory.join(include), generated_directory.join(include)] {
                    if candidate.is_file() {
                        pending.push(candidate);
                        break;
                    }
                }
            }
        }
        for path in &seen {
            let relative = path.strip_prefix(&self.source_directory).unwrap_or(path);
            hasher.update(relative.to_string_lossy().as_bytes());
            hasher.update(&fs::read(path)?);
        }
        Ok(hasher.finalize().into())
    }

    fn compile_file(
        &self,
        source_path: PathBuf,
        enum_paths: &EnumPaths,
        gpu_type_names: &GpuTypeNames,
    ) -> anyhow::Result<FileOutcome> {
        let relative = source_path.strip_prefix(&self.source_directory).context("source is not in src_dir")?;
        let relative_str = relative.to_str().context("source path is not utf-8")?.replace('\\', "/");
        let kernel_path: KernelPath = relative
            .with_extension("")
            .components()
            .map(|component| component.as_os_str().to_str().unwrap().to_string())
            .collect();

        let output_base_path = self.output_directory.join(relative).with_extension("");
        fs::create_dir_all(output_base_path.parent().context("cannot get output directory")?)?;
        let bindings_file = output_base_path.with_extension("rs");
        let cached_file = output_base_path.with_extension("cached");

        let file_cache_key = self.file_cache_key(&source_path)?;
        if let Ok(contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&contents)
            && cached.cache_key == file_cache_key
            && (!cached.has_bindings || bindings_file.exists())
        {
            debug_log!("amdgpu cached: {relative_str}");
            return Ok(FileOutcome {
                kernel_path,
                public_kernels: cached.public_kernels,
                bindings_file: cached.has_bindings.then_some(bindings_file),
                status: cached.status,
            });
        }

        let reference_kernels = self.reference.get(&kernel_path).cloned().unwrap_or_default();

        let deferred = DEFERRED_FILES.iter().find(|(path, _)| relative_str.strip_suffix(".metal") == Some(path));
        let analysis = match deferred {
            Some((_, reason)) => Err(anyhow::anyhow!("deferred: {reason}")),
            None => self.toolchain.analyze(&source_path, gpu_type_names),
        };
        let (public_kernels, has_bindings, status) = match analysis {
            Err(error) => {
                let what = if deferred.is_some() {
                    "not compiled"
                } else {
                    "analysis failed"
                };
                if reference_kernels.is_empty() {
                    (
                        Box::default(),
                        false,
                        format!("{what}, no public kernels: {}", first_lines(&format!("{error:#}"), 2)),
                    )
                } else {
                    self.write_stubs(&bindings_file, &reference_kernels)?;
                    let names = reference_kernels.iter().map(|k| k.name.as_ref()).join(", ");
                    (
                        reference_kernels,
                        true,
                        format!("{what}, stubs for {names}: {}", first_lines(&format!("{error:#}"), 2)),
                    )
                }
            },
            Ok(kernel_infos) if kernel_infos.is_empty() => (Box::default(), false, "no kernels".into()),
            Ok(kernel_infos) => self.compile_kernels(
                &source_path,
                &output_base_path,
                &bindings_file,
                &relative_str,
                &kernel_infos,
                enum_paths,
            )?,
        };

        let cached = Cached {
            cache_key: file_cache_key,
            public_kernels: public_kernels.clone(),
            has_bindings,
            status: status.clone(),
        };
        fs::write(&cached_file, serde_json::to_vec_pretty(&cached)?).context("cannot write cache file")?;

        Ok(FileOutcome {
            kernel_path,
            public_kernels,
            bindings_file: has_bindings.then_some(bindings_file),
            status,
        })
    }

    fn compile_kernels(
        &self,
        source_path: &Path,
        output_base_path: &Path,
        bindings_file: &Path,
        relative_str: &str,
        kernel_infos: &[MetalKernelInfo],
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<(Box<[Kernel]>, bool, String)> {
        let mut compiled = Vec::new();
        let mut stubbed = Vec::new();
        let mut problems = Vec::new();
        for kernel in kernel_infos {
            match kernel_wrappers(kernel, enum_paths) {
                Ok(variants) => compiled.push((kernel, variants)),
                Err(error) => {
                    problems.push(format!("{}: {error:#}", kernel.name));
                    if let Some(public) = kernel.to_kernel() {
                        stubbed.push(public);
                    }
                },
            }
        }

        let variants: Vec<&VariantWrapper> = compiled.iter().flat_map(|(_, variants)| variants.iter()).collect();
        let (code_objects, shard_errors) = if variants.is_empty() {
            (Vec::new(), Vec::new())
        } else {
            self.compile_shards(source_path, output_base_path, &variants)
        };
        problems.extend(shard_errors);

        let code_objects_const =
            format_ident!("AMDGPU_CO_{}", blake3::hash(relative_str.as_bytes()).to_hex().to_uppercase());
        let code_object_items = code_objects.iter().map(|code_object| match code_object {
            Some(path) => {
                let path = path.to_str().expect("code object path is not utf-8");
                quote! { include_bytes!(#path) }
            },
            None => quote! { &[] },
        });
        let num_shards = code_objects.len();

        let mut items = Vec::new();
        for (kernel, _) in compiled.iter() {
            let (tokens, _associated_type) = bindgen(kernel, enum_paths, &code_objects_const)
                .with_context(|| format!("cannot generate bindings for {}", kernel.name))?;
            items.push(tokens);
        }
        for kernel in stubbed.iter() {
            items.push(bindgen_stub(kernel)?);
        }

        let code_objects_tokens = if num_shards > 0 {
            quote! { #[allow(dead_code)] const #code_objects_const: [&[u8]; #num_shards] = [#(#code_object_items),*]; }
        } else {
            quote! {}
        };
        write_tokens(quote! { #code_objects_tokens #(#items)* }, bindings_file).context("cannot write bindings")?;

        let public_kernels: Box<[Kernel]> = kernel_infos.iter().filter_map(|kernel| kernel.to_kernel()).collect();
        let failed_shards = code_objects.iter().filter(|c| c.is_none()).count();
        let status = if problems.is_empty() {
            format!("ok ({} variants / {} shards)", variants.len(), num_shards)
        } else {
            format!(
                "{} variants / {} shards, {} failed shards{}:\n{}",
                variants.len(),
                num_shards,
                failed_shards,
                if stubbed.is_empty() {
                    String::new()
                } else {
                    format!(", stubs for {}", stubbed.iter().map(|k| k.name.as_ref()).join(", "))
                },
                problems.join("\n")
            )
        };
        Ok((public_kernels, true, status))
    }
}

impl Compiler for AmdgpuCompiler {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        let gpu_type_names: GpuTypeNames = gpu_types
            .files
            .iter()
            .flat_map(|file| {
                file.types.iter().filter_map(move |ty| {
                    let name = match ty {
                        GpuType::Enum(ty) => ty.name.to_string(),
                        GpuType::OptionSet(ty) => ty.name.clone(),
                        GpuType::Struct(ty) => ty.name.to_string(),
                        GpuType::Constant(_) => return None,
                    };
                    Some((name.clone(), format!("uzu::{}::{name}", file.name)))
                })
            })
            .collect();

        let sources: Vec<PathBuf> = WalkDir::new(&self.source_directory)
            .into_iter()
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry.file_type().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("metal")
            })
            .map(|entry| entry.into_path())
            .sorted()
            .collect();

        // deep clang ASTs: generous stacks for the JSON walk
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(self.jobs)
            .stack_size(64 << 20)
            .build()
            .context("cannot create thread pool")?;

        let outcomes: Vec<FileOutcome> = pool.install(|| {
            sources
                .par_iter()
                .map(|path| {
                    self.compile_file(path.clone(), enum_paths, &gpu_type_names)
                        .with_context(|| format!("cannot compile {}", path.display()))
                })
                .collect::<anyhow::Result<_>>()
        })?;

        let mut problems = 0;
        for outcome in outcomes.iter() {
            if !outcome.status.starts_with("ok") && outcome.status != "no kernels" {
                problems += 1;
                warn(&format!("amdgpu {}: {}", outcome.kernel_path.join("/"), outcome.status));
            }
        }
        debug_log!("amdgpu: {} files, {} with problems", outcomes.len(), problems);

        let files: Vec<(PathBuf, Vec<syn::Ident>)> = outcomes
            .iter()
            .filter_map(|o| o.bindings_file.clone().map(|f| (f, Vec::new())))
            .sorted_by(|a, b| a.0.cmp(&b.0))
            .collect();
        let public_kernels: Vec<&Kernel> = outcomes.iter().flat_map(|o| o.public_kernels.iter()).collect();
        let tokens = bindgen_global(&files, &public_kernels).context("cannot generate bindings")?;
        write_tokens(tokens, self.output_directory.with_extension("rs")).context("cannot write bindings")?;
        native::compile_native(&self.toolchain, &self.native_directory, &self.output_directory, warn)?;

        Ok(outcomes.into_iter().map(|o| (o.kernel_path, o.public_kernels)).collect())
    }
}
