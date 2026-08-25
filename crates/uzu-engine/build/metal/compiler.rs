use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, future::try_join_all, stream};
use itertools::{Itertools, izip};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;
use xxhash_rust::xxh3::xxh3_64;

use super::{
    ast::MetalKernelInfo,
    bindgen::bindgen_global,
    toolchain::MetalToolchain,
    wrapper::{KernelWrappers, VariantWrapper, wrappers},
};
use crate::{
    common::{
        caching, codegen::write_tokens, compiler::Compiler, enum_paths::EnumPaths, gpu_types::GpuTypes,
        identifiers::KernelPath, kernel::Kernel,
    },
    debug_log,
    metal::gpu_types::gpu_type_gen,
};

const MIN_VARIANTS_PER_SHARD: usize = 8;
const MAX_VARIANTS_PER_SHARD: usize = 64;

fn shard_footers(kernel_wrappers: &[KernelWrappers]) -> Vec<String> {
    let total_variants: usize = kernel_wrappers.iter().map(|kernel| kernel.variants.len()).sum();
    let min_shards = total_variants.div_ceil(MAX_VARIANTS_PER_SHARD);
    let ncpu = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(1);
    let num_shards = total_variants.div_ceil(MIN_VARIANTS_PER_SHARD).clamp(min_shards, min_shards.max(ncpu));
    let num_shards = if num_shards >= ncpu {
        num_shards.div_ceil(ncpu) * ncpu
    } else {
        num_shards
    };

    let mut footers = vec![String::new(); num_shards];
    for kernel in kernel_wrappers {
        let mut emit = |index: usize, variants: Vec<&VariantWrapper>| {
            let footer = &mut footers[index];
            footer.push_str(kernel.header.as_deref().unwrap_or(""));
            footer.push_str(&variants.iter().map(|variant| variant.source.as_ref()).join(""));
            footer.push_str(kernel.footer.as_deref().unwrap_or(""));
        };

        if num_shards == 1 {
            emit(0, kernel.variants.iter().collect());
        } else {
            for (index, variants) in kernel
                .variants
                .iter()
                .into_group_map_by(|variant| (xxh3_64(variant.name.as_bytes()) % num_shards as u64) as usize)
                .into_iter()
                .sorted_unstable_by_key(|(index, _)| *index)
            {
                emit(index, variants);
            }
        }
    }
    footers
}

#[derive(Serialize, Deserialize)]
struct Cached {
    cache_key: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    public_kernels: Box<[Kernel]>,
    has_kernels: bool,
    num_variants: usize,
    num_shards: usize,
}

#[derive(Debug)]
pub struct MetalCompiler {
    source_directory: PathBuf,
    gpu_types_directory: PathBuf,
    output_directory: PathBuf,
    metallib_compressed: bool,
    toolchain: MetalToolchain,
    cache_key: [u8; blake3::OUT_LEN],
}

impl MetalCompiler {
    pub async fn new() -> anyhow::Result<Self> {
        let source_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/metal/kernel");
        println!("cargo::rerun-if-changed={}", source_directory.display());

        let gpu_types_directory = source_directory.join("generated");

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("metal");
        fs::create_dir_all(&output_directory)
            .with_context(|| format!("cannot create {}", output_directory.display()))?;

        let metallib_compressed = match env::var("OPT_LEVEL").context("missing OPT_LEVEL")?.as_str() {
            "0" | "1" | "2" => false, // treat opt-level 0/1/2 as debug/test build where size doesn't matter
            _ => true,                // treat everything else (3,s,z) as release build where size matters
        };

        let toolchain = MetalToolchain::from_env_with_include_dir(Some(gpu_types_directory.clone()))
            .await
            .context("cannot create toolchain")?;

        let cache_key = {
            let build_system_hash = caching::build_system_hash().context("cannot get build system hash")?;

            let mut hasher = blake3::Hasher::new();
            hasher.update(build_system_hash.as_bytes());
            hasher.update(toolchain.cache_key());
            hasher.update(&[u8::from(metallib_compressed)]);

            hasher.finalize().into()
        };

        Ok(Self {
            source_directory,
            gpu_types_directory,
            output_directory,
            metallib_compressed,
            toolchain,
            cache_key,
        })
    }

    fn emit_rerun_if_changed_for_dependency(
        &self,
        path: &str,
    ) {
        if !Path::new(path).starts_with(&self.source_directory) {
            println!("cargo::rerun-if-changed={path}");
        }
    }

    async fn compile(
        &self,
        source_path: PathBuf,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<(KernelPath, Box<[Kernel]>, bool)> {
        let source_path_relative =
            source_path.strip_prefix(&self.source_directory).context("source is not in src_dir")?;
        let source_path_relative_str = source_path_relative.to_str().context("source path is not utf-8")?;
        debug_log!("compile start: {source_path_relative_str}");

        let kernel_path: KernelPath = source_path_relative
            .with_extension("")
            .components()
            .map(|component| component.as_os_str().to_str().unwrap().to_string())
            .collect();

        let output_base_path = self.output_directory.join(source_path_relative).with_extension("");
        fs::create_dir_all(output_base_path.parent().context("cannot get output directory")?)
            .context("cannot create output directory")?;

        let bindgen_file = output_base_path.with_extension("rs");
        let cached_file = output_base_path.with_extension("cached");

        if let Ok(cached_contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&cached_contents)
            && cached.cache_key == self.cache_key
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path.as_ref()).map(|contents| blake3::hash(&contents).as_bytes() == hash).unwrap_or(false)
            })
        {
            for path in cached.dependency_hashes.keys() {
                self.emit_rerun_if_changed_for_dependency(path);
            }
            let sharding = if cached.has_kernels {
                format!(" ({} variants / {} shards)", cached.num_variants, cached.num_shards)
            } else {
                Default::default()
            };
            debug_log!("compile cached: {source_path_relative_str}{sharding}");
            return Ok((kernel_path, cached.public_kernels, cached.has_kernels));
        }

        let (metal_kernel_infos, dependencies) = self
            .toolchain
            .analyze(&source_path)
            .await
            .with_context(|| format!("cannot analyze {source_path_relative_str}"))?;

        let kernel_infos: Vec<MetalKernelInfo> = metal_kernel_infos.collect();

        let dependency_hashes = dependencies
            .map(|path| {
                self.emit_rerun_if_changed_for_dependency(&path);
                Ok((
                    path.clone(),
                    blake3::hash(&fs::read(path.as_ref()).with_context(|| format!("cannot read {path}"))?).into(),
                ))
            })
            .collect::<anyhow::Result<HashMap<Box<str>, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        let mut num_variants = 0;
        let mut num_shards = 0;
        if !kernel_infos.is_empty() {
            let (kernel_wrappers, specialize_indices) =
                wrappers(&kernel_infos, enum_paths).context("cannot generate kernel wrappers")?;

            num_variants = kernel_wrappers.iter().map(|kernel| kernel.variants.len()).sum();
            let footers = shard_footers(&kernel_wrappers);
            num_shards = footers.len();

            let metallib_files: Vec<PathBuf> = match num_shards {
                1 => vec![output_base_path.with_extension("metallib")],
                num_shards => {
                    (0..num_shards).map(|i| output_base_path.with_extension(format!("shard{i}.metallib"))).collect()
                },
            };

            let metallib_maybe_compressed_files: Vec<PathBuf> = metallib_files
                .iter()
                .map(|file| {
                    if self.metallib_compressed {
                        file.with_added_extension("zst")
                    } else {
                        file.clone()
                    }
                })
                .collect();

            let compile_outputs = try_join_all(izip!(&footers, &metallib_files, &metallib_maybe_compressed_files).map(
                |(footer, metallib_file, compressed_file)| {
                    let source_path = &source_path;
                    async move {
                        let warnings = self.toolchain.compile(source_path, footer, metallib_file).await?;

                        if self.metallib_compressed {
                            let metallib_file = metallib_file.clone();
                            let compressed_file = compressed_file.clone();
                            tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
                                let metallib_source = fs::read(metallib_file)?;
                                fs::write(compressed_file, zstd::encode_all(metallib_source.as_slice(), 22)?)?;
                                Ok(())
                            })
                            .await??;
                        }

                        anyhow::Ok(warnings)
                    }
                },
            ))
            .await
            .with_context(|| format!("cannot compile {source_path_relative_str}"))?;

            for warnings in compile_outputs.into_iter().flatten() {
                for line in warnings.lines() {
                    println!("cargo::warning={line}");
                }
            }

            let library_const =
                format_ident!("MTLB_{}", blake3::hash(source_path_relative_str.as_bytes()).to_hex().to_uppercase());
            let metallib_file_strs = metallib_maybe_compressed_files
                .iter()
                .map(|file| file.to_str().context("metallib path is not utf-8"))
                .collect::<anyhow::Result<Vec<_>>>()?;

            let bindings = kernel_infos
                .iter()
                .map(|kernel| {
                    super::bindgen::bindgen(
                        kernel,
                        &specialize_indices,
                        enum_paths,
                        &library_const,
                        num_shards,
                        self.metallib_compressed,
                    )
                    .with_context(|| format!("cannot generate bindings for {}", kernel.name))
                    .map(|(tokens, _associated_type)| tokens)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let tokens = quote! {
                const #library_const: [&[u8]; #num_shards] = [#(include_bytes!(#metallib_file_strs)),*];

                #(#bindings)*
            };

            write_tokens(tokens, &bindgen_file).context("cannot write bindings")?;
        }

        let public_kernels: Box<[Kernel]> = kernel_infos.iter().filter_map(|kernel| kernel.to_kernel()).collect();
        let has_kernels = !kernel_infos.is_empty();

        let cached = Cached {
            cache_key: self.cache_key,
            dependency_hashes,
            public_kernels: public_kernels.clone(),
            has_kernels,
            num_variants,
            num_shards,
        };
        fs::write(&cached_file, serde_json::to_vec_pretty(&cached).context("cannot serialize cache")?)
            .context("cannot write cache file")?;

        let sharding = if has_kernels {
            format!(" ({num_variants} variants / {num_shards} shards)")
        } else {
            Default::default()
        };
        debug_log!("compile end: {source_path_relative_str}{sharding}");

        Ok((kernel_path, public_kernels, has_kernels))
    }
}

#[async_trait]
impl Compiler for MetalCompiler {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        gpu_type_gen(&self.gpu_types_directory, gpu_types).await.context("cannot generate shared gpu types")?;

        let metal_sources: Vec<PathBuf> = WalkDir::new(&self.source_directory)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.path().extension().and_then(|s| s.to_str()) == Some("metal"))
            .map(|e| e.into_path())
            .collect();

        let num_concurrent_compiles = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) * 2;

        let compiled: Vec<(KernelPath, Box<[Kernel]>, bool)> = stream::iter(metal_sources)
            .map(|path| async move {
                self.compile(path.clone(), enum_paths)
                    .await
                    .with_context(|| format!("cannot compile {}", path.display()))
            })
            .buffer_unordered(num_concurrent_compiles)
            .try_collect()
            .await?;

        let mut kernels_bindgen = compiled
            .iter()
            .filter(|(_path, _kernels, has_kernels)| *has_kernels)
            .map(|(path, kernels, _has_kernels)| {
                (self.output_directory.join(path.join("/")).with_extension("rs"), kernels.as_ref())
            })
            .collect::<Vec<(PathBuf, &[Kernel])>>();
        kernels_bindgen.sort_by(|(a_path, _a_kernels), (b_path, _b_kernels)| a_path.cmp(b_path));

        let tokens = bindgen_global(&kernels_bindgen).context("cannot generate bindings")?;
        write_tokens(tokens, self.output_directory.with_extension("rs")).context("cannot write bindings")?;

        Ok(compiled.into_iter().map(|(path, kernels, _has_kernels)| (path, kernels)).collect())
    }
}
