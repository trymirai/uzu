use std::{collections::HashMap, env, fs, path::PathBuf};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use super::{ast::MetalKernelInfo, bindgen::bindgen_global, toolchain::MetalToolchain, wrapper::wrappers};
use crate::{
    common::{
        caching, codegen::write_tokens, compiler::Compiler, enum_paths::EnumPaths, gpu_types::GpuTypes,
        identifiers::KernelPath, kernel::Kernel,
    },
    debug_log,
    metal::gpu_types::gpu_type_gen,
};

#[derive(Serialize, Deserialize)]
struct Cached {
    buildsystem_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    public_kernels: Box<[Kernel]>,
    has_kernels: bool,
}

#[derive(Debug)]
pub struct MetalCompiler {
    source_directory: PathBuf,
    gpu_types_directory: PathBuf,
    output_directory: PathBuf,
    toolchain: MetalToolchain,
}

impl MetalCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let source_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/metal/kernel");

        let gpu_types_directory = source_directory.join("generated");

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("metal");
        fs::create_dir_all(&output_directory)
            .with_context(|| format!("cannot create {}", output_directory.display()))?;

        let toolchain = MetalToolchain::from_env_with_include_dir(Some(gpu_types_directory.clone()))
            .context("cannot create toolchain")?;

        Ok(Self {
            source_directory,
            gpu_types_directory,
            output_directory,
            toolchain,
        })
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

        let object_file = output_base_path.with_extension("air");
        let metallib_file = output_base_path.with_extension("metallib");
        let bindgen_file = output_base_path.with_extension("rs");
        let cached_file = output_base_path.with_extension("cached");

        let buildsystem_hash = caching::build_system_hash().context("cannot get build system hash")?.as_bytes();

        if let Ok(cached_contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&cached_contents)
            && &cached.buildsystem_hash == buildsystem_hash
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path.as_ref()).map(|contents| blake3::hash(&contents).as_bytes() == hash).unwrap_or(false)
            })
        {
            debug_log!("compile cached: {source_path_relative_str}");
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
                Ok((
                    path.clone(),
                    blake3::hash(&fs::read(path.as_ref()).with_context(|| format!("cannot read {path}"))?).into(),
                ))
            })
            .collect::<anyhow::Result<HashMap<Box<str>, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        if !kernel_infos.is_empty() {
            let (wrapper_strs, specialize_indices) =
                wrappers(&kernel_infos, enum_paths).context("cannot generate kernel wrappers")?;

            let mut footer = String::new();
            for wrapper in wrapper_strs.iter() {
                footer.push_str(wrapper);
            }

            let compile_output = self
                .toolchain
                .compile(&source_path, &footer, &object_file)
                .await
                .with_context(|| format!("cannot compile {source_path_relative_str}"))?;

            if let Some(warnings) = &compile_output {
                for line in warnings.lines() {
                    println!("cargo::warning={line}");
                }
            }

            let link_output = self
                .toolchain
                .link(&object_file, &metallib_file)
                .await
                .with_context(|| format!("cannot link {source_path_relative_str}"))?;

            if let Some(warnings) = &link_output {
                for line in warnings.lines() {
                    println!("cargo::warning={line}");
                }
            }

            let library_const =
                format_ident!("MTLB_{}", blake3::hash(source_path_relative_str.as_bytes()).to_hex().to_uppercase());
            let metallib_file_str = metallib_file.to_str().context("metallib path is not utf-8")?;

            let bindings = kernel_infos
                .iter()
                .map(|kernel| {
                    super::bindgen::bindgen(kernel, &specialize_indices, enum_paths, &library_const)
                        .with_context(|| format!("cannot generate bindings for {}", kernel.name))
                        .map(|(tokens, _associated_type)| tokens)
                })
                .collect::<anyhow::Result<Vec<_>>>()?;

            let tokens = quote! {
                const #library_const: &[u8] = include_bytes!(#metallib_file_str);

                #(#bindings)*
            };

            write_tokens(tokens, &bindgen_file).context("cannot write bindings")?;
        }

        let public_kernels: Box<[Kernel]> = kernel_infos.iter().filter_map(|kernel| kernel.to_kernel()).collect();
        let has_kernels = !kernel_infos.is_empty();

        let cached = Cached {
            buildsystem_hash: *buildsystem_hash,
            dependency_hashes,
            public_kernels: public_kernels.clone(),
            has_kernels,
        };
        fs::write(&cached_file, serde_json::to_vec_pretty(&cached).context("cannot serialize cache")?)
            .context("cannot write cache file")?;

        debug_log!("compile end: {source_path_relative_str}");

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
