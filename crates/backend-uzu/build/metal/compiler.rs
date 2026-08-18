use std::{
    collections::HashMap,
    env, fmt, fs,
    path::{Path, PathBuf},
    time::Instant,
};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use quote::{format_ident, quote};
use serde::{Deserialize, Serialize};
use walkdir::WalkDir;

use super::{
    artifact_cache::{self, SharedArtifactCache, SourceIndex},
    ast::MetalKernelInfo,
    bindgen::bindgen_global,
    cache_protocol::Stage,
    toolchain::MetalToolchain,
    wrapper::{SpecializeBaseIndices, wrappers},
};
use crate::{
    build_warning,
    common::{
        caching, codegen::write_tokens, compiler::Compiler, enum_paths::EnumPaths, envs, gpu_types::GpuTypes,
        identifiers::KernelPath, kernel::Kernel,
    },
    debug_log,
    metal::gpu_types::gpu_type_gen,
};

/// Distribution compression level retained from the original release build.
const METAL_ZSTD_LEVEL: i32 = 22;

#[derive(Serialize, Deserialize, Clone)]
struct Cached {
    schema: u32,
    compile_schema_hash: [u8; blake3::OUT_LEN],
    toolchain_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    artifact_hash: [u8; blake3::OUT_LEN],
    binding_hash: [u8; blake3::OUT_LEN],
    zstd_level: i32,
    compressed: bool,
    public_kernels: Box<[Kernel]>,
    has_kernels: bool,
}

enum LocalCache {
    Hit(Cached),
    Rebind(MissReason),
    Miss(MissReason),
}

#[derive(Clone)]
enum MissReason {
    MissingMetadata,
    Schema,
    CompileSchema,
    MissingBinding,
    Dependency(Box<str>),
    Toolchain,
    Compression,
    MissingArtifact,
    CorruptArtifact,
}

impl fmt::Display for MissReason {
    fn fmt(
        &self,
        f: &mut fmt::Formatter<'_>,
    ) -> fmt::Result {
        match self {
            Self::MissingMetadata => write!(f, "missing metadata"),
            Self::Schema => write!(f, "build-system/schema change"),
            Self::CompileSchema => write!(f, "compile schema change"),
            Self::MissingBinding => write!(f, "missing or stale bindings"),
            Self::Dependency(path) => write!(f, "named dependency change: {path}"),
            Self::Toolchain => write!(f, "compiler/SDK/flags change"),
            Self::Compression => write!(f, "compression settings change"),
            Self::MissingArtifact => write!(f, "missing artifact"),
            Self::CorruptArtifact => write!(f, "corrupt artifact"),
        }
    }
}

struct StageClock {
    stages: Vec<(String, u128)>,
    mark: Instant,
    started: Instant,
}

impl StageClock {
    fn new() -> Self {
        let now = Instant::now();
        Self {
            stages: Vec::new(),
            mark: now,
            started: now,
        }
    }

    fn lap(
        &mut self,
        name: &str,
    ) {
        let now = Instant::now();
        self.stages.push((name.to_string(), now.duration_since(self.mark).as_millis()));
        self.mark = now;
    }

    fn total_ms(&self) -> u128 {
        self.started.elapsed().as_millis()
    }
}

struct CompiledSource {
    kernel_path: KernelPath,
    public_kernels: Box<[Kernel]>,
    has_kernels: bool,
    source: String,
    cache_hit: bool,
    miss_reason: Option<String>,
    total_ms: u128,
    stages: Vec<(String, u128)>,
}

struct Analysis {
    kernel_infos: Vec<MetalKernelInfo>,
    specialize_indices: SpecializeBaseIndices,
    public_kernels: Box<[Kernel]>,
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
    footer: String,
    from_index: bool,
}

struct EnsuredArtifact {
    origin: &'static str,
    hash: [u8; blake3::OUT_LEN],
}

#[derive(Debug)]
pub struct MetalCompiler {
    source_directory: PathBuf,
    gpu_types_directory: PathBuf,
    output_directory: PathBuf,
    metallib_compressed: bool,
    zstd_level: i32,
    toolchain: MetalToolchain,
    toolchain_hash: blake3::Hash,
    analyzer_hash: blake3::Hash,
    linker_hash: blake3::Hash,
    compile_schema_hash: blake3::Hash,
    gpu_types_hash: blake3::Hash,
    shared_cache: SharedArtifactCache,
    bypass_shared_cache: bool,
}

impl MetalCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let manifest_directory = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?);
        let source_directory = manifest_directory.join("src/backends/metal/kernel");
        let gpu_types_directory = source_directory.join("generated");

        let output_directory = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("metal");
        fs::create_dir_all(&output_directory)
            .with_context(|| format!("cannot create {}", output_directory.display()))?;

        let metallib_compressed = match env::var("OPT_LEVEL").context("missing OPT_LEVEL")?.as_str() {
            "0" | "1" | "2" => false, // treat opt-level 0/1/2 as debug/test build where size doesn't matter
            _ => true,                // treat everything else (3,s,z) as release build where size matters
        };

        let zstd_level = METAL_ZSTD_LEVEL;

        let toolchain = MetalToolchain::from_env_with_include_dir(Some(gpu_types_directory.clone()))
            .context("cannot create toolchain")?;
        let toolchain_hash = toolchain.identity_hash().context("cannot hash metal toolchain")?;
        let analyzer_hash = toolchain.analyzer_identity_hash().context("cannot hash metal analyzer")?;
        let linker_hash = toolchain.linker_identity_hash().context("cannot hash metal linker")?;

        // Build-source and dependency changes are rare compared with host runtime edits. Hashing the
        // conservative generator boundary keeps cache hits correct without tying them to Cargo's build-script binary.
        let workspace_directory =
            manifest_directory.parent().and_then(Path::parent).context("backend-uzu is not inside the workspace")?;
        let compile_schema_hash = caching::hash_paths([
            manifest_directory.join("build"),
            manifest_directory.join("Cargo.toml"),
            workspace_directory.join("Cargo.lock"),
        ])
        .context("cannot hash compile schema")?;
        let gpu_types_hash = caching::hash_paths([manifest_directory.join("src/backends/common/gpu_types")])
            .context("cannot hash gpu types")?;

        Ok(Self {
            source_directory,
            gpu_types_directory,
            output_directory,
            metallib_compressed,
            zstd_level,
            toolchain,
            toolchain_hash,
            analyzer_hash,
            linker_hash,
            compile_schema_hash,
            gpu_types_hash,
            shared_cache: SharedArtifactCache::new()?,
            // BUILD_CLEAN keeps its upstream meaning: force regeneration, then refresh the shared cache.
            bypass_shared_cache: envs::build_clean(),
        })
    }

    fn check_local_cache(
        &self,
        cached_file: &Path,
        artifact_file: &Path,
        bindgen_file: &Path,
    ) -> LocalCache {
        let Ok(bytes) = fs::read(cached_file) else {
            return LocalCache::Miss(MissReason::MissingMetadata);
        };
        let Ok(cached) = serde_json::from_slice::<Cached>(&bytes) else {
            return LocalCache::Miss(MissReason::Schema);
        };
        if cached.schema != caching::METAL_CACHE_SCHEMA {
            return LocalCache::Miss(MissReason::Schema);
        }
        if &cached.compile_schema_hash != self.compile_schema_hash.as_bytes() {
            return LocalCache::Miss(MissReason::CompileSchema);
        }
        if &cached.toolchain_hash != self.toolchain_hash.as_bytes() {
            return LocalCache::Miss(MissReason::Toolchain);
        }
        for (path, hash) in &cached.dependency_hashes {
            match fs::read(path.as_ref()) {
                Ok(contents) if blake3::hash(&contents).as_bytes() == hash => {},
                _ => return LocalCache::Miss(MissReason::Dependency(path.clone())),
            }
        }
        if cached.compressed != self.metallib_compressed || cached.zstd_level != self.effective_zstd_level() {
            return LocalCache::Miss(MissReason::Compression);
        }
        if cached.has_kernels {
            let Ok(artifact) = fs::read(artifact_file) else {
                return LocalCache::Miss(MissReason::MissingArtifact);
            };
            if blake3::hash(&artifact).as_bytes() != &cached.artifact_hash {
                return LocalCache::Miss(MissReason::CorruptArtifact);
            }
            match fs::read(bindgen_file) {
                Ok(binding) if blake3::hash(&binding).as_bytes() == &cached.binding_hash => {},
                _ => return LocalCache::Rebind(MissReason::MissingBinding),
            }
        }
        LocalCache::Hit(cached)
    }

    fn effective_zstd_level(&self) -> i32 {
        if self.metallib_compressed {
            self.zstd_level
        } else {
            0
        }
    }

    fn write_local_cache(
        &self,
        cached_file: &Path,
        dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
        artifact_hash: [u8; blake3::OUT_LEN],
        binding_hash: [u8; blake3::OUT_LEN],
        public_kernels: Box<[Kernel]>,
        has_kernels: bool,
    ) -> anyhow::Result<()> {
        let cached = Cached {
            schema: caching::METAL_CACHE_SCHEMA,
            compile_schema_hash: *self.compile_schema_hash.as_bytes(),
            toolchain_hash: *self.toolchain_hash.as_bytes(),
            dependency_hashes,
            artifact_hash,
            binding_hash,
            zstd_level: self.effective_zstd_level(),
            compressed: self.metallib_compressed,
            public_kernels,
            has_kernels,
        };
        caching::write_atomic(cached_file, serde_json::to_vec_pretty(&cached).context("cannot serialize cache")?)
            .context("cannot write cache file")
    }

    fn write_bindings(
        &self,
        source_path_relative_str: &str,
        kernel_infos: &[MetalKernelInfo],
        specialize_indices: &SpecializeBaseIndices,
        enum_paths: &EnumPaths,
        metallib_maybe_compressed_file: &Path,
        bindgen_file: &Path,
    ) -> anyhow::Result<[u8; blake3::OUT_LEN]> {
        let library_const =
            format_ident!("MTLB_{}", blake3::hash(source_path_relative_str.as_bytes()).to_hex().to_uppercase());
        let metallib_maybe_compressed_file_str =
            metallib_maybe_compressed_file.to_str().context("metallib path is not utf-8")?;

        let bindings = kernel_infos
            .iter()
            .map(|kernel| {
                super::bindgen::bindgen(
                    kernel,
                    specialize_indices,
                    enum_paths,
                    &library_const,
                    self.metallib_compressed,
                )
                .with_context(|| format!("cannot generate bindings for {}", kernel.name))
                .map(|(tokens, _associated_type)| tokens)
            })
            .collect::<anyhow::Result<Vec<_>>>()?;

        let tokens = quote! {
            const #library_const: &[u8] = include_bytes!(#metallib_maybe_compressed_file_str);

            #(#bindings)*
        };

        write_tokens(tokens, bindgen_file).context("cannot write bindings")?;
        Ok(*caching::hash_file(bindgen_file)?.as_bytes())
    }

    async fn compress_metallib(
        &self,
        metallib_file: PathBuf,
        compressed_file: PathBuf,
    ) -> anyhow::Result<()> {
        let zstd_level = self.zstd_level;
        tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
            let metallib_source = fs::read(&metallib_file)?;
            let metallib_compressed = zstd::encode_all(metallib_source.as_slice(), zstd_level)?;
            caching::write_atomic(&compressed_file, metallib_compressed)?;
            Ok(())
        })
        .await??;
        Ok(())
    }

    async fn compile_air(
        &self,
        source_path: &Path,
        source_path_relative_str: &str,
        footer: &str,
        object_file: &Path,
    ) -> anyhow::Result<()> {
        let compile_output = self
            .toolchain
            .compile(source_path, footer, object_file)
            .await
            .with_context(|| format!("cannot compile {source_path_relative_str}"))?;

        if let Some(warnings) = &compile_output {
            for line in warnings.lines() {
                println!("cargo::warning={line}");
            }
        }
        Ok(())
    }

    async fn link_metallib(
        &self,
        source_path_relative_str: &str,
        object_file: &Path,
        metallib_file: &Path,
    ) -> anyhow::Result<()> {
        let link_output = self
            .toolchain
            .link(object_file, metallib_file)
            .await
            .with_context(|| format!("cannot link {source_path_relative_str}"))?;

        if let Some(warnings) = &link_output {
            for line in warnings.lines() {
                println!("cargo::warning={line}");
            }
        }
        Ok(())
    }

    async fn resolve_analysis(
        &self,
        source_path: &Path,
        source_path_relative_str: &str,
        enum_paths: &EnumPaths,
        clock: &mut StageClock,
    ) -> anyhow::Result<Analysis> {
        let source_hash = caching::hash_file(source_path)?;
        let index_key = artifact_cache::index_key(
            source_path_relative_str,
            &source_hash,
            &self.compile_schema_hash,
            &self.analyzer_hash,
            &self.gpu_types_hash,
        );

        if !self.bypass_shared_cache
            && let Some(index) = artifact_cache::load_source_index(&self.shared_cache, &index_key)?
            && index.compile_schema_hash == *self.compile_schema_hash.as_bytes()
            && index.analyzer_hash == *self.analyzer_hash.as_bytes()
            && index.gpu_types_hash == *self.gpu_types_hash.as_bytes()
            && index.source_hash == *source_hash.as_bytes()
            && artifact_cache::dependencies_match(&index.dependency_hashes)
        {
            clock.lap("index");
            return Ok(Analysis {
                kernel_infos: index.kernel_infos,
                specialize_indices: index.specialize_indices,
                public_kernels: index.public_kernels,
                dependency_hashes: index.dependency_hashes,
                footer: index.footer,
                from_index: true,
            });
        }

        let _lock = self.shared_cache.lock(Stage::Index, &index_key).await?;
        if !self.bypass_shared_cache
            && let Some(index) = artifact_cache::load_source_index(&self.shared_cache, &index_key)?
            && index.compile_schema_hash == *self.compile_schema_hash.as_bytes()
            && index.analyzer_hash == *self.analyzer_hash.as_bytes()
            && index.gpu_types_hash == *self.gpu_types_hash.as_bytes()
            && index.source_hash == *source_hash.as_bytes()
            && artifact_cache::dependencies_match(&index.dependency_hashes)
        {
            clock.lap("index");
            return Ok(Analysis {
                kernel_infos: index.kernel_infos,
                specialize_indices: index.specialize_indices,
                public_kernels: index.public_kernels,
                dependency_hashes: index.dependency_hashes,
                footer: index.footer,
                from_index: true,
            });
        }

        let (metal_kernel_infos, dependencies) = self
            .toolchain
            .analyze(source_path)
            .await
            .with_context(|| format!("cannot analyze {source_path_relative_str}"))?;
        let kernel_infos: Vec<MetalKernelInfo> = metal_kernel_infos.collect();
        clock.lap("analyze");

        let dependency_hashes = dependencies
            .map(|path| {
                Ok((
                    path.clone(),
                    blake3::hash(&fs::read(path.as_ref()).with_context(|| format!("cannot read {path}"))?).into(),
                ))
            })
            .collect::<anyhow::Result<HashMap<Box<str>, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        let (footer, specialize_indices) = if kernel_infos.is_empty() {
            (String::new(), SpecializeBaseIndices::new())
        } else {
            let (wrapper_strs, specialize_indices) =
                wrappers(&kernel_infos, enum_paths).context("cannot generate kernel wrappers")?;
            let mut footer = String::new();
            for wrapper in wrapper_strs.iter() {
                footer.push_str(wrapper);
            }
            clock.lap("wrappers");
            (footer, specialize_indices)
        };

        let public_kernels: Box<[Kernel]> = kernel_infos.iter().filter_map(|kernel| kernel.to_kernel()).collect();
        let index = SourceIndex {
            schema: caching::METAL_CACHE_SCHEMA,
            source_hash: *source_hash.as_bytes(),
            compile_schema_hash: *self.compile_schema_hash.as_bytes(),
            analyzer_hash: *self.analyzer_hash.as_bytes(),
            gpu_types_hash: *self.gpu_types_hash.as_bytes(),
            dependency_hashes: dependency_hashes.clone(),
            footer: footer.clone(),
            kernel_infos: kernel_infos.clone(),
            specialize_indices: specialize_indices.clone(),
            public_kernels: public_kernels.clone(),
        };
        artifact_cache::store_source_index(&self.shared_cache, &index_key, &index)?;
        clock.lap("index_store");

        Ok(Analysis {
            kernel_infos,
            specialize_indices,
            public_kernels,
            dependency_hashes,
            footer,
            from_index: false,
        })
    }

    async fn ensure_air(
        &self,
        source_path: &Path,
        source_path_relative_str: &str,
        analysis: &Analysis,
        object_file: &Path,
        clock: &mut StageClock,
    ) -> anyhow::Result<EnsuredArtifact> {
        let key = artifact_cache::air_key(
            source_path_relative_str,
            &analysis.dependency_hashes,
            &analysis.footer,
            &self.compile_schema_hash,
            &self.toolchain_hash,
        );
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Air, &key)?
        {
            caching::hard_link_atomic(&artifact.path, object_file)?;
            clock.lap("shared_air");
            return Ok(EnsuredArtifact {
                origin: "shared_air",
                hash: artifact.hash,
            });
        }
        let _lock = self.shared_cache.lock(Stage::Air, &key).await?;
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Air, &key)?
        {
            caching::hard_link_atomic(&artifact.path, object_file)?;
            clock.lap("shared_air");
            return Ok(EnsuredArtifact {
                origin: "shared_air",
                hash: artifact.hash,
            });
        }
        caching::remove_file_if_exists(object_file)?;
        self.compile_air(source_path, source_path_relative_str, &analysis.footer, object_file).await?;
        clock.lap("compile");
        let hash = self.shared_cache.store(Stage::Air, &key, object_file)?;
        clock.lap("air_store");
        Ok(EnsuredArtifact {
            origin: "compile",
            hash,
        })
    }

    async fn ensure_metallib(
        &self,
        source_path_relative_str: &str,
        air_hash: [u8; blake3::OUT_LEN],
        object_file: &Path,
        metallib_file: &Path,
        clock: &mut StageClock,
    ) -> anyhow::Result<EnsuredArtifact> {
        let air_hash = blake3::Hash::from_bytes(air_hash);
        let key = artifact_cache::metallib_key(&air_hash, &self.linker_hash);
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Metallib, &key)?
        {
            caching::hard_link_atomic(&artifact.path, metallib_file)?;
            clock.lap("shared_metallib");
            return Ok(EnsuredArtifact {
                origin: "shared_metallib",
                hash: artifact.hash,
            });
        }
        let _lock = self.shared_cache.lock(Stage::Metallib, &key).await?;
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Metallib, &key)?
        {
            caching::hard_link_atomic(&artifact.path, metallib_file)?;
            clock.lap("shared_metallib");
            return Ok(EnsuredArtifact {
                origin: "shared_metallib",
                hash: artifact.hash,
            });
        }
        caching::remove_file_if_exists(metallib_file)?;
        self.link_metallib(source_path_relative_str, object_file, metallib_file).await?;
        clock.lap("link");
        let hash = self.shared_cache.store(Stage::Metallib, &key, metallib_file)?;
        clock.lap("metallib_store");
        Ok(EnsuredArtifact {
            origin: "link",
            hash,
        })
    }

    async fn ensure_zstd(
        &self,
        metallib_hash: [u8; blake3::OUT_LEN],
        metallib_file: &Path,
        compressed_file: &Path,
        clock: &mut StageClock,
    ) -> anyhow::Result<EnsuredArtifact> {
        let metallib_hash = blake3::Hash::from_bytes(metallib_hash);
        let key = artifact_cache::zstd_key(&metallib_hash, self.zstd_level);
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Zstd, &key)?
        {
            caching::hard_link_atomic(&artifact.path, compressed_file)?;
            clock.lap("shared_zstd");
            return Ok(EnsuredArtifact {
                origin: "shared_zstd",
                hash: artifact.hash,
            });
        }
        let _lock = self.shared_cache.lock(Stage::Zstd, &key).await?;
        if !self.bypass_shared_cache
            && let Some(artifact) = self.shared_cache.lookup(Stage::Zstd, &key)?
        {
            caching::hard_link_atomic(&artifact.path, compressed_file)?;
            clock.lap("shared_zstd");
            return Ok(EnsuredArtifact {
                origin: "shared_zstd",
                hash: artifact.hash,
            });
        }
        self.compress_metallib(metallib_file.to_path_buf(), compressed_file.to_path_buf()).await?;
        clock.lap("compress");
        let hash = self.shared_cache.store(Stage::Zstd, &key, compressed_file)?;
        clock.lap("zstd_store");
        Ok(EnsuredArtifact {
            origin: "compress",
            hash,
        })
    }

    async fn compile(
        &self,
        source_path: PathBuf,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<CompiledSource> {
        let mut clock = StageClock::new();
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
        let metallib_maybe_compressed_file = if self.metallib_compressed {
            metallib_file.with_added_extension("zst")
        } else {
            metallib_file.clone()
        };
        let bindgen_file = output_base_path.with_extension("rs");
        let cached_file = output_base_path.with_extension("cached");

        let local = self.check_local_cache(&cached_file, &metallib_maybe_compressed_file, &bindgen_file);
        clock.lap("local_cache");

        if let LocalCache::Hit(cached) = local {
            debug_log!("compile cached: {source_path_relative_str}");
            return Ok(CompiledSource {
                kernel_path,
                public_kernels: cached.public_kernels,
                has_kernels: cached.has_kernels,
                source: source_path_relative_str.to_string(),
                cache_hit: true,
                miss_reason: None,
                total_ms: clock.total_ms(),
                stages: clock.stages,
            });
        }

        let miss_reason = match &local {
            LocalCache::Hit(_) => unreachable!(),
            LocalCache::Rebind(reason) | LocalCache::Miss(reason) => reason.clone(),
        };

        let analysis = self.resolve_analysis(&source_path, source_path_relative_str, enum_paths, &mut clock).await?;

        if analysis.kernel_infos.is_empty() {
            self.write_local_cache(
                &cached_file,
                analysis.dependency_hashes,
                [0; blake3::OUT_LEN],
                [0; blake3::OUT_LEN],
                Box::new([]),
                false,
            )?;
            clock.lap("cache_store");
            return Ok(CompiledSource {
                kernel_path,
                public_kernels: Box::new([]),
                has_kernels: false,
                source: source_path_relative_str.to_string(),
                cache_hit: analysis.from_index,
                miss_reason: Some(miss_reason.to_string()),
                total_ms: clock.total_ms(),
                stages: clock.stages,
            });
        }

        if let LocalCache::Rebind(_) = local {
            let binding_hash = self.write_bindings(
                source_path_relative_str,
                &analysis.kernel_infos,
                &analysis.specialize_indices,
                enum_paths,
                &metallib_maybe_compressed_file,
                &bindgen_file,
            )?;
            let artifact_hash = *caching::hash_file(&metallib_maybe_compressed_file)?.as_bytes();
            self.write_local_cache(
                &cached_file,
                analysis.dependency_hashes,
                artifact_hash,
                binding_hash,
                analysis.public_kernels.clone(),
                true,
            )?;
            clock.lap("bindgen");
            return Ok(CompiledSource {
                kernel_path,
                public_kernels: analysis.public_kernels,
                has_kernels: true,
                source: source_path_relative_str.to_string(),
                cache_hit: false,
                miss_reason: Some(miss_reason.to_string()),
                total_ms: clock.total_ms(),
                stages: clock.stages,
            });
        }

        let air = self.ensure_air(&source_path, source_path_relative_str, &analysis, &object_file, &mut clock).await?;
        let metallib =
            self.ensure_metallib(source_path_relative_str, air.hash, &object_file, &metallib_file, &mut clock).await?;
        let artifact = if self.metallib_compressed {
            self.ensure_zstd(metallib.hash, &metallib_file, &metallib_maybe_compressed_file, &mut clock).await?
        } else {
            metallib
        };
        let produced_from = if artifact.origin.starts_with("shared_") && !air.origin.starts_with("shared_") {
            air.origin
        } else {
            artifact.origin
        };

        let binding_hash = self.write_bindings(
            source_path_relative_str,
            &analysis.kernel_infos,
            &analysis.specialize_indices,
            enum_paths,
            &metallib_maybe_compressed_file,
            &bindgen_file,
        )?;
        clock.lap("bindgen");

        self.write_local_cache(
            &cached_file,
            analysis.dependency_hashes,
            artifact.hash,
            binding_hash,
            analysis.public_kernels.clone(),
            true,
        )?;

        debug_log!("compile end: {source_path_relative_str} via {produced_from}");

        Ok(CompiledSource {
            kernel_path,
            public_kernels: analysis.public_kernels,
            has_kernels: true,
            source: source_path_relative_str.to_string(),
            cache_hit: produced_from.starts_with("shared_"),
            miss_reason: Some(format!("{miss_reason}; recovered via {produced_from}")),
            total_ms: clock.total_ms(),
            stages: clock.stages,
        })
    }
}

#[async_trait]
impl Compiler for MetalCompiler {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
        enum_paths: &EnumPaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        let gpu_types_started = Instant::now();
        gpu_type_gen(&self.gpu_types_directory, gpu_types).await.context("cannot generate shared gpu types")?;
        build_warning!("gpu-type generation {}ms", gpu_types_started.elapsed().as_millis());

        let metal_sources: Vec<PathBuf> = WalkDir::new(&self.source_directory)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.path().extension().and_then(|s| s.to_str()) == Some("metal"))
            .map(|e| e.into_path())
            .collect();

        let num_concurrent_compiles = std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4) * 2;
        build_warning!("compiling {} metal sources, concurrency {num_concurrent_compiles}", metal_sources.len());

        let compiled: Vec<CompiledSource> = stream::iter(metal_sources)
            .map(|path| async move {
                self.compile(path.clone(), enum_paths)
                    .await
                    .with_context(|| format!("cannot compile {}", path.display()))
            })
            .buffer_unordered(num_concurrent_compiles)
            .try_collect()
            .await?;

        let hits = compiled.iter().filter(|source| source.cache_hit).count();
        let misses = compiled.len() - hits;
        let slowest = compiled.iter().max_by_key(|source| source.total_ms);
        build_warning!(
            "metal summary: {hits} hit, {misses} miss, {} sources; slowest {} ({}ms)",
            compiled.len(),
            slowest.map(|source| source.source.as_str()).unwrap_or("-"),
            slowest.map(|source| source.total_ms).unwrap_or(0)
        );
        for source in &compiled {
            if source.cache_hit {
                debug_log!("metal hit {}: {}ms {:?}", source.source, source.total_ms, source.stages);
            } else {
                build_warning!(
                    "metal miss {}: {} ({}ms) {:?}",
                    source.source,
                    source.miss_reason.as_deref().unwrap_or("unknown"),
                    source.total_ms,
                    source.stages
                );
            }
        }

        let mut kernels_bindgen = compiled
            .iter()
            .filter(|source| source.has_kernels)
            .map(|source| {
                (
                    self.output_directory.join(source.kernel_path.join("/")).with_extension("rs"),
                    source.public_kernels.as_ref(),
                )
            })
            .collect::<Vec<(PathBuf, &[Kernel])>>();
        kernels_bindgen.sort_by(|(a_path, _a_kernels), (b_path, _b_kernels)| a_path.cmp(b_path));

        let tokens = bindgen_global(&kernels_bindgen).context("cannot generate bindings")?;
        write_tokens(tokens, self.output_directory.with_extension("rs")).context("cannot write bindings")?;

        Ok(compiled.into_iter().map(|source| (source.kernel_path, source.public_kernels)).collect())
    }
}
