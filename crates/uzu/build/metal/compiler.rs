use std::collections::HashMap;
use std::iter::{once, zip};
use std::path::Path;
use std::{env, fs, path::PathBuf};

use anyhow::{Context, bail};
use futures::future::try_join_all;
use futures::{StreamExt, TryStreamExt, stream};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::task::spawn_blocking;
use walkdir::WalkDir;

use async_trait::async_trait;

use crate::common::compiler::Compiler;
use crate::common::{caching, envs};
use crate::debug_log;

use super::ast::{MetalArgumentType, MetalKernelInfo};
use super::toolchain::MetalToolchain;

#[derive(Serialize, Deserialize, Debug)]
struct ObjectInfo {
    object_path: PathBuf,
    kernels: Box<[MetalKernelInfo]>,
    buildsystem_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
}

fn is_nax_source(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains("_nax"))
        .unwrap_or(false)
}

fn wrappers(kernel: &MetalKernelInfo) -> anyhow::Result<Box<[Box<str>]>> {
    let mut kernel_wrappers = Vec::new();

    for specialization_variant in kernel
        .specializations
        .as_ref()
        .map(|(t, v)| zip(std::iter::repeat(t), v.iter()).map(Some).collect())
        .unwrap_or(vec![None])
    {
        let (wrapper_name, underlying_name) =
            if let Some((_, variant)) = specialization_variant {
                (
                    format!("{}_{}", kernel.name, variant),
                    format!("{}<{}>", kernel.name, variant),
                )
            } else {
                (kernel.name.to_string(), kernel.name.to_string())
            };

        let max_total_threads_per_threadgroup = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Axis(_, l))
                | Ok(MetalArgumentType::Threads(l)) => Some(format!("({l})")),
                _ => None,
            })
            .collect::<Vec<_>>();

        let max_total_threads_per_threadgroup =
            if !max_total_threads_per_threadgroup.is_empty() {
                max_total_threads_per_threadgroup.join(" * ")
            } else {
                "1".to_string()
            };

        let mut wrapper_arguments = kernel
            .arguments
            .iter()
            .filter_map(|a| match a.argument_type() {
                Ok(MetalArgumentType::Buffer)
                | Ok(MetalArgumentType::Constant(_)) => Some(format!(
                    "{} {}",
                    a.c_type
                        .split_whitespace()
                        .map(|token| {
                            if let Some((typename, variant)) =
                                specialization_variant
                                && token == typename.as_ref()
                            {
                                variant
                            } else {
                                token
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" "),
                    a.name
                )),
                _ => None,
            })
            .collect::<Vec<_>>();

        if kernel.has_axis() {
            if kernel.has_groups() || kernel.has_threads() {
                bail!("mixing groups/threads and axis is not supported");
            }

            wrapper_arguments.push(
                "uint3 __dsl_axis_idx [[thread_position_in_grid]]".into(),
            );
        }

        if kernel.has_groups() {
            wrapper_arguments.push(
                "uint3 __dsl_group_idx [[threadgroup_position_in_grid]]".into(),
            );
        }

        if kernel.has_threads() {
            wrapper_arguments.push(
                "uint3 __dsl_thread_idx [[thread_position_in_threadgroup]]"
                    .into(),
            );
        }

        let wrapper_arguments = wrapper_arguments.join(", ");

        let shared_definitions = kernel.arguments.iter().filter_map(|a| {
            if let Ok(MetalArgumentType::Shared(len)) = a.argument_type() {
                Some(format!(
                    "{} {}[{}]",
                    a.c_type.replace('*', ""),
                    a.name,
                    len.as_ref(),
                ))
            } else {
                None
            }
        });

        let underlying_arguments = {
            let mut group_axis_letters = ["x", "y", "z"].iter();
            let mut thread_axis_letters = ["x", "y", "z"].iter();

            kernel
                .arguments
                .iter()
                .map(|a| match a.argument_type().unwrap() {
                    MetalArgumentType::Buffer
                    | MetalArgumentType::Constant(_)
                    | MetalArgumentType::Shared(_) => a.name.to_string(),
                    MetalArgumentType::Axis(..) => {
                        format!(
                            "__dsl_axis_idx.{}",
                            group_axis_letters.next().unwrap()
                        )
                    },
                    MetalArgumentType::Groups(_) => {
                        format!(
                            "__dsl_group_idx.{}",
                            group_axis_letters.next().unwrap()
                        )
                    },
                    MetalArgumentType::Threads(_) => {
                        format!(
                            "__dsl_thread_idx.{}",
                            thread_axis_letters.next().unwrap()
                        )
                    },
                })
                .collect::<Vec<_>>()
                .join(", ")
        };

        let underlying_call =
            format!("{underlying_name}({underlying_arguments})");

        let wrapper_body = shared_definitions
            .chain(once(underlying_call))
            .map(|l| format!("  {l};\n"))
            .collect::<Vec<_>>()
            .join("");

        kernel_wrappers.push(
            format!(
                "\n[[kernel, max_total_threads_per_threadgroup({max_total_threads_per_threadgroup})]] void {wrapper_name}({wrapper_arguments}) {{\n{wrapper_body}}}\n"
            )
            .into(),
        );
    }

    Ok(kernel_wrappers.into())
}

async fn hash_dependencies(
    dependencies: impl Iterator<Item = Box<str>>
) -> anyhow::Result<HashMap<Box<str>, [u8; blake3::OUT_LEN]>> {
    let futures: Vec<_> = dependencies
        .map(|path| async move {
            let mut file = tokio::fs::File::open(path.as_ref())
                .await
                .with_context(|| format!("cannot open file {}", path))?;
            let mut contents = Vec::new();
            file.read_to_end(&mut contents)
                .await
                .with_context(|| format!("cannot read file {}", path))?;
            let hash = spawn_blocking(move || blake3::hash(&contents))
                .await
                .context("hash task panicked")?;
            Ok::<_, anyhow::Error>((path, hash.into()))
        })
        .collect();
    try_join_all(futures).await.map(|v| v.into_iter().collect())
}

fn objects_hash<'a>(
    objects: impl IntoIterator<Item = &'a ObjectInfo>
) -> anyhow::Result<blake3::Hash> {
    let mut hasher = blake3::Hasher::new();
    hasher.update(
        caching::build_system_hash()
            .context("cannot get build system hash")?
            .as_bytes(),
    );
    let mut paths: Vec<_> =
        objects.into_iter().map(|o| &o.object_path).collect();
    paths.sort();
    for path in paths {
        let path_bytes = path.to_string_lossy();
        let path_bytes = path_bytes.as_bytes();
        hasher.update(&(path_bytes.len() as u32).to_le_bytes());
        hasher.update(path_bytes);
        let contents = fs::read(path)
            .with_context(|| format!("cannot read {}", path.display()))?;
        hasher.update(&(contents.len() as u32).to_le_bytes());
        hasher.update(&contents);
    }
    Ok(hasher.finalize())
}

#[derive(Debug)]
pub struct MetalCompiler {
    src_dir: PathBuf,
    build_dir: PathBuf,
    out_dir: PathBuf,
    toolchain: MetalToolchain,
}

impl MetalCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let src_dir = PathBuf::from(
            env::var("CARGO_MANIFEST_DIR")
                .context("missing CARGO_MANIFEST_DIR")?,
        )
        .join("src");

        let out_dir =
            PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
        let build_dir = out_dir.join("metal");
        fs::create_dir_all(&build_dir).with_context(|| {
            format!("cannot create {}", build_dir.display())
        })?;

        let toolchain =
            MetalToolchain::from_env().context("cannot create toolchain")?;

        Ok(Self {
            src_dir,
            build_dir,
            out_dir,
            toolchain,
        })
    }

    async fn compile(
        &self,
        source_path: PathBuf,
    ) -> anyhow::Result<ObjectInfo> {
        let buildsystem_hash = caching::build_system_hash()
            .context("cannot get build system cache")?
            .as_bytes()
            .clone();

        let source_path_pretty =
            source_path.file_name().unwrap().to_str().unwrap();
        debug_log!("compile start: {source_path_pretty}");

        let source_path_display = source_path.display().to_string();
        let source_path_hash =
            blake3::hash(source_path.to_string_lossy().as_bytes());
        let source_file_name = source_path
            .with_extension("")
            .file_name()
            .context("source path has no file name")?
            .to_os_string();

        let build_dir = self.build_dir.join(source_path_hash.to_string());
        let objectinfo_path =
            build_dir.join(&source_file_name).with_extension("objectinfo");

        // Check cache
        if build_dir.exists() {
            if let Ok(contents) = tokio::fs::read(&objectinfo_path).await
                && let Ok(cached) =
                    serde_json::from_slice::<ObjectInfo>(&contents)
                && buildsystem_hash == cached.buildsystem_hash
                && let Ok(dependency_hashes) =
                    hash_dependencies(cached.dependency_hashes.keys().cloned())
                        .await
                && dependency_hashes == cached.dependency_hashes
            {
                if envs::build_debug() {
                    let kernel_list = cached
                        .kernels
                        .iter()
                        .map(|k| k.name.as_ref())
                        .collect::<Vec<_>>()
                        .join(", ");
                    if kernel_list.is_empty() {
                        debug_log!("compile cached: {source_path_pretty}");
                    } else {
                        debug_log!(
                            "compile cached: {source_path_pretty} (kernels: [{kernel_list}])"
                        );
                    }
                }
                return Ok(cached);
            }
            fs::remove_dir_all(&build_dir).ok();
        }

        fs::create_dir_all(&build_dir).with_context(|| {
            format!("cannot create build directory {}", build_dir.display())
        })?;

        // Analyze source
        let (metal_kernel_infos, dependencies) =
            self.toolchain.analyze(&source_path).await.with_context(|| {
                format!("cannot analyze {}", source_path_display)
            })?;

        let kernel_infos: Vec<MetalKernelInfo> = metal_kernel_infos.collect();

        // Generate footer with kernel wrappers
        let mut footer = String::new();
        for kernel_info in &kernel_infos {
            for wrapper in wrappers(kernel_info)
                .with_context(|| {
                    format!("cannot generate wrappers for {}", kernel_info.name)
                })?
                .iter()
            {
                footer.push_str(wrapper);
            }
        }

        // Compile
        let object_path =
            build_dir.join(&source_file_name).with_extension("air");

        let compile_output = self
            .toolchain
            .compile(&source_path, &footer, &object_path)
            .await
            .with_context(|| {
                format!("cannot compile {}", source_path_display)
            })?;

        if let Some(warnings) = &compile_output {
            for line in warnings.lines() {
                println!("cargo::warning={line}");
            }
        }

        let dependency_hashes = hash_dependencies(dependencies)
            .await
            .context("cannot hash dependencies")?;

        let object_info = ObjectInfo {
            object_path,
            kernels: kernel_infos.into_boxed_slice(),
            buildsystem_hash,
            dependency_hashes,
        };

        fs::write(
            &objectinfo_path,
            serde_json::to_string_pretty(&object_info)
                .context("failed to serialize object info")?
                .as_bytes(),
        )
        .with_context(|| {
            format!("cannot write object info {}", objectinfo_path.display())
        })?;

        if envs::build_debug() {
            let kernel_list = object_info
                .kernels
                .iter()
                .map(|k| k.name.as_ref())
                .collect::<Vec<_>>()
                .join(", ");
            if kernel_list.is_empty() {
                debug_log!("compile end: {source_path_pretty}");
            } else {
                debug_log!(
                    "compile end: {source_path_pretty} (kernels: [{kernel_list}])"
                );
            }
        }

        Ok(object_info)
    }

    async fn link<'a>(
        &self,
        objects: impl IntoIterator<Item = &'a ObjectInfo> + Clone,
    ) -> anyhow::Result<PathBuf> {
        let library_path = self.out_dir.join("default.metallib");
        let hash_path = self.out_dir.join("default.metallib.hash");

        let hash = objects_hash(objects.clone())?;

        if let Ok(cached_hash) = fs::read_to_string(&hash_path)
            && cached_hash == hash.to_string()
        {
            debug_log!("linking cached");
            return Ok(library_path);
        }

        debug_log!("linking start");

        let link_output = self
            .toolchain
            .link(objects.into_iter().map(|o| &o.object_path), &library_path)
            .await
            .context("cannot link objects")?;

        if let Some(warnings) = &link_output {
            for line in warnings.lines() {
                println!("cargo::warning={line}");
            }
        }

        fs::write(&hash_path, hash.to_string()).with_context(|| {
            format!("cannot write hash file {}", hash_path.display())
        })?;

        debug_log!("linking end");

        Ok(library_path)
    }

    fn bindgen<'a>(
        &self,
        objects: impl IntoIterator<Item = &'a ObjectInfo> + Clone,
    ) -> anyhow::Result<()> {
        let out_path = self.out_dir.join("dsl.rs");
        let hash_path = self.out_dir.join("dsl.rs.hash");

        let hash = objects_hash(objects.clone())?;

        if let Ok(cached_hash) = fs::read_to_string(&hash_path)
            && cached_hash == hash.to_string()
        {
            debug_log!("bindgen cached");
            return Ok(());
        }

        debug_log!("bindgen start");

        let bindings = objects
            .into_iter()
            .flat_map(|o| &o.kernels)
            .map(|k| {
                super::bindgen::bindgen(k).with_context(|| {
                    format!("cannot generate bindings for {}", k.name)
                })
            })
            .collect::<anyhow::Result<Vec<TokenStream>>>()?;

        let tokens = quote! { #(#bindings)* };

        let parsed =
            syn::parse2(tokens).context("cannot parse generated bindings")?;
        fs::write(&out_path, prettyplease::unparse(&parsed)).with_context(
            || format!("cannot write bindings file {}", out_path.display()),
        )?;

        fs::write(&hash_path, hash.to_string()).with_context(|| {
            format!("cannot write hash file {}", hash_path.display())
        })?;

        debug_log!("bindgen end");

        Ok(())
    }
}

#[async_trait]
impl Compiler for MetalCompiler {
    async fn build(&self) -> anyhow::Result<()> {
        let nax_enabled = cfg!(feature = "metal-nax");
        debug_log!(
            "metal nax {}",
            if nax_enabled {
                "enabled"
            } else {
                "disabled"
            }
        );

        let metal_sources: Vec<PathBuf> = WalkDir::new(&self.src_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.file_type().is_file()
                    && e.path().extension().and_then(|s| s.to_str())
                        == Some("metal")
            })
            .map(|e| e.into_path())
            .filter(|p| nax_enabled || !is_nax_source(p))
            .collect();

        let num_concurrent_compiles =
            std::thread::available_parallelism().map(|x| x.get()).unwrap_or(4)
                * 2;

        let objects: Vec<ObjectInfo> = stream::iter(metal_sources)
            .map(|p| self.compile(p))
            .buffer_unordered(num_concurrent_compiles)
            .try_collect()
            .await
            .context("cannot compile metal sources")?;

        self.link(&objects).await.context("cannot link objects")?;
        self.bindgen(&objects).context("cannot generate bindings")?;

        Ok(())
    }
}
