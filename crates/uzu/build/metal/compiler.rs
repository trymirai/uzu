use std::collections::HashMap;
use std::iter::{once, zip};
use std::{env, fs, path::PathBuf};

use anyhow::{Context, bail};
use futures::future::try_join_all;
use serde::{Deserialize, Serialize};
use tokio::io::AsyncReadExt;
use tokio::task::spawn_blocking;

use crate::debug_log;

use super::ast::{MetalArgumentType, MetalKernelInfo};
use super::toolchain::MetalToolchain;

#[derive(Serialize, Deserialize, Debug)]
pub struct ObjectInfo {
    pub object_path: PathBuf,
    pub kernels: Box<[MetalKernelInfo]>,
    pub dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
}

pub fn wrappers(kernel: &MetalKernelInfo) -> anyhow::Result<Box<[Box<str>]>> {
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
            .collect::<Vec<_>>()
            .join(" * ");

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

#[derive(Debug)]
pub struct MetalCompiler {
    out_dir: PathBuf,
    build_dir: PathBuf,
    toolchain: MetalToolchain,
}

impl MetalCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let out_dir =
            PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);
        let build_dir = out_dir.join("metal");
        fs::create_dir_all(&build_dir).with_context(|| {
            format!("cannot create {}", build_dir.display())
        })?;

        let toolchain =
            MetalToolchain::from_env().context("cannot create toolchain")?;

        Ok(Self {
            out_dir,
            build_dir,
            toolchain,
        })
    }

    pub async fn compile(
        &self,
        source_path: PathBuf,
    ) -> anyhow::Result<(ObjectInfo, bool)> {
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
                && let Ok(current_hashes) =
                    hash_dependencies(cached.dependency_hashes.keys().cloned())
                        .await
                && current_hashes == cached.dependency_hashes
            {
                if cfg!(feature = "build-debug") {
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
                return Ok((cached, true));
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

        if cfg!(feature = "build-debug") {
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

        Ok((object_info, false))
    }

    pub async fn link(
        &self,
        objects: impl IntoIterator<Item = &ObjectInfo>,
    ) -> anyhow::Result<PathBuf> {
        debug_log!("linking start");

        let library_path = self.out_dir.join("default.metallib");

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

        debug_log!("linking end");

        Ok(library_path)
    }
}
