use std::{
    collections::HashMap,
    env, fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, future::try_join_all, stream};
use proc_macro2::TokenStream;
use quote::quote;
use serde::{Deserialize, Serialize};
use tokio::{io::AsyncReadExt, task::spawn_blocking};
use walkdir::WalkDir;

use super::{
    ast::{MetalKernelInfo, MetalStructInfo},
    toolchain::MetalToolchain,
    wrapper::{SpecializeBaseIndices, wrappers},
};
use crate::{
    common::{
        caching, codegen::write_tokens, compiler::Compiler, envs,
        kernel::Kernel,
    },
    debug_log,
};

#[derive(Serialize, Deserialize, Debug)]
struct ObjectInfo {
    src_rel_path: PathBuf,
    object_path: PathBuf,
    kernels: Box<[MetalKernelInfo]>,
    structs: Box<[MetalStructInfo]>,
    specialize_indices: SpecializeBaseIndices,
    buildsystem_hash: [u8; blake3::OUT_LEN],
    dependency_hashes: HashMap<Box<str>, [u8; blake3::OUT_LEN]>,
}

impl ObjectInfo {
    fn kernels(&self) -> (Box<[Box<str>]>, Box<[Kernel]>) {
        let src_rel_path: Box<[Box<str>]> = self
            .src_rel_path
            .with_extension("")
            .as_os_str()
            .to_str()
            .unwrap()
            .split("/")
            .map(|s| s.to_string().into_boxed_str())
            .collect();

        let kernels = self.kernels.iter().map(|ki| ki.to_kernel()).collect();

        (src_rel_path, kernels)
    }
}

fn is_nax_source(path: &Path) -> bool {
    path.file_name()
        .and_then(|s| s.to_str())
        .map(|s| s.contains("_nax"))
        .unwrap_or(false)
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
        .join("src/backends/metal/kernel");

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

        let source_path_rel = source_path
            .strip_prefix(&self.src_dir)
            .context("source is not in src_dir")?;

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

        let (metal_kernel_infos, metal_struct_infos, dependencies) =
            self.toolchain.analyze(&source_path).await.with_context(|| {
                format!("cannot analyze {}", source_path_display)
            })?;

        let kernel_infos: Vec<MetalKernelInfo> = metal_kernel_infos.collect();
        let struct_infos: Vec<MetalStructInfo> = metal_struct_infos.collect();

        let (wrapper_strs, specialize_indices) = wrappers(&kernel_infos)
            .context("cannot generate kernel wrappers")?;

        let mut footer = String::new();
        for wrapper in wrapper_strs.iter() {
            footer.push_str(wrapper);
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
            src_rel_path: source_path_rel.into(),
            object_path,
            kernels: kernel_infos.into_boxed_slice(),
            structs: struct_infos.into_boxed_slice(),
            specialize_indices,
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

    fn generate_struct_bindings(
        &self,
        structs: &[MetalStructInfo],
    ) -> anyhow::Result<TokenStream> {
        if structs.is_empty() {
            return Ok(quote! {});
        }

        let mut header_content = String::from("#include <stdint.h>\n\n");
        for struct_info in structs {
            header_content
                .push_str(&format!("struct {} {{\n", struct_info.name));
            for field in struct_info.fields.iter() {
                header_content.push_str(&format!(
                    "    {} {};\n",
                    field.c_type, field.name
                ));
            }
            header_content.push_str("};\n\n");
        }

        let temp_header = self.build_dir.join("dsl_structs.h");
        fs::write(&temp_header, header_content)
            .context("cannot write temp header for struct bindgen")?;

        let mut builder = bindgen::Builder::default()
            .header(temp_header.to_string_lossy())
            .clang_arg("-x")
            .clang_arg("c")
            .derive_default(true)
            .derive_copy(true)
            .derive_debug(true)
            .use_core()
            .layout_tests(true);

        for struct_info in structs {
            builder = builder.allowlist_type(struct_info.name.as_ref());
        }

        let bindings = builder
            .generate()
            .context("bindgen failed to generate struct bindings")?;

        let bindings_str = bindings.to_string();
        let syntax_tree: syn::File = syn::parse_str(&bindings_str)
            .context("failed to parse bindgen output")?;

        Ok(quote! { #syntax_tree })
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

        let all_structs: Vec<MetalStructInfo> = objects
            .clone()
            .into_iter()
            .flat_map(|o| o.structs.iter().cloned())
            .collect();

        let struct_bindings = self
            .generate_struct_bindings(&all_structs)
            .context("cannot generate struct bindings")?;

        let (bindings, associated_types) = objects
            .into_iter()
            .flat_map(|o| o.kernels.iter().map(|k| (k, &o.specialize_indices)))
            .map(|(k, specialize_indices)| {
                super::bindgen::bindgen(k, specialize_indices).with_context(
                    || format!("cannot generate bindings for {}", k.name),
                )
            })
            .collect::<anyhow::Result<(Vec<TokenStream>, Vec<TokenStream>)>>(
            )?;

        let tokens = quote! {
            use crate::backends::metal::{
                ComputeEncoderSetValue,
                KernelDataType,
                MTLContext,
                MTLDataType,
                MTLError,
                MTLFunctionConstantValues,
                MTLSize,
                ProtocolObject,
                Retained,
                metal_extensions::{ComputeEncoderConditional, LibraryPipelineExtensions},
            };
            use metal::{MTLBuffer, MTLComputeCommandEncoder, MTLComputePipelineState};

            #struct_bindings

            #(#bindings)*

            pub struct MetalKernels;

            impl crate::backends::common::kernel::Kernels for MetalKernels {
                type Backend = crate::backends::metal::Metal;

                #(#associated_types)*
            }
        };

        write_tokens(tokens, &out_path).context("cannot write bindings")?;

        fs::write(&hash_path, hash.to_string()).with_context(|| {
            format!("cannot write hash file {}", hash_path.display())
        })?;

        debug_log!("bindgen end");

        Ok(())
    }
}

#[async_trait]
impl Compiler for MetalCompiler {
    async fn build(
        &self
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>> {
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

        Ok(objects.iter().map(|o| o.kernels()).collect())
    }
}
