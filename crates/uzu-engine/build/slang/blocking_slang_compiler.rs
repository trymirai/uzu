use std::{collections::HashMap, ffi::CString, fs, iter::once, path::PathBuf, sync::Arc};

use anyhow::Context;
use shader_slang::{CompilerOptions, ComponentType, GlobalSession, Session, SessionDesc};

use crate::{
    common::{caching, gpu_type_paths::GpuTypePaths, kernel::Kernel},
    debug_log,
    slang::{
        Cached, SlangCompilerFields, SlangTarget, reflection::SlangKernel,
        slang_sys_ext::SessionLoadModuleWithDiagnosticsExt,
    },
};

pub struct BlockingSlangCompiler<T: SlangTarget> {
    session: Session,
    global_session: GlobalSession,
    fields: SlangCompilerFields<T>,
}

unsafe impl<T: SlangTarget> Send for BlockingSlangCompiler<T> {}

impl<T: SlangTarget> BlockingSlangCompiler<T> {
    pub fn new(fields: SlangCompilerFields<T>) -> anyhow::Result<Self> {
        let global_session = GlobalSession::new().context("cannot create global session")?;

        let compiler_options = CompilerOptions::default()
            .optimization(fields.optimization_level)
            .emit_spirv_directly(true)
            .vulkan_use_entry_point_name(true);

        let search_path = CString::new(fields.source_directory.to_string_lossy().as_bytes())
            .context("cannot convert src_dir to CString")?;
        let search_paths = [search_path.as_ptr()];

        let target_desc = T::create_target_desc(&global_session);
        let targets = [target_desc];

        let session_desc =
            SessionDesc::default().options(&compiler_options).search_paths(&search_paths).targets(&targets);

        let session = global_session.create_session(&session_desc).context("cannot create session")?;

        Ok(Self {
            session,
            global_session,
            fields,
        })
    }

    pub fn compile(
        &self,
        source_path: PathBuf,
        gpu_type_map: Arc<HashMap<String, String>>,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<(Box<[Box<str>]>, Box<[Kernel]>)> {
        let source_path_relative =
            source_path.strip_prefix(&self.fields.source_directory).context("source is not in src_dir")?;
        let source_path_relative_str = source_path_relative.to_str().unwrap();
        debug_log!("compile start: {source_path_relative_str}");

        let source_path_relative_segments: Box<[Box<str>]> = source_path_relative
            .with_extension("")
            .components()
            .map(|s| s.as_os_str().to_str().unwrap().to_string().into_boxed_str())
            .collect();

        let output_base_path = self.fields.output_directory.join(source_path_relative).with_extension("");
        fs::create_dir_all(output_base_path.parent().context("cannot get output directory")?)
            .context("cannot create output directory")?;

        let wrapper_file = output_base_path.with_extension("slang");
        let cached_file = output_base_path.with_extension("cached");

        let buildsystem_hash: &[u8; blake3::OUT_LEN] =
            caching::build_system_hash().context("cannot get build system hash")?.as_bytes();

        if let Ok(cached_contents) = fs::read(&cached_file)
            && let Ok(cached) = serde_json::from_slice::<Cached>(&cached_contents)
            && &cached.buildsystem_hash == buildsystem_hash
            && cached.compiler_version == self.global_session.build_tag_string()
            && cached.dependency_hashes.iter().all(|(path, hash)| {
                fs::read(path)
                    .map(|dependency_contents| blake3::hash(&dependency_contents).as_bytes() == hash)
                    .unwrap_or(false)
            })
        {
            debug_log!("compile cached: {source_path_relative_str}");
            return Ok((source_path_relative_segments, cached.public_kernels));
        }

        let (source_module, source_diagnostics) =
            self.session.load_module_with_diagnostics(&source_path).context("cannot load source module")?;

        if let Some(source_diagnostics) = source_diagnostics {
            for source_diagnostics_line in source_diagnostics.lines() {
                println!("cargo::warning={source_diagnostics_line}");
            }
        }

        let dependency_hashes = source_module
            .dependency_file_paths()
            .map(|d| Ok((d.into(), blake3::hash(&fs::read(d).with_context(|| format!("cannot read {d}"))?).into())))
            .collect::<anyhow::Result<HashMap<String, [u8; blake3::OUT_LEN]>>>()
            .context("cannot hash dependencies")?;

        let common_gpu_type_map = gpu_type_map.keys().map(|name| (name.clone(), name.clone())).collect();
        let kernels = source_module
            .module_reflection()
            .children()
            .filter_map(|decl| SlangKernel::from_reflection(decl, &common_gpu_type_map).transpose())
            .collect::<anyhow::Result<Vec<SlangKernel>>>()
            .context("cannot collect kernels")?;

        if !kernels.is_empty() {
            let wrapper_contents = T::wrapper(&source_path, &kernels).context("cannot generate wrappers")?;

            fs::write(&wrapper_file, wrapper_contents.as_bytes()).context("cannot write wrapper file")?;

            let (wrapper_module, wrapper_diagnostics) =
                self.session.load_module_with_diagnostics(&wrapper_file).context("cannot load wrapper module")?;

            if let Some(wrapper_diagnostics) = wrapper_diagnostics {
                for wrapper_diagnostics_line in wrapper_diagnostics.lines() {
                    println!("cargo::warning={wrapper_diagnostics_line}");
                }
            }

            let entry_points =
                wrapper_module.entry_points().map(|entry_point| entry_point.into()).collect::<Vec<ComponentType>>();

            let wrapper_component: ComponentType = wrapper_module.into();

            let composite_component = self
                .session
                .create_composite_component_type(
                    &once(wrapper_component).chain(entry_points).collect::<Vec<ComponentType>>(),
                )
                .context("cannot create composite component")?;

            let linked_component = composite_component.link().context("cannot link wrapper module")?;

            T::emit(&source_module, &kernels, &gpu_type_map, gpu_type_paths, &linked_component, &output_base_path)
                .context("cannot emit kernels")?;
        }

        let public_kernels = kernels
            .iter()
            .filter_map(|slang_kernel| {
                slang_kernel
                    .to_common(&common_gpu_type_map)
                    .with_context(|| format!("cannot convert {} to common kernel", slang_kernel.name))
                    .transpose()
            })
            .collect::<anyhow::Result<Box<[Kernel]>>>()?;

        fs::write(
            &cached_file,
            &serde_json::to_vec_pretty(&Cached {
                buildsystem_hash: *buildsystem_hash,
                compiler_version: self.global_session.build_tag_string().to_owned(),
                dependency_hashes,
                public_kernels: public_kernels.clone(),
            })
            .context("cannot serialize cache")?,
        )
        .context("cannot write cache file")?;

        debug_log!("compile end: {source_path_relative_str}");

        Ok((source_path_relative_segments, public_kernels))
    }
}
