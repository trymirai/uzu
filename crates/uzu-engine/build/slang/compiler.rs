use std::{collections::HashMap, fs, iter::once, path::PathBuf, sync::Mutex};

use anyhow::Context;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::{
    common::{
        codegen::write_tokens,
        compiler::Compiler,
        gpu_type_paths::GpuTypePaths,
        gpu_types::GpuTypes,
        identifiers::{KernelName, KernelPath},
        kernel::Kernel,
    },
    slang::{
        SlangGpuTypes,
        compiler_internal::{BlockingSlangCompiler, SlangCompilerFields, SlangTarget},
        gpu_type_gen,
    },
};

pub struct SlangCompiler<T: SlangTarget> {
    fields: SlangCompilerFields<T>,
    blocking_compiler_pool: Mutex<Vec<BlockingSlangCompiler<T>>>,
}

impl<T: SlangTarget> SlangCompiler<T> {
    pub fn new() -> anyhow::Result<Self> {
        let fields = SlangCompilerFields::new()?;
        let blocking_compiler_pool = Mutex::new(Vec::new());

        Ok(Self {
            fields,
            blocking_compiler_pool,
        })
    }

    fn with_blocking_compiler<R, F>(
        &self,
        f: F,
    ) -> anyhow::Result<R>
    where
        F: FnOnce(&mut BlockingSlangCompiler<T>) -> anyhow::Result<R>,
    {
        let blocking_compiler = if let Some(blocking_compiler) = self.blocking_compiler_pool.lock().unwrap().pop() {
            blocking_compiler
        } else {
            BlockingSlangCompiler::new(self.fields.clone()).context("cannot create blocking compiler")?
        };

        let mut blocking_compiler = blocking_compiler;
        let result = f(&mut blocking_compiler);

        self.blocking_compiler_pool.lock().unwrap().push(blocking_compiler);

        result
    }
}

impl<T: SlangTarget> Compiler for SlangCompiler<T> {
    fn build(
        &self,
        gpu_types: &GpuTypes,
        gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        let private_gpu_types =
            T::PRIVATE_GPU_TYPES.map(GpuTypes::scan).transpose().context("cannot scan private gpu types")?;
        let mut paths = gpu_type_paths.clone();
        if let Some(private_gpu_types) = &private_gpu_types {
            paths.extend(private_gpu_types).context("cannot extend gpu type paths")?;
        }
        let definitions = once(gpu_types).chain(&private_gpu_types).collect::<Vec<_>>();
        // Before any module loads or cache check, so cached dependency hashes see the current types.
        let map = gpu_type_gen(&T::gpu_types_directory(&self.fields), &definitions)
            .context("cannot generate shared gpu types")?;
        let gpu_types = SlangGpuTypes {
            map,
            paths,
            definitions,
        };

        let slang_sources: Vec<PathBuf> = WalkDir::new(&self.fields.source_directory)
            .into_iter()
            .filter_map(|res| res.ok())
            .filter(|entry| {
                entry.path().is_file()
                    && entry.path().extension().and_then(|s| s.to_str()) == Some("slang")
                    && fs::read(entry.path()).map(|p| !p.starts_with(b"implementing")).unwrap_or(true)
            })
            .map(|e| e.into_path())
            .collect();

        let kernels = slang_sources
            .into_par_iter()
            .map(|path| {
                self.with_blocking_compiler(|compiler| compiler.compile(path.clone(), &gpu_types))
                    .with_context(|| format!("cannot compile {}", path.display()))
            })
            .collect::<anyhow::Result<Vec<(Box<[Box<str>]>, Box<[Kernel]>, Box<[KernelName]>)>>>()?;

        let mut kernels_bindgen = kernels
            .iter()
            .filter(|(_segments, public, private)| !public.is_empty() || !private.is_empty())
            .map(|(segments, public, private)| {
                (self.fields.output_directory.join(segments.join("/")), public.as_ref(), private.as_ref())
            })
            .collect::<Vec<(PathBuf, &[Kernel], &[KernelName])>>();

        kernels_bindgen.sort_by(|(a_path, ..), (b_path, ..)| a_path.cmp(b_path));

        let bindgen_tokens = T::bindgen_global(&self.fields, &kernels_bindgen).context("cannot bindgen global")?;
        write_tokens(bindgen_tokens, self.fields.output_directory.with_extension("rs"))
            .context("cannot write global bindgen file")?;

        Ok(kernels
            .into_iter()
            .filter(|(_, kernels, _)| !kernels.is_empty())
            .map(|(segments, kernels, _)| (segments.iter().map(|segment| segment.to_string()).collect(), kernels))
            .collect())
    }
}
