use std::{
    collections::HashMap,
    fs,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Context;
use rayon::prelude::*;
use walkdir::WalkDir;

use crate::{
    common::{
        codegen::write_tokens, compiler::Compiler, gpu_type_paths::GpuTypePaths, gpu_types::GpuTypes,
        identifiers::KernelPath, kernel::Kernel,
    },
    slang::{
        compiler_internal::{BlockingSlangCompiler, SlangCompilerFields, SlangTarget},
        gpu_types::gpu_type_gen,
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
        _gpu_type_paths: &GpuTypePaths,
    ) -> anyhow::Result<HashMap<KernelPath, Box<[Kernel]>>> {
        let gpu_type_map = Arc::new(
            gpu_type_gen(&self.fields.source_directory.join("generated"), gpu_types)
                .context("cannot generate shared gpu types")?,
        );

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
                self.with_blocking_compiler(|compiler| compiler.compile(path.clone(), gpu_type_map.clone()))
                    .with_context(|| format!("cannot compile {}", path.display()))
            })
            .collect::<anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>>()?;

        let mut kernels_bindgen = kernels
            .iter()
            .filter(|(_segments, kernels)| !kernels.is_empty())
            .map(|(segments, kernels)| {
                (self.fields.output_directory.join(segments.join("/")).with_extension("rs"), kernels.as_ref())
            })
            .collect::<Vec<(PathBuf, &[Kernel])>>();

        kernels_bindgen.sort_by(|(a_path, _a_kernels), (b_path, _b_kernels)| a_path.cmp(b_path));

        let bindgen_tokens = T::bindgen_global(&kernels_bindgen).context("cannot bindgen global")?;
        write_tokens(bindgen_tokens, self.fields.output_directory.with_extension("rs"))
            .context("cannot write global bindgen file")?;

        Ok(kernels
            .into_iter()
            .filter(|(_, kernels)| !kernels.is_empty())
            .map(|(segments, kernels)| (segments.iter().map(|segment| segment.to_string()).collect(), kernels))
            .collect())
    }
}
