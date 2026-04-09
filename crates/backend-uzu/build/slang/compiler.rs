use std::{collections::HashMap, fs, path::PathBuf, sync::Arc, thread};

use anyhow::Context;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use itertools::Itertools;
use tokio::{sync::Mutex, task::spawn_blocking};
use walkdir::WalkDir;

use crate::{
    common::{codegen::write_tokens, compiler::Compiler, gpu_types::GpuTypes, kernel::Kernel},
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

    async fn with_blocking_compiler<R, F>(
        &self,
        f: F,
    ) -> anyhow::Result<R>
    where
        R: Send + 'static,
        F: FnOnce(&mut BlockingSlangCompiler<T>) -> anyhow::Result<R> + Send + 'static,
    {
        let blocking_compiler = if let Some(blocking_compiler) = self.blocking_compiler_pool.lock().await.pop() {
            blocking_compiler
        } else {
            BlockingSlangCompiler::new(self.fields.clone()).context("cannot create blocking compiler")?
        };

        let (blocking_compiler, result) = spawn_blocking(move || {
            let mut blocking_compiler = blocking_compiler;
            let result = f(&mut blocking_compiler);
            (blocking_compiler, result)
        })
        .await
        .context("spawn_blocking failed")?;

        self.blocking_compiler_pool.lock().await.push(blocking_compiler);

        result
    }
}

#[async_trait]
impl<T: SlangTarget> Compiler for SlangCompiler<T> {
    async fn build(
        &self,
        gpu_types: &GpuTypes,
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>> {
        let gpu_type_map = Arc::new(
            gpu_type_gen(&self.fields.source_directory.join("generated"), gpu_types)
                .await
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

        let num_concurrent_compiles = thread::available_parallelism().map(|x| x.get()).unwrap_or(4) * 2;

        let gpu_type_map_ref = &gpu_type_map;

        let kernels = stream::iter(slang_sources)
            .map(|path| async move {
                let path_clone = path.clone();
                let gpu_type_map_clone = gpu_type_map_ref.clone();

                self.with_blocking_compiler(|bc| bc.compile(path_clone, gpu_type_map_clone))
                    .await
                    .with_context(|| format!("cannot compile {}", path.display()))
            })
            .buffer_unordered(num_concurrent_compiles)
            .try_collect::<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>()
            .await?;

        let mut kernels_bindgen = kernels
            .iter()
            .filter(|(_segments, kernels)| !kernels.is_empty())
            .map(|(segments, kernels)| {
                (self.fields.output_directory.join(segments.join("/")).with_extension("rs"), kernels.as_ref())
            })
            .collect::<Vec<(PathBuf, &[Kernel])>>();

        kernels_bindgen.sort_by(|(a_path, _a_kernels), (b_path, _b_kernels)| a_path.cmp(b_path));

        let bindgen_tokens = T::bindgen_global(&kernels_bindgen).context("cannot bindgen global")?;
        write_tokens(bindgen_tokens, &self.fields.output_directory.with_extension("rs"))
            .context("cannot write global bindgen file")?;

        Ok(kernels)
    }
}
