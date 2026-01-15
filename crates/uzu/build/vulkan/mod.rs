use std::env;
use std::path::PathBuf;
use futures::stream::{self, StreamExt};
use walkdir::WalkDir;

mod core;
mod specialize;

pub async fn main() {
    let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let comp_shader_paths = WalkDir::new(&src_dir).into_iter()
        .filter_map(|res| res.ok())
        .filter(|entry| {
            entry.path().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("comp")
        })
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();
    comp_shader_paths.iter().for_each(|file_path| {
        let file_name = file_path.to_str().unwrap();
        println!("cargo:rerun-if-changed={}", file_name);
    });

    let tasks = comp_shader_paths.into_iter()
        .map(|file_path| async move {
            core::compile_vulkan_shader(&file_path)
                .await
                .map_err(|e| format!("failed to compile {}: {}", file_path.display(), e))
        });

    let max_concurrency = std::thread::available_parallelism()
        .map(|x| x.get())
        .unwrap_or(4);
    let results = stream::iter(tasks)
        .buffer_unordered(max_concurrency)
        .collect::<Vec<_>>()
        .await;

    let mut had_error = false;
    for res in results {
        if let Err(e) = res {
            eprintln!("{e}");
            had_error = true;
        }
    }
    if had_error {
        panic!("Shader compilation failed");
    }
}