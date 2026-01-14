use std::{env, fs};
use std::path::PathBuf;
use futures::stream::{self, StreamExt};
use shaderc::{CompileOptions, Compiler};
use walkdir::WalkDir;

fn get_spv_file_name(file_name: &str) -> String {
    format!("{file_name}.spv")
}

async fn compile_vulkan_shader(
    file_path: &PathBuf
) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(file_path)?;

    let compiler = Compiler::new()?;
    let mut options = CompileOptions::new().expect("Failed to create shader compiler options");
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_3 as u32);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);
    options.set_include_callback(move |include_file_name, _include_type, _in_file_name, _depth| {
        let src_dir = file_path.parent().unwrap();
        let include_file_path_buf = src_dir.join(include_file_name);
        let include_file_path = match include_file_path_buf.file_name() {
            Some(path) => path.to_os_string().into_string().unwrap(),
            None => return Err("Can not get file name from path".into())
        };

        let include_file_content = match fs::read_to_string(include_file_path_buf) {
            Ok(content) => content,
            Err(err) => return Err(format!("Failed to read include file: {}", err))
        };

        Ok(shaderc::ResolvedInclude {
            resolved_name: include_file_path,
            content: include_file_content,
        })
    });

    let artifact = compiler.compile_into_spirv(
        &source,
        shaderc::ShaderKind::Compute,
        &file_path.to_string_lossy(),
        "main",
        Some(&options)
    )?;
    let out_path = get_spv_file_name(&file_path.to_string_lossy().to_string().as_str());
    fs::write(&out_path, artifact.as_binary_u8())?;

    Ok(())
}

pub async fn compile_vulkan_shaders() {
    let src_dir = env::current_dir().unwrap().join("shaders");
    let comp_shader_paths = WalkDir::new(&src_dir).into_iter()
        .filter_map(|res| res.ok())
        .filter(|entry| {
            entry.path().is_file() && entry.path().extension().and_then(|s| s.to_str()) == Some("comp")
        })
        .map(|entry| entry.into_path())
        .collect::<Vec<_>>();

    println!("cargo:rerun-if-changed={}", src_dir.display());
    comp_shader_paths.iter().for_each(|file_path| {
        let file_name = file_path.to_str().unwrap();
        println!("cargo:rerun-if-changed={}", file_name);
        println!("cargo:rerun-if-changed={}", get_spv_file_name(&file_name));
    });

    let tasks = comp_shader_paths.into_iter()
        .map(|file_path| async move {
            compile_vulkan_shader(&file_path)
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
            eprintln!("{}", e);
            had_error = true;
        }
    }
    if had_error {
        panic!("Shader compilation failed");
    }
}
