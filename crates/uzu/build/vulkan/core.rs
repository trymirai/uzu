use std::path::PathBuf;
use shaderc::{CompileOptions, Compiler};
use crate::vulkan::specialize;
use crate::vulkan::specialize::ShaderSpecializations;

struct CompilationRequest {
    pub source: String,
    pub options: CompileOptions<'static>,
    pub out_file_path_str: String
}

fn get_common_compile_options(
    file_dir: PathBuf
) -> Result<CompileOptions<'static>, Box<dyn std::error::Error>> {
    let mut options = CompileOptions::new()?;
    options.set_target_env(shaderc::TargetEnv::Vulkan, shaderc::EnvVersion::Vulkan1_2 as u32);
    options.set_optimization_level(shaderc::OptimizationLevel::Performance);

    options.set_include_callback(move |include_file_name, _include_type, _in_file_name, _depth| {
        let include_file_path_buf = file_dir.join(include_file_name);
        let include_file_path = match include_file_path_buf.file_name() {
            Some(path) => path.to_os_string().into_string().unwrap(),
            None => return Err("Can not get file name from path".into())
        };

        let include_file_content = match std::fs::read_to_string(include_file_path_buf) {
            Ok(content) => content,
            Err(err) => return Err(format!("Failed to read include file: {}", err))
        };

        Ok(shaderc::ResolvedInclude {
            resolved_name: include_file_path,
            content: include_file_content,
        })
    });

    Ok(options)
}

fn fill_comp_requests_with_specializations(
    requests: &mut Vec<CompilationRequest>,
    specializations: &ShaderSpecializations,
    common_options: &CompileOptions<'static>,
    source: &str,
    file_path: &PathBuf
) -> Result<(), Box<dyn std::error::Error>> {
    if specializations.types.is_empty() {
        return Ok(())
    }

    let file_dir = file_path.parent().unwrap();

    let all_definitions = specializations.get_all_types_definitions();
    for _definitions in all_definitions {
        let file_name = file_path.file_stem().unwrap().to_str().unwrap().to_string();
        let options = CompileOptions::from(common_options.clone());

        let out_file_path = file_dir.join(file_name).with_extension(".spv");
        let request = CompilationRequest {
            source: source.to_string(),
            options,
            out_file_path_str: out_file_path.to_str().unwrap().to_string()
        };
        requests.push(request);
    }

    Ok(())
}

pub async fn compile_vulkan_shader(
    file_path: &PathBuf
) -> Result<(), Box<dyn std::error::Error>> {
    let source = std::fs::read_to_string(file_path)?;

    let compiler = Compiler::new()?;
    let dir = file_path.parent().unwrap().to_path_buf();
    let common_options = get_common_compile_options(dir)?;

    let mut comp_requests: Vec<CompilationRequest> = Vec::new();
    let specializations = specialize::get_shader_specializations(source.as_str());
    fill_comp_requests_with_specializations(
        &mut comp_requests,
        &specializations,
        &common_options,
        source.as_str(),
        file_path
    )?;

    if comp_requests.is_empty() {
        let out_file_path = file_path.with_extension("spv");
        let request = CompilationRequest {
            source,
            options: common_options,
            out_file_path_str: out_file_path.to_str().unwrap().to_string()
        };
        comp_requests.push(request);
    }

    for request in comp_requests {
        let artifact = compiler.compile_into_spirv(
            &request.source,
            shaderc::ShaderKind::Compute,
            &file_path.to_string_lossy(),
            "main",
            Some(&request.options)
        )?;
        std::fs::write(&request.out_file_path_str, artifact.as_binary_u8())?;
    }

    Ok(())
}