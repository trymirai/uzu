use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::OnceLock;
use regex::Regex;
use shaderc::CompileOptions;
use crate::vulkan::core;

static SPECIALIZE_REGEX: OnceLock<Regex> = OnceLock::new();

#[derive(Debug)]
pub struct ShaderSpecializations {
    pub types: HashMap<String, Vec<String>>
}

impl ShaderSpecializations {
    pub fn get_all_types_definitions(&self) -> Vec<Vec<(&str, &str)>> {
        let mut combos: Vec<Vec<(&str, &str)>> = vec![vec![]];
        let specs = &self.types;

        for (name, types) in specs {
            let mut next = Vec::with_capacity(combos.len() * types.len());
            for combo in &combos {
                for t in types {
                    let mut c = combo.clone();
                    c.push((name, t));
                    next.push(c);
                }
            }
            combos = next;
        }

        combos
    }
}

fn get_specialize_types_regex<'a>() -> &'a Regex {
    SPECIALIZE_REGEX.get_or_init(|| {
        Regex::new(r"^\s*//\s*SPECIALIZE\s*\(([A-Za-z_]\w*)(?:,\s*([A-Za-z_]\w*(?:,\s*[A-Za-z_]\w*)*))?\)$").unwrap()
    })
}

/// Parses source code for strings that matches regex [SPECIALIZE_REGEX]:
/// "// SPECIALIZE(TYPE_NAME, type_1, type_2, ... type_N)".
///
/// For example "SPECIALIZE(BUFFER_TYPE, float, float16_t)"
fn get_shader_specializations(
    source: &str
) -> ShaderSpecializations {
    let types_regex = get_specialize_types_regex();

    let mut types: HashMap<String, Vec<String>> = HashMap::new();
    for line in source.lines() {
        let mut non_whitespace_line = String::from(line);
        non_whitespace_line.retain(|c| !c.is_whitespace());
        if let Some(caps) = types_regex.captures(non_whitespace_line.as_str()) {
            let define_type_name = caps.get(1).unwrap().as_str().to_string();
            let possible_types = caps.get(2).unwrap().as_str()
                .split(",")
                .map(|s| s.to_string())
                .collect::<Vec<String>>();
            types.entry(define_type_name)
                .or_insert_with(|| possible_types);
        }
    }

    ShaderSpecializations {
        types
    }
}

pub fn fill_comp_requests_with_specializations(
    requests: &mut Vec<core::CompilationRequest>,
    common_options: &CompileOptions<'static>,
    source: &str,
    file_path: &PathBuf
) -> Result<(), Box<dyn std::error::Error>> {
    let specializations = get_shader_specializations(source);
    if specializations.types.is_empty() {
        return Ok(())
    }

    let file_dir = file_path.parent().unwrap();

    let all_definitions = specializations.get_all_types_definitions();
    for definitions in all_definitions {
        let mut out_file_name = file_path.file_stem().unwrap().to_str().unwrap().to_string();
        let mut options = CompileOptions::from(common_options.clone());
        let mut source = source.to_string();
        let mut extensions = HashSet::new();

        // fill options and extensions
        for (type_name, type_value) in definitions {
            let mut actual_type = type_value;
            if actual_type == "float16_t" {
                extensions.insert(core::GL_EXT_SHADER_16BIT_STORAGE);
                extensions.insert(core::GL_EXT_SHADER_EXPLICIT_ARITHMETIC_TYPES_FLOAT_16);
            }
            options.add_macro_definition(type_name, Some(actual_type));
            out_file_name.push('_');
            out_file_name.push_str(&type_value);
        }

        // insert extensions in source code
        if !extensions.is_empty() {
            let mut first_pos_to_find_newline = 0;
            if let Some(pos) = source.rfind("#extension") {
                first_pos_to_find_newline = pos;
            }

            let mut ext_insert_pos = source[first_pos_to_find_newline..].find('\n').unwrap() + 1;
            for ext in extensions {
                source.insert_str(ext_insert_pos, ext);
                ext_insert_pos += ext.len();
                source.insert(ext_insert_pos, '\n');
                ext_insert_pos += 1;
            }
        }

        // build new request
        let out_file_path = file_dir.join(out_file_name).with_extension("spv");
        let request = core::CompilationRequest {
            source,
            options,
            out_file_path_str: out_file_path.to_str().unwrap().to_string()
        };
        requests.push(request);
    }

    Ok(())
}