use std::collections::HashMap;
use std::sync::OnceLock;
use regex::Regex;

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

static SPECIALIZE_REGEX: OnceLock<Regex> = OnceLock::new();

fn get_specialize_types_regex<'a>() -> &'a Regex {
    SPECIALIZE_REGEX.get_or_init(|| {
        Regex::new(r"^\s*//\s*SPECIALIZE\s*\(([A-Za-z_]\w*)(?:,\s*([A-Za-z_]\w*(?:,\s*[A-Za-z_]\w*)*))?\)$").unwrap()
    })
}

pub fn get_shader_specializations(
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