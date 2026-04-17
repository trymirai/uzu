use std::{env, fs, path::Path};

const CONFIG_TYPES: &[&str] = &["parsing", "rendering", "tokens", "ordering"];

fn raw_string_delimiter(content: &str) -> String {
    let mut hashes = 1;
    loop {
        let closing = format!("\"{}", "#".repeat(hashes));
        if !content.contains(&closing) {
            return "#".repeat(hashes);
        }
        hashes += 1;
    }
}

fn main() {
    #[cfg(feature = "bindings-napi")]
    napi_build::setup();

    let manifest_directory = env::var("CARGO_MANIFEST_DIR").unwrap();
    let configs_directory = Path::new(&manifest_directory).join("configs");

    println!("cargo:rerun-if-changed={}", configs_directory.display());

    let out_directory = env::var("OUT_DIR").unwrap();
    let destination = Path::new(&out_directory).join("bundled_configs.rs");

    let mut code = String::new();

    for config_type in CONFIG_TYPES {
        let type_directory = configs_directory.join(config_type);
        if !type_directory.exists() {
            continue;
        }

        let mut entries: Vec<(String, String)> = Vec::new();

        for entry in fs::read_dir(&type_directory).expect("Failed to read config directory") {
            let path = entry.unwrap().path();
            if path.extension().map(|extension| extension == "json").unwrap_or(false) {
                let name = path.file_stem().unwrap().to_string_lossy().to_string();

                let content = fs::read_to_string(&path).unwrap();
                let _: serde_json::Value = serde_json::from_str(&content).unwrap_or_else(|error| {
                    panic!("Invalid JSON in {}: {}", path.display(), error);
                });

                entries.push((name, content));
            }
        }

        entries.sort_by(|left, right| left.0.cmp(&right.0));

        let constant_name = format!("BUNDLED_{}_CONFIGS", config_type.to_uppercase());

        code.push_str(&format!("pub const {constant_name}: &[(&str, &str)] = &[\n"));
        for (name, content) in &entries {
            let delimiter = raw_string_delimiter(content);
            code.push_str(&format!("    (\"{name}\", r{delimiter}\"{content}\"{delimiter}),\n",));
        }
        code.push_str("];\n\n");
    }

    let configs_json_path = configs_directory.join("configs.json");
    if configs_json_path.exists() {
        let content = fs::read_to_string(&configs_json_path).unwrap();
        let configs: Vec<serde_json::Value> = serde_json::from_str(&content).unwrap_or_else(|error| {
            panic!("Invalid JSON in {}: {}", configs_json_path.display(), error);
        });

        code.push_str("pub const BUNDLED_CONFIG_MAPPINGS: &[(&str, &str, &str, &str, &str)] = &[\n");
        for config in &configs {
            let name = config["name"].as_str().unwrap();
            let parsing = config["parsing"].as_str().unwrap();
            let rendering = config["rendering"].as_str().unwrap();
            let tokens = config["tokens"].as_str().unwrap();
            let ordering = config["ordering"].as_str().unwrap();
            code.push_str(&format!(
                "    (\"{name}\", \"{parsing}\", \"{rendering}\", \"{tokens}\", \"{ordering}\"),\n",
            ));
        }
        code.push_str("];\n\n");
    }

    fs::write(&destination, code).unwrap();
}
