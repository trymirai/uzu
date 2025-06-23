use std::{
    collections::HashSet,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

fn main() {
    compile_metal_shaders();
}

fn compile_metal_shaders() {
    // Determine the target platform
    let target = env::var("TARGET").unwrap_or_default();

    // Exit early if not compiling for Apple platforms
    if !target.contains("apple") {
        println!(
            "cargo:warning=Not on Apple platform, skipping Metal shader compilation"
        );
        return;
    }

    // Set SDK based on target
    let sdk = if target.contains("ios") {
        "iphoneos"
    } else if target.contains("apple-darwin") {
        if env::var("CARGO_CFG_TARGET_FEATURE")
            .unwrap_or_default()
            .contains("catalyst")
        {
            "maccatalyst"
        } else {
            "macosx"
        }
    } else {
        println!(
            "cargo:warning=Unsupported Apple platform, skipping Metal shader compilation"
        );
        return;
    };

    println!("cargo:info=Compiling Metal shaders for {}", sdk);

    // Find all .metal files
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest_dir.join("src");
    let metal_files = find_metal_files(&src_dir);

    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found in src directory");
        return;
    }

    println!("cargo:info=Found {} Metal shader files", metal_files.len());

    // Create output directory
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metallib_dir = out_dir.join("metallib");
    fs::create_dir_all(&metallib_dir).unwrap();

    // Check if xcrun is available
    if !Command::new("xcrun")
        .arg("--version")
        .status()
        .is_ok_and(|s| s.success())
    {
        panic!(
            "xcrun not found. Metal shader compilation requires macOS with Xcode tools installed."
        );
    }

    // Create platform-specific metallib path
    let platform_suffix = if sdk == "macosx" {
        "macos"
    } else {
        "ios"
    };
    let metallib_path =
        metallib_dir.join(format!("shaders_{}.metallib", platform_suffix));

    // Determine if we need to recompile based on source file timestamps
    let should_compile = needs_compilation(&metal_files, &metallib_path);

    if should_compile {
        // Compile the metal files
        compile_metal_files(&metal_files, &metallib_dir, sdk, &src_dir);

        // Write the current target to a file for future comparison
        let target_file = metallib_path.with_extension("target");
        if let Err(e) = fs::write(&target_file, &target) {
            println!("cargo:warning=Failed to write target file: {}", e);
        }
    } else {
        println!("cargo:info=Using existing metallib as it's up to date");
    }

    // Read metallib content - use the platform-specific one
    let metallib_data = fs::read(&metallib_path).unwrap_or_else(|_| {
        // Fall back to standard name if platform-specific one doesn't exist
        fs::read(metallib_dir.join("shaders.metallib")).unwrap()
    });

    // Write the metallib data as a Rust file with a static byte array
    let rust_output = out_dir.join("metal_lib.rs");
    write_metallib_as_bytes(&metallib_data, &rust_output);

    // Tell Cargo to rerun this script if any .metal files change
    for file in &metal_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }

    // Also rerun if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
    // Also rerun if ffi.rs changes
    println!("cargo:rerun-if-changed=src/ffi.rs");
}

fn find_metal_files(dir: &Path) -> Vec<PathBuf> {
    let mut metal_files = Vec::new();

    if dir.is_dir() {
        for entry in fs::read_dir(dir).unwrap() {
            let entry = entry.unwrap();
            let path = entry.path();

            if path.is_dir() {
                metal_files.extend(find_metal_files(&path));
            } else if let Some(ext) = path.extension() {
                if ext == "metal" {
                    metal_files.push(path);
                }
            }
        }
    }

    metal_files
}

fn needs_compilation(
    metal_files: &[PathBuf],
    metallib_path: &Path,
) -> bool {
    // If metallib doesn't exist, we need to compile
    if !metallib_path.exists() {
        return true;
    }

    // Check if target has changed since last build
    let target = env::var("TARGET").unwrap_or_default();
    let target_file = metallib_path.with_extension("target");

    if let Ok(contents) = fs::read_to_string(&target_file) {
        if contents != target {
            println!(
                "cargo:warning=Target changed from {} to {}, recompiling Metal shaders",
                contents, target
            );
            return true;
        }
    } else {
        // No target file, recompile
        return true;
    }

    // Get the modification time of the metallib
    let metallib_mtime = fs::metadata(metallib_path)
        .and_then(|meta| meta.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);

    // Check if any metal file is newer than the metallib
    for metal_file in metal_files {
        if let Ok(metal_mtime) =
            fs::metadata(metal_file).and_then(|meta| meta.modified())
        {
            if metal_mtime > metallib_mtime {
                return true;
            }
        } else {
            // If we can't get the mtime, be safe and recompile
            return true;
        }
    }

    false
}

fn compile_metal_files(
    metal_files: &[PathBuf],
    out_dir: &Path,
    sdk: &str,
    src_dir: &Path,
) {
    // Create temporary directory for intermediate .air files
    let air_dir = out_dir.join("air");
    fs::create_dir_all(&air_dir).unwrap();

    // Collect unique include directories (parent directories of all metal files)
    let mut include_dirs = HashSet::new();
    for metal_file in metal_files {
        if let Some(parent) = metal_file.parent() {
            include_dirs.insert(parent.to_path_buf());
        }
    }
    // Always include the src directory
    include_dirs.insert(src_dir.to_path_buf());

    // Get appropriate optimization level
    let opt_level = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".to_string());
    let metal_opt = match opt_level.as_str() {
        "0" => "-O0",
        "1" => "-O1",
        "2" | "3" | "s" | "z" => "-O2", // Metal only has up to -O2
        _ => "-O0",
    };

    // Platform-specific compilation flags
    let is_macos = sdk == "macosx";
    let platform_flags = if is_macos {
        // macOS specific flags
        // Set minimum macOS version to 14.0 for Metal 3.1 and bfloat support
        vec!["-mmacosx-version-min=14.0"]
    } else {
        // iOS specific flags
        // Set minimum iOS version to 17.0 for Metal 3.1 and bfloat support
        vec!["-mios-version-min=17.0"]
    };

    println!(
        "cargo:info=Using platform-specific flags for {}: {:?}",
        sdk, platform_flags
    );

    // Step 1: Compile each .metal file to .air
    let mut air_files = Vec::new();
    for metal_file in metal_files {
        let file_stem = metal_file.file_stem().unwrap().to_str().unwrap();

        // Create platform-specific AIR files to avoid conflicts
        let platform_prefix = if is_macos {
            "macos_"
        } else {
            "ios_"
        };
        let air_file =
            air_dir.join(format!("{}_{}.air", platform_prefix, file_stem));
        air_files.push(air_file.clone());

        let mut command = Command::new("xcrun");
        command.args(["-sdk", sdk, "metal", metal_opt]);

        // Use platform-neutral Metal 3.1 standard for both macOS and iOS
        let metal_std = "metal3.1";
        command.args([&format!("-std={}", metal_std)]);

        // Add platform-specific flags
        for flag in &platform_flags {
            command.arg(flag);
        }

        // Add all include directories
        for include_dir in &include_dirs {
            command.args(["-I", include_dir.to_str().unwrap()]);
        }

        // Debug build gets better error messages
        if opt_level == "0" {
            command.arg("-gline-tables-only");

            // Add extra debug options for macOS
            if is_macos {
                command.args(["-frecord-sources", "-fpreserve-invariance"]);
            }
        }

        // Add the source file and output
        command.args([
            "-c",
            metal_file.to_str().unwrap(),
            "-o",
            air_file.to_str().unwrap(),
        ]);

        // Print the command for debugging purposes
        let cmd_str = format!("{:?}", command);
        println!("cargo:info=Running: {}", cmd_str);

        let status =
            command.status().expect("Failed to execute metal compiler");

        if !status.success() {
            panic!("Failed to compile {} to .air", metal_file.display());
        }
    }

    // Step 2: Link all .air files into a single .metallib
    let platform_suffix = if is_macos {
        "macos"
    } else {
        "ios"
    };
    let metallib_path =
        out_dir.join(format!("shaders_{}.metallib", platform_suffix));

    let mut command = Command::new("xcrun");
    command.args(["-sdk", sdk, "metallib"]);

    for air_file in &air_files {
        command.arg(air_file.to_str().unwrap());
    }

    command.args(["-o", metallib_path.to_str().unwrap()]);

    // Print the command for debugging purposes
    let cmd_str = format!("{:?}", command);
    println!("cargo:info=Running: {}", cmd_str);

    let status = command.status().expect("Failed to execute metallib");

    if !status.success() {
        panic!("Failed to link .air files into .metallib");
    }

    // Create a symbolic link or copy to the standard name for backward compatibility
    let standard_metallib_path = out_dir.join("shaders.metallib");
    if let Err(e) = fs::copy(&metallib_path, &standard_metallib_path) {
        println!(
            "cargo:warning=Failed to create standard metallib link: {}",
            e
        );
    }

    // Clean up .air files after successful compilation
    for air_file in &air_files {
        if let Err(e) = fs::remove_file(air_file) {
            println!(
                "cargo:warning=Failed to remove {}: {}",
                air_file.display(),
                e
            );
        }
    }

    println!(
        "cargo:info=Successfully compiled Metal shaders to {}",
        metallib_path.display()
    );
}

fn write_metallib_as_bytes(
    metallib_data: &[u8],
    output_path: &Path,
) {
    let mut file = fs::File::create(output_path).unwrap();

    // Write the header of the Rust file
    writeln!(file, "// This file is auto-generated. Do not edit.").unwrap();
    writeln!(file).unwrap();
    writeln!(file, "pub const METAL_LIBRARY_DATA: &[u8] = &[").unwrap();

    // Write the byte array
    const BYTES_PER_LINE: usize = 16;

    for chunk in metallib_data.chunks(BYTES_PER_LINE) {
        write!(file, "    ").unwrap();
        for byte in chunk {
            write!(file, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(file).unwrap();
    }

    // Close the array
    writeln!(file, "];").unwrap();
}
