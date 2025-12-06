use std::{
    collections::HashSet,
    env, fs,
    io::Write,
    path::{Path, PathBuf},
    process::Command,
    time::SystemTime,
};

fn main() {
    // Generate Rust bindings from shared Metal/C headers
    generate_metal_bindings();

    if cfg!(feature = "metal-shaders") {
        println!("cargo:rerun-if-env-changed=MY_API_LEVEL");
        compile_metal_shaders();
    } else {
        write_empty_metallib();
    }
}

/// Generate Rust bindings from shared C headers using bindgen.
/// These headers are also used by Metal shaders, ensuring type consistency.
fn generate_metal_bindings() {
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let matmul_dir = manifest_dir.join("src/backends/metal/kernel/matmul");
    let types_header = matmul_dir.join("shared_types.h");

    // Only regenerate if header changed
    println!("cargo:rerun-if-changed={}", types_header.display());

    let bindings = bindgen::Builder::default()
        .header(types_header.to_string_lossy())
        // Generate for C mode (no Metal-specific stuff)
        .clang_arg("-x")
        .clang_arg("c")
        // Derive common traits
        .derive_default(true)
        .derive_copy(true)
        // Only generate types we care about
        .allowlist_type("GEMMParams")
        .allowlist_type("GEMMAddMMParams")
        // Use core types
        .use_core()
        .generate()
        .expect("Failed to generate bindings from shared_types.h");

    // Write bindings to the source directory so it can be checked in
    let bindings_path = matmul_dir.join("shared_types.rs");
    bindings
        .write_to_file(&bindings_path)
        .expect("Failed to write shared_types.rs");
}

fn compile_metal_shaders() {
    // Determine target platform (only proceed on Apple platforms)
    let target = env::var("TARGET").unwrap_or_default();
    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    let target_env = env::var("CARGO_CFG_TARGET_ENV").unwrap_or_default();

    // Choose appropriate SDK: macosx, iphoneos, iphonesimulator, or maccatalyst
    let sdk = if target_os == "ios" {
        if target.contains("macabi") || target_env == "macabi" {
            "maccatalyst"
        } else if target.contains("ios")
            && (target.contains("86_64") || target_env == "sim")
        {
            // Use iPhoneSimulator SDK for simulator targets (x86_64 or explicit 'sim')
            "iphonesimulator"
        } else {
            "iphoneos"
        }
    } else if target_os == "macos" {
        "macosx"
    } else {
        println!(
            "cargo:warning=Not an Apple platform, skipping Metal shader compilation"
        );
        return;
    };

    println!("cargo:info=Compiling Metal shaders for SDK: {}", sdk);

    // Gather all .metal files in the project (src directory and subdirs)
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let src_dir = manifest_dir.join("src");
    let metal_files = find_metal_files(&src_dir);
    if metal_files.is_empty() {
        println!("cargo:warning=No .metal files found; nothing to compile");
        return;
    }
    println!("cargo:info=Found {} Metal shader files", metal_files.len());

    // Prepare output directory for metallib
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metallib_dir = out_dir.join("metallib");
    fs::create_dir_all(&metallib_dir).unwrap();

    // Require xcrun (should be present on macOS with Xcode)
    if !Command::new("xcrun")
        .arg("--version")
        .status()
        .map(|s| s.success())
        .unwrap_or(false)
    {
        panic!(
            "xcrun not found. Please install Xcode command-line tools to compile Metal shaders."
        );
    }

    // Platform-specific metallib path (separate for macos vs ios for caching)
    let platform_suffix = if sdk == "macosx" || sdk == "maccatalyst" {
        "macos"
    } else {
        "ios"
    };
    let metallib_path =
        metallib_dir.join(format!("shaders_{}.metallib", platform_suffix));

    // Decide whether to compile (if metallib missing or any source changed)
    let should_compile =
        needs_compilation(&metal_files, &metallib_path, &target);
    if should_compile {
        compile_metal_files(&metal_files, &metallib_dir, sdk, &src_dir);
        // Record the target triple for which we built the metallib
        let _ = fs::write(metallib_path.with_extension("target"), &target);
    } else {
        println!(
            "cargo:info=Metal shaders are up-to-date; skipping compilation"
        );
    }

    // Read the (new or existing) metallib bytes
    let metallib_data = fs::read(&metallib_path)
        .or_else(|_| fs::read(metallib_dir.join("shaders.metallib")))
        .expect("Failed to read metallib data");
    // Write bytes into a Rust source file as a static array
    write_metallib_as_bytes(&metallib_data, &out_dir.join("metal_lib.rs"));

    // Tell Cargo when to re-run this build script: all shader files, plus build.rs and any FFI headers
    let mut sorted_files: Vec<&PathBuf> = metal_files.iter().collect();
    sorted_files.sort();
    for file in sorted_files {
        println!("cargo:rerun-if-changed={}", file.display());
    }
    println!("cargo:rerun-if-changed=build.rs");
}

// Recursively find all .metal files in the given directory
fn find_metal_files(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                files.extend(find_metal_files(&path));
            } else if path.extension().and_then(|s| s.to_str()) == Some("metal")
            {
                files.push(path);
            }
        }
    }
    files
}

// Check timestamps and target to determine if recompilation is needed
fn needs_compilation(
    metal_files: &[PathBuf],
    metallib_path: &Path,
    target: &str,
) -> bool {
    // If output metallib doesn't exist, need to compile
    if !metallib_path.exists() {
        return true;
    }
    // If the previously built target differs (target changed, e.g. arch or OS)
    if let Ok(prev_target) =
        fs::read_to_string(metallib_path.with_extension("target"))
    {
        if prev_target != target {
            println!(
                "cargo:warning=Target changed from {} to {}, recompiling shaders",
                prev_target, target
            );
            return true;
        }
    }
    // If any source file is newer than the metallib, recompile
    let metallib_mtime = fs::metadata(metallib_path)
        .and_then(|m| m.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    for metal_file in metal_files {
        if let Ok(src_mtime) =
            fs::metadata(metal_file).and_then(|m| m.modified())
        {
            if src_mtime > metallib_mtime {
                return true;
            }
        } else {
            // If any file's metadata can't be read, assume we should rebuild
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
    // Create temp directory for .air files
    let air_dir = out_dir.join("air");
    fs::create_dir_all(&air_dir).unwrap();

    // Collect include directories (parents of each metal file + src dir)
    let mut include_dirs = HashSet::new();
    include_dirs.insert(src_dir.to_path_buf());
    for metal_file in metal_files {
        if let Some(parent) = metal_file.parent() {
            include_dirs.insert(parent.to_path_buf());
        }
    }

    // Optimization level: Metal supports up to -O2
    let opt = env::var("OPT_LEVEL").unwrap_or_else(|_| "0".into());
    let metal_opt_flag = match opt.as_str() {
        "0" => "-O0",
        "1" => "-O1",
        _ => "-O2", // treat levels 2,3,s,z as O2 for metal
    };

    // Platform-specific min version flags (Metal 3.1 requires macOS 14 / iOS 17)
    let platform_flags = if sdk == "macosx" || sdk == "maccatalyst" {
        vec!["-mmacosx-version-min=14.0"]
    } else {
        vec!["-mios-version-min=17.0"]
    };

    // Compile each .metal source to an intermediate .air file
    let mut air_files = Vec::new();
    for metal_file in metal_files {
        let name = metal_file.file_stem().unwrap().to_str().unwrap();
        // Name .air with platform prefix to avoid collisions
        let prefix = if sdk == "macosx" || sdk == "maccatalyst" {
            "macos_"
        } else {
            "ios_"
        };
        let air_path = air_dir.join(format!("{}{}.air", prefix, name));
        air_files.push(air_path.clone());

        // Invoke Metal compiler
        let mut cmd = Command::new("xcrun");
        cmd.args(&["-sdk", sdk, "metal", metal_opt_flag]);
        cmd.arg(format!("-std={}", "metal3.1")); // target Metal 3.1 shading language
        for flag in &platform_flags {
            cmd.arg(flag);
        }
        for inc in &include_dirs {
            cmd.args(&["-I", inc.to_str().unwrap()]);
        }
        if opt == "0" {
            cmd.arg("-gline-tables-only");
            if sdk == "macosx" {
                cmd.args(&["-frecord-sources", "-fpreserve-invariance"]);
            }
        }
        cmd.args(&[
            "-c",
            metal_file.to_str().unwrap(),
            "-o",
            air_path.to_str().unwrap(),
        ]);

        println!(
            "cargo:info=Running Metal compiler: {}",
            cmd.get_args()
                .map(|a| a.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        );
        if !cmd.status().expect("Failed to run metal compiler").success() {
            panic!(
                "Metal shader compilation failed for {}",
                metal_file.display()
            );
        }
    }

    // Link all .air files into a single .metallib library
    let metallib_path =
        out_dir.join(if sdk == "macosx" || sdk == "maccatalyst" {
            "shaders_macos.metallib"
        } else {
            "shaders_ios.metallib"
        });
    let mut lib_cmd = Command::new("xcrun");
    lib_cmd.args(&["-sdk", sdk, "metallib"]);
    for air in &air_files {
        lib_cmd.arg(air);
    }
    lib_cmd.args(&["-o", metallib_path.to_str().unwrap()]);
    println!(
        "cargo:info=Linking Metal libraries: {}",
        lib_cmd
            .get_args()
            .map(|a| a.to_string_lossy())
            .collect::<Vec<_>>()
            .join(" ")
    );
    if !lib_cmd.status().expect("Failed to run metallib linker").success() {
        panic!("Failed to link Metal shader libraries");
    }

    // Clean up intermediate .air files
    for air in air_files {
        let _ = fs::remove_file(air);
    }
    println!(
        "cargo:info=Metal shaders compiled successfully to {}",
        metallib_path.display()
    );
}

fn write_metallib_as_bytes(
    metallib_data: &[u8],
    out_path: &Path,
) {
    let mut f = fs::File::create(out_path).expect("Cannot create metal_lib.rs");
    writeln!(f, "// AUTO-GENERATED Metal library byte array").unwrap();
    writeln!(f, "pub const METAL_LIBRARY_DATA: &[u8] = &[").unwrap();
    for chunk in metallib_data.chunks(16) {
        write!(f, "    ").unwrap();
        for byte in chunk {
            write!(f, "0x{:02x}, ", byte).unwrap();
        }
        writeln!(f).unwrap();
    }
    writeln!(f, "];").unwrap();
}

fn write_empty_metallib() {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let metal_lib_rs = out_dir.join("metal_lib.rs");

    if let Some(parent) = metal_lib_rs.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(metal_lib_rs, b"pub const METAL_LIBRARY_DATA: &[u8] = &[];\n")
        .expect("Cannot create stub metal_lib.rs");
}
