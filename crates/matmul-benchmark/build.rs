use std::{
    env,
    fs::{self, create_dir_all},
    path::{Path, PathBuf},
    process::Command,
};

use cmake::Config as CMakeConfig;
use walkdir::WalkDir;

const DEFAULT_MLX_GIT_URL: &str = "https://github.com/ml-explore/mlx.git";
const DEFAULT_MLX_GIT_REF: &str = "v0.30.1";

fn abs_path<P: AsRef<Path>>(p: P) -> PathBuf {
    if p.as_ref().is_absolute() {
        p.as_ref().to_path_buf()
    } else {
        env::current_dir().expect("current_dir failed").join(p)
    }
}

fn looks_like_mlx_repo_root(dir: &Path) -> bool {
    dir.join("CMakeLists.txt").exists()
        && dir.join("mlx").exists()
        && dir.join("mlx/mlx.h").exists()
}

fn is_truthy_env(name: &str) -> bool {
    let Ok(v) = env::var(name) else {
        return false;
    };
    matches!(
        v.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

fn cargo_offline() -> bool {
    is_truthy_env("CARGO_NET_OFFLINE") || is_truthy_env("MLX_RS_OFFLINE")
}

fn cache_dir(out_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("MLX_RS_CACHE_DIR") {
        return abs_path(p);
    }

    if let Ok(p) = env::var("CARGO_HOME") {
        return abs_path(p).join("mlx-rs-cache");
    }

    if let Ok(p) = env::var("HOME") {
        return PathBuf::from(p).join(".cache/mlx-rs");
    }
    if let Ok(p) = env::var("LOCALAPPDATA") {
        return PathBuf::from(p).join("mlx-rs");
    }

    out_dir.join("mlx-rs-cache")
}

fn run_checked(
    mut cmd: Command,
    what: &str,
) {
    let output = cmd.output().unwrap_or_else(|e| {
        panic!("Failed to run {}: {}", what, e);
    });
    if !output.status.success() {
        let stdout = String::from_utf8_lossy(&output.stdout);
        let stderr = String::from_utf8_lossy(&output.stderr);
        panic!(
            "{} failed (exit={:?})\n--- stdout ---\n{}\n--- stderr ---\n{}\n",
            what,
            output.status.code(),
            stdout,
            stderr
        );
    }
}

fn pins_toml_path(manifest_dir: &Path) -> PathBuf {
    if let Ok(p) = env::var("MLX_RS_PINS_TOML") {
        return abs_path(p);
    }
    manifest_dir.join("mlx-pins.toml")
}

#[derive(Debug, Clone)]
struct Pins {
    repo_url: Option<String>,
    repo_ref: Option<String>,
}

fn parse_pins(pins_path: &Path) -> Pins {
    let contents = fs::read_to_string(pins_path).unwrap_or_else(|e| {
        panic!(
            "Failed to read pins file at {}: {}. \
             Update the pins file or set MLX_RS_PINS_TOML to a valid path.",
            pins_path.display(),
            e
        )
    });

    #[derive(Debug)]
    enum Section {
        None,
        Repo,
    }

    let mut section = Section::None;
    let mut pins = Pins {
        repo_url: None,
        repo_ref: None,
    };

    for raw in contents.lines() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            let header = &line[1..line.len() - 1];
            if header == "repo" {
                section = Section::Repo;
            } else {
                section = Section::None;
            }
            continue;
        }

        let Some((k, v)) = line.split_once('=') else {
            continue;
        };
        let key = k.trim();
        let mut val = v.trim();
        if val.starts_with('"') && val.ends_with('"') && val.len() >= 2 {
            val = &val[1..val.len() - 1];
        }

        match &mut section {
            Section::Repo => match key {
                "url" => pins.repo_url = Some(val.to_string()),
                "ref" | "rev" => pins.repo_ref = Some(val.to_string()),
                _ => {},
            },
            Section::None => {},
        }
    }

    pins
}

fn pinned_mlx_git(pins: &Pins) -> (String, String) {
    let url = env::var("MLX_GIT_URL")
        .ok()
        .or_else(|| pins.repo_url.clone())
        .unwrap_or_else(|| DEFAULT_MLX_GIT_URL.to_string());
    let rev = env::var("MLX_GIT_REF")
        .ok()
        .or_else(|| pins.repo_ref.clone())
        .unwrap_or_else(|| DEFAULT_MLX_GIT_REF.to_string());
    (url, rev)
}

fn maybe_clear_cmake_build_dir(
    build_dir: &Path,
    source_dir: &Path,
) {
    let cache = build_dir.join("CMakeCache.txt");
    let Ok(contents) = fs::read_to_string(&cache) else {
        return;
    };
    let src =
        source_dir.canonicalize().unwrap_or_else(|_| source_dir.to_path_buf());
    for line in contents.lines() {
        if !line.starts_with("CMAKE_HOME_DIRECTORY") {
            continue;
        }
        let needs_cleanup = match line.split('=').next_back() {
            Some(cmake_home) => fs::canonicalize(cmake_home)
                .ok()
                .map(|cmake_home| cmake_home != src)
                .unwrap_or(true),
            None => true,
        };
        if needs_cleanup {
            let _ = fs::remove_dir_all(build_dir);
        }
        break;
    }
}

fn ensure_git_checkout_cached(
    name: &str,
    url: &str,
    rev: &str,
    cache_dir: &Path,
) -> PathBuf {
    let checkout_dir = cache_dir.join(format!("{}-{}", name, rev));
    let marker = checkout_dir.join(".mlx_rs_fetched");
    if marker.exists() {
        return checkout_dir;
    }

    if checkout_dir.exists() {
        let _ = fs::remove_dir_all(&checkout_dir);
    }
    create_dir_all(cache_dir).expect("Failed to create cache dir");

    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("clone").arg(url).arg(&checkout_dir);
            c
        },
        &format!("git clone {} into cache", name),
    );
    run_checked(
        {
            let mut c = Command::new("git");
            c.arg("-C").arg(&checkout_dir).arg("checkout").arg(rev);
            c
        },
        &format!("git checkout {}@{}", name, rev),
    );

    let _ = fs::write(&marker, rev);
    checkout_dir
}

fn ensure_mlx_repo(
    out_dir: &Path,
    manifest_dir: &Path,
    repo_url: &str,
    repo_ref: &str,
) -> PathBuf {
    // First, check for explicit MLX_SRC_DIR environment variable
    if let Ok(p) = env::var("MLX_SRC_DIR") {
        let p = abs_path(p);
        if !looks_like_mlx_repo_root(&p) {
            panic!(
                "MLX_SRC_DIR={} does not look like an MLX repo root \
                 (expected CMakeLists.txt + mlx/ + mlx/mlx.h)",
                p.display()
            );
        }
        return p;
    }

    // Second, check for local external/mlx in the workspace
    let workspace_root = manifest_dir.parent().and_then(|p| p.parent());
    if let Some(root) = workspace_root {
        let local_mlx = root.join("external/mlx");
        if looks_like_mlx_repo_root(&local_mlx) {
            println!(
                "cargo:warning=mlx-rs: using local MLX at {}",
                local_mlx.display()
            );
            return local_mlx;
        }
    }

    if cargo_offline() {
        panic!(
            "MLX sources not found locally and Cargo is offline. \
             Set MLX_SRC_DIR to a checked-out MLX repo or build with network access."
        );
    }

    let cache = cache_dir(out_dir);
    println!(
        "cargo:warning=mlx-rs: fetching MLX {}@{} into {}",
        repo_url,
        repo_ref,
        cache.display()
    );
    ensure_git_checkout_cached("mlx", repo_url, repo_ref, &cache)
}

fn find_mlx_lib_dir(root: &Path) -> Option<PathBuf> {
    let static_candidates = ["libmlx.a", "mlx.lib"];

    for entry in
        WalkDir::new(root).max_depth(6).into_iter().filter_map(Result::ok)
    {
        if !entry.file_type().is_file() {
            continue;
        }

        let name = entry.file_name().to_string_lossy();
        if static_candidates.iter().any(|c| name == *c) {
            return entry.path().parent().map(|p| p.to_path_buf());
        }
    }

    None
}

#[derive(Debug, Clone)]
struct BuildContext {
    #[allow(dead_code)]
    manifest_dir: PathBuf,
    mlx_src_dir: PathBuf,
    out_dir: PathBuf,
    src_include_dir: PathBuf,
    target: String,
}

fn collect_build_context() -> BuildContext {
    println!("cargo:rerun-if-env-changed=MLX_SRC_DIR");
    println!("cargo:rerun-if-env-changed=MLX_RS_PINS_TOML");
    println!("cargo:rerun-if-env-changed=MLX_GIT_URL");
    println!("cargo:rerun-if-env-changed=MLX_GIT_REF");
    println!("cargo:rerun-if-env-changed=MLX_RS_CACHE_DIR");
    println!("cargo:rerun-if-env-changed=MLX_RS_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_NET_OFFLINE");
    println!("cargo:rerun-if-env-changed=CARGO_HOME");

    let manifest_dir = abs_path(
        env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set"),
    );
    let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));

    let pins_path = pins_toml_path(&manifest_dir);
    println!("cargo:rerun-if-changed={}", pins_path.display());
    let pins = parse_pins(&pins_path);

    let (repo_url, repo_ref) = pinned_mlx_git(&pins);
    let mlx_src_dir =
        ensure_mlx_repo(&out_dir, &manifest_dir, &repo_url, &repo_ref);

    println!("cargo:rerun-if-changed={}", mlx_src_dir.join("mlx").display());
    println!(
        "cargo:rerun-if-changed={}",
        mlx_src_dir.join("CMakeLists.txt").display()
    );

    let src_include_dir = manifest_dir.join("src");
    let target = env::var("TARGET").unwrap_or_default();

    BuildContext {
        manifest_dir,
        mlx_src_dir,
        out_dir,
        src_include_dir,
        target,
    }
}

fn build_mlx_cmake(ctx: &BuildContext) -> PathBuf {
    let cmake_build_dir = ctx.out_dir.join("build");
    maybe_clear_cmake_build_dir(&cmake_build_dir, &ctx.mlx_src_dir);
    create_dir_all(&cmake_build_dir).ok();

    let mut cmake_config = CMakeConfig::new(&ctx.mlx_src_dir);
    cmake_config.out_dir(&ctx.out_dir);

    // Disable features we don't need
    cmake_config.define("MLX_BUILD_TESTS", "OFF");
    cmake_config.define("MLX_BUILD_EXAMPLES", "OFF");
    cmake_config.define("MLX_BUILD_BENCHMARKS", "OFF");
    cmake_config.define("MLX_BUILD_PYTHON_BINDINGS", "OFF");

    // Enable Metal backend only (CPU backend has compatibility issues with some Xcode versions)
    cmake_config.define("MLX_BUILD_METAL", "ON");
    cmake_config.define("MLX_BUILD_CPU", "OFF");

    // Build static library
    cmake_config.define("BUILD_SHARED_LIBS", "OFF");

    // C++ settings
    cmake_config.define("CMAKE_CXX_STANDARD", "17");
    cmake_config.define("CMAKE_CXX_STANDARD_REQUIRED", "ON");
    cmake_config.define("CMAKE_CXX_EXTENSIONS", "OFF");

    // Disable LTO to avoid linking issues
    cmake_config.define("CMAKE_INTERPROCEDURAL_OPTIMIZATION", "OFF");

    let is_msvc = ctx.target.contains("msvc");
    if !is_msvc {
        cmake_config.cflag("-fno-lto");
        cmake_config.cxxflag("-fno-lto");
    }

    let build_profile =
        match env::var("PROFILE").unwrap_or_else(|_| "release".into()).as_str()
        {
            "debug" => "Debug",
            "release" => "Release",
            _ => "RelWithDebInfo",
        };
    cmake_config.profile(build_profile);

    // macOS specific settings
    if ctx.target.contains("apple-darwin") {
        let arch = if ctx.target.contains("aarch64") {
            "arm64"
        } else {
            "x86_64"
        };
        cmake_config.define("CMAKE_OSX_ARCHITECTURES", arch);
        // Force deployment target to avoid cmake crate confusing SDK version with deployment target
        let dep_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "14.0".into());
        cmake_config.define("CMAKE_OSX_DEPLOYMENT_TARGET", &dep_target);
        // Also set the env var so cmake-rs doesn't override with bad values
        unsafe { env::set_var("MACOSX_DEPLOYMENT_TARGET", &dep_target) };
    }

    cmake_config.build_target("mlx").build()
}

fn link_mlx_static(
    ctx: &BuildContext,
    destination_path: &Path,
) {
    let cmake_build_dir = ctx.out_dir.join("build");
    let lib_search_dir = find_mlx_lib_dir(&cmake_build_dir)
        .or_else(|| find_mlx_lib_dir(destination_path))
        .unwrap_or_else(|| destination_path.join("lib"));
    println!("cargo:rustc-link-search=native={}", lib_search_dir.display());
    println!("cargo:rustc-link-lib=static=mlx");

    // Link Apple frameworks required by MLX
    println!("cargo:rustc-link-lib=framework=Metal");
    println!("cargo:rustc-link-lib=framework=Foundation");
    println!("cargo:rustc-link-lib=framework=QuartzCore");
    println!("cargo:rustc-link-lib=framework=Accelerate");

    // Link C++ standard library
    println!("cargo:rustc-link-lib=c++");

    // Set minimum macOS version for linking to match what MLX was built with
    if ctx.target.contains("apple") {
        let dep_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "14.0".into());
        println!("cargo:rustc-link-arg=-mmacosx-version-min={}", dep_target);
        // Link clang runtime for __isPlatformVersionAtLeast and other builtins
        // Find clang resource directory for runtime libraries
        if let Ok(output) =
            Command::new("clang").args(["--print-resource-dir"]).output()
        {
            if output.status.success() {
                let resource_dir =
                    String::from_utf8_lossy(&output.stdout).trim().to_string();
                let lib_dir = format!("{}/lib/darwin", resource_dir);
                println!("cargo:rustc-link-search=native={}", lib_dir);
                println!("cargo:rustc-link-lib=static=clang_rt.osx");
            }
        }
    }
}

fn build_autocxx_bridge(
    ctx: &BuildContext,
    metal_cpp_dir: &Path,
) {
    println!("cargo:rerun-if-changed=src/lib.rs");
    println!("cargo:rerun-if-changed=src/cxx_utils.hpp");
    println!("cargo:rerun-if-changed=src/cxx_utils/matmul.hpp");

    let extra_clang_args = vec!["-std=c++17".to_string()];
    let extra_clang_args_refs: Vec<&str> =
        extra_clang_args.iter().map(|s| s.as_str()).collect();

    let mut autocxx_builder = autocxx_build::Builder::new(
        "src/lib.rs",
        &[&ctx.src_include_dir, &ctx.mlx_src_dir, metal_cpp_dir],
    )
    .extra_clang_args(&extra_clang_args_refs)
    .build()
    .expect("autocxx build failed");

    autocxx_builder
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-copy")
        .include(&ctx.src_include_dir)
        .include(&ctx.mlx_src_dir)
        .include(metal_cpp_dir);

    autocxx_builder.compile("mlx_rs_bridge");
}

fn find_metal_cpp_dir(out_dir: &Path) -> PathBuf {
    // metal-cpp is fetched by MLX's CMake into the build directory
    for entry in
        WalkDir::new(out_dir).max_depth(6).into_iter().filter_map(Result::ok)
    {
        let path = entry.path();
        if path.ends_with("Metal/Metal.hpp") {
            if let Some(parent) = path.parent().and_then(|p| p.parent()) {
                return parent.to_path_buf();
            }
        }
    }
    // Fallback
    out_dir.join("build/_deps/metal_cpp-src")
}

fn main() {
    // Set MACOSX_DEPLOYMENT_TARGET early to ensure all compilation uses consistent version
    // MLX requires macOS 14.0+
    let target = env::var("TARGET").unwrap_or_default();
    if target.contains("apple") {
        let dep_target = env::var("MACOSX_DEPLOYMENT_TARGET")
            .unwrap_or_else(|_| "14.0".into());
        // Ensure minimum 14.0 for MLX compatibility
        let version: f32 = dep_target.parse().unwrap_or(14.0);
        let final_target = if version < 14.0 {
            "14.0"
        } else {
            &dep_target
        };
        unsafe { env::set_var("MACOSX_DEPLOYMENT_TARGET", final_target) };
    }

    let ctx = collect_build_context();
    let destination_path = build_mlx_cmake(&ctx);
    link_mlx_static(&ctx, &destination_path);

    let metal_cpp_dir = find_metal_cpp_dir(&ctx.out_dir);
    build_autocxx_bridge(&ctx, &metal_cpp_dir);
}
