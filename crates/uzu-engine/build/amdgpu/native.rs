//! AMD-only kernels written for RDNA directly (`src/backends/amdgpu/kernel/native/*.clcpp`), next to the MSL
//! kernels: shapes of work the DSL kernels cannot express on this hardware (WMMA operands built in registers).
//! One code object per source, cached by toolchain and source. A source that does not compile leaves an empty
//! code object with a build warning: the runtime then reports its kernels unavailable and the backend keeps the
//! DSL path. `OUT_DIR/amdgpu_native.rs` names the code objects `<STEM>` in upper case.

use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Context;
use itertools::Itertools;

use super::toolchain::AmdgpuToolchain;

pub fn compile_native(
    toolchain: &AmdgpuToolchain,
    source_directory: &Path,
    output_directory: &Path,
    warn: impl Fn(&str),
) -> anyhow::Result<()> {
    let object_directory = output_directory.join("native");
    fs::create_dir_all(&object_directory).with_context(|| format!("cannot create {}", object_directory.display()))?;
    let sources: Vec<PathBuf> = fs::read_dir(source_directory)
        .with_context(|| format!("cannot read {}", source_directory.display()))?
        .filter_map(|entry| entry.ok().map(|entry| entry.path()))
        .filter(|path| path.extension().and_then(|extension| extension.to_str()) == Some("clcpp"))
        .sorted()
        .collect();

    let mut constants = Vec::new();
    for source in sources {
        let stem = source.file_stem().and_then(|stem| stem.to_str()).context("native source name is not utf-8")?;
        let code_object = object_directory.join(format!("{stem}.hsaco"));
        let cache_file = object_directory.join(format!("{stem}.key"));
        let key = {
            let mut hasher = blake3::Hasher::new();
            hasher.update(toolchain.cache_key());
            hasher.update(&fs::read(&source).with_context(|| format!("cannot read {}", source.display()))?);
            hasher.finalize().to_hex().to_string()
        };
        let cached = code_object.exists() && fs::read_to_string(&cache_file).is_ok_and(|cached| cached == key);
        if !cached {
            let _ = fs::remove_file(&cache_file);
            match toolchain.compile_native(&source, &object_directory.join(format!("{stem}.o")), &code_object) {
                Ok(_) => fs::write(&cache_file, &key).context("cannot write native cache key")?,
                Err(error) => {
                    warn(&format!("amdgpu native {stem}: {}", format!("{error:#}").lines().take(4).join(" | ")));
                    let _ = fs::remove_file(&code_object);
                },
            }
        }
        let bytes = if code_object.exists() {
            format!("include_bytes!({:?})", code_object.display().to_string())
        } else {
            "&[]".into()
        };
        constants.push(format!("pub const {}: &[u8] = {bytes};\n", stem.to_uppercase()));
    }
    fs::write(output_directory.with_file_name("amdgpu_native.rs"), constants.concat())
        .context("cannot write native code object constants")
}
