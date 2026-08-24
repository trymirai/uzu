use std::{
    env::{self},
    ffi::OsString,
    fs,
    path::{Path, PathBuf},
    process::Stdio,
};

use anyhow::{Context, bail};
use serde::Deserialize;
use tempfile::NamedTempFile;
use tokio::{io::AsyncWriteExt, process::Command};

use super::ast::{MetalAstKind, MetalAstNode, MetalKernelInfo};

#[derive(Debug)]
pub enum MetalSdk {
    MacOSX,
    MacCatalyst,
    IPhoneOS,
    IPhoneSimulator,
}

impl MetalSdk {
    pub fn from_parts(
        target: &str,
        target_os: &str,
        target_env: &str,
    ) -> anyhow::Result<Self> {
        if target_os == "ios" {
            if target.contains("macabi") || target_env == "macabi" {
                Ok(Self::MacCatalyst)
            } else if target.contains("ios") && (target.contains("86_64") || target_env == "sim") {
                Ok(Self::IPhoneSimulator)
            } else {
                Ok(Self::IPhoneOS)
            }
        } else if target_os == "macos" {
            Ok(Self::MacOSX)
        } else {
            bail!("cannot find matching metal sdk for ({target:?}, {target_os:?}, {target_env:?})");
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let target = env::var("TARGET").context("missing TARGET")?;
        let target_os = env::var("CARGO_CFG_TARGET_OS").context("missing CARGO_CFG_TARGET_OS")?;
        let target_env = env::var("CARGO_CFG_TARGET_ENV").context("missing CARGO_CFG_TARGET_ENV")?;

        let sdk = Self::from_parts(&target, &target_os, &target_env)?;

        Ok(sdk)
    }

    pub fn to_str(&self) -> &'static str {
        match self {
            Self::MacOSX => "macosx",
            Self::MacCatalyst => "maccatalyst",
            Self::IPhoneOS => "iphoneos",
            Self::IPhoneSimulator => "iphonesimulator",
        }
    }

    pub fn os(&self) -> &'static str {
        match self {
            Self::MacOSX | Self::MacCatalyst => "macosx",
            Self::IPhoneOS | Self::IPhoneSimulator => "ios",
        }
    }
}

#[derive(Debug)]
pub enum MetalStd {
    Metal4_0,
}

impl MetalStd {
    pub fn to_str(&self) -> &'static str {
        match self {
            Self::Metal4_0 => "metal4.0",
        }
    }

    pub fn min_os(&self) -> &'static str {
        match self {
            Self::Metal4_0 => "26.4",
        }
    }
}

#[derive(Debug)]
pub struct MetalToolchain {
    sdk: MetalSdk,
    std: MetalStd,
    opt_flags: Box<[OsString]>,
    extra_options: Box<[OsString]>,
    include_dirs: Box<[PathBuf]>,
    cache_key: [u8; blake3::OUT_LEN],
}

impl MetalToolchain {
    pub async fn from_env_with_include_dir(include_dir: Option<PathBuf>) -> anyhow::Result<Self> {
        let sdk = MetalSdk::from_env().context("cannot get sdk")?;
        let std = MetalStd::Metal4_0;

        let opt_level_flags = match env::var("OPT_LEVEL").context("missing OPT_LEVEL")?.as_str() {
            "0" => vec![OsString::from("-O1")], // matmul kernels compiled with -O0 are broken and require a reboot to unfreeze the os
            _ => vec![OsString::from("-O2")],   // treat levels everything else (1,2,3,s,z) as O2 for metal
        };

        let debug_flags = match env::var("DEBUG").context("missing DEBUG")?.as_str() {
            "false" => vec![],
            "true" => vec![
                OsString::from("-gline-tables-only"), // debug with line tables only
                OsString::from("-frecord-sources"),   // include source code
            ],
            debug => bail!("Unknown DEBUG value {debug}"),
        };

        let opt_flags = [opt_level_flags, debug_flags].concat().into_boxed_slice();

        let extra_options: Box<[OsString]> =
            Box::new([OsString::from(format!("-m{}-version-min={}", sdk.os(), std.min_os()))]);

        let include_dirs = include_dir.into_iter().collect();

        let cache_key = {
            let mut hasher = blake3::Hasher::new();

            hasher.update(sdk.to_str().as_bytes());
            hasher.update(b"\0");
            hasher.update(std.to_str().as_bytes());
            hasher.update(b"\0");
            for flag in opt_flags.iter().chain(extra_options.iter()) {
                hasher.update(flag.as_encoded_bytes());
                hasher.update(b"\0");
            }

            for args in [&["metal", "--version"][..], &["--show-sdk-version"][..], &["--show-sdk-build-version"][..]] {
                let output = Command::new("xcrun")
                    .args(["-sdk", sdk.to_str()])
                    .args(args)
                    .output()
                    .await
                    .with_context(|| format!("cannot execute xcrun {}", args.join(" ")))?;
                if !output.status.success() {
                    bail!("xcrun {} failed: {}", args.join(" "), String::from_utf8_lossy(&output.stderr));
                }
                hasher.update(&output.stdout);
                hasher.update(b"\0");
                hasher.update(&output.stderr);
                hasher.update(b"\0");
            }

            hasher.finalize().into()
        };

        Ok(Self {
            sdk,
            std,
            opt_flags,
            extra_options,
            include_dirs,
            cache_key,
        })
    }

    fn xcrun(&self) -> Command {
        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", self.sdk.to_str()]);
        cmd
    }

    pub fn cache_key(&self) -> &[u8; blake3::OUT_LEN] {
        &self.cache_key
    }

    fn add_include_dirs(
        &self,
        cmd: &mut Command,
    ) {
        for dir in self.include_dirs.iter() {
            cmd.arg("-I").arg(dir);
        }
    }

    pub async fn analyze(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<(impl Iterator<Item = MetalKernelInfo>, impl Iterator<Item = Box<str>>)> {
        let path = path.as_ref();

        let depfile_path = NamedTempFile::new().context("cannot create temporary file")?;

        let mut cmd = self.xcrun();
        cmd.arg("metal")
            .args(["-x", "metal"])
            .arg(format!("-std={}", self.std.to_str()))
            .args(self.extra_options.as_ref());

        self.add_include_dirs(&mut cmd);

        cmd.arg("-DDSL_ANALYZE")
            .arg(path)
            .arg("-fsyntax-only")
            .args(["-MMD", "-MF"])
            .arg(depfile_path.path())
            .args(["-Xclang", "-ast-dump=json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());

        let analyze_output = cmd.output().await.context("cannot execute metal analyzer")?;

        if !analyze_output.status.success() {
            bail!("metal analyzer failed: {}", String::from_utf8_lossy(&analyze_output.stderr));
        }

        let source_contents = fs::read_to_string(path).context("cannot read source file")?;

        let kernel_infos = tokio::task::spawn_blocking(move || {
            let mut deserializer = serde_json::Deserializer::from_slice(&analyze_output.stdout);
            deserializer.disable_recursion_limit();
            let ast_root = MetalAstNode::deserialize(&mut deserializer).context("cannot deserialize ast dump")?;

            if !matches!(&ast_root.kind, MetalAstKind::TranslationUnitDecl) {
                bail!(
                    "unexpected kind of ast root: MetalAstKind::TranslationUnitDecl expected, but {:?} found",
                    ast_root.kind
                );
            }

            ast_root
                .inner
                .into_iter()
                .filter_map(|node| MetalKernelInfo::from_ast_node_and_source(node, &source_contents).transpose())
                .collect::<anyhow::Result<Vec<_>>>()
                .context("cannot parse kernel infos from AST")
        })
        .await??;

        let depfile_contents = fs::read_to_string(depfile_path.path()).context("cannot read depfile")?;

        let dependencies = depfile::parse(&depfile_contents)
            .map_err(|e| anyhow::anyhow!("cannot parse depfile: {e}"))?
            .iter()
            .flat_map(|(_, d)| d)
            .map(|f| f.as_ref().into())
            .collect::<Vec<_>>()
            .into_iter();

        Ok((kernel_infos.into_iter(), dependencies))
    }

    pub async fn compile(
        &self,
        source: impl AsRef<Path>,
        footer: impl AsRef<str>,
        output: impl AsRef<Path>,
    ) -> anyhow::Result<Option<Box<str>>> {
        let mut cmd = self.xcrun();
        cmd.arg("metal")
            .args(["-x", "metal"])
            .arg(format!("-std={}", self.std.to_str()))
            .args(self.extra_options.as_ref())
            .args(self.opt_flags.as_ref());

        self.add_include_dirs(&mut cmd);

        cmd.arg("-include")
            .arg(source.as_ref())
            .arg("-")
            .arg("-o")
            .arg(output.as_ref())
            .stdin(Stdio::piped())
            .stderr(Stdio::piped());

        let mut compile_child = cmd.spawn().context("cannot execute metal compiler")?;

        compile_child
            .stdin
            .as_mut()
            .context("metal compiler stdin missing")?
            .write_all(footer.as_ref().as_bytes())
            .await
            .context("cannot write to metal compiler stdin")?;

        let compile_output = compile_child.wait_with_output().await.context("cannot wait on metal compiler")?;

        let stderr = String::from_utf8_lossy(&compile_output.stderr).into_owned().into_boxed_str();

        if !compile_output.status.success() {
            bail!("metal compiler failed: {stderr}");
        }

        let warnings = if !stderr.is_empty() {
            Some(stderr)
        } else {
            None
        };

        Ok(warnings)
    }
}
