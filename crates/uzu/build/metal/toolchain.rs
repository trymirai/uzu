use std::{
    env::{self},
    ffi::OsString,
    fs,
    path::Path,
    process::Stdio,
};

use anyhow::{Context, bail};
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
            } else if target.contains("ios")
                && (target.contains("86_64") || target_env == "sim")
            {
                Ok(Self::IPhoneSimulator)
            } else {
                Ok(Self::IPhoneOS)
            }
        } else if target_os == "macos" {
            Ok(Self::MacOSX)
        } else {
            bail!(
                "cannot find matching metal sdk for ({target:?}, {target_os:?}, {target_env:?})"
            );
        }
    }

    pub fn from_env() -> anyhow::Result<Self> {
        let target = env::var("TARGET").context("missing TARGET")?;
        let target_os = env::var("CARGO_CFG_TARGET_OS")
            .context("missing CARGO_CFG_TARGET_OS")?;
        let target_env = env::var("CARGO_CFG_TARGET_ENV")
            .context("missing CARGO_CFG_TARGET_ENV")?;

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
            MetalStd::Metal4_0 => "metal4.0",
        }
    }

    pub fn min_os(&self) -> &'static str {
        match self {
            MetalStd::Metal4_0 => "26.0",
        }
    }
}

impl Default for MetalStd {
    fn default() -> Self {
        Self::Metal4_0
    }
}

#[derive(Debug)]
pub struct MetalToolchain {
    sdk: MetalSdk,
    std: MetalStd,
    opt_flags: Box<[OsString]>,
    extra_options: Box<[OsString]>,
}

impl MetalToolchain {
    pub fn from_env() -> anyhow::Result<Self> {
        let sdk = MetalSdk::from_env().context("cannot get sdk")?;
        let std = MetalStd::default();

        let opt_flags = match env::var("OPT_LEVEL")
            .context("missing OPT_LEVEL")?
            .as_str()
        {
            "0" => {
                [
                    OsString::from("-O1"), // matmul kernels compiled with -O0 are broken and require a reboot to unfreeze the os
                    OsString::from("-gline-tables-only"), // debug with line tables only
                    OsString::from("-frecord-sources"),   // include source code
                ]
                .into()
            },
            "1" => [OsString::from("-O1")].into(),
            _ => [OsString::from("-O2")].into(), // treat levels 2,3,s,z as O2 for metal
        };

        let extra_options = Box::new([
            OsString::from(format!(
                "-m{}-version-min={}",
                sdk.os(),
                std.min_os()
            )),
            // NOTE: Previous build.rs script didn't forward metal compiler warnings to cargo warnings, the new one does. This is temporary to avoid warning spam without modifying unrelated things in the same pull request as introducing the dsl and converting kernels
            OsString::from("-Wno-sign-compare"),
            OsString::from("-Wno-macro-redefined"),
            OsString::from("-Wno-unused-variable"),
        ]);

        Ok(Self {
            sdk,
            std,
            opt_flags,
            extra_options,
        })
    }

    fn xcrun(&self) -> Command {
        let mut cmd = Command::new("xcrun");
        cmd.args(["-sdk", self.sdk.to_str()]);
        cmd
    }

    pub async fn analyze(
        &self,
        path: impl AsRef<Path>,
    ) -> anyhow::Result<(
        impl Iterator<Item = MetalKernelInfo>,
        impl Iterator<Item = Box<str>>,
    )> {
        let path = path.as_ref();

        let depfile_path =
            NamedTempFile::new().context("cannot create temporary file")?;

        let analyze_output = self
            .xcrun()
            .arg("metal")
            .args(["-x", "metal"])
            .arg(format!("-std={}", self.std.to_str()))
            .args(self.extra_options.as_ref())
            .arg(path)
            .arg("-fsyntax-only")
            .args(["-MMD", "-MF"])
            .arg(depfile_path.path())
            .args(["-Xclang", "-ast-dump=json"])
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .await
            .context("cannot execute metal analyzer")?;

        if !analyze_output.status.success() {
            bail!(
                "metal analyzer failed: {}",
                String::from_utf8_lossy(&analyze_output.stderr)
            );
        }

        let ast_root =
            serde_json::from_slice::<MetalAstNode>(&analyze_output.stdout)
                .context("cannot deserialize ast dump")?;

        if !matches!(&ast_root.kind, MetalAstKind::TranslationUnitDecl) {
            bail!(
                "unexpected kind of ast root: MetalAstKind::TranslationUnitDecl expected, but {:?} found",
                ast_root.kind
            );
        }

        let source_contents =
            fs::read_to_string(path).context("cannot read source file")?;

        let kernel_infos = ast_root
            .inner
            .into_iter()
            .filter_map(|node| {
                MetalKernelInfo::from_ast_node_and_source(
                    node,
                    &source_contents,
                )
                .transpose()
            })
            .collect::<anyhow::Result<Vec<_>>>()
            .context("cannot parse kernel infos from AST")?
            .into_iter();

        let depfile_contents = fs::read_to_string(depfile_path.path())
            .context("cannot read depfile")?;

        let dependencies = depfile::parse(&depfile_contents)
            .map_err(|e| anyhow::anyhow!("cannot parse depfile: {e}"))?
            .iter()
            .flat_map(|(_, d)| d)
            .map(|f| f.as_ref().into())
            .collect::<Vec<_>>()
            .into_iter();

        Ok((kernel_infos, dependencies))
    }

    pub async fn compile(
        &self,
        source: impl AsRef<Path>,
        footer: impl AsRef<str>,
        output: impl AsRef<Path>,
    ) -> anyhow::Result<Option<Box<str>>> {
        let mut compile_child = self
            .xcrun()
            .arg("metal")
            .arg("-c")
            .args(["-x", "metal"])
            .arg(format!("-std={}", self.std.to_str()))
            .args(self.extra_options.as_ref())
            .args(self.opt_flags.as_ref())
            .arg("-include")
            .arg(source.as_ref())
            .arg("-")
            .arg("-o")
            .arg(output.as_ref())
            .stdin(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("cannot execute metal compiler")?;

        compile_child
            .stdin
            .as_mut()
            .context("metal compiler stdin missing")?
            .write_all(footer.as_ref().as_bytes())
            .await
            .context("cannot write to metal compiler stdin")?;

        let compile_output = compile_child
            .wait_with_output()
            .await
            .context("cannot wait on metal compiler")?;

        let stderr = String::from_utf8_lossy(&compile_output.stderr)
            .into_owned()
            .into_boxed_str();

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

    pub async fn link(
        &self,
        objects: impl IntoIterator<Item = impl AsRef<Path>>,
        output: impl AsRef<Path>,
    ) -> anyhow::Result<Option<Box<str>>> {
        let link_output = self
            .xcrun()
            .arg("metallib")
            .args(
                objects.into_iter().map(|p| p.as_ref().as_os_str().to_owned()),
            )
            .arg("-o")
            .arg(output.as_ref())
            .output()
            .await
            .context("cannot execute metal linker")?;

        let stderr = String::from_utf8_lossy(&link_output.stderr)
            .into_owned()
            .into_boxed_str();

        if !link_output.status.success() {
            bail!("metal linker failed: {stderr}");
        }

        let warnings = if !stderr.is_empty() {
            Some(stderr)
        } else {
            None
        };

        Ok(warnings)
    }
}
