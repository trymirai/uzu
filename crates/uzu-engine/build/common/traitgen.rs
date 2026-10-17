use std::{collections::HashMap, env, path::PathBuf};

use anyhow::{Context, bail};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::common::{
    codegen::write_tokens,
    gpu_type_paths::GpuTypePaths,
    identifiers::KernelPath,
    kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType},
};

pub fn traitgen(
    kernel: &Kernel,
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<(TokenStream, TokenStream)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{kernel_name}Kernel");

    let params = kernel
        .parameters
        .iter()
        .map(|p| {
            let name = format_ident!("{}", p.name.as_ref());
            let ty = match &p.ty {
                KernelParameterType::Type => quote! { crate::data_type::DataType },
                KernelParameterType::Value(ty) => {
                    let ty = gpu_type_paths
                        .resolve_type(ty)
                        .with_context(|| format!("invalid type for parameter `{}`", p.name))?;
                    quote! { #ty }
                },
            };

            Ok(quote! { #name: #ty })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    let mut args = kernel
        .arguments
        .iter()
        .map(|a| {
            let name = format_ident!("{}", a.name.as_ref());

            let mut ty = match &a.ty {
                KernelArgumentType::Buffer(access) => match access {
                    KernelBufferAccess::Read => {
                        quote! { impl crate::backends::common::BufferRef<Backend = Self::Backend> }
                    },
                    KernelBufferAccess::ReadWrite => {
                        quote! { impl crate::backends::common::BufferMut<Backend = Self::Backend> }
                    },
                },
                KernelArgumentType::Constant(ty) => {
                    let ty = gpu_type_paths
                        .resolve_type(ty)
                        .with_context(|| format!("invalid type for argument `{}`", a.name))?;
                    quote! { #ty }
                },
            };

            if a.conditional {
                ty = quote! { Option<#ty> };
            }

            Ok(quote! { #name: #ty })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;

    args.push(quote! { command_buffer: &mut <<Self::Backend as crate::backends::common::Backend>::CommandBuffer as crate::backends::common::CommandBuffer>::Encoding });

    let kernel_trait = quote! {
        #[allow(clippy::style, clippy::complexity, clippy::perf)]
        pub trait #trait_name: Sized + Send + Sync {
            type Backend: crate::backends::common::Backend<Kernels: Kernels<#trait_name = Self>>;

            #[allow(non_snake_case)]
            fn new(context: &<Self::Backend as crate::backends::common::Backend>::Context #(, #params)*) -> Result<Self, <Self::Backend as crate::backends::common::Backend>::Error>;

            fn encode(&self, #(#args),*);
        }
    };

    let kernel_type = quote! { type #trait_name: #trait_name<Backend = Self::Backend>; };

    Ok((kernel_trait, kernel_type))
}

pub fn traitgen_all(
    backends_kernels: Vec<HashMap<KernelPath, Box<[Kernel]>>>,
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);

    let mut kernels: HashMap<KernelPath, Box<[Kernel]>> = HashMap::new();

    let mut mismatches = Vec::new();

    for backend_kernels in backends_kernels {
        for (file_path, file_kernels) in backend_kernels {
            if let Some(cached_kernels) = kernels.get(&file_path) {
                if cached_kernels != &file_kernels {
                    mismatches.push(format!(
                        "kernel metadata mismatch at {}:\nfirst:  {cached_kernels:?}\nsecond: {file_kernels:?}",
                        file_path.join("::")
                    ));
                }
            } else {
                kernels.insert(file_path, file_kernels);
            }
        }
    }

    if !mismatches.is_empty() {
        bail!("{}", mismatches.join("\n\n"));
    }

    let mut kernel_traits = Vec::new();
    let mut kernel_types = Vec::new();

    for (file_path, file_kernels) in kernels.into_iter().sorted_by_key(|(p, _k)| p.join("::")) {
        for kernel in &file_kernels {
            let (tr, ty) = traitgen(kernel, gpu_type_paths).with_context(|| {
                format!("cannot generate shared trait for kernel `{}` in `{}`", kernel.name, file_path.join("::"))
            })?;
            kernel_traits.push(tr);
            kernel_types.push(ty);
        }
    }

    let kernel_traits = quote! {
        #(#kernel_traits)*

        macro_rules! autogen_kernels {
            () => {
                #(#kernel_types)*
            }
        }
    };

    write_tokens(kernel_traits, out_dir.join("traits.rs")).context("cannot write kernel traits")?;

    Ok(())
}
