use std::{collections::HashMap, env, path::PathBuf};

use anyhow::{Context, bail};
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::Type;

use super::kernel::Kernel;
use crate::common::{
    codegen::write_tokens,
    kernel::{KernelArgumentType, KernelBufferAccess, KernelParameterType},
};

pub fn traitgen(kernel: &Kernel) -> (TokenStream, TokenStream) {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{kernel_name}Kernel");

    let params = kernel.parameters.iter().map(|p| {
        let name = format_ident!("{}", p.name.as_ref());
        let ty = match &p.ty {
            KernelParameterType::Type => quote! { crate::DataType },
            KernelParameterType::Value(ty) => {
                let ty: Type = syn::parse_str(ty.as_ref()).unwrap();
                quote! { #ty }
            },
        };

        quote! { #name: #ty }
    });

    let args = kernel
        .arguments
        .iter()
        .map(|a| {
            let name = format_ident!("{}", a.name.as_ref());

            let mut ty = match &a.ty {
                KernelArgumentType::Buffer(access) => match access {
                    KernelBufferAccess::Read => {
                        quote! { &crate::backends::common::Allocation<Self::Backend> }
                    },
                    KernelBufferAccess::ReadWrite => {
                        quote! { &mut crate::backends::common::Allocation<Self::Backend> }
                    },
                },
                KernelArgumentType::Constant(ty) => {
                    let ty: Type = syn::parse_str(ty.as_ref()).unwrap();
                    quote! { &[#ty] }
                },
                KernelArgumentType::Scalar(ty) => {
                    let ty: Type = syn::parse_str(ty.as_ref()).unwrap();
                    quote! { #ty }
                },
            };

            if a.conditional {
                ty = quote! { Option<#ty> };
            }

            quote! { #name: #ty }
        })
        .collect::<Vec<_>>();

    let kernel_trait = quote! {
        pub trait #trait_name: Sized {
            type Backend: crate::backends::common::Backend<Kernels: Kernels<#trait_name = Self>>;

            fn new(context: &<Self::Backend as crate::backends::common::Backend>::Context #(, #params)*) -> Result<Self, <Self::Backend as crate::backends::common::Backend>::Error>;

            fn encode(&self, #(#args ,)* encoder: &mut crate::backends::common::Encoder<Self::Backend>);
        }
    };

    let kernel_type = quote! { type #trait_name: #trait_name<Backend = Self::Backend>; };

    (kernel_trait, kernel_type)
}

pub fn traitgen_all(backends_kernels: Vec<HashMap<Box<[Box<str>]>, Box<[Kernel]>>>) -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);

    let mut kernels: HashMap<Box<[Box<str>]>, Box<[Kernel]>> = HashMap::new();

    for backend_kernels in backends_kernels {
        for (file_path, file_kernels) in backend_kernels {
            if let Some(cached_kernels) = kernels.get(&file_path) {
                if cached_kernels != &file_kernels {
                    bail!("{cached_kernels:?} != {file_kernels:?}");
                }
            } else {
                kernels.insert(file_path, file_kernels);
            }
        }
    }

    let mut kernel_traits = Vec::new();
    let mut kernel_types = Vec::new();

    for (_file_path, file_kernels) in kernels.into_iter().sorted_by_key(|(p, _k)| p.join("::")) {
        for (tr, ty) in file_kernels.iter().map(traitgen) {
            kernel_traits.push(tr);
            kernel_types.push(ty);
        }
    }

    let kernel_traits = quote! {
        #(#kernel_traits)*

        pub trait Kernels: Sized {
            type Backend: crate::backends::common::Backend<Kernels = Self>;

            #(#kernel_types)*
        }
    };

    write_tokens(kernel_traits, out_dir.join("traits.rs")).context("cannot write kernel traits")?;

    Ok(())
}
