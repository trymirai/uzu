use std::{collections::HashMap, env, path::PathBuf};

use anyhow::{Context, bail};
use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{Lifetime, Type};

use super::{identifiers::KernelPath, kernel::Kernel};
use crate::common::{
    codegen::write_tokens,
    kernel::{KernelArgumentType, KernelBufferAccess, KernelParameterType},
    utils::get_generic_name_stream,
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

    let (encode_lifetime_generics, type_generics, mut args) = kernel
        .arguments
        .iter()
        .map(|a| {
            let name = format_ident!("{}", a.name.as_ref());

            let (lifetime_generic, type_generic, mut ty) = match &a.ty {
                KernelArgumentType::Buffer(access) => {
                    let buffer_lifetime = Lifetime::new(&format!("'{}", a.name.as_ref()), Span::call_site());
                    let buffer_type = get_generic_name_stream(a.name.as_ref());
                    (
                        Some(quote! { #buffer_lifetime }),
                        Some(quote! { #buffer_type }),
                        match access {
                            KernelBufferAccess::Read => quote! { impl BufferArg<#buffer_lifetime, #buffer_type> },
                            KernelBufferAccess::ReadWrite => {
                                quote! { impl BufferArgMut<#buffer_lifetime, #buffer_type> }
                            },
                        },
                    )
                },
                KernelArgumentType::Constant(ty) => {
                    let ty: Type = syn::parse_str(ty.as_ref()).unwrap();
                    (None, None, quote! { #ty })
                },
            };

            if a.conditional {
                ty = quote! { Option<#ty> };
            }

            (lifetime_generic, type_generic, quote! { #name: #ty })
        })
        .collect::<(Vec<_>, Vec<_>, Vec<_>)>();

    let mut encode_generics = encode_lifetime_generics.into_iter().flatten().collect::<Vec<_>>();
    let mut where_generics: Vec<TokenStream> = Vec::new();

    encode_generics.push(quote! { 'encoder });
    args.push(quote! { encoder: &'encoder mut crate::backends::common::Encoder<Self::Backend> });

    type_generics.iter().flatten().for_each(|generic| {
        encode_generics.push(quote! { #generic });
        where_generics.push(quote! { #generic: Buffer<Backend = Self::Backend> });
    });

    let maybe_where_block = if where_generics.is_empty() {
        quote! {}
    } else {
        quote! { where #(#where_generics),* }
    };

    let kernel_trait = quote! {
        pub trait #trait_name: Sized {
            type Backend: crate::backends::common::Backend<Kernels: Kernels<#trait_name = Self>>;

            fn new(context: &<Self::Backend as crate::backends::common::Backend>::Context #(, #params)*) -> Result<Self, <Self::Backend as crate::backends::common::Backend>::Error>;

            fn encode<#(#encode_generics),*>(&self, #(#args),*)#maybe_where_block;
        }
    };

    let kernel_type = quote! { type #trait_name: #trait_name<Backend = Self::Backend>; };

    (kernel_trait, kernel_type)
}

pub fn traitgen_all(backends_kernels: Vec<HashMap<KernelPath, Box<[Kernel]>>>) -> anyhow::Result<()> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?);

    let mut kernels: HashMap<KernelPath, Box<[Kernel]>> = HashMap::new();

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
        use crate::backends::common::{
            Buffer,
            buffer_range::{AsBufferRangeMut, AsBufferRangeRef},
        };

        pub trait BufferArg<'a, B: Buffer> {
            fn into_parts(self) -> (&'a B, usize, usize);
        }

        impl<'a, B: Buffer, T: AsBufferRangeRef<Buffer = B>> BufferArg<'a, B> for &'a T {
            fn into_parts(self) -> (&'a B, usize, usize) {
                let buffer_range = self.as_buffer_range_ref();
                let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
                (buffer, range.start, range.end - range.start)
            }
        }

        impl<'a, B: Buffer, T: AsBufferRangeRef<Buffer = B>> BufferArg<'a, B> for (&'a T, usize) {
            fn into_parts(self) -> (&'a B, usize, usize) {
                let buffer_range = self.0.as_buffer_range_ref();
                let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
                (buffer, range.start + self.1, range.end - range.start - self.1)
            }
        }

        pub trait BufferArgMut<'a, B: Buffer> {
            fn into_parts(self) -> (&'a B, usize, usize);
        }

        impl<'a, B: Buffer, T: AsBufferRangeMut<Buffer = B>> BufferArgMut<'a, B> for &'a mut T {
            fn into_parts(self) -> (&'a B, usize, usize) {
                let buffer_range = self.as_buffer_range_mut();
                let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
                (buffer, range.start, range.end - range.start)
            }
        }

        impl<'a, B: Buffer, T: AsBufferRangeMut<Buffer = B>> BufferArgMut<'a, B> for (&'a mut T, usize) {
            fn into_parts(self) -> (&'a B, usize, usize) {
                let buffer_range = self.0.as_buffer_range_mut();
                let (buffer, range) = (buffer_range.buffer(), buffer_range.range());
                (buffer, range.start + self.1, range.end - range.start - self.1)
            }
        }

        #(#kernel_traits)*

        pub trait Kernels: Sized {
            type Backend: crate::backends::common::Backend<Kernels = Self>;

            #(#kernel_types)*
        }
    };

    write_tokens(kernel_traits, out_dir.join("traits.rs")).context("cannot write kernel traits")?;

    Ok(())
}
