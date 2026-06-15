use std::path::Path;

use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use crate::common::kernel::Kernel;

pub fn bindgen_global(kernels: &[(impl AsRef<Path>, &[Kernel])]) -> anyhow::Result<TokenStream> {
    let includes = kernels.iter().map(|(path, _kernels)| {
        let path = path.as_ref().to_str().unwrap();

        quote! {
            include!(#path);
        }
    });

    let types = kernels.iter().flat_map(|(_path, kernels)| kernels.iter()).map(|kernel| {
        let kernel_trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
        let kernel_struct_name = format_ident!("{}WebGPUKernel", kernel.name.as_ref());

        quote! {
            type #kernel_trait_name = #kernel_struct_name;
        }
    });

    let tokens = quote! {
        #(#includes)*

        pub struct WebGPUKernels;

        impl crate::backends::common::kernel::Kernels for WebGPUKernels {
            type Backend = crate::backends::webgpu::WebGPU;

            #(#types)*
        }
    };

    Ok(tokens)
}
