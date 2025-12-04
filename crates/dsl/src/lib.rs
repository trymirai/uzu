use proc_macro2::{Literal, TokenStream};
use quote::{format_ident, quote};

use crate::compiler::CompiledArgType;

mod compiler;

#[proc_macro]
pub fn dsl(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let base_dir = std::env::current_dir().expect("couldn't get current dir");
    let caller_dir = input
        .clone()
        .into_iter()
        .next()
        .expect("no tokens in TokenStream")
        .span()
        .local_file()
        .expect("couldn't get caller file path")
        .parent()
        .expect("couldn't get caller's directory")
        .to_path_buf();
    let shader_file = syn::parse_macro_input!(input as syn::LitStr).value();
    let shader_path = base_dir.join(caller_dir).join(shader_file);

    let compiled = compiler::compile(&shader_path);

    let dependency_guards = compiled.dependencies.iter().enumerate().map(
        |(dep_index, dep_path)| {
            let dep_name = format_ident!("_REBUILD_GUARD_{dep_index}");
            let dep_path = &**dep_path;
            quote! { const #dep_name: &[u8] = include_bytes!(#dep_path); }
        },
    );

    let mtlb = Literal::byte_string(&compiled.mtlb);

    let kernel_name = compiled.name.as_ref();
    let struct_name = format_ident!("{kernel_name}Kernel");

    let (encode_args_defs, encode_args_sets): (Vec<_>, Vec<_>) = compiled
        .args
        .iter()
        .enumerate()
        .map(|(arg_index, (arg_name, arg_type))| {
            let arg_index = arg_index as u64;
            let arg_name = format_ident!("{}", arg_name.as_ref());

            match arg_type {
                CompiledArgType::Buffer => {
                    let def = quote! { #arg_name: &metal::Buffer };
                    let set = quote! { compute_encoder.set_buffer(#arg_index, Some(#arg_name), 0); };

                    (def, set)
                },
                CompiledArgType::Constant(arg_dtype) => {
                    let arg_dtype = format_ident!("{arg_dtype}");
                    let def = quote! { #arg_name: #arg_dtype };
                    let set = quote! { compute_encoder.set_bytes(#arg_index, size_of::<#arg_dtype>() as u64, std::ptr::addr_of!(#arg_name).cast::<std::ffi::c_void>()); };

                    (def, set)
                },
            }
        })
        .unzip();

    let global_size: Vec<TokenStream> = compiled
        .global_size
        .iter()
        .map(|x| syn::parse_str(x.as_ref()).unwrap())
        .collect();
    let local_size: Vec<TokenStream> = compiled
        .local_size
        .iter()
        .map(|x| syn::parse_str(x.as_ref()).unwrap())
        .collect();

    let tokens = quote! {
        struct #struct_name {
            pipeline: metal::ComputePipelineState,
        }

        impl #struct_name {
            #(#dependency_guards)*
            const MTLB: &[u8] =  #mtlb;

            fn new(context: &crate::backends::metal::MTLContext, data_type: crate::backends::metal::KernelDataType) -> Result<Self, crate::backends::metal::MTLError> {
                let lib = context.device.new_library_with_data(Self::MTLB)?;
                let func = lib.get_function(&format!("{}_{}", #kernel_name, data_type.function_name_suffix()), None)?;
                let pipeline = context.device.new_compute_pipeline_state_with_function(&func)?;
                Ok(Self { pipeline })
            }

            fn encode(&self, #(#encode_args_defs, )*compute_encoder: &metal::ComputeCommandEncoderRef) {
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_args_sets)*
                compute_encoder.dispatch_thread_groups(
                    metal::MTLSize::new(#((#global_size) as u64, )*),
                    metal::MTLSize::new(#((#local_size) as u64, )*),
                );
            }
        }
    };

    // eprintln!("{}", tokens);

    tokens.into()
}
