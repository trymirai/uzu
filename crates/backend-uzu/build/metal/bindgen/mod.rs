mod arguments;
mod dispatch;
mod specialize;
mod trait_wiring;
mod variant_path_rewriter;
mod variants;

use anyhow::Result;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use self::variant_path_rewriter::VariantPathRewriter;
use super::{ast::MetalKernelInfo, enum_path_rewrite::EnumPathRewriter, wrapper::SpecializeBaseIndices};
use crate::common::mangling::dynamic_mangle;

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
    enum_path_rewriter: &EnumPathRewriter,
) -> Result<(TokenStream, Option<TokenStream>)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{}Kernel", kernel_name);
    let struct_name = format_ident!("{}MetalKernel", kernel_name);

    let variant_binds = variants::parse(kernel)?;
    let argument_emissions = arguments::parse(kernel, enum_path_rewriter)?;
    let specialize_emission = specialize::parse(kernel, specialize_indices.get(&kernel.name).copied(), kernel_name)?;
    let trait_wiring = trait_wiring::build(kernel, &trait_name, &struct_name);

    let mut variant_path_rewriter = VariantPathRewriter::new(&variant_binds, kernel_name);
    let dispatch_emission = dispatch::parse(kernel, &mut variant_path_rewriter)?;
    let referenced_parameter_names = variant_path_rewriter.finish();

    let conditional_buffer_fields: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.struct_field()).collect();
    let conditional_buffer_initializers: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.struct_initializer()).collect();
    let encode_argument_definitions: Vec<TokenStream> =
        argument_emissions.iter().map(|argument| argument.encode_argument_definition()).collect();
    let encode_lifetimes: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_lifetime()).collect();
    let encode_deconstructs: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_deconstruct()).collect();
    let encode_set_calls: Vec<TokenStream> = argument_emissions.iter().map(|argument| argument.encode_set()).collect();
    let encode_accesses_call = arguments::encode_accesses_call(&argument_emissions);

    let variant_struct_fields: Vec<TokenStream> =
        variant_binds.iter().filter_map(|variant| variant.struct_field(&referenced_parameter_names)).collect();
    let variant_struct_initializers: Vec<TokenStream> =
        variant_binds.iter().filter_map(|variant| variant.struct_initializer(&referenced_parameter_names)).collect();
    let variant_constructor_arguments: Vec<TokenStream> =
        variant_binds.iter().map(|variant| variant.constructor_argument()).collect();
    let variant_kernel_format: Vec<TokenStream> = variant_binds.iter().map(|variant| variant.kernel_format()).collect();
    let entry_name = dynamic_mangle(kernel_name, variant_kernel_format);

    let specialize_arguments = specialize_emission.constructor_arguments();
    let function_constants_initialization = specialize_emission.function_constants_initialization();
    let function_constants_argument = specialize_emission.function_constants_argument();
    let cache_key = specialize_emission.cache_key();

    let dispatch_code = &dispatch_emission.dispatch_code;
    let empty_dispatch_guards = &dispatch_emission.empty_dispatch_guards;

    let trait_implementation_for = &trait_wiring.trait_implementation_for;
    let associate_backend = &trait_wiring.associate_backend;
    let method_visibility = &trait_wiring.method_visibility;

    let kernel_tokens = quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            #(#conditional_buffer_fields,)*
            #(#variant_struct_fields,)*
        }

        impl #trait_implementation_for #struct_name {
            #associate_backend

            #method_visibility fn new(
                context: &MetalContext
                #(, #variant_constructor_arguments)*
                #(, #specialize_arguments)*
            ) -> Result<Self, MetalError> {
                let entry_name = #entry_name;
                #function_constants_initialization
                let pipeline = context.compute_pipeline_state(#cache_key, &entry_name, #function_constants_argument)?;
                Ok(Self { pipeline #(, #conditional_buffer_initializers)* #(, #variant_struct_initializers)* })
            }

            #method_visibility fn encode<#(#encode_lifetimes, )* 'encoder>(
                &self,
                #(#encode_argument_definitions, )*
                encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::metal::Metal>,
            ) {
                #empty_dispatch_guards
                #(#encode_deconstructs)*
                #encode_accesses_call
                let compute_encoder = encoder.as_command_buffer_mut().ensure_compute();
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_set_calls)*
                #dispatch_code
            }
        }
    };

    Ok((kernel_tokens, trait_wiring.associated_type))
}
