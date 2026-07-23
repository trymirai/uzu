mod arguments;
mod dispatch;
mod host_expression_rewriter;
mod specialize;
mod trait_wiring;
mod variants;

use anyhow::{Context, Result};
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use self::host_expression_rewriter::HostExpressionRewriter;
use super::{
    ast::MetalKernelInfo,
    wrapper::{SpecializeBaseIndices, accepted_variants},
};
use crate::common::enum_paths::EnumPaths;

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
    enum_paths: &EnumPaths,
) -> Result<(TokenStream, Option<TokenStream>)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{}Kernel", kernel_name);
    let struct_name = format_ident!("{}MetalKernel", kernel_name);

    let variant_binds = variants::parse(kernel)?;
    let accepted_variants = accepted_variants(kernel);
    let first_variant =
        accepted_variants.first().context(format!("kernel {kernel_name}: all variants rejected by constraints"))?;
    let request_emission = variants::request(kernel, &variant_binds, &accepted_variants, enum_paths)?;
    let specialize_emission =
        specialize::parse(kernel, specialize_indices.get(&kernel.name).copied(), kernel_name, enum_paths)?;
    let mut host_expression_rewriter =
        HostExpressionRewriter::new(&variant_binds, enum_paths, specialize_emission.argument_names(), kernel_name);
    let argument_emissions = arguments::parse(kernel, enum_paths, &mut host_expression_rewriter)?;
    let trait_wiring = trait_wiring::build(kernel, &trait_name, &struct_name);

    let dispatch_emission = dispatch::parse(kernel, &mut host_expression_rewriter)?;
    let referenced_parameter_names = host_expression_rewriter.finish();

    let conditional_buffer_fields: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.struct_field()).collect();
    let conditional_buffer_initializers: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.struct_initializer()).collect();
    let mut encode_argument_definitions: Vec<TokenStream> =
        argument_emissions.iter().filter_map(|argument| argument.encode_argument_definition()).collect();
    let mut encode_lifetimes: Vec<TokenStream> =
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
    let request_tokens = request_emission.as_ref().map(|request| &request.tokens);
    let request_initializer = request_emission.as_ref().map(|request| &request.initializer);
    let request_name = format_ident!("{kernel_name}Request");
    let request_fields = variant_binds.iter().map(|bind| &bind.field_name);
    let request_parameters = variant_binds.iter().map(|bind| &bind.parameter_name);
    let request_deconstruct = request_emission.as_ref().map(|_| {
        quote! {
            #[allow(unused_variables, non_snake_case, non_shorthand_field_patterns)]
            let #request_name {
                #(#request_fields: #request_parameters,)*
            } = request;
        }
    });

    let specialize_arguments = specialize_emission.constructor_arguments();
    let specialize_names =
        specialize_emission.argument_names().into_iter().map(|name| format_ident!("{name}")).collect::<Vec<_>>();
    let specialize::RetainedSpecializations {
        wrapper_fields: retained_specialization_fields,
        wrapper_initializers: retained_specialization_initializers,
    } = specialize_emission.retain_referenced(&referenced_parameter_names);
    let function_constants_initialization = specialize_emission.function_constants_initialization();
    let function_constants_argument = specialize_emission.function_constants_argument();
    let cache_key = specialize_emission.cache_key();

    let dispatch_code = &dispatch_emission.dispatch_code;
    let empty_dispatch_guards = &dispatch_emission.empty_dispatch_guards;

    let trait_implementation_for = &trait_wiring.trait_implementation_for;
    let associate_backend = &trait_wiring.associate_backend;
    let method_visibility = &trait_wiring.method_visibility;

    encode_lifetimes.push(quote! { 'encoder });
    encode_argument_definitions.push(quote! {
        encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::metal::Metal>
    });

    let build_from_request = request_emission.as_ref().map(|_| quote! {
        impl #struct_name {
            pub(crate) fn from_request(
                context: &MetalContext,
                request: #request_name
                #(, #specialize_arguments)*
            ) -> Result<Self, MetalError> {
                let entry_name = request.resolve()?;
                #request_deconstruct
                #function_constants_initialization
                let pipeline = context.compute_pipeline_state(#cache_key, entry_name, #function_constants_argument)?;
                Ok(Self {
                    pipeline
                    #(, #conditional_buffer_initializers)*
                    #(, #variant_struct_initializers)*
                    #(, #retained_specialization_initializers)*
                })
            }
        }
    });

    let new_body = if request_emission.is_some() {
        quote! { Self::from_request(context, #request_initializer #(, #specialize_names)*) }
    } else {
        let entry_name = &first_variant.entry_name;
        quote! {
            let entry_name = #entry_name;
            #function_constants_initialization
            let pipeline = context.compute_pipeline_state(#cache_key, entry_name, #function_constants_argument)?;
            Ok(Self {
                pipeline
                #(, #conditional_buffer_initializers)*
                #(, #variant_struct_initializers)*
                #(, #retained_specialization_initializers)*
            })
        }
    };

    let kernel_tokens = quote! {
        #request_tokens

        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            #(#conditional_buffer_fields,)*
            #(#variant_struct_fields,)*
            #(#retained_specialization_fields,)*
        }

        #[allow(clippy::style, clippy::complexity, clippy::perf)]
        impl #trait_implementation_for #struct_name {
            #associate_backend

            #method_visibility fn new(
                context: &MetalContext
                #(, #variant_constructor_arguments)*
                #(, #specialize_arguments)*
            ) -> Result<Self, MetalError> {
                #new_body
            }

            #method_visibility fn encode<#(#encode_lifetimes),*>(
                &self,
                #(#encode_argument_definitions),*
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

        #build_from_request
    };

    Ok((kernel_tokens, trait_wiring.associated_type))
}
