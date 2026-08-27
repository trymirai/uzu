mod arguments;
mod dispatch;
mod host_expression_rewriter;
mod specialize;
mod trait_wiring;
mod variants;

use anyhow::Result;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};

use self::host_expression_rewriter::HostExpressionRewriter;
use super::{ast::MetalKernelInfo, wrapper::SpecializeBaseIndices};
use crate::common::{enum_paths::EnumPaths, kernel::Kernel, mangling::dynamic_mangle};

pub fn bindgen(
    kernel: &MetalKernelInfo,
    specialize_indices: &SpecializeBaseIndices,
    enum_paths: &EnumPaths,
    library_const: &proc_macro2::Ident,
    num_shards: usize,
    library_compressed: bool,
) -> Result<(TokenStream, Option<TokenStream>)> {
    let kernel_name = kernel.name.as_ref();
    let trait_name = format_ident!("{}Kernel", kernel_name);
    let struct_name = format_ident!("{}MetalKernel", kernel_name);

    let variant_binds = variants::parse(kernel)?;
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
    let (encode_set_calls, encode_set_uses_binding_slot): (Vec<TokenStream>, Vec<bool>) =
        argument_emissions.iter().filter_map(|argument| argument.encode_set()).collect();
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
    let specialize::RetainedSpecializations {
        wrapper_fields: retained_specialization_fields,
        wrapper_initializers: retained_specialization_initializers,
    } = specialize_emission.retain_referenced(&referenced_parameter_names);
    let function_constants_initialization = specialize_emission.function_constants_initialization();
    let function_constants_argument = specialize_emission.function_constants_argument();
    let cache_key = specialize_emission.cache_key();

    let dispatch_code = &dispatch_emission.dispatch_code;
    let empty_dispatch_guards = &dispatch_emission.empty_dispatch_guards;

    let library_data = if num_shards == 1 {
        quote! { #library_const[0] }
    } else {
        let num_shards = num_shards as u64;
        quote! { #library_const[(xxhash_rust::xxh3::xxh3_64(entry_name.as_bytes()) % #num_shards) as usize] }
    };

    // TODO: overpessimistic, not all are always used at the same time, should really go over variants
    let max_buffer_bind_count = encode_set_uses_binding_slot.iter().filter(|v| **v).count();
    assert!(max_buffer_bind_count <= 31, "metal 4 doesn't support more than 31 bindings");

    let trait_implementation_for = &trait_wiring.trait_implementation_for;
    let associate_backend = &trait_wiring.associate_backend;
    let method_visibility = &trait_wiring.method_visibility;

    encode_lifetimes.push(quote! { 'encoder });
    encode_argument_definitions.push(quote! {
        encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::metal::Metal>
    });

    let kernel_tokens = quote! {
        pub struct #struct_name {
            pipeline: Retained<ProtocolObject<dyn MTLComputePipelineState>>,
            argument_table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
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
                let entry_name = #entry_name;
                #function_constants_initialization
                let pipeline = context.compute_pipeline_state(#library_data, #library_compressed, #cache_key, &entry_name, #function_constants_argument)?;
                let argument_table_descriptor = MTL4ArgumentTableDescriptor::new();
                argument_table_descriptor.set_max_buffer_bind_count(#max_buffer_bind_count);
                let argument_table = context.device.new_argument_table_with_descriptor(&argument_table_descriptor).map_err(|e| MetalError::CannotCreateArgumentTable(e.to_string()))?;
                Ok(Self {
                    pipeline,
                    argument_table
                    #(, #conditional_buffer_initializers)*
                    #(, #variant_struct_initializers)*
                    #(, #retained_specialization_initializers)*
                })
            }

            #method_visibility fn encode<#(#encode_lifetimes),*>(
                &self,
                #(#encode_argument_definitions),*
            ) {
                #empty_dispatch_guards
                encoder.push_debug_group(#kernel_name);
                #(#encode_deconstructs)*
                #encode_accesses_call
                let compute_encoder = &encoder.as_command_buffer_mut().compute_encoder;
                compute_encoder.set_compute_pipeline_state(&self.pipeline);
                #(#encode_set_calls)*
                compute_encoder.set_argument_table(Some(&self.argument_table));
                #dispatch_code
                let command_encoder: &ProtocolObject<dyn MTL4CommandEncoder> = compute_encoder.as_ref();
                command_encoder.pop_debug_group();
            }
        }
    };

    Ok((kernel_tokens, trait_wiring.associated_type))
}

pub fn bindgen_global(kernels: &[(impl AsRef<std::path::Path>, &[Kernel])]) -> Result<TokenStream> {
    let includes = kernels.iter().map(|(path, _kernels)| {
        let path = path.as_ref().to_str().expect("bindings path is not utf-8");

        quote! {
            include!(#path);
        }
    });

    let associated_types = kernels.iter().flat_map(|(_path, kernels)| kernels.iter()).map(|kernel| {
        let trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
        let struct_name = format_ident!("{}MetalKernel", kernel.name.as_ref());

        quote! {
            type #trait_name = #struct_name;
        }
    });

    let tokens = quote! {
        use metal::{
            MTL4ArgumentTable, MTL4ArgumentTableDescriptor, MTL4CommandEncoder, MTL4ComputeCommandEncoder, MTLBuffer,
            MTLComputePipelineState, MTLDeviceExt, MTLFunctionConstantValues, MTLSize,
        };
        use objc2::{rc::Retained, runtime::ProtocolObject};

        use crate::backends::{
            common::{BufferArg, BufferGpuAddressRangeExt},
            metal::{
                context::MetalContext,
                error::MetalError,
                metal_extensions::{FunctionConstantValuesSetValue, MetalDataTypeExt},
            },
        };

        #(#includes)*

        macro_rules! autogen_kernels {
            () => {
                #(#associated_types)*
            }
        }
    };

    Ok(tokens)
}
