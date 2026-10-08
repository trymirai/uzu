use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail, ensure};
use proc_macro2::{TokenStream, TokenTree};
use quote::{format_ident, quote};
use syn::{Expr, Ident, Type};

use super::{Error, SlangArgumentType, SlangEntryPointAbi, SlangKernelInfo, wrapper::specialization_name};
use crate::common::kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType};

/// Generated file holding a kernel's binding type, next to its SPIR-V.
pub fn binding_file(
    output_base: &Path,
    kernel_name: &str,
) -> PathBuf {
    output_base.with_file_name(binding_module(kernel_name).to_string()).with_extension("rs")
}

/// Module declarations and re-exports of every generated binding from `(file, kernel name, test)`; bindings of
/// `[[Test]]` kernels exist only in test builds.
pub fn bindgen_umbrella(bindings: &[(String, String, bool)]) -> TokenStream {
    let modules = bindings.iter().map(|(file, kernel_name, test)| {
        let test = test.then(|| quote! { #[cfg(test)] });
        let module = binding_module(kernel_name);
        let ty = format_ident!("{kernel_name}VulkanKernel");
        quote! {
            #test
            #[path = #file]
            mod #module;
            #test
            pub use #module::#ty;
        }
    });
    quote! { #(#modules)* }
}

/// Raw typed binding of a public kernel. Its constructor and `encode` follow the common `Kernel`
/// signature; the push-constant block and specialization data follow each entry point's reflected ABI.
pub fn bindgen(
    info: &SlangKernelInfo,
    kernel: &Kernel,
    variants: &[(Vec<&'static str>, SlangEntryPointAbi)],
    spirv_file: &str,
) -> Result<TokenStream, Error> {
    let kernel_name: &str = &kernel.name;
    let struct_name = format_ident!("{kernel_name}VulkanKernel");
    let (_, abi) = variants.first().context("kernel has no entry points")?;
    let layout = |abi: &SlangEntryPointAbi| {
        let fields = abi.fields.iter().map(|field| (field.name.clone(), field.offset, field.size)).collect::<Vec<_>>();
        (abi.block_size, abi.group_size, abi.specialization_ids.clone(), fields)
    };
    ensure!(variants.iter().all(|(_, other)| layout(other) == layout(abi)), "{kernel_name}: variants differ in layout");

    let type_parameters = kernel
        .parameters
        .iter()
        .filter(|parameter| matches!(parameter.ty, KernelParameterType::Type))
        .map(|parameter| format_ident!("{}", parameter.name.as_ref()))
        .collect::<Vec<_>>();
    let mut specializations = Vec::new();
    for parameter in &kernel.parameters {
        let KernelParameterType::Value(ty) = &parameter.ty else {
            continue;
        };
        ensure!(ty.as_ref() == "bool", "{kernel_name}: specialization '{}' is not bool", parameter.name);
        let constant = specialization_name(kernel_name, &parameter.name);
        let (_, id) = abi
            .specialization_ids
            .iter()
            .find(|(name, _)| *name == constant)
            .with_context(|| format!("{kernel_name}: no specialization constant '{constant}'"))?;
        specializations.push((format_ident!("{}", parameter.name.as_ref()), *id));
    }

    let mut strides = BTreeSet::new();
    for (_, abi) in variants {
        for (pointee, stride, _) in abi.fields.iter().filter_map(|field| field.pointee.as_ref()) {
            strides.insert((data_type(pointee)?.to_string(), *stride));
        }
    }
    let stride_checks = strides.iter().map(|(data_type, stride)| {
        let data_type = format_ident!("{data_type}");
        quote! { const _: () = assert!(crate::data_type::DataType::#data_type.size_in_bytes() == #stride); }
    });

    ensure!(kernel.arguments.len() == abi.fields.len(), "{kernel_name}: arguments and reflected fields differ");
    let mut encode_arguments = Vec::new();
    let mut packing = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for argument in &kernel.arguments {
        let name = format_ident!("{}", argument.name.as_ref());
        let field = abi
            .fields
            .iter()
            .find(|field| field.name == argument.name.as_ref())
            .with_context(|| format!("{kernel_name}: no reflected field for '{name}'"))?;
        let (start, end) = (field.offset, field.offset + field.size);
        let (argument_type, bytes) = match &argument.ty {
            KernelArgumentType::Buffer(access) => {
                ensure!(field.size == 8 && field.pointee.is_some(), "{kernel_name}: '{name}' is not a device address");
                let declared = if argument.conditional {
                    quote! { #name.clone() }
                } else {
                    quote! { Some(#name.clone()) }
                };
                match access {
                    KernelBufferAccess::Read => reads.push(declared),
                    KernelBufferAccess::ReadWrite => writes.push(declared),
                }
                let address = if argument.conditional {
                    quote! { #name.as_ref().map_or(0, |(buffer, range)| buffer.device_address() + range.start) }
                } else {
                    quote! { (#name.0.device_address() + #name.1.start) }
                };
                (quote! { (&std::sync::Arc<VkBuffer>, std::ops::Range<u64>) }, quote! { #address.to_ne_bytes() })
            },
            KernelArgumentType::Constant(ty) => {
                let size = match ty.as_ref() {
                    "u32" | "i32" | "f32" => 4,
                    other => bail!("{kernel_name}: unsupported constant type '{other}' for '{name}'"),
                };
                ensure!(field.size == size && field.pointee.is_none(), "{kernel_name}: '{name}' is not a {ty}");
                let ty: Type = syn::parse_str(ty)?;
                (quote! { #ty }, quote! { #name.to_ne_bytes() })
            },
        };
        encode_arguments.push(if argument.conditional {
            quote! { #name: Option<#argument_type> }
        } else {
            quote! { #name: #argument_type }
        });
        packing.push(quote! { __dsl_block[#start..#end].copy_from_slice(&#bytes); });
    }

    let mut conditions = Vec::new();
    let (mut axes, mut groups) = (Vec::new(), Vec::new());
    let host_expression = |text: &str| {
        syn::parse_str::<Expr>(text).with_context(|| format!("{kernel_name}: malformed host expression '{text}'"))
    };
    for argument in info.arguments() {
        match argument.argument_type()? {
            SlangArgumentType::Axis(total, _) => axes.push(host_expression(&total)?),
            SlangArgumentType::Groups(count) => groups.push(host_expression(&count)?),
            _ => {},
        }
        if let Some(condition) = argument.condition()? {
            let name = format_ident!("{}", argument.name()?);
            let condition = host_expression(condition)?;
            let message =
                format!("{kernel_name}: argument '{name}' must be present exactly when {}", quote! { #condition });
            conditions.push(quote! { assert_eq!(#name.is_some(), #condition, "{}", #message); });
        }
    }
    // The wrapper admits exactly one mode with at most three dimensions. Axis totals count threads and divide by the
    // group size; Groups counts are workgroups directly.
    let axis = !axes.is_empty();
    let grid = if axis {
        axes
    } else {
        groups
    };
    ensure!(abi.group_size.iter().all(|&size| size > 0), "{kernel_name}: reflected group size {:?}", abi.group_size);
    let dispatch = (0..3)
        .map(|index| match (index < grid.len(), abi.group_size[index]) {
            (true, size) if axis => Ok(quote! { __dsl_grid[#index].div_ceil(#size) }),
            (true, _) => Ok(quote! { __dsl_grid[#index] }),
            (false, size) if !axis || size == 1 => Ok(quote! { 1 }),
            (false, size) => bail!("{kernel_name}: reflected group size {size} on undispatched axis {index}"),
        })
        .collect::<Result<Vec<_>, Error>>()?;
    let grid_count = grid.len();
    let [group_x, group_y, group_z] = abi.group_size;
    let invocations = group_x
        .checked_mul(group_y)
        .and_then(|invocations| invocations.checked_mul(group_z))
        .with_context(|| format!("{kernel_name}: work group {:?} overflows u32 invocations", abi.group_size))?;

    let host_expressions = quote! { #(#conditions)* #(#grid)* };
    let referenced = specializations
        .iter()
        .map(|(name, _)| name)
        .filter(|name| references(host_expressions.clone(), name))
        .collect::<Vec<_>>();

    let entry_arms = variants
        .iter()
        .map(|(types, abi)| {
            let data_types = types.iter().map(|ty| data_type(ty)).collect::<Result<Vec<_>, _>>()?;
            let entry_point = &abi.name;
            Ok(quote! { (#(crate::data_type::DataType::#data_types,)*) => #entry_point })
        })
        .collect::<Result<Vec<_>, Error>>()?;
    // A kernel without type parameters has its single entry point; the others select one by data types.
    let entry_selection = match type_parameters.is_empty() {
        true => {
            let entry_point = &abi.name;
            quote! { #entry_point }
        },
        false => quote! {
            match (#(#type_parameters,)*) {
                #(#entry_arms,)*
                _ => return Err(Error::KernelVariant { kernel: #kernel_name, data_types: Box::new([#(#type_parameters),*]) }),
            }
        },
    };

    let specialization_count = specializations.len();
    let specialization_names = specializations.iter().map(|(name, _)| name).collect::<Vec<_>>();
    let specialization_entries = specializations.iter().enumerate().map(|(index, (_, id))| {
        let offset = 4 * index as u32;
        quote! { vk::SpecializationMapEntry::default().constant_id(#id).offset(#offset).size(4) }
    });
    let block_size = abi.block_size;
    let block_size_u32 = u32::try_from(block_size)?;
    let dispatch_message = format!("{kernel_name} dispatch");
    let (read_count, write_count) = (reads.len(), writes.len());

    Ok(quote! {
        use ash::vk;

        use crate::backends::vulkan::{Error, VkBuffer, VkCommandBufferEncoding, VkComputePipeline, VkContext, VkShader};

        const SPIRV: &[u8] = include_bytes!(#spirv_file);

        #(#stride_checks)*

        pub struct #struct_name {
            pipeline: std::sync::Arc<VkComputePipeline>,
            #(#referenced: bool,)*
        }

        impl #struct_name {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub fn new(
                context: &std::sync::Arc<VkContext>
                #(, #type_parameters: crate::data_type::DataType)*
                #(, #specialization_names: bool)*
            ) -> Result<Self, Error> {
                let entry_point = #entry_selection;
                let limits = &context.physical_device().properties.limits;
                let size = [#group_x, #group_y, #group_z];
                if size.iter().zip(limits.max_compute_work_group_size).any(|(&size, limit)| size > limit)
                    || #invocations > limits.max_compute_work_group_invocations
                {
                    return Err(Error::WorkGroupSize {
                        size,
                        limit: limits.max_compute_work_group_size,
                        invocations: limits.max_compute_work_group_invocations,
                    });
                }
                let shader = VkShader::new(context.clone(), SPIRV)?;
                let specialization_data: [[u8; 4]; #specialization_count] =
                    [#(vk::Bool32::from(#specialization_names).to_ne_bytes()),*];
                let specialization_entries: [vk::SpecializationMapEntry; #specialization_count] = [#(#specialization_entries),*];
                let specialization =
                    vk::SpecializationInfo::default().map_entries(&specialization_entries).data(specialization_data.as_flattened());
                let pipeline =
                    VkComputePipeline::new(context.clone(), shader.module(), &[], #block_size_u32, entry_point, &specialization)?;
                Ok(Self {
                    pipeline: std::sync::Arc::new(pipeline),
                    #(#referenced,)*
                })
            }

            /// Records one dispatch. Optional-argument presence is asserted before anything is recorded,
            /// and an empty dispatch records nothing. Contract violations the runtime detects panic.
            ///
            /// # Safety
            /// Every range must start at an element of the constructed data type, aligned to it, and
            /// cover every element the kernel indexes for these scalar arguments. Ranges written by the
            /// kernel must not alias other arguments unless the kernel defines that aliasing. Range
            /// bounds are checked; shader indexing within them is not.
            #[allow(clippy::too_many_arguments)]
            pub unsafe fn encode(
                &self,
                #(#encode_arguments,)*
                command_buffer: &mut VkCommandBufferEncoding,
            ) {
                #(let #referenced = self.#referenced;)*
                #(#conditions)*
                let __dsl_grid: [u32; #grid_count] = [#(#grid),*];
                if __dsl_grid.contains(&0) {
                    return;
                }
                let mut __dsl_block = [0u8; #block_size];
                #(#packing)*
                let __dsl_reads: [Option<(&std::sync::Arc<VkBuffer>, std::ops::Range<u64>)>; #read_count] = [#(#reads),*];
                let __dsl_writes: [Option<(&std::sync::Arc<VkBuffer>, std::ops::Range<u64>)>; #write_count] = [#(#writes),*];
                unsafe {
                    command_buffer.encode_dispatch(
                        &self.pipeline,
                        &__dsl_block,
                        [#(#dispatch),*],
                        __dsl_reads.into_iter().flatten(),
                        __dsl_writes.into_iter().flatten(),
                    )
                }
                .expect(#dispatch_message);
            }
        }
    })
}

fn binding_module(kernel_name: &str) -> Ident {
    let mut module = String::new();
    for (index, character) in kernel_name.chars().enumerate() {
        if character.is_ascii_uppercase() && index > 0 {
            module.push('_');
        }
        module.push(character.to_ascii_lowercase());
    }
    format_ident!("{module}_vulkan_kernel")
}

fn data_type(slang_type: &str) -> Result<Ident, Error> {
    Ok(format_ident!(
        "{}",
        match slang_type {
            "float" => "F32",
            "half" => "F16",
            "bf16" => "BF16",
            "uint" => "U32",
            "int" => "I32",
            other => bail!("no DataType for Slang type '{other}'"),
        }
    ))
}

fn references(
    tokens: TokenStream,
    name: &Ident,
) -> bool {
    tokens.into_iter().any(|token| match token {
        TokenTree::Ident(ident) => ident == *name,
        TokenTree::Group(group) => references(group.stream(), name),
        _ => false,
    })
}
