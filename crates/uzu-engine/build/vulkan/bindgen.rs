use std::{
    collections::{BTreeSet, HashMap},
    iter::once,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail, ensure};
use itertools::Itertools;
use proc_macro2::{TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use shader_slang::ScalarType;
use syn::{Expr, GenericArgument, Ident, PathArguments, Type, parse_quote};

use crate::{
    common::{
        expr_rewrite::{qualify_enum_path, rewrite_paths_with},
        gpu_type_paths::{GpuTypeKind, GpuTypePaths},
        gpu_types::{GpuType, GpuTypeEnum, GpuTypeStructFieldType, GpuTypes},
        kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType},
    },
    slang::{SlangArgumentType, SlangKernel, slang2rust, specialization_name},
    vulkan::{SlangEntryPointAbi, SlangEntryPointVariant, SlangFieldAbi, vulkan_target::AXIS_FOLD_GROUPS},
};

/// Generated file holding a kernel's binding type, next to its SPIR-V.
pub fn binding_file(
    output_base: &Path,
    kernel_name: &str,
) -> PathBuf {
    output_base.with_file_name(binding_module(kernel_name).to_string()).with_extension("rs")
}

/// Module declarations and re-exports of every generated binding from `(file, kernel name, public)`.
/// `autogen_kernels!` names the binding of every public kernel as its `Kernels` associated type.
pub fn bindgen_umbrella(bindings: &[(String, String, bool)]) -> TokenStream {
    let modules = bindings.iter().map(|(file, kernel_name, _)| {
        let module = binding_module(kernel_name);
        let ty = format_ident!("{kernel_name}VulkanKernel");
        quote! {
            #[path = #file]
            mod #module;
            pub use #module::#ty;
        }
    });
    let kernel_types = bindings.iter().filter(|(.., public)| *public).map(|(_, kernel_name, _)| {
        let kernel_trait = format_ident!("{kernel_name}Kernel");
        let ty = format_ident!("{kernel_name}VulkanKernel");
        quote! { type #kernel_trait = crate::backends::vulkan::#ty; }
    });
    quote! {
        #(#modules)*

        macro_rules! autogen_kernels {
            () => {
                #(#kernel_types)*
            }
        }
    }
}

/// Raw typed binding of a kernel: the common `Kernel` signature over each entry point's reflected ABI, with its SPIR-V
/// and, for an entry point that stores BF16, the ABI and SPIR-V built with the native BF16 type.
pub fn bindgen(
    info: &SlangKernel,
    kernel: &Kernel,
    variants: &[SlangEntryPointVariant],
    gpu_type_paths: &GpuTypePaths,
    gpu_types: &[&GpuTypes],
) -> anyhow::Result<TokenStream> {
    let kernel_name: &str = &kernel.name;
    let struct_name = format_ident!("{kernel_name}VulkanKernel");
    let first = variants.first().context("kernel has no entry points")?;
    let abi = &first.abi;
    let layout = |abi: &SlangEntryPointAbi| {
        let fields = abi.fields.iter().map(|field| (field.name.clone(), field.offset, field.size)).collect::<Vec<_>>();
        (abi.block_size, abi.group_size, abi.specialization_ids.clone(), fields)
    };
    ensure!(
        variant_abis(variants).all(|other| layout(other) == layout(abi)),
        "{kernel_name}: variants differ in layout"
    );
    // Vulkan 1.4 guarantees every device 256 bytes of push constants.
    ensure!(abi.block_size <= 256, "{kernel_name}: push-constant block of {} bytes exceeds 256", abi.block_size);

    let type_parameters = kernel
        .parameters
        .iter()
        .filter(|parameter| matches!(parameter.ty, KernelParameterType::Type))
        .map(|parameter| format_ident!("{}", parameter.name.as_ref()))
        .collect::<Vec<_>>();
    let specializations = specialization_constants(kernel_name, kernel, abi, gpu_type_paths)?;
    let pipeline_variants = pipeline_variant_argument(kernel_name, info, kernel, abi, gpu_types, gpu_type_paths)?;
    let mut layout_checks = stride_checks(kernel, variants, gpu_type_paths)?;
    ensure!(
        kernel.arguments.len() == abi.fields.len() + usize::from(pipeline_variants.is_some()),
        "{kernel_name}: arguments and reflected fields differ"
    );
    let ArgumentEncoding {
        parameters,
        packing,
        reads,
        writes,
    } = encode_arguments(
        kernel_name,
        kernel,
        variants,
        abi,
        pipeline_variants.as_ref(),
        gpu_types,
        gpu_type_paths,
        &mut layout_checks,
    )?;
    let (conditions, specialization_conditions) = presence_checks(kernel_name, info, gpu_type_paths)?;
    let (new_preconditions, encode_preconditions) =
        preconditions(kernel_name, kernel, info, gpu_types, gpu_type_paths)?;
    let Dispatch {
        grid,
        group_sizes,
        axis_sizes,
        dimensions,
    } = dispatch_plan(kernel_name, info, abi, gpu_type_paths)?;
    let grid_count = grid.len();

    let host_expressions = quote! { #(#conditions)* #(#encode_preconditions)* #(#grid)* #axis_sizes };
    let references = |name: &Ident| find_ident(host_expressions.clone(), &|ident| ident == name).is_some();
    let (referenced, referenced_types): (Vec<_>, Vec<_>) = specializations
        .iter()
        .filter(|specialization| references(&specialization.name))
        .map(|specialization| (&specialization.name, &specialization.ty))
        .unzip();
    let (specialization_names, specialization_types): (Vec<_>, Vec<_>) =
        specializations.iter().map(|specialization| (&specialization.name, &specialization.ty)).unzip();

    // A pinned subgroup size, which the workgroup must fill with whole subgroups within the device's subgroup limit.
    let required_subgroup_size = match info
        .required_subgroup_size
        .as_deref()
        .map(|text| host_expression(kernel_name, gpu_type_paths, text))
        .transpose()?
    {
        Some(size) => quote! { Some(#size) },
        None => quote! { None },
    };
    let PipelineCode {
        field: pipeline_field,
        construction,
        lookup: pipeline,
    } = pipeline_code(
        pipeline_variants.as_ref(),
        &specializations,
        &referenced,
        abi.block_size,
        &required_subgroup_size,
    )?;
    let block_size = abi.block_size;
    let read_count = reads.len();
    let write_count = writes.len();

    // A public kernel's common trait owns `new` and `encode`, so its raw binding takes other names that never shadow the
    // trait methods on the concrete type.
    let new_fn = if info.public {
        format_ident!("create")
    } else {
        format_ident!("new")
    };
    let encode_fn = if info.public {
        format_ident!("encode_ranges")
    } else {
        format_ident!("encode")
    };
    let entry_selection = select_entry(kernel_name, &type_parameters, first, variants)?;
    let trait_impl = if info.public {
        public_trait_impl(
            kernel_name,
            kernel,
            gpu_type_paths,
            &struct_name,
            &type_parameters,
            &specializations,
            &new_fn,
            &encode_fn,
        )?
    } else {
        quote! {}
    };

    Ok(quote! {
        use crate::backends::vulkan::{
            Error, VkBuffer, VkCommandBufferEncoding, VkComputePipeline, VkComputePipelineKey, VkContext,
            check_work_group,
        };


        #(#layout_checks)*

        #[allow(non_snake_case)]
        pub struct #struct_name {
            #pipeline_field,
            #(#referenced: #referenced_types,)*
        }

        impl #struct_name {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub fn #new_fn(
                context: &std::sync::Arc<VkContext>
                #(, #type_parameters: crate::data_type::DataType)*
                #(, #specialization_names: #specialization_types)*
            ) -> Result<Self, Error> {
                let (entry_point, spirv) = #entry_selection;
                #(#specialization_conditions)*
                #(#new_preconditions)*
                check_work_group(
                    context.physical_device(),
                    #kernel_name,
                    [#(#group_sizes),*],
                    #required_subgroup_size,
                )?;
                #construction
            }

            /// Records one dispatch. Optional-argument presence is asserted before anything is recorded,
            /// and an empty dispatch records nothing. Violated preconditions panic; a failed upload, an
            /// out-of-bounds range or a grid past the device limit ends the recording, as a device error does.
            ///
            /// # Safety
            /// Every range must start at an element of the constructed data type, aligned to it, and
            /// cover every element the kernel indexes for these scalar arguments. Ranges written by the
            /// kernel must not alias other arguments unless the kernel defines that aliasing. Range
            /// bounds are checked; shader indexing within them is not.
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub unsafe fn #encode_fn(
                &self,
                #(#parameters,)*
                command_buffer: &mut VkCommandBufferEncoding,
            ) {
                #(let #referenced = self.#referenced;)*
                #(#conditions)*
                #(#encode_preconditions)*
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
                        #kernel_name,
                        #pipeline,
                        &__dsl_block,
                        #dimensions,
                        __dsl_reads.into_iter().flatten(),
                        __dsl_writes.into_iter().flatten(),
                    )
                }
            }
        }

        #trait_impl
    })
}

/// A specialization constant: the parameter's name and Rust type, the value it writes and its constant id.
struct Specialization {
    name: Ident,
    ty: Type,
    value: TokenStream,
    id: u32,
}

/// The `[[PipelineVariants]]` argument: its name, one path per variant of its canonical enum, and its constant id.
struct PipelineVariants {
    selector: Ident,
    paths: Vec<TokenStream>,
    id: u32,
}

/// The parameters, push-constant packing and buffer ranges of `encode`, one per kernel argument in declaration order.
struct ArgumentEncoding {
    parameters: Vec<TokenStream>,
    packing: Vec<TokenStream>,
    reads: Vec<TokenStream>,
    writes: Vec<TokenStream>,
}

/// The dispatch of a kernel: its grid, the workgroup size, the grid's `Axis` sizes if it has any, and the dispatch
/// dimensions as an expression.
struct Dispatch {
    grid: Vec<Expr>,
    group_sizes: Vec<Expr>,
    axis_sizes: Option<TokenStream>,
    dimensions: TokenStream,
}

/// The pipeline field of the kernel, its construction in `new`, and the pipeline `encode` dispatches with.
struct PipelineCode {
    field: TokenStream,
    construction: TokenStream,
    lookup: TokenStream,
}

/// The specialization constants of the kernel, each written from its parameter's value.
fn specialization_constants(
    kernel_name: &str,
    kernel: &Kernel,
    abi: &SlangEntryPointAbi,
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<Vec<Specialization>> {
    // Specializations are `bool`, `u32` or a canonical enum (`uint` in Slang); an absent optional one writes 0.
    let mut specializations = Vec::new();
    for parameter in &kernel.parameters {
        let KernelParameterType::Value(text) = &parameter.ty else {
            continue;
        };
        let name = format_ident!("{}", parameter.name.as_ref());
        let ty: Type = gpu_type_paths.resolve_type(text)?;
        let value = match option_inner(&ty) {
            Some(inner) => match syn::parse2(word(kernel_name, gpu_type_paths, &name, inner, &["u32"])?)? {
                Expr::Paren(inner) => {
                    let inner = inner.expr;
                    quote! { #name.map_or(0, |#name| #inner) }
                },
                _ => quote! { #name.unwrap_or(0) },
            },
            None => word(kernel_name, gpu_type_paths, &name, &ty, &["u32", "bool"])?,
        };
        let id = specialization_id(kernel_name, abi, &parameter.name)?;
        specializations.push(Specialization {
            name,
            ty,
            value,
            id,
        });
    }
    Ok(specializations)
}

/// The `[[PipelineVariants]]` argument, if the kernel has one.
fn pipeline_variant_argument(
    kernel_name: &str,
    info: &SlangKernel,
    kernel: &Kernel,
    abi: &SlangEntryPointAbi,
    gpu_types: &[&GpuTypes],
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<Option<PipelineVariants>> {
    let mut pipeline_variants = None;
    for argument in info.arguments.iter().filter(|argument| argument.pipeline_variants) {
        let name = argument.name.as_str();
        let Some(KernelArgumentType::Value(text)) =
            kernel.arguments.iter().find(|candidate| candidate.name.as_ref() == name).map(|argument| &argument.ty)
        else {
            bail!("{kernel_name}: PipelineVariants argument '{name}' is not a uniform");
        };
        let ty = gpu_type_paths.resolve_type(text)?;
        let short =
            (gpu_type_paths.kind_for(text) == Some(GpuTypeKind::Enum)).then_some(&**text).with_context(|| {
                format!("{kernel_name}: PipelineVariants argument '{name}' has type '{text}', not a canonical GPU enum")
            })?;
        let canonical =
            canonical_enum(gpu_types, short).with_context(|| format!("{kernel_name}: no canonical enum '{short}'"))?;
        let paths = canonical
            .variants
            .iter()
            .map(|variant| {
                let variant = format_ident!("{}", variant.name.as_ref());
                quote! { #ty::#variant }
            })
            .collect::<Vec<_>>();
        pipeline_variants = Some(PipelineVariants {
            selector: format_ident!("{name}"),
            paths,
            id: specialization_id(kernel_name, abi, name)?,
        });
    }
    Ok(pipeline_variants)
}

/// The asserts that every buffer of a primitive type has the stride of its data type. Buffers of canonical structs
/// are checked by `canonical_struct`.
fn stride_checks(
    kernel: &Kernel,
    variants: &[SlangEntryPointVariant],
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<Vec<TokenStream>> {
    let mut strides = BTreeSet::new();
    for abi in variant_abis(variants) {
        for field in &abi.fields {
            let buffer = kernel.arguments.iter().any(|argument| {
                argument.name.as_ref() == field.name && matches!(argument.ty, KernelArgumentType::Buffer(_))
            });
            if buffer
                && let Some(pointee) = &field.layout
                && gpu_type_paths.kind_for(&pointee.name) != Some(GpuTypeKind::Struct)
            {
                strides.insert((data_type(&pointee.name)?.to_string(), pointee.stride));
            }
        }
    }
    Ok(strides
        .iter()
        .map(|(data_type, stride)| {
            let data_type = format_ident!("{data_type}");
            quote! { const _: () = assert!(crate::data_type::DataType::#data_type.size_in_bytes() == #stride); }
        })
        .collect())
}

/// The canonical struct `field` holds or points to. Its fields and layout are checked against Slang's reflection, and
/// the asserts are appended to `layout_checks`.
fn canonical_struct<'a>(
    kernel_name: &str,
    variants: &[SlangEntryPointVariant],
    gpu_types: &[&GpuTypes],
    gpu_type_paths: &'a GpuTypePaths,
    field: &SlangFieldAbi,
    layout_checks: &mut Vec<TokenStream>,
) -> anyhow::Result<Option<(&'a str, Type, usize, usize)>> {
    let Some(layout) = &field.layout else {
        return Ok(None);
    };
    let pointee = &layout.name;
    let Some(path) =
        gpu_type_paths.full_path_for(pointee).filter(|_| gpu_type_paths.kind_for(pointee) == Some(GpuTypeKind::Struct))
    else {
        return Ok(None);
    };
    let name = &field.name;
    ensure!(
        variant_abis(variants).all(|other| other.fields.iter().any(|other| other == field)),
        "{kernel_name}: variants differ in the layout of '{name}'"
    );
    let canonical = canonical_types(gpu_types)
        .find_map(|candidate| match candidate {
            GpuType::Struct(candidate) if candidate.name.as_ref() == pointee => Some(candidate),
            _ => None,
        })
        .with_context(|| format!("{kernel_name}: no canonical struct '{pointee}'"))?;
    let fields = &layout.fields;
    ensure!(
        canonical.fields.iter().map(|canonical| canonical.name.as_ref()).eq(fields.iter().map(|field| &field.name)),
        "{kernel_name}: '{pointee}' fields differ from the canonical struct"
    );
    let scalar = |text: &str| match text {
        "u32" => Some(ScalarType::Uint32),
        "f32" => Some(ScalarType::Float32),
        _ => None,
    };
    for (canonical, reflected) in canonical.fields.iter().zip(fields) {
        let (element, length) = match &canonical.ty {
            GpuTypeStructFieldType::Scalar(element) => (element, None),
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => (element, Some((*length, 4))),
        };
        ensure!(
            scalar(element) == Some(reflected.scalar) && reflected.array == length,
            "{kernel_name}: field {pointee}.{} is {element} {length:?} in Rust but {:?} {:?} in Slang",
            reflected.name,
            reflected.scalar,
            reflected.array
        );
    }
    let ty: Type = syn::parse_str(path)?;
    let message = format!("{kernel_name}: {path} does not have the layout Slang reflects");
    let stride = layout.stride;
    let alignment = layout.alignment;
    layout_checks.push(quote! {
        const _: () = assert!(size_of::<#ty>() == #stride && align_of::<#ty>() == #alignment, #message);
    });
    for (canonical, reflected) in canonical.fields.iter().zip(fields) {
        let field_type: Type = match &canonical.ty {
            GpuTypeStructFieldType::Scalar(element) => syn::parse_str(element)?,
            GpuTypeStructFieldType::Array {
                element,
                length,
            } => syn::parse_str(&format!("[{element}; {length}]"))?,
        };
        let field_name = format_ident!("{}", reflected.name);
        let offset = reflected.offset;
        let size = reflected.size;
        layout_checks.push(quote! {
            const _: () = assert!(
                std::mem::offset_of!(#ty, #field_name) == #offset && size_of::<#field_type>() == #size,
                #message
            );
        });
    }
    Ok(Some((path, ty, stride, alignment)))
}

/// Encodes every kernel argument: its `encode` parameter, its push-constant bytes, and its buffer ranges.
fn encode_arguments(
    kernel_name: &str,
    kernel: &Kernel,
    variants: &[SlangEntryPointVariant],
    abi: &SlangEntryPointAbi,
    pipeline_variants: Option<&PipelineVariants>,
    gpu_types: &[&GpuTypes],
    gpu_type_paths: &GpuTypePaths,
    layout_checks: &mut Vec<TokenStream>,
) -> anyhow::Result<ArgumentEncoding> {
    let mut parameters = Vec::new();
    let mut packing = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for argument in &kernel.arguments {
        let name = format_ident!("{}", argument.name.as_ref());
        // The pipeline-selecting argument has no push-constant field.
        if let Some(PipelineVariants {
            selector,
            ..
        }) = pipeline_variants
            && let KernelArgumentType::Value(text) = &argument.ty
            && *selector == name
        {
            let ty: Type = gpu_type_paths.resolve_type(text)?;
            parameters.push(quote! { #name: #ty });
            continue;
        }
        let field = abi
            .fields
            .iter()
            .find(|field| field.name == argument.name.as_ref())
            .with_context(|| format!("{kernel_name}: no reflected field for '{name}'"))?;
        let start = field.offset;
        let end = start + field.size;
        let canonical = canonical_struct(kernel_name, variants, gpu_types, gpu_type_paths, field, layout_checks)?;
        let (argument_type, bytes) = match (&argument.ty, canonical) {
            (KernelArgumentType::Buffer(access), _) => {
                ensure!(field.size == 8 && field.layout.is_some(), "{kernel_name}: '{name}' is not a device address");
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
            // A canonical struct by value: its bytes in the push constants, which an absent optional leaves zero.
            (KernelArgumentType::Value(text), Some((path, ty, stride, _)))
                if gpu_type_paths.full_path_for(text) == Some(path) =>
            {
                ensure!(field.size == stride, "{kernel_name}: '{name}' of {} bytes is not one {path}", field.size);
                let cast = quote! { bytemuck::cast::<#ty, [u8; #stride]> };
                let bytes = if argument.conditional {
                    quote! { #name.map_or([0; #stride], #cast) }
                } else {
                    quote! { #cast(#name) }
                };
                (quote! { #ty }, bytes)
            },
            // A `[[Constant]]` pointer is uploaded into the command buffer's memory; an empty slice passes address 0.
            (KernelArgumentType::Value(text), Some((path, ty, _, alignment))) => {
                ensure!(
                    field.size == 8
                        && !argument.conditional
                        && gpu_type_paths.resolve_type(text)? == parse_quote! { &[#ty] },
                    "{kernel_name}: '{name}' of type '{text}' is not a device address of a canonical '{path}' slice"
                );
                // The encoder's upload allocator aligns ranges to at most 64 bytes.
                ensure!(alignment <= 64, "{kernel_name}: '{path}' needs alignment {alignment} above 64");
                packing.push(quote! { let #name = command_buffer.upload(#name); });
                reads.push(quote! { #name.as_ref().map(|(buffer, range)| (buffer, range.clone())) });
                let address =
                    quote! { #name.as_ref().map_or(0, |(buffer, range)| buffer.device_address() + range.start) };
                (quote! { &[#ty] }, quote! { #address.to_ne_bytes() })
            },
            (KernelArgumentType::Value(text), _) => {
                ensure!(field.size == 4 && field.layout.is_none(), "{kernel_name}: '{name}' is not a 4-byte {text}");
                let ty: Type = gpu_type_paths.resolve_type(text)?;
                let value = word(kernel_name, gpu_type_paths, &name, &ty, &["u32", "i32", "f32", "bool"])?;
                // An absent optional constant leaves its bytes zero.
                let bytes = if argument.conditional {
                    quote! { #name.map_or([0; 4], |#name| #value.to_ne_bytes()) }
                } else {
                    quote! { #value.to_ne_bytes() }
                };
                (quote! { #ty }, bytes)
            },
        };
        parameters.push(if argument.conditional {
            quote! { #name: Option<#argument_type> }
        } else {
            quote! { #name: #argument_type }
        });
        packing.push(quote! { __dsl_block[#start..#end].copy_from_slice(&#bytes); });
    }
    Ok(ArgumentEncoding {
        parameters,
        packing,
        reads,
        writes,
    })
}

/// Optional specializations are checked by the constructor, other optional arguments by `encode`. Returns the
/// constructor's checks, then `encode`'s.
fn presence_checks(
    kernel_name: &str,
    info: &SlangKernel,
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<(Vec<TokenStream>, Vec<TokenStream>)> {
    let mut conditions = Vec::new();
    let mut specialization_conditions = Vec::new();
    for argument in &info.arguments {
        let (SlangArgumentType::Buffer {
            condition,
            ..
        }
        | SlangArgumentType::Uniform {
            condition,
        }
        | SlangArgumentType::Specialize {
            condition,
        }) = &argument.argument_type
        else {
            continue;
        };
        let Some(condition) = condition else {
            continue;
        };
        let name = format_ident!("{}", argument.name);
        let condition = host_expression(kernel_name, gpu_type_paths, condition)?;
        let message =
            format!("{kernel_name}: argument '{name}' must be present exactly when {}", quote! { #condition });
        let check = quote! { assert_eq!(#name.is_some(), #condition, "{}", #message); };
        match argument.argument_type {
            SlangArgumentType::Specialize {
                ..
            } => specialization_conditions.push(check),
            _ => conditions.push(check),
        }
    }
    Ok((conditions, specialization_conditions))
}

/// The host preconditions as `(new, encode)` checks. `new` returns an error before anything is created, `encode`
/// asserts before recording.
fn preconditions(
    kernel_name: &str,
    kernel: &Kernel,
    info: &SlangKernel,
    gpu_types: &[&GpuTypes],
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<(Vec<TokenStream>, Vec<TokenStream>)> {
    let names = |constants: bool| {
        let parameters = kernel.parameters.iter().filter(|parameter| match parameter.ty {
            KernelParameterType::Value(_) => true,
            KernelParameterType::Type => !constants,
        });
        let parameters = parameters.map(|parameter| parameter.name.as_ref());
        let uniforms =
            kernel.arguments.iter().filter(|argument| constants && matches!(argument.ty, KernelArgumentType::Value(_)));
        parameters.chain(uniforms.map(|argument| argument.name.as_ref())).map(str::to_string).collect::<BTreeSet<_>>()
    };
    let is_variant = |path: &syn::Path| {
        let segments = path.segments.iter().map(|segment| segment.ident.to_string());
        segments.collect_tuple().is_some_and(|(head, variant): (String, String)| {
            canonical_enum(gpu_types, &head)
                .is_some_and(|candidate| candidate.variants.iter().any(|known| *known.name == *variant))
        })
    };
    let capability = |path: &syn::Path| path.segments.len() == 2 && path.segments[0].ident == "VkDeviceCapabilities";
    let is_constant = |path: &syn::Path| {
        let text = path.to_token_stream().to_string().replace(' ', "");
        gpu_types.iter().any(|types| {
            types.files.iter().any(|file| {
                file.types.iter().any(|ty| {
                    matches!(ty, GpuType::Constant(constant)
                        if text == format!("{}::{}::{}", types.module_path, file.name, constant.name))
                })
            })
        })
    };
    let mut new_preconditions = Vec::new();
    let mut encode_preconditions = Vec::new();
    for (phase, text) in &info.preconditions {
        let phase = phase.as_str();
        let text = text.as_str();
        let construction = phase == "new";
        let mut known = names(!construction);
        if construction {
            known.insert("context".to_owned());
        }
        let mut unknown = Vec::new();
        let mut parsed = syn::parse_str::<Expr>(text)
            .with_context(|| format!("{kernel_name}: malformed {phase} Precondition '{text}'"))?;
        rewrite_paths_with(&mut parsed, |path| {
            let single = path.get_ident().is_some_and(|ident| known.contains(&ident.to_string()));
            if !single && !is_variant(path) && !is_constant(path) && !(construction && capability(path)) {
                unknown.push(path.to_token_stream().to_string());
            }
            None
        });
        ensure!(unknown.is_empty(), "{kernel_name}: {phase} Precondition '{text}' names unknown {unknown:?}");
        let mut condition = host_expression(kernel_name, gpu_type_paths, text)?;
        rewrite_paths_with(&mut condition, |path| {
            capability(path).then(|| parse_quote! { crate::backends::vulkan::#path })
        });
        match phase {
            "new" => new_preconditions.push(quote! {
                if !(#condition) {
                    return Err(crate::backends::vulkan::VkKernelError::KernelPrecondition {
                        kernel: #kernel_name,
                        condition: #text,
                    }.into());
                }
            }),
            _ => {
                let message = format!("{kernel_name}: precondition {text} violated");
                encode_preconditions.push(quote! { assert!(#condition, "{}", #message); });
            },
        }
    }
    Ok((new_preconditions, encode_preconditions))
}

/// The grid, workgroup size and dispatch dimensions. Axis totals count threads and divide by the group size; Groups
/// counts are workgroups.
fn dispatch_plan(
    kernel_name: &str,
    info: &SlangKernel,
    abi: &SlangEntryPointAbi,
    gpu_type_paths: &GpuTypePaths,
) -> anyhow::Result<Dispatch> {
    let mut axes = Vec::new();
    let mut groups = Vec::new();
    let mut group_sizes = Vec::new();
    for argument in &info.arguments {
        match &argument.argument_type {
            SlangArgumentType::Axis {
                threads: total,
                threads_in_group: size,
            } => {
                axes.push(host_expression(kernel_name, gpu_type_paths, total)?);
                group_sizes.push(host_expression(kernel_name, gpu_type_paths, size)?);
            },
            SlangArgumentType::Groups {
                groups: count,
            } => groups.push(host_expression(kernel_name, gpu_type_paths, count)?),
            SlangArgumentType::Threads {
                threads: size,
            } => group_sizes.push(host_expression(kernel_name, gpu_type_paths, size)?),
            SlangArgumentType::Buffer {
                ..
            }
            | SlangArgumentType::Uniform {
                ..
            }
            | SlangArgumentType::Specialize {
                ..
            }
            | SlangArgumentType::ThreadContext => {},
        }
    }
    let axis = !axes.is_empty();
    let grid = if axis {
        axes
    } else {
        groups
    };
    while group_sizes.len() < 3 {
        group_sizes.push(parse_quote! { 1u32 });
    }
    ensure!(group_sizes.len() == 3, "{kernel_name}: workgroup has more than three dimensions");
    let counts = (0..3)
        .map(|index| match (index < grid.len(), abi.group_size[index]) {
            (true, _) if axis => {
                let size = &group_sizes[index];
                Ok(quote! { __dsl_grid[#index].div_ceil(#size) })
            },
            (true, _) => Ok(quote! { __dsl_grid[#index] }),
            (false, size) if !axis || size == 1 => Ok(quote! { 1 }),
            (false, size) => bail!("{kernel_name}: reflected group size {size} on undispatched axis {index}"),
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    // A single axis wider than AXIS_FOLD_GROUPS workgroups folds into rows of them, which the wrapper unfolds.
    let dimensions = if axis && grid.len() == 1 {
        let groups = &counts[0];
        let fold = AXIS_FOLD_GROUPS;
        quote! {{
            let __dsl_groups: u32 = #groups;
            if __dsl_groups > #fold { [#fold, __dsl_groups.div_ceil(#fold), 1] } else { [__dsl_groups, 1, 1] }
        }}
    } else {
        quote! { [#(#counts),*] }
    };
    let axis_sizes = axis.then(|| quote! { #(#group_sizes)* });
    Ok(Dispatch {
        grid,
        group_sizes,
        axis_sizes,
        dimensions,
    })
}

/// The pipeline field, its construction in `new` and its lookup in `encode`: one pipeline, or one per
/// `[[PipelineVariants]]` value with its constant in the last specialization slot.
fn pipeline_code(
    pipeline_variants: Option<&PipelineVariants>,
    specializations: &[Specialization],
    referenced: &[&Ident],
    block_size: usize,
    required_subgroup_size: &TokenStream,
) -> anyhow::Result<PipelineCode> {
    let block_size = u32::try_from(block_size)?;
    let specialization = specializations
        .iter()
        .map(|specialization| {
            let id = specialization.id;
            let value = &specialization.value;
            quote! { (#id, #value) }
        })
        .collect::<Vec<_>>();
    let key = |variant: Option<TokenStream>| {
        quote! {
            VkComputePipelineKey {
                spirv,
                entry_point,
                specialization: vec![#(#specialization,)* #variant],
                push_constant_size: #block_size,
                required_subgroup_size: #required_subgroup_size,
            }
        }
    };
    Ok(match pipeline_variants {
        None => {
            let key = key(None);
            PipelineCode {
                field: quote! { pipeline: std::sync::Arc<VkComputePipeline> },
                construction: quote! {
                    let pipeline = context.compute_pipeline(#key)?;
                    Ok(Self {
                        pipeline,
                        #(#referenced,)*
                    })
                },
                lookup: quote! { &self.pipeline },
            }
        },
        Some(PipelineVariants {
            selector,
            paths,
            id,
        }) => {
            let count = paths.len();
            let indices = 0..count;
            let keys = paths.iter().map(|path| key(Some(quote! { (#id, #path as u32) })));
            PipelineCode {
                field: quote! { pipelines: [std::sync::Arc<VkComputePipeline>; #count] },
                construction: quote! {
                    let pipelines = [#(context.compute_pipeline(#keys)?),*];
                    Ok(Self {
                        pipelines,
                        #(#referenced,)*
                    })
                },
                lookup: quote! { match #selector { #(#paths => &self.pipelines[#indices],)* } },
            }
        },
    })
}

/// The `(entry point name, SPIR-V bytes)` pair `new` builds from: the single entry point of a kernel without type
/// parameters, or the entry point the data types select.
fn select_entry(
    kernel_name: &str,
    type_parameters: &[Ident],
    first: &SlangEntryPointVariant,
    variants: &[SlangEntryPointVariant],
) -> anyhow::Result<TokenStream> {
    // A kernel without type parameters has its single entry point; the others select one by data types.
    if type_parameters.is_empty() {
        let entry_point = &first.abi.name;
        let spirv = entry_spirv(&first.spirv, &first.native_bf16);
        return Ok(quote! { (#entry_point, #spirv) });
    }
    let entry_arms = variants
        .iter()
        .map(|variant| {
            let data_types = variant.types.iter().map(|ty| data_type(ty)).collect::<Result<Vec<_>, _>>()?;
            let entry_point = &variant.abi.name;
            let spirv = entry_spirv(&variant.spirv, &variant.native_bf16);
            Ok(quote! { (#(crate::data_type::DataType::#data_types,)*) => (#entry_point, #spirv) })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(quote! {
        match (#(#type_parameters,)*) {
            #(#entry_arms,)*
            _ => return Err(crate::backends::vulkan::VkKernelError::KernelVariant {
                kernel: #kernel_name,
                data_types: Box::new([#(#type_parameters),*]),
            }.into()),
        }
    })
}

/// The kernel's common trait, implemented through the raw binding: `new` and `encode` forward to `new_fn` and
/// `encode_fn` with the specialization constants and the buffers resolved from the trait's arguments.
fn public_trait_impl(
    kernel_name: &str,
    kernel: &Kernel,
    gpu_type_paths: &GpuTypePaths,
    struct_name: &Ident,
    type_parameters: &[Ident],
    specializations: &[Specialization],
    new_fn: &Ident,
    encode_fn: &Ident,
) -> anyhow::Result<TokenStream> {
    let kernel_trait = format_ident!("{kernel_name}Kernel");
    let new_parameters = kernel
        .parameters
        .iter()
        .map(|parameter| {
            let name = format_ident!("{}", parameter.name.as_ref());
            Ok(match &parameter.ty {
                KernelParameterType::Type => quote! { #name: crate::data_type::DataType },
                KernelParameterType::Value(text) => {
                    let ty: Type = gpu_type_paths.resolve_type(text)?;
                    quote! { #name: #ty }
                },
            })
        })
        .collect::<anyhow::Result<Vec<_>>>()?;
    let mut trait_arguments = Vec::new();
    let mut conversions = Vec::new();
    let mut call_arguments = Vec::new();
    for argument in &kernel.arguments {
        let name = format_ident!("{}", argument.name.as_ref());
        let ty = match &argument.ty {
            KernelArgumentType::Buffer(access) => {
                let reference = match access {
                    KernelBufferAccess::Read => quote! { crate::backends::common::BufferRef },
                    KernelBufferAccess::ReadWrite => quote! { crate::backends::common::BufferMut },
                };
                let resolve = quote! {{
                    let (buffer, range) = #reference::parts(#name);
                    VkBuffer::resolve(buffer, range)
                }};
                conversions.push(if argument.conditional {
                    quote! { let #name = #name.map(|#name| #resolve); }
                } else {
                    quote! { let #name = #resolve; }
                });
                quote! { impl #reference<Backend = crate::backends::vulkan::Vulkan> }
            },
            KernelArgumentType::Value(text) => {
                let ty: Type = gpu_type_paths.resolve_type(text)?;
                quote! { #ty }
            },
        };
        trait_arguments.push(if argument.conditional {
            quote! { #name: Option<#ty> }
        } else {
            quote! { #name: #ty }
        });
        call_arguments.push(name);
    }
    let specialization_names = specializations.iter().map(|specialization| &specialization.name);
    Ok(quote! {
        #[allow(clippy::style, clippy::complexity, clippy::perf)]
        impl crate::backends::common::kernel::#kernel_trait for #struct_name {
            type Backend = crate::backends::vulkan::Vulkan;

            #[allow(non_snake_case)]
            fn new(context: &VkContext #(, #new_parameters)*) -> Result<Self, Error> {
                #struct_name::#new_fn(&context.arc() #(, #type_parameters)* #(, #specialization_names)*)
            }

            fn encode(&self, #(#trait_arguments,)* command_buffer: &mut VkCommandBufferEncoding) {
                #(#conversions)*
                // SAFETY: as for the CPU kernels, the common contract makes callers pass ranges that cover every
                // element the kernel indexes, aliasing only where the kernel defines it.
                unsafe { #struct_name::#encode_fn(self, #(#call_arguments,)* command_buffer) }
            }
        }
    })
}

/// Every type of the canonical GPU type files, across the modules that declare them.
fn canonical_types<'a>(gpu_types: &[&'a GpuTypes]) -> impl Iterator<Item = &'a GpuType> {
    gpu_types.iter().flat_map(|types| types.files.iter()).flat_map(|file| &file.types)
}

/// The canonical enum named `short`.
fn canonical_enum<'a>(
    gpu_types: &[&'a GpuTypes],
    short: &str,
) -> Option<&'a GpuTypeEnum> {
    canonical_types(gpu_types).find_map(|candidate| match candidate {
        GpuType::Enum(candidate) if candidate.name.as_ref() == short => Some(candidate),
        _ => None,
    })
}

/// Every entry point's ABI, each variant's native-BF16 ABI included.
fn variant_abis(variants: &[SlangEntryPointVariant]) -> impl Iterator<Item = &SlangEntryPointAbi> {
    variants.iter().flat_map(|variant| once(&variant.abi).chain(variant.native_bf16.iter().map(|(abi, _)| abi)))
}

/// The specialization id of `argument`, the constant named for it in the kernel's wrapper.
fn specialization_id(
    kernel_name: &str,
    abi: &SlangEntryPointAbi,
    argument: &str,
) -> anyhow::Result<u32> {
    let constant = specialization_name(kernel_name, argument);
    abi.specialization_ids
        .iter()
        .find_map(|(name, id)| (*name == constant).then_some(*id))
        .with_context(|| format!("{kernel_name}: no specialization constant '{constant}'"))
}

/// A host expression from its reflected text, with enum paths qualified.
fn host_expression(
    kernel_name: &str,
    gpu_type_paths: &GpuTypePaths,
    text: &str,
) -> anyhow::Result<Expr> {
    let mut expression =
        syn::parse_str::<Expr>(text).with_context(|| format!("{kernel_name}: malformed host expression '{text}'"))?;
    rewrite_paths_with(&mut expression, |path| qualify_enum_path(path, gpu_type_paths));
    Ok(expression)
}

// A 32-bit word of one of `scalars` or a canonical enum, by discriminant, as Slang lays it out; a `bool` is 0 or 1.
fn word(
    kernel_name: &str,
    gpu_type_paths: &GpuTypePaths,
    name: &Ident,
    ty: &Type,
    scalars: &[&str],
) -> anyhow::Result<TokenStream> {
    let text = ty.to_token_stream().to_string().replace(" :: ", "::");
    match text.as_str() {
        _ if canonical_enum_name(gpu_type_paths, &text).is_some() => Ok(quote! { (#name as u32) }),
        "bool" if scalars.contains(&"bool") => Ok(quote! { u32::from(#name) }),
        scalar if scalars.contains(&scalar) => Ok(quote! { #name }),
        other => bail!("{kernel_name}: unsupported type '{other}' for '{name}'"),
    }
}

// The native-BF16 SPIR-V on devices with the BF16 type.
fn entry_spirv(
    object: &str,
    native: &Option<(SlangEntryPointAbi, String)>,
) -> TokenStream {
    match native {
        Some((_, native)) => quote! {
            if context.physical_device().capabilities.contains(crate::backends::vulkan::VkDeviceCapabilities::BF16_TYPE) {
                include_bytes!(#native) as &[u8]
            } else {
                include_bytes!(#object) as &[u8]
            }
        },
        None => quote! { include_bytes!(#object) as &[u8] },
    }
}

/// The `DataType` of a Slang scalar type.
fn data_type(slang_type: &str) -> anyhow::Result<Ident> {
    slang2rust(slang_type, &HashMap::new())
        .map(|rust_type| format_ident!("{}", rust_type.to_uppercase()))
        .with_context(|| format!("no DataType for GPU type '{slang_type}'"))
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

/// `T` of an `Option<T>` type.
fn option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.iter().exactly_one().ok().filter(|segment| segment.ident == "Option")?;
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    match arguments.args.iter().exactly_one().ok()? {
        GenericArgument::Type(inner) => Some(inner),
        _ => None,
    }
}

/// The short name of `path` when it is the full path of a canonical GPU enum.
fn canonical_enum_name<'a>(
    gpu_type_paths: &GpuTypePaths,
    path: &'a str,
) -> Option<&'a str> {
    let (_, short) = path.rsplit_once("::")?;
    (gpu_type_paths.full_path_for(short) == Some(path) && gpu_type_paths.kind_for(short) == Some(GpuTypeKind::Enum))
        .then_some(short)
}

/// The first identifier among `tokens`, groups included, that satisfies `predicate`.
fn find_ident(
    tokens: TokenStream,
    predicate: &impl Fn(&Ident) -> bool,
) -> Option<Ident> {
    tokens.into_iter().find_map(|token| match token {
        TokenTree::Ident(ident) => Some(ident).filter(predicate),
        TokenTree::Group(group) => find_ident(group.stream(), predicate),
        _ => None,
    })
}
