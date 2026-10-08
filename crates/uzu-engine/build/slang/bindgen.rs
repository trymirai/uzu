use std::{
    collections::BTreeSet,
    path::{Path, PathBuf},
};

use anyhow::{Context, bail, ensure};
use itertools::Itertools;
use proc_macro2::{TokenStream, TokenTree};
use quote::{ToTokens, format_ident, quote};
use shader_slang::ScalarType;
use syn::{Expr, GenericArgument, Ident, PathArguments, Type, parse_quote};

use super::{
    Error, SlangArgumentType, SlangEntryPointAbi, SlangFieldAbi, SlangKernelInfo, wrapper::specialization_name,
};
use crate::common::{
    enum_paths::{EnumPaths, GpuTypeKind},
    expr_rewrite::rewrite_paths_with,
    gpu_types::{GpuType, GpuTypeStructFieldType, GpuTypes},
    kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType},
};

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

/// Raw typed binding of a kernel. Its constructor and `encode` follow the common `Kernel`
/// signature; the push-constant block and specialization data follow each entry point's reflected ABI. A
/// `[[PipelineVariants]]` argument gets one pipeline per value of its canonical enum, which `encode` selects.
pub fn bindgen(
    info: &SlangKernelInfo,
    kernel: &Kernel,
    variants: &[(Vec<&'static str>, SlangEntryPointAbi)],
    spirv_file: &str,
    enum_paths: &EnumPaths,
    gpu_types: &GpuTypes,
) -> Result<TokenStream, Error> {
    let kernel_name: &str = &kernel.name;
    let struct_name = format_ident!("{kernel_name}VulkanKernel");
    let (_, abi) = variants.first().context("kernel has no entry points")?;
    let layout = |abi: &SlangEntryPointAbi| {
        let fields = abi.fields.iter().map(|field| (field.name.clone(), field.offset, field.size)).collect::<Vec<_>>();
        (abi.block_size, abi.group_size, abi.specialization_ids.clone(), fields)
    };
    ensure!(variants.iter().all(|(_, other)| layout(other) == layout(abi)), "{kernel_name}: variants differ in layout");
    // Vulkan guarantees every device 128 bytes of push constants.
    ensure!(abi.block_size <= 128, "{kernel_name}: push-constant block of {} bytes exceeds 128", abi.block_size);

    let type_parameters = kernel
        .parameters
        .iter()
        .filter(|parameter| matches!(parameter.ty, KernelParameterType::Type))
        .map(|parameter| format_ident!("{}", parameter.name.as_ref()))
        .collect::<Vec<_>>();
    // A 32-bit word of one of `scalars` or a canonical enum, by discriminant, as Slang lays it out; a `bool` is 0 or 1.
    let word = |name: &Ident, ty: &Type, scalars: &[&str]| -> Result<TokenStream, Error> {
        let text = ty.to_token_stream().to_string().replace(" :: ", "::");
        let canonical_enum = text.rsplit_once("::").is_some_and(|(_, short)| {
            enum_paths.full_path_for(short) == Some(text.as_str())
                && enum_paths.kind_for(short) == Some(GpuTypeKind::Enum)
        });
        match text.as_str() {
            _ if canonical_enum => Ok(quote! { (#name as u32) }),
            "bool" if scalars.contains(&"bool") => Ok(quote! { u32::from(#name) }),
            scalar if scalars.contains(&scalar) => Ok(quote! { #name }),
            other => bail!("{kernel_name}: unsupported type '{other}' for '{name}'"),
        }
    };
    // Specializations are `bool`, `u32` or a canonical enum, which Slang declares as `uint`. An absent optional one
    // writes 0, which its presence condition keeps the kernel from reading.
    let mut specializations = Vec::new();
    for parameter in &kernel.parameters {
        let KernelParameterType::Value(text) = &parameter.ty else {
            continue;
        };
        let name = format_ident!("{}", parameter.name.as_ref());
        let ty: Type = syn::parse_str(text)?;
        let value = match option_inner(&ty) {
            _ if text.as_ref() == "bool" => quote! { vk::Bool32::from(#name) },
            Some(inner) => match syn::parse2(word(&name, inner, &["u32"])?)? {
                Expr::Paren(inner) => {
                    let inner = inner.expr;
                    quote! { #name.map_or(0, |#name| #inner) }
                },
                _ => quote! { #name.unwrap_or(0) },
            },
            None => word(&name, &ty, &["u32"])?,
        };
        let constant = specialization_name(kernel_name, &parameter.name);
        let (_, id) = abi
            .specialization_ids
            .iter()
            .find(|(name, _)| *name == constant)
            .with_context(|| format!("{kernel_name}: no specialization constant '{constant}'"))?;
        specializations.push((name, ty, value, *id));
    }

    // `(argument, variant paths, specialization constant id)` of the `[[PipelineVariants]]` argument, whose values are
    // the canonical enum's own variants, by name.
    let mut pipeline_variants = None;
    for argument in info.arguments() {
        if !argument.pipeline_variants()? {
            continue;
        }
        let name = argument.name()?;
        let Some(KernelArgumentType::Constant(text)) =
            kernel.arguments.iter().find(|candidate| candidate.name.as_ref() == name).map(|argument| &argument.ty)
        else {
            bail!("{kernel_name}: PipelineVariants argument '{name}' is not a uniform");
        };
        let ty: Type = syn::parse_str(text)?;
        let short = text.rsplit_once("::").map_or(text.as_ref(), |(_, short)| short);
        ensure!(
            enum_paths.full_path_for(short) == Some(text.as_ref())
                && enum_paths.kind_for(short) == Some(GpuTypeKind::Enum),
            "{kernel_name}: PipelineVariants argument '{name}' has type '{text}', not a canonical GPU enum"
        );
        let canonical = gpu_types
            .files
            .iter()
            .flat_map(|file| &file.types)
            .find_map(|candidate| match candidate {
                GpuType::Enum(candidate) if candidate.name.as_ref() == short => Some(candidate),
                _ => None,
            })
            .with_context(|| format!("{kernel_name}: no canonical enum '{short}'"))?;
        let paths = canonical
            .variants
            .iter()
            .map(|variant| {
                let variant = format_ident!("{}", variant.name.as_ref());
                quote! { #ty::#variant }
            })
            .collect::<Vec<_>>();
        let constant = specialization_name(kernel_name, name);
        let (_, id) = abi
            .specialization_ids
            .iter()
            .find(|(name, _)| *name == constant)
            .with_context(|| format!("{kernel_name}: no specialization constant '{constant}'"))?;
        pipeline_variants = Some((format_ident!("{name}"), paths, *id));
    }

    // Buffers of canonical structs are checked by `canonical_struct`, those of primitives by their data type's size.
    let mut strides = BTreeSet::new();
    for (_, abi) in variants {
        for field in &abi.fields {
            let buffer = kernel.arguments.iter().any(|argument| {
                argument.name.as_ref() == field.name && matches!(argument.ty, KernelArgumentType::Buffer(_))
            });
            if let (true, Some((pointee, stride, ..))) = (buffer, &field.layout)
                && (enum_paths.full_path_for(pointee).is_none() || enum_paths.kind_for(pointee).is_some())
            {
                strides.insert((data_type(pointee)?.to_string(), *stride));
            }
        }
    }
    let stride_checks = strides.iter().map(|(data_type, stride)| {
        let data_type = format_ident!("{data_type}");
        quote! { const _: () = assert!(crate::data_type::DataType::#data_type.size_in_bytes() == #stride); }
    });

    ensure!(
        kernel.arguments.len() == abi.fields.len() + usize::from(pipeline_variants.is_some()),
        "{kernel_name}: arguments and reflected fields differ"
    );
    // The canonical struct `field` holds by value, slices or addresses, with its path, type, stride and alignment; `None`
    // for other fields. Its fields must be the canonical struct's, by name, of the reflected scalar type and array shape,
    // the same in every variant, and the canonical Rust type must have Slang's reflected layout, field by field. `bool`,
    // 1 byte in Rust and 4 in Slang, has no shared one.
    let mut layout_checks = Vec::new();
    let mut canonical_struct = |field: &SlangFieldAbi| -> Result<Option<(&str, Type, usize, usize)>, Error> {
        let Some((pointee, stride, alignment, fields)) = &field.layout else {
            return Ok(None);
        };
        let Some(path) = enum_paths.full_path_for(pointee).filter(|_| enum_paths.kind_for(pointee).is_none()) else {
            return Ok(None);
        };
        let name = &field.name;
        ensure!(
            variants.iter().all(|(_, other)| other.fields.iter().any(|other| other == field)),
            "{kernel_name}: variants differ in the layout of '{name}'"
        );
        let canonical = gpu_types
            .files
            .iter()
            .flat_map(|file| &file.types)
            .find_map(|candidate| match candidate {
                GpuType::Struct(candidate) if candidate.name.as_ref() == pointee => Some(candidate),
                _ => None,
            })
            .with_context(|| format!("{kernel_name}: no canonical struct '{pointee}'"))?;
        ensure!(
            canonical.fields.iter().map(|canonical| canonical.name.as_ref()).eq(fields.iter().map(|(name, ..)| name)),
            "{kernel_name}: '{pointee}' fields differ from the canonical struct"
        );
        let scalar = |text: &str| match text {
            "u32" => Some(ScalarType::Uint32),
            "f32" => Some(ScalarType::Float32),
            _ => None,
        };
        for (canonical, (field_name, _, _, reflected, array)) in canonical.fields.iter().zip(fields) {
            let (element, length) = match &canonical.ty {
                GpuTypeStructFieldType::Scalar(element) => (element, None),
                GpuTypeStructFieldType::Array {
                    element,
                    length,
                } => (element, Some((*length, 4))),
            };
            ensure!(
                scalar(element) == Some(*reflected) && *array == length,
                "{kernel_name}: field {pointee}.{field_name} is {element} {length:?} in Rust but {reflected:?} {array:?} \
                 in Slang"
            );
        }
        let ty: Type = syn::parse_str(path)?;
        let message = format!("{kernel_name}: {path} does not have the layout Slang reflects");
        layout_checks.push(quote! {
            const _: () = assert!(size_of::<#ty>() == #stride && align_of::<#ty>() == #alignment, #message);
        });
        for (canonical, (field_name, offset, size, ..)) in canonical.fields.iter().zip(fields) {
            let field_type: Type = match &canonical.ty {
                GpuTypeStructFieldType::Scalar(element) => syn::parse_str(element)?,
                GpuTypeStructFieldType::Array {
                    element,
                    length,
                } => syn::parse_str(&format!("[{element}; {length}]"))?,
            };
            let field_name = format_ident!("{field_name}");
            layout_checks.push(quote! {
                const _: () = assert!(
                    std::mem::offset_of!(#ty, #field_name) == #offset && size_of::<#field_type>() == #size,
                    #message
                );
            });
        }
        Ok(Some((path, ty, *stride, *alignment)))
    };
    let mut encode_arguments = Vec::new();
    let mut packing = Vec::new();
    let mut reads = Vec::new();
    let mut writes = Vec::new();
    for argument in &kernel.arguments {
        let name = format_ident!("{}", argument.name.as_ref());
        // The pipeline-selecting argument has no push-constant field.
        if let (Some((selector, ..)), KernelArgumentType::Constant(text)) = (&pipeline_variants, &argument.ty)
            && *selector == name
        {
            let ty: Type = syn::parse_str(text)?;
            encode_arguments.push(quote! { #name: #ty });
            continue;
        }
        let field = abi
            .fields
            .iter()
            .find(|field| field.name == argument.name.as_ref())
            .with_context(|| format!("{kernel_name}: no reflected field for '{name}'"))?;
        let (start, end) = (field.offset, field.offset + field.size);
        let (argument_type, bytes) = match (&argument.ty, canonical_struct(field)?) {
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
            (KernelArgumentType::Constant(text), Some((path, ty, stride, _))) if text.as_ref() == path => {
                ensure!(field.size == stride, "{kernel_name}: '{name}' of {} bytes is not one {path}", field.size);
                let cast = quote! { bytemuck::cast::<#ty, [u8; #stride]> };
                let bytes = if argument.conditional {
                    quote! { #name.map_or([0; #stride], #cast) }
                } else {
                    quote! { #cast(#name) }
                };
                (quote! { #ty }, bytes)
            },
            // A `[[HostSlice]]`: the slice is copied into memory of the command buffer once the dispatch is known to
            // record, and the shader reads it through its address; an empty slice passes address 0 and reads nothing.
            (KernelArgumentType::Constant(text), Some((path, ty, _, alignment))) => {
                ensure!(
                    field.size == 8 && !argument.conditional && text.as_ref() == format!("&[{path}]"),
                    "{kernel_name}: '{name}' of type '{text}' is not a device address of a canonical '{path}' slice"
                );
                // The encoder's upload allocator aligns ranges to at most 64 bytes.
                ensure!(alignment <= 64, "{kernel_name}: '{path}' needs alignment {alignment} above 64");
                let message = format!("{kernel_name} upload of {name}");
                packing.push(quote! { let #name = command_buffer.upload(#name).expect(#message); });
                reads.push(quote! { #name.as_ref().map(|(buffer, range)| (buffer, range.clone())) });
                let address =
                    quote! { #name.as_ref().map_or(0, |(buffer, range)| buffer.device_address() + range.start) };
                (quote! { &[#ty] }, quote! { #address.to_ne_bytes() })
            },
            (KernelArgumentType::Constant(text), None) => {
                ensure!(field.size == 4 && field.layout.is_none(), "{kernel_name}: '{name}' is not a 4-byte {text}");
                let ty: Type = syn::parse_str(text)?;
                let value = word(&name, &ty, &["u32", "i32", "f32", "bool"])?;
                // An absent optional constant leaves its bytes zero.
                let bytes = if argument.conditional {
                    quote! { #name.map_or([0; 4], |#name| #value.to_ne_bytes()) }
                } else {
                    quote! { #value.to_ne_bytes() }
                };
                (quote! { #ty }, bytes)
            },
        };
        encode_arguments.push(if argument.conditional {
            quote! { #name: Option<#argument_type> }
        } else {
            quote! { #name: #argument_type }
        });
        packing.push(quote! { __dsl_block[#start..#end].copy_from_slice(&#bytes); });
    }

    // Optional specializations are checked by the constructor, other optional arguments by `encode`.
    let (mut conditions, mut specialization_conditions) = (Vec::new(), Vec::new());
    let (mut axes, mut groups) = (Vec::new(), Vec::new());
    let host_expression = |text: &str| {
        let mut expression = syn::parse_str::<Expr>(text)
            .with_context(|| format!("{kernel_name}: malformed host expression '{text}'"))?;
        // `Enum::Variant` paths name canonical GPU types.
        rewrite_paths_with(&mut expression, |path| {
            let (head, variant) = (path.segments.first()?, path.segments.iter().skip(1));
            let canonical: syn::Path = syn::parse_str(enum_paths.full_path_for(&head.ident.to_string())?).ok()?;
            (path.segments.len() > 1).then(|| parse_quote! { #canonical #(:: #variant)* })
        });
        Ok::<_, Error>(expression)
    };
    for argument in info.arguments() {
        let argument_type = argument.argument_type()?;
        match &argument_type {
            SlangArgumentType::Axis(total, _) => axes.push(host_expression(total)?),
            SlangArgumentType::Groups(count) => groups.push(host_expression(count)?),
            _ => {},
        }
        if let Some(condition) = argument.condition()? {
            let name = format_ident!("{}", argument.name()?);
            let condition = host_expression(condition)?;
            let message =
                format!("{kernel_name}: argument '{name}' must be present exactly when {}", quote! { #condition });
            let check = quote! { assert_eq!(#name.is_some(), #condition, "{}", #message); };
            match argument_type {
                SlangArgumentType::Specialize(_) => specialization_conditions.push(check),
                _ => conditions.push(check),
            }
        }
    }
    // Preconditions: the constructor's over specializations return an error before anything is created; `encode`'s,
    // also over uniforms, assert after the presence checks and before anything is recorded. Single names must be
    // specializations, for `new` also type parameters and for `encode` uniforms; longer paths variants of canonical GPU
    // enums.
    let names = |constants: bool| {
        let parameters = kernel.parameters.iter().filter(|parameter| match parameter.ty {
            KernelParameterType::Value(_) => true,
            KernelParameterType::Type => !constants,
        });
        let parameters = parameters.map(|parameter| parameter.name.as_ref());
        let uniforms = kernel
            .arguments
            .iter()
            .filter(|argument| constants && matches!(argument.ty, KernelArgumentType::Constant(_)));
        parameters.chain(uniforms.map(|argument| argument.name.as_ref())).map(str::to_string).collect::<BTreeSet<_>>()
    };
    let is_variant = |path: &syn::Path| {
        let [head, variant] = [0, 1].map(|index| path.segments.get(index).map(|segment| segment.ident.to_string()));
        gpu_types.files.iter().flat_map(|file| &file.types).any(|candidate| match candidate {
            GpuType::Enum(candidate) => {
                path.segments.len() == 2
                    && Some(candidate.name.as_ref()) == head.as_deref()
                    && candidate.variants.iter().any(|known| Some(known.name.as_ref()) == variant.as_deref())
            },
            _ => false,
        })
    };
    let (mut new_preconditions, mut encode_preconditions) = (Vec::new(), Vec::new());
    for (phase, text) in info.preconditions()? {
        let known = names(phase == "encode");
        let mut unknown = Vec::new();
        let mut parsed = syn::parse_str::<Expr>(text)
            .with_context(|| format!("{kernel_name}: malformed {phase} Precondition '{text}'"))?;
        rewrite_paths_with(&mut parsed, |path| {
            let single = path.get_ident().is_some_and(|ident| known.contains(&ident.to_string()));
            if !single && !is_variant(path) {
                unknown.push(path.to_token_stream().to_string());
            }
            None
        });
        ensure!(unknown.is_empty(), "{kernel_name}: {phase} Precondition '{text}' names unknown {unknown:?}");
        let condition = host_expression(text)?;
        match phase {
            "new" => new_preconditions.push(quote! {
                if !(#condition) {
                    return Err(Error::KernelPrecondition { kernel: #kernel_name, condition: #text });
                }
            }),
            _ => {
                let message = format!("{kernel_name}: precondition {text} violated");
                encode_preconditions.push(quote! { assert!(#condition, "{}", #message); });
            },
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

    let host_expressions = quote! { #(#conditions)* #(#encode_preconditions)* #(#grid)* };
    let (referenced, referenced_types): (Vec<_>, Vec<_>) = specializations
        .iter()
        .filter(|(name, ..)| references(host_expressions.clone(), name))
        .map(|(name, ty, ..)| (name, ty))
        .unzip();

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
    let (specialization_names, specialization_types, specialization_values) =
        specializations.iter().map(|(name, ty, value, _)| (name, ty, value)).multiunzip::<(Vec<_>, Vec<_>, Vec<_>)>();
    let specialization_entries = specializations.iter().enumerate().map(|(index, (.., id))| {
        let offset = 4 * index as u32;
        quote! { vk::SpecializationMapEntry::default().constant_id(#id).offset(#offset).size(4) }
    });
    let block_size = abi.block_size;
    let block_size_u32 = u32::try_from(block_size)?;
    // One pipeline, or one per `[[PipelineVariants]]` value from the same shader module, its constant in the last
    // specialization slot; `encode` selects by an exhaustive match on the canonical variants.
    let (pipeline_field, pipelines, pipeline) = match &pipeline_variants {
        None => (
            quote! { pipeline: std::sync::Arc<VkComputePipeline> },
            quote! {
                let specialization_data: [[u8; 4]; #specialization_count] =
                    [#(#specialization_values.to_ne_bytes()),*];
                let specialization_entries: [vk::SpecializationMapEntry; #specialization_count] = [#(#specialization_entries),*];
                let specialization =
                    vk::SpecializationInfo::default().map_entries(&specialization_entries).data(specialization_data.as_flattened());
                let pipeline =
                    VkComputePipeline::new(context.clone(), shader.module(), &[], #block_size_u32, entry_point, &specialization)?;
                Ok(Self {
                    pipeline: std::sync::Arc::new(pipeline),
                    #(#referenced,)*
                })
            },
            quote! { &self.pipeline },
        ),
        Some((selector, paths, id)) => {
            let (count, offset, indices) = (paths.len(), 4 * specialization_count as u32, 0..paths.len());
            (
                quote! { pipelines: [std::sync::Arc<VkComputePipeline>; #count] },
                quote! {
                    let mut specialization_data: [[u8; 4]; #specialization_count + 1] =
                        [#(#specialization_values.to_ne_bytes(),)* [0; 4]];
                    let specialization_entries: [vk::SpecializationMapEntry; #specialization_count + 1] = [
                        #(#specialization_entries,)*
                        vk::SpecializationMapEntry::default().constant_id(#id).offset(#offset).size(4)
                    ];
                    let pipelines = [#({
                        specialization_data[#specialization_count] = (#paths as u32).to_ne_bytes();
                        let specialization = vk::SpecializationInfo::default()
                            .map_entries(&specialization_entries)
                            .data(specialization_data.as_flattened());
                        std::sync::Arc::new(VkComputePipeline::new(
                            context.clone(), shader.module(), &[], #block_size_u32, entry_point, &specialization,
                        )?)
                    }),*];
                    Ok(Self {
                        pipelines,
                        #(#referenced,)*
                    })
                },
                quote! { match #selector { #(#paths => &self.pipelines[#indices],)* } },
            )
        },
    };
    let dispatch_message = format!("{kernel_name} dispatch");
    let (read_count, write_count) = (reads.len(), writes.len());

    Ok(quote! {
        use ash::vk;

        use crate::backends::vulkan::{Error, VkBuffer, VkCommandBufferEncoding, VkComputePipeline, VkContext, VkShader};

        const SPIRV: &[u8] = include_bytes!(#spirv_file);

        #(#stride_checks)*
        #(#layout_checks)*

        pub struct #struct_name {
            #pipeline_field,
            #(#referenced: #referenced_types,)*
        }

        impl #struct_name {
            #[allow(non_snake_case, clippy::too_many_arguments)]
            pub fn new(
                context: &std::sync::Arc<VkContext>
                #(, #type_parameters: crate::data_type::DataType)*
                #(, #specialization_names: #specialization_types)*
            ) -> Result<Self, Error> {
                #(#specialization_conditions)*
                #(#new_preconditions)*
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
                #pipelines
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
                        #pipeline,
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
            "uint8_t" => "U8",
            "int8_t" => "I8",
            other => bail!("no DataType for Slang type '{other}'"),
        }
    ))
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
