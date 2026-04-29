use std::{
    collections::{HashMap, hash_map::Entry},
    path::Path,
};

use anyhow::{Context, bail, ensure};
use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use shader_slang::{
    ComponentType, LayoutRules, ParameterCategory, TypeKind,
    reflection::{Shader, TypeLayout},
};
use syn::{Expr, Ident, Lifetime, Type};

use crate::{
    common::{kernel::KernelParameterType, mangling::static_mangle},
    debug_log,
    slang::{SlangArgumentType, SlangBufferAccess, SlangKernel, SlangParameterType, Specializer, slang2rust},
};

pub fn bindgen_file(
    linked_component: &ComponentType,
    kernels: &[SlangKernel],
    gpu_type_map: &HashMap<String, String>,
    object_file: &Path,
) -> anyhow::Result<TokenStream> {
    let shader = linked_component.layout(0).context("cannot get layout")?;
    let object_file = object_file.to_str().unwrap();
    let object_constant = format_ident!("WGSL_{}", blake3::hash(object_file.as_bytes()).to_hex().to_uppercase());

    let generated_kernels = kernels
        .iter()
        .map(|kernel| bindgen_kernel(linked_component, shader, kernel, gpu_type_map, &object_constant))
        .collect::<anyhow::Result<Vec<TokenStream>>>()?;

    Ok(quote! {
        const #object_constant: &str = include_str!(#object_file);

        #(#generated_kernels)*
    })
}

fn bindgen_kernel(
    linked_component: &ComponentType,
    shader: &Shader,
    kernel: &SlangKernel,
    gpu_type_map: &HashMap<String, String>,
    object_constant: &Ident,
) -> anyhow::Result<TokenStream> {
    let kernel_trait_name = format_ident!("{}Kernel", kernel.name.as_str());
    let kernel_struct_name = format_ident!("{}WebGPUKernel", kernel.name.as_str());

    let globals = shader.global_params_type_layout().unwrap();

    struct SpecializationConstant {
        index: usize,
        name: Ident,
    }

    let (
        kernel_new_arguments_definitions,
        wrapper_new_arguments_definitions,
        kernel_new_match_arguments,
        kernel_wrapper_new_arguments,
        specialization_constants,
    ) = kernel
        .parameters
        .iter()
        .map(|parameter| match &parameter.ty {
            SlangParameterType::Type {
                variants: _,
            } => {
                let name = format_ident!("{}", parameter.name.as_str());
                Ok(Some((
                    quote! { #[allow(non_snake_case)] #name: crate::DataType },
                    None,
                    Some(quote! { #name }),
                    None,
                    None,
                )))
            },
            SlangParameterType::Value {
                value_type,
                variants: _,
            } => {
                let name = format_ident!("{}", parameter.name.as_str());

                let ty: Type = slang2rust(value_type, gpu_type_map)
                    .with_context(|| format!("cannot convert to rust the type of parameter {}", parameter.name))
                    .and_then(|ty| {
                        syn::parse_str(&ty)
                            .with_context(|| format!("cannot parse the type of parameter {}", parameter.name))
                    })?;

                Ok(Some((quote! { #[allow(non_snake_case)] #name: #ty }, None, Some(quote! { #name }), None, None)))
            },
            SlangParameterType::GroupShared {
                ..
            } => Ok(None),
        })
        .chain(kernel.arguments.iter().map(|argument| {
            if let SlangArgumentType::Specialize = argument.argument_type {
                let slang_name = format!("{}_specialize_{}", kernel.name, argument.name.as_str());
                let wgsl_idx = globals
                    .field_by_index(globals.find_field_index_by_name(&slang_name) as u32)
                    .unwrap()
                    .offset(ParameterCategory::SpecializationConstant);
                let name = format_ident!("{}", argument.name.as_str());
                let ty: Type = argument.rust_type(gpu_type_map).and_then(|ty| {
                    syn::parse_str(&ty).with_context(|| format!("cannot parse the type of argument {}", argument.name))
                })?;
                Ok(Some((
                    quote! { #name: #ty },
                    Some(quote! { #[allow(unused)] #name: #ty }),
                    None,
                    Some(quote! { #name }),
                    Some(SpecializationConstant {
                        index: wgsl_idx,
                        name,
                    }),
                )))
            } else {
                Ok(None)
            }
        }))
        .filter_map(|x| x.transpose())
        .collect::<anyhow::Result<(
            Vec<TokenStream>,
            Vec<Option<TokenStream>>,
            Vec<Option<TokenStream>>,
            Vec<Option<TokenStream>>,
            Vec<Option<SpecializationConstant>>,
        )>>()?;
    let wrapper_new_arguments_definitions =
        wrapper_new_arguments_definitions.into_iter().flatten().collect::<Vec<TokenStream>>();
    let kernel_new_match_arguments = kernel_new_match_arguments.into_iter().flatten().collect::<Vec<TokenStream>>();
    let kernel_wrapper_new_arguments = kernel_wrapper_new_arguments.into_iter().flatten().collect::<Vec<TokenStream>>();
    let specialization_constants = specialization_constants.into_iter().flatten().collect::<Vec<_>>();

    let mut indirect_used = false;

    let (encode_generics, kernel_encode_arguments_definitions, wrapper_encode_arguments_definitions, kernel_wrapper_call_arguments) = kernel
        .arguments
        .iter()
        .map(|argument| {
            match &argument.argument_type {
                SlangArgumentType::Buffer {
                    access_type: _,
                    condition,
                }
                | SlangArgumentType::Constant {
                    condition,
                } => {
                    let argument_name = format_ident!("{}", argument.name);

                    let (generic, mut kernel_argument_type, mut wrapper_argument_type, mut kernel_wrapper_call_argument) = match &argument.argument_type {
                        SlangArgumentType::Buffer {
                            access_type:
                                access_type @ (SlangBufferAccess::Read {
                                    is_constant: false,
                                }
                                | SlangBufferAccess::ReadWrite),
                            condition: _,
                        } => {
                            let buffer_lifetime = Lifetime::new(&format!("'{}", argument.name.as_str()), Span::call_site());

                            let buffer_arg_type = match access_type {
                                SlangBufferAccess::Read { .. } => quote! { BufferArg },
                                SlangBufferAccess::ReadWrite => quote! { BufferArgMut },
                            };

                            (
                                Some(quote! { #buffer_lifetime }),
                                quote! { impl crate::backends::common::kernel::#buffer_arg_type<#buffer_lifetime, crate::backends::webgpu::buffer::WebGPUBuffer> },
                                quote! { (&#buffer_lifetime crate::backends::webgpu::buffer::WebGPUBuffer, usize, usize) },
                                quote! { #argument_name.into_parts() },
                            )
                        },
                        SlangArgumentType::Constant {
                            condition: _,
                        }
                        | SlangArgumentType::Buffer {
                            access_type:
                                SlangBufferAccess::Read {
                                    is_constant: true,
                                },
                            condition: _,
                        } => {
                            let ty: Type = argument.rust_type(gpu_type_map).and_then(|ty| {
                                syn::parse_str(&ty).with_context(|| format!("cannot parse the type of argument {}", argument.name))
                            })?;

                            (None, quote! { #ty }, quote! { #ty }, quote! { #argument_name })
                        },
                        _ => unreachable!(),
                    };

                    if let Some(_condition) = condition {
                        kernel_argument_type = quote! { Option<#kernel_argument_type> };
                        wrapper_argument_type = quote! { Option<#wrapper_argument_type> };
                        kernel_wrapper_call_argument = quote! { #argument_name.map(|#argument_name| #kernel_wrapper_call_argument) };
                    }

                    Ok(Some((
                        generic,
                        quote! { #argument_name: #kernel_argument_type },
                        quote! { #[allow(unused)] #argument_name: #wrapper_argument_type },
                        quote! { #kernel_wrapper_call_argument },
                    )))
                },
                SlangArgumentType::Groups { groups } if !indirect_used && groups == "INDIRECT" => {
                    indirect_used = true;

                    Ok(Some((
                        Some(quote! { '__dsl_indirect_dispatch_buffer }),
                        quote! { __dsl_indirect_dispatch_buffer: impl crate::backends::common::kernel::BufferArg<'__dsl_indirect_dispatch_buffer, crate::backends::webgpu::buffer::WebGPUBuffer> },
                        quote! { #[allow(unused)] __dsl_indirect_dispatch_buffer: (&'__dsl_indirect_dispatch_buffer crate::backends::webgpu::buffer::WebGPUBuffer, usize, usize) },
                        quote! { __dsl_indirect_dispatch_buffer.into_parts() },
                    )))
                },
                _ => Ok(None)
            }
        })
        .filter_map(|x| x.transpose())
        .collect::<anyhow::Result<(Vec<Option<TokenStream>>, Vec<TokenStream>, Vec<TokenStream>, Vec<TokenStream>)>>()?;

    let encode_generics: Vec<TokenStream> = encode_generics.into_iter().flatten().collect();

    let wrapper_trait_name = format_ident!("{}KernelTrait", kernel.name.as_str());

    let mut wrappers = Vec::new();
    let mut kernel_new_match_arms: Vec<TokenStream> = Vec::new();

    for variant in kernel.variants() {
        let specializer = Specializer::new(&variant);
        let mangled_name = static_mangle(&kernel.name, variant.iter().map(|(_k, v)| *v));
        let wrapper_struct_name = format_ident!("{}KernelWrapper{}", kernel.name.as_str(), mangled_name);

        let (entry_point_index, entry_point) = shader
            .entry_points()
            .enumerate()
            .find(|(_, entry_point)| entry_point.name() == Some(mangled_name.as_str()))
            .with_context(|| format!("cannot get entry point {mangled_name}"))?;
        let metadata = linked_component
            .entry_point_metadata(entry_point_index as i64, 0)
            .with_context(|| format!("cannot get metadata for entry point {mangled_name}"))?;
        let binding_base = entry_point.var_layout().unwrap().binding_index() as usize;

        enum BindGroupEntryBindingType {
            StorageBuffer {
                read_only: bool,
            },
            UniformBuffer,
        }

        #[derive(Debug)]
        pub enum SourceTypeKind {
            Struct {
                fields: Vec<(String, usize, SourceType)>,
            },
            Array {
                element_type: Box<SourceType>,
                element_stride: usize,
                length: usize,
            },
            Scalar,
        }

        #[derive(Debug)]
        pub struct SourceType {
            kind: SourceTypeKind,
            size: usize,
        }

        impl SourceType {
            fn from_type_layout(type_layout: &TypeLayout) -> anyhow::Result<Self> {
                let type_categories = type_layout.categories().collect::<Vec<_>>();
                ensure!(&type_categories == &[ParameterCategory::Uniform]);

                let kind = match type_layout.kind() {
                    TypeKind::Struct => {
                        let fields = type_layout
                            .fields()
                            .map(|field| {
                                Ok((
                                    field.name().unwrap().to_string(),
                                    field.offset(ParameterCategory::Uniform),
                                    Self::from_type_layout(field.type_layout().unwrap())?,
                                ))
                            })
                            .collect::<anyhow::Result<_>>()?;

                        SourceTypeKind::Struct {
                            fields,
                        }
                    },
                    TypeKind::Array => {
                        let element_type = Self::from_type_layout(type_layout.element_type_layout().unwrap())?;

                        SourceTypeKind::Array {
                            element_type: Box::new(element_type),
                            element_stride: type_layout.element_stride(ParameterCategory::Uniform),
                            length: type_layout.element_count().unwrap(),
                        }
                    },
                    TypeKind::Scalar => SourceTypeKind::Scalar,
                    type_layout_kind => bail!("unexpected type layout kind: {type_layout_kind:?}"),
                };

                let size = type_layout.size(ParameterCategory::Uniform);

                Ok(Self {
                    kind,
                    size,
                })
            }

            fn generate_write(
                &self,
                destination: TokenStream,
                source: TokenStream,
            ) -> TokenStream {
                self.generate_write_with_depth(destination, source, 0)
            }

            fn generate_write_with_depth(
                &self,
                destination: TokenStream,
                source: TokenStream,
                depth: usize,
            ) -> TokenStream {
                match &self.kind {
                    SourceTypeKind::Struct {
                        fields,
                    } => {
                        let writes = fields.iter().map(|(field_name, field_offset, field_type)| {
                            let field_ident = format_ident!("{}", field_name);
                            field_type.generate_write_with_depth(
                                quote! { #destination[#field_offset..] },
                                quote! { #source.#field_ident },
                                depth,
                            )
                        });

                        quote! { #(#writes)* }
                    },
                    SourceTypeKind::Array {
                        element_type,
                        element_stride,
                        length,
                    } => {
                        let index_ident = format_ident!("__dsl_index_{depth}");
                        let next_depth = depth + 1;
                        let element_write = element_type.generate_write_with_depth(
                            quote! { #destination[#index_ident * #element_stride..] },
                            quote! { #source[#index_ident] },
                            next_depth,
                        );

                        quote! {
                            for #index_ident in 0..#length {
                                #element_write
                            }
                        }
                    },
                    SourceTypeKind::Scalar => {
                        let size = self.size;

                        quote! {
                            {
                                let __dsl_source = &(#source);
                                let __dsl_source_bytes = unsafe {
                                    std::slice::from_raw_parts(
                                        __dsl_source as *const _ as *const u8,
                                        std::mem::size_of_val(__dsl_source),
                                    )
                                };
                                debug_assert!(__dsl_source_bytes.len() <= #size);
                                #destination[..__dsl_source_bytes.len()].copy_from_slice(__dsl_source_bytes);
                            }
                        }
                    },
                }
            }
        }

        enum BindGroupEntrySourceType {
            Buffer {
                conditional: bool,
            },
            Constant {
                buffer_size: usize,
                fields: Vec<(usize, String, bool, SourceType)>,
            },
            Slice {
                element_type: SourceType,
            },
        }

        struct BindGroupEntry {
            buffer_name: String,
            binding_type: BindGroupEntryBindingType,
            source_type: BindGroupEntrySourceType,
        }

        let mut dsl_uniform_index = 0;
        let mut bind_group_entries = HashMap::new();

        for parameter in entry_point.parameters() {
            let parameter_name = parameter.name().unwrap();
            let parameter_category = parameter.category().unwrap();

            let kernel_argument = kernel.arguments.iter().find(|argument| &argument.name == parameter_name);

            match parameter_category {
                ParameterCategory::DescriptorTableSlot => {
                    let SlangArgumentType::Buffer {
                        access_type,
                        condition,
                    } = &kernel_argument.unwrap().argument_type
                    else {
                        unreachable!();
                    };

                    let binding = binding_base + parameter.offset(ParameterCategory::DescriptorTableSlot);

                    let Entry::Vacant(vacant) = bind_group_entries.entry(binding as u32) else {
                        bail!("binding {binding} repeated")
                    };

                    let buffer_name = parameter_name.to_string();

                    let read_only = matches!(access_type, SlangBufferAccess::Read { .. });

                    let source_type = if matches!(
                        access_type,
                        SlangBufferAccess::Read {
                            is_constant: true
                        }
                    ) {
                        let element_type = parameter.type_layout().unwrap().resource_result_type().unwrap();
                        let element_type_layout =
                            shader.type_layout(element_type, LayoutRules::DefaultStructuredBuffer).unwrap();

                        BindGroupEntrySourceType::Slice {
                            element_type: SourceType::from_type_layout(element_type_layout)?,
                        }
                    } else {
                        BindGroupEntrySourceType::Buffer {
                            conditional: condition.is_some(),
                        }
                    };

                    vacant.insert(BindGroupEntry {
                        buffer_name,
                        binding_type: BindGroupEntryBindingType::StorageBuffer {
                            read_only,
                        },
                        source_type,
                    });
                },
                ParameterCategory::Uniform => {
                    let SlangArgumentType::Constant {
                        condition,
                    } = &kernel_argument.unwrap().argument_type
                    else {
                        unreachable!();
                    };

                    let binding = binding_base + parameter.offset(ParameterCategory::DescriptorTableSlot);
                    let offset = parameter.offset(ParameterCategory::Uniform);

                    let source_type = SourceType::from_type_layout(parameter.type_layout().unwrap())?;
                    let field_buffer_size = (offset + source_type.size).next_multiple_of(16);

                    match bind_group_entries.entry(binding as u32) {
                        Entry::Occupied(mut occupied) => {
                            if !matches!(occupied.get().binding_type, BindGroupEntryBindingType::UniformBuffer) {
                                bail!("binding {binding} repeated as uniform and non-uniform")
                            }
                            let BindGroupEntrySourceType::Constant {
                                buffer_size,
                                fields,
                            } = &mut occupied.get_mut().source_type
                            else {
                                bail!("binding {binding} repeated as buffer and constant")
                            };
                            *buffer_size = usize::max(*buffer_size, field_buffer_size);
                            fields.push((offset, parameter_name.to_string(), condition.is_some(), source_type))
                        },
                        Entry::Vacant(vacant) => {
                            let buffer_name = format!("__dsl_uniform_{}", dsl_uniform_index);
                            dsl_uniform_index += 1;

                            vacant.insert(BindGroupEntry {
                                buffer_name,
                                binding_type: BindGroupEntryBindingType::UniformBuffer,
                                source_type: BindGroupEntrySourceType::Constant {
                                    buffer_size: field_buffer_size,
                                    fields: vec![(
                                        offset,
                                        parameter_name.to_string(),
                                        condition.is_some(),
                                        source_type,
                                    )],
                                },
                            });
                        },
                    }
                },
                ParameterCategory::None => (),
                parameter_category => panic!("unsupported parameter category: {parameter_category:?}"),
            }
        }

        let mut bind_group_entries = bind_group_entries.into_iter().collect::<Vec<_>>();
        bind_group_entries.sort_by_key(|(idx, _)| *idx);

        let bind_group_layout_entries = bind_group_entries.iter().map(|(binding, entry)| {
            let buffer_binding_type = match entry.binding_type {
                BindGroupEntryBindingType::StorageBuffer {
                    read_only,
                } => {
                    quote! { wgpu::BufferBindingType::Storage { read_only: #read_only } }
                },
                BindGroupEntryBindingType::UniformBuffer => quote! { wgpu::BufferBindingType::Uniform },
            };

            quote! {
                wgpu::BindGroupLayoutEntry {
                    binding: #binding,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: #buffer_binding_type,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }
            }
        });

        let (prologue, bind_group_entries) = bind_group_entries.iter().map(|(binding, entry)| {
            let buffer_name_ident = format_ident!("{}", entry.buffer_name);

            let prologue = match &entry.source_type {
                BindGroupEntrySourceType::Buffer { conditional: true } => {
                    let (buffer_maybe_mut, buffer_arg_type, buffer_ref_type) = if matches!(entry.binding_type, BindGroupEntryBindingType::StorageBuffer { read_only: false }) {
                        (quote!{ mut }, quote! { BufferArgMut }, quote! { as_mut })
                    } else {
                        (quote! {}, quote! { BufferArg }, quote! { as_ref })
                    };

                    Some(quote! {
                        let #buffer_maybe_mut __dsl_dummy_allocation = #buffer_name_ident.is_none().then(|| encoder.allocate_scratch(256).unwrap());
                        let #buffer_name_ident = #buffer_name_ident.unwrap_or_else(|| {
                            use crate::backends::common::kernel::#buffer_arg_type;

                            (__dsl_dummy_allocation.#buffer_ref_type().unwrap()).into_parts()
                        });
                    })
                }
                BindGroupEntrySourceType::Constant { buffer_size, fields } => {
                    let field_writes = fields.iter().map(|(offset, name, conditional, source_type)| {
                        let name = format_ident!("{}", name);
                        let write = source_type.generate_write(quote! { buffer_slice[#offset..] }, quote! { #name });

                        if *conditional {
                            quote! {
                                if let Some(#name) = #name {
                                    #write
                                }
                            }
                        } else {
                            write
                        }
                    });

                    Some(quote! {
                        let __dsl_dummy_allocation = encoder.allocate_constant(#buffer_size).unwrap();
                        let #buffer_name_ident = {
                            use crate::backends::common::{kernel::BufferArg, Buffer};

                            let (buffer, offset, size) = __dsl_dummy_allocation.into_parts();
                            let buffer_slice = unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, size) };

                            buffer_slice.fill(0);
                            #(#field_writes)*

                            (buffer, offset, size)
                        };
                    })
                }
                BindGroupEntrySourceType::Slice { element_type } => {
                    let element_size = element_type.size;
                    let element_write = element_type.generate_write(
                        quote! { buffer_slice[__dsl_offset..] },
                        quote! { #buffer_name_ident[__dsl_index] },
                    );

                    Some(quote! {
                        let __dsl_dummy_allocation = encoder.allocate_constant(#element_size * #buffer_name_ident.len()).unwrap();
                        let #buffer_name_ident = {
                            use crate::backends::common::{kernel::BufferArg, Buffer};

                            let (buffer, offset, size) = __dsl_dummy_allocation.into_parts();
                            let buffer_slice = unsafe { std::slice::from_raw_parts_mut(buffer.cpu_ptr().as_ptr() as *mut u8, size) };

                            buffer_slice.fill(0);
                            for __dsl_index in 0..#buffer_name_ident.len() {
                                let __dsl_offset = __dsl_index * #element_size;
                                #element_write
                            }

                            (buffer, offset, size)
                        };
                    })
                }
                _ => None,
            };

            (
                prologue,
                quote! {
                    wgpu::BindGroupEntry {
                        binding: #binding,
                        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &#buffer_name_ident.0.buffer, offset: #buffer_name_ident.1 as u64, size: Some(std::num::NonZero::new(#buffer_name_ident.2 as u64).unwrap()) }),
                    }
                }
            )
        }).collect::<(Vec<_>, Vec<_>)>();

        let prologue = prologue.into_iter().flatten();

        let wrapper_specialization_constants = specialization_constants
            .iter()
            .map(|specialization_constant| {
                let used = metadata
                    .is_parameter_location_used(
                        ParameterCategory::SpecializationConstant,
                        0,
                        specialization_constant.index as u64,
                    )
                    .with_context(|| {
                        format!(
                            "cannot query specialization constant {} for entry point {mangled_name}",
                            specialization_constant.index
                        )
                    })?;

                Ok(used.then(|| {
                    let wgsl_idx = specialization_constant.index.to_string();
                    let name = &specialization_constant.name;
                    quote! { (#wgsl_idx, #name.into()) }
                }))
            })
            .collect::<anyhow::Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect::<Vec<_>>();

        let mut axis_dispatch_counts = Vec::new();
        let mut group_dispatch_counts = Vec::new();
        let mut has_indirect_dispatch = false;

        for argument in &kernel.arguments {
            match &argument.argument_type {
                SlangArgumentType::Axis {
                    threads,
                    threads_in_group,
                } => {
                    let threads = specializer.specialize(threads);
                    let threads_in_group = specializer.specialize(threads_in_group);
                    let threads: Expr = syn::parse_str(&threads)
                        .with_context(|| format!("cannot parse axis threads expression {threads:?}"))?;
                    let threads_in_group: Expr = syn::parse_str(&threads_in_group)
                        .with_context(|| format!("cannot parse axis group size expression {threads_in_group:?}"))?;

                    axis_dispatch_counts.push(quote! { ((#threads) as u32).div_ceil((#threads_in_group) as u32) });
                },
                SlangArgumentType::Groups {
                    groups,
                } if groups == "INDIRECT" => {
                    has_indirect_dispatch = true;
                },
                SlangArgumentType::Groups {
                    groups,
                } => {
                    let groups = specializer.specialize(groups);
                    let groups: Expr = syn::parse_str(&groups)
                        .with_context(|| format!("cannot parse groups expression {groups:?}"))?;

                    group_dispatch_counts.push(quote! { (#groups) as u32 });
                },
                _ => (),
            }
        }

        let dispatch = if has_indirect_dispatch {
            quote! {
                compute_pass.dispatch_workgroups_indirect(
                    &__dsl_indirect_dispatch_buffer.0.buffer,
                    __dsl_indirect_dispatch_buffer.1 as u64,
                );
            }
        } else {
            let dispatch_counts = if axis_dispatch_counts.is_empty() {
                group_dispatch_counts
            } else {
                axis_dispatch_counts
            };
            let mut dispatch_counts = dispatch_counts.into_iter();
            let dispatch_x = dispatch_counts.next().unwrap_or_else(|| quote! { 1u32 });
            let dispatch_y = dispatch_counts.next().unwrap_or_else(|| quote! { 1u32 });
            let dispatch_z = dispatch_counts.next().unwrap_or_else(|| quote! { 1u32 });

            quote! {
                compute_pass.dispatch_workgroups(#dispatch_x, #dispatch_y, #dispatch_z);
            }
        };

        wrappers.push(quote! {
            #[allow(non_camel_case_types)]
            struct #wrapper_struct_name {
                pipeline: wgpu::ComputePipeline,
                bind_group_layout: wgpu::BindGroupLayout,
            }

            impl #wrapper_struct_name {
                fn new(
                    context: &crate::backends::webgpu::context::WebGPUContext,
                    #(#wrapper_new_arguments_definitions ,)*
                ) -> Result<Self, crate::backends::webgpu::error::WebGPUError> {
                    let shader_module = context.get_shader_module(#object_constant);
                    let bind_group_layout = context.get_bind_group_layout(vec![#(#bind_group_layout_entries ,)*]);
                    let pipeline_layout = context.get_pipeline_layout(bind_group_layout.clone());
                    let constants = vec![#(#wrapper_specialization_constants ,)*];
                    let pipeline = context.get_pipeline(shader_module, pipeline_layout, constants, #mangled_name);

                    Ok(Self { pipeline, bind_group_layout })
                }
            }

            impl #wrapper_trait_name for #wrapper_struct_name {
                fn encode<#(#encode_generics ,)* 'encoder>(
                    &self,
                    #(#wrapper_encode_arguments_definitions ,)*
                    encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::webgpu::WebGPU>,
                ) {
                    #(#prologue)*
                    let bind_group_entries = vec![#(#bind_group_entries ,)*];
                    let bind_group = encoder.context().device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: None,
                        layout: &self.bind_group_layout,
                        entries: &bind_group_entries,
                    });

                    let compute_pass = encoder.as_command_buffer_mut().ensure_compute();
                    compute_pass.set_pipeline(&self.pipeline);
                    compute_pass.set_bind_group(0, Some(&bind_group), &[]);
                    #dispatch
                }
            }
        });
        let kernel_new_match_arm_conditions = kernel
            .parameters
            .iter()
            .filter_map(|kernel_parameter| kernel_parameter.to_common(gpu_type_map).unwrap())
            .zip(variant.iter())
            .map(|(kernel_parameter, (variant_name, variant_value))| {
                assert!(kernel_parameter.name.as_ref() == *variant_name);

                match kernel_parameter.ty {
                    KernelParameterType::Type => {
                        let variant_value: Expr =
                            syn::parse_str(&slang2rust(*variant_value, gpu_type_map).unwrap().to_uppercase()).unwrap();

                        quote! { crate::DataType::#variant_value }
                    },
                    KernelParameterType::Value(_) => {
                        let variant_value: Expr = syn::parse_str(*variant_value).unwrap();

                        quote! { #variant_value }
                    },
                }
            });
        kernel_new_match_arms.push(quote! {
            (#(#kernel_new_match_arm_conditions ,)*) => Ok(Self(Box::new(#wrapper_struct_name::new(context, #(#kernel_wrapper_new_arguments ,)*)?)))
        })
    }

    let (public_trait_impl, public_type_backend, private_pub) = if kernel.public {
        (
            quote! { crate::backends::common::kernel::#kernel_trait_name for },
            quote! { type Backend = crate::backends::webgpu::WebGPU; },
            quote! {},
        )
    } else {
        (quote! {}, quote! {}, quote! { pub })
    };

    Ok(quote! {
        #(#wrappers)*

        trait #wrapper_trait_name {
            fn encode<#(#encode_generics ,)* 'encoder>(
                &self,
                #(#wrapper_encode_arguments_definitions ,)*
                encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::webgpu::WebGPU>,
            );
        }

        pub struct #kernel_struct_name(Box<dyn #wrapper_trait_name>);

        impl #public_trait_impl #kernel_struct_name {
            #public_type_backend

            #private_pub fn new(
                context: &crate::backends::webgpu::context::WebGPUContext,
                #(#kernel_new_arguments_definitions ,)*
            ) -> Result<Self, crate::backends::webgpu::error::WebGPUError> {
                match (#(#kernel_new_match_arguments ,)*) {
                    #(#kernel_new_match_arms ,)*
                    #[allow(unreachable_patterns)]
                    __dsl_configuration => Err(crate::backends::webgpu::error::WebGPUError::UnsupportedKernelVariant(format!("{__dsl_configuration:?}"))),
                }
            }
            #private_pub fn encode<#(#encode_generics ,)* 'encoder>(
                &self,
                #(#kernel_encode_arguments_definitions ,)*
                encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::webgpu::WebGPU>,
            ) {
                self.0.encode(#(#kernel_wrapper_call_arguments ,)* encoder);
            }
        }
    })
}
