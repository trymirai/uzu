use std::{collections::HashMap, env, fs, path::PathBuf};

use anyhow::{Context, bail};
use async_trait::async_trait;
use itertools::Itertools;
use proc_macro2::{Span, TokenStream};
use quote::{ToTokens, format_ident, quote};
use syn::{
    Expr, FnArg, GenericArgument, GenericParam, Ident, Item, ItemFn, Lifetime, PathArguments, Type,
    punctuated::Punctuated, token::Comma,
};
use walkdir::WalkDir;

use crate::common::{
    codegen::write_tokens,
    compiler::Compiler,
    gpu_types::GpuTypes,
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelBufferAccess, KernelParameter, KernelParameterType},
};

#[derive(PartialEq, Debug)]
pub enum FunctionArgumentType {
    Buffer(KernelBufferAccess),
    Constant(Type, Option<Expr>),
    Scalar(Type),
    Specialization(Type),
}

#[derive(PartialEq, Debug)]
pub struct FunctionArgument {
    pub name: Ident,
    pub conditional: Option<Expr>,
    pub ty: FunctionArgumentType,
}

impl FunctionArgument {
    fn to_kernel_argument(&self) -> Option<KernelArgument> {
        Some(KernelArgument {
            name: self.name.to_string().into_boxed_str(),
            conditional: self.conditional.is_some(),
            ty: match &self.ty {
                FunctionArgumentType::Buffer(access) => KernelArgumentType::Buffer(access.clone()),
                FunctionArgumentType::Constant(ty, None) => KernelArgumentType::Constant(
                    format!("&[{}]", ty.to_token_stream().to_string().replace(" :: ", "::")).into_boxed_str(),
                ),
                FunctionArgumentType::Constant(ty, Some(sz)) => KernelArgumentType::Constant(
                    format!(
                        "&[{}; {}]",
                        ty.to_token_stream().to_string().replace(" :: ", "::"),
                        sz.to_token_stream().to_string(),
                    )
                    .into_boxed_str(),
                ),
                FunctionArgumentType::Scalar(ty) => KernelArgumentType::Constant(
                    ty.to_token_stream().to_string().replace(" :: ", "::").into_boxed_str(),
                ),
                FunctionArgumentType::Specialization(_) => {
                    return None;
                },
            },
        })
    }

    fn to_kernel_parameter(&self) -> Option<KernelParameter> {
        Some(KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionArgumentType::Specialization(ty) => {
                    KernelParameterType::Value(ty.to_token_stream().to_string().into_boxed_str())
                },
                _ => {
                    return None;
                },
            },
        })
    }
}

#[derive(PartialEq, Debug)]
pub enum FunctionParameterType {
    Type,
    Value(Type),
}

#[derive(PartialEq, Debug)]
pub struct FunctionParameter {
    pub name: Ident,
    pub ty: FunctionParameterType,
}

impl FunctionParameter {
    fn to_kernel_parameter(&self) -> KernelParameter {
        KernelParameter {
            name: self.name.to_string().into_boxed_str(),
            ty: match &self.ty {
                FunctionParameterType::Type => KernelParameterType::Type,
                FunctionParameterType::Value(ty) => {
                    KernelParameterType::Value(ty.to_token_stream().to_string().into_boxed_str())
                },
            },
        }
    }
}

pub struct CpuCompiler {
    src_dir: PathBuf,
    build_dir: PathBuf,
}

impl CpuCompiler {
    pub fn new() -> anyhow::Result<Self> {
        let src_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").context("missing CARGO_MANIFEST_DIR")?)
            .join("src/backends/cpu/kernel");

        let build_dir = PathBuf::from(env::var("OUT_DIR").context("missing OUT_DIR")?).join("cpu");
        fs::create_dir_all(&build_dir).with_context(|| format!("cannot create {}", build_dir.display()))?;

        Ok(Self {
            src_dir,
            build_dir,
        })
    }

    fn compile(
        &self,
        source_path: PathBuf,
    ) -> anyhow::Result<(Box<[Box<str>]>, Box<[Kernel]>)> {
        let src_rel_path: Box<[Box<str>]> = source_path
            .strip_prefix(&self.src_dir)
            .context("source is not in src_dir")?
            .with_extension("")
            .as_os_str()
            .to_str()
            .unwrap()
            .split("/")
            .map(|s| s.to_string().into_boxed_str())
            .collect();

        let source_contents = fs::read_to_string(&source_path).context("cannot read the source file")?;
        let source_ast = syn::parse_file(&source_contents).context("cannot parse ast")?;

        let kernels = source_ast
            .items
            .into_iter()
            .filter_map(|item| {
                if let Item::Fn(ifn) = item
                    && ifn.attrs.iter().any(|attr| attr.path().is_ident("kernel"))
                {
                    Some(self.compile_kernel(ifn))
                } else {
                    None
                }
            })
            .collect::<anyhow::Result<_>>()?;

        Ok((src_rel_path, kernels))
    }

    fn compile_kernel(
        &self,
        ifn: ItemFn,
    ) -> anyhow::Result<Kernel> {
        let mut kernel_ident = None;
        let mut function_variants = Vec::new();
        let mut function_constraints: Vec<Expr> = Vec::new();

        for attr in ifn.attrs {
            match attr.path().get_ident().map(|i| i.to_string()).as_ref().map(|s| s.as_ref()) {
                Some("kernel") => {
                    if kernel_ident.is_some() {
                        bail!("Multiple kernel attributes!");
                    }

                    kernel_ident = Some(attr.parse_args::<Ident>().context("cannot parse kernel attribute arg")?);
                },
                Some("variants") => {
                    let mut args = attr
                        .parse_args_with(Punctuated::<Expr, Comma>::parse_terminated)
                        .context("cannot parse variants attribute args")?
                        .into_iter();

                    function_variants.push((
                        syn::parse2::<Ident>(args.next().context("variant must have a name")?.into_token_stream())
                            .unwrap(),
                        args.collect::<Box<[_]>>(),
                    ));
                },
                Some("constraint") => {
                    let expr = attr.parse_args::<Expr>().context("cannot parse constraint attribute")?;
                    function_constraints.push(expr);
                },
                _ => bail!("Unexpected attr {attr:?}"),
            }
        }

        let Some(kernel_ident) = kernel_ident else {
            bail!("Not a kernel")
        };

        let function_ident = ifn.sig.ident;
        let function_parameters = ifn
            .sig
            .generics
            .params
            .into_iter()
            .map(|parameter| {
                Ok(match parameter {
                    GenericParam::Type(parameter) => FunctionParameter {
                        name: parameter.ident,
                        ty: FunctionParameterType::Type,
                    },
                    GenericParam::Const(parameter) => FunctionParameter {
                        name: parameter.ident,
                        ty: FunctionParameterType::Value(parameter.ty),
                    },
                    parameter => {
                        bail!("unsupported kernel parameter type: {parameter:?}")
                    },
                })
            })
            .collect::<anyhow::Result<Box<[FunctionParameter]>>>()?;
        let function_arguments = ifn
            .sig
            .inputs
            .into_iter()
            .map(|argument| {
                let FnArg::Typed(argument) = argument else {
                    bail!("self argument in a kernel is not supported");
                };

                let name = syn::parse2(argument.pat.into_token_stream()).context("cannot parse the argument name")?;

                let specialize = argument.attrs.iter().any(|attr| attr.path().is_ident("specialize"));

                let conditional = argument
                    .attrs
                    .iter()
                    .find_map(|attr| attr.path().is_ident("optional").then(|| attr.parse_args::<Expr>().unwrap()));

                let ty = if specialize {
                    FunctionArgumentType::Specialization(*argument.ty)
                } else if conditional.is_some() {
                    let Type::Path(ty) = *argument.ty else {
                        bail!("conditional argument must be a type path");
                    };
                    if ty.path.segments.len() != 1 {
                        bail!("conditional argument type path must have one segment");
                    }
                    let seg = &ty.path.segments[0];
                    if seg.ident.to_string() != "Option" {
                        bail!("conditional argument type must be Option<...>");
                    }
                    let PathArguments::AngleBracketed(option_arguments) = &seg.arguments else {
                        bail!("conditional argument type must be angle bracketed");
                    };
                    if option_arguments.args.len() != 1 {
                        bail!("conditional argument type Option must have one generic argument");
                    }
                    let generic_argument = &option_arguments.args[0];
                    let GenericArgument::Type(inner_ty) = generic_argument else {
                        bail!("conditional argument type Option must have a type argument")
                    };
                    Self::parse_type(inner_ty.clone()).context("failed to parse conditional argument type")?
                } else {
                    Self::parse_type(*argument.ty).context("failed to parse argument type")?
                };

                Ok(FunctionArgument {
                    name,
                    conditional,
                    ty,
                })
            })
            .collect::<anyhow::Result<Box<[FunctionArgument]>>>()?;

        let kernel_parameters = function_parameters
            .iter()
            .map(|p| p.to_kernel_parameter())
            .chain(function_arguments.iter().flat_map(|p| p.to_kernel_parameter()))
            .collect::<Box<[KernelParameter]>>();

        let kernel_arguments =
            function_arguments.iter().flat_map(|p| p.to_kernel_argument()).collect::<Box<[KernelArgument]>>();

        if function_parameters.len() != function_variants.len() {
            bail!(
                "Kernel function has {} generics != {} #[variants(...)]!",
                function_parameters.len(),
                function_variants.len()
            );
        }

        for (parameter, (variant_name, _)) in std::iter::zip(function_parameters.iter(), function_variants.iter()) {
            if &parameter.name != variant_name {
                bail!(
                    "Parameter name doesn't match variant name: {} | {}",
                    parameter.name.to_string(),
                    variant_name.to_string(),
                );
            }
        }

        // === Bindgen ===

        let trait_ident = format_ident!("{kernel_ident}Kernel");
        let struct_ident = format_ident!("{kernel_ident}CpuKernel");

        let (struct_fields_defs, struct_fields_sets): (Vec<TokenStream>, Vec<TokenStream>) = function_parameters
            .iter()
            .map(|parameter| {
                let ident = &parameter.name;

                let ty = match &parameter.ty {
                    FunctionParameterType::Type => quote! { crate::DataType },
                    FunctionParameterType::Value(ty) => quote! { #ty },
                };

                (quote! { #ident: #ty }, quote! { #ident })
            })
            .chain(function_arguments.iter().flat_map(|argument| {
                let FunctionArgumentType::Specialization(ty) = &argument.ty else {
                    return None;
                };

                let ident = &argument.name;

                Some((quote! { #ident: #ty }, quote! { #ident }))
            }))
            .collect();

        let parameter_args: Vec<TokenStream> = kernel_parameters
            .iter()
            .map(|parameter| {
                let parameter_ident: Ident = syn::parse_str(parameter.name.as_ref())?;

                Ok(match &parameter.ty {
                    KernelParameterType::Type => quote! { #[allow(non_snake_case)] #parameter_ident: crate::DataType },
                    KernelParameterType::Value(ty) => {
                        let ty: syn::Type = syn::parse_str(ty.as_ref()).unwrap();
                        quote! { #[allow(non_snake_case)] #parameter_ident: #ty }
                    },
                })
            })
            .collect::<anyhow::Result<_>>()?;

        let (encode_generics, encode_args_defs): (Vec<_>, Vec<_>) = kernel_arguments
            .iter()
            .map(|argument| {
                let argument_ident: Ident = syn::parse_str(argument.name.as_ref()).context("cannot parse ident")?;

                let (generic, mut ty) = match &argument.ty {
                    KernelArgumentType::Buffer(access) => {
                        let buffer_lifetime = Lifetime::new(&format!("'{}", argument.name.as_ref()), Span::call_site());
                        (
                            Some(quote! { #buffer_lifetime }),
                            match access {
                                KernelBufferAccess::Read => quote! { impl crate::backends::common::kernel::BufferArg<#buffer_lifetime, std::cell::UnsafeCell<std::pin::Pin<Box<[u8]>>>> },
                                KernelBufferAccess::ReadWrite => quote! { impl crate::backends::common::kernel::BufferArgMut<#buffer_lifetime, std::cell::UnsafeCell<std::pin::Pin<Box<[u8]>>>> },
                            },
                        )
                    },
                    KernelArgumentType::Constant(ty) => {
                        let ty: Type = syn::parse_str(ty.as_ref()).context("cannot parse type")?;
                        (None, quote! { #ty })
                    },
                };

                if argument.conditional {
                    ty = quote! { Option<#ty> };
                }

                Ok((generic, quote! { #argument_ident: #ty }))
            })
            .collect::<anyhow::Result<_>>()?;

        let encode_generics = encode_generics.into_iter().flatten().collect::<Vec<_>>();

        let argument_copies = function_arguments
            .iter()
            .flat_map(|argument| {
                let argument_ident = &argument.name;
                match &argument.ty {
                    FunctionArgumentType::Buffer(access) => {
                        let (buffer_ptr, buffer_ptr_wrapper) = match access {
                            KernelBufferAccess::Read => {
                                (quote! { (&*__dsl_buffer.get()).as_ptr() }, quote! { crate::utils::pointers::SendPtr })
                            },
                            KernelBufferAccess::ReadWrite => (
                                quote! { (&mut *__dsl_buffer.get()).as_mut_ptr() },
                                quote! { crate::utils::pointers::SendPtrMut },
                            ),
                        };

                        if argument.conditional.is_some() {
                            Some(quote! {
                                let #argument_ident = #argument_ident.map(|__dsl_buffer_impl| unsafe {
                                    let (__dsl_buffer, __dsl_offset) = __dsl_buffer_impl.into_parts();

                                    #buffer_ptr_wrapper(#buffer_ptr.byte_add(__dsl_offset))
                                });
                            })
                        } else {
                            Some(quote! {
                                let #argument_ident = unsafe {
                                    let (__dsl_buffer, __dsl_offset) = #argument_ident.into_parts();

                                    #buffer_ptr_wrapper(#buffer_ptr.byte_add(__dsl_offset))
                                };
                            })
                        }
                    },
                    FunctionArgumentType::Constant(_, None) => {
                        Some(quote! { let #argument_ident = #argument_ident.to_vec().into_boxed_slice(); })
                    },
                    FunctionArgumentType::Constant(_, Some(_)) => {
                        Some(quote! { let #argument_ident = Box::new(*#argument_ident); })
                    },
                    FunctionArgumentType::Scalar(_) => None,
                    FunctionArgumentType::Specialization(_) => {
                        Some(quote! { let #argument_ident = self.#argument_ident; })
                    },
                }
            })
            .collect::<Vec<_>>();

        let make_encode = |generics: TokenStream| -> TokenStream {
            let monomorphized_function = if !generics.is_empty() {
                quote! { self::#function_ident::<#generics> }
            } else {
                quote! { self::#function_ident }
            };

            let function_call_args = function_arguments.iter().map(|argument| {
                let argument_ident = &argument.name;

                match &argument.ty {
                    FunctionArgumentType::Buffer(_) => {
                        if argument.conditional.is_some() {
                            quote! { #argument_ident.map(|p| p.as_ptr() as _) }
                        } else {
                            quote! { #argument_ident.as_ptr() as _ }
                        }
                    },
                    FunctionArgumentType::Constant(_, _) => quote! { &*#argument_ident },
                    FunctionArgumentType::Scalar(_) | FunctionArgumentType::Specialization(_) => {
                        quote! { #argument_ident }
                    },
                }
            });

            let function_call_args_joined = function_call_args.fold(quote! {}, |a, b| {
                if !a.is_empty() && !b.is_empty() {
                    quote! {#a , #b}
                } else {
                    quote! {#a #b}
                }
            });

            quote! {
                encoder.as_command_buffer_mut().push_command(move || #monomorphized_function(#function_call_args_joined));
            }
        };

        let encode_body = if function_parameters.len() > 0 {
            let mut parameter_idents = function_parameters.iter().map(|p| p.name.clone()).fold(quote! {}, |a, b| {
                if !a.is_empty() {
                    quote! {#a , self.#b}
                } else {
                    quote! {self.#b}
                }
            });

            if function_parameters.len() > 1 {
                parameter_idents = quote! { (#parameter_idents) };
            }

            let constraint_strs: Vec<String> =
                function_constraints.iter().map(|c| c.to_token_stream().to_string()).collect();

            let match_arms = function_parameters
                .iter()
                .zip(function_variants.iter())
                .map(|(parameter, (_, variants))| {
                    variants.iter().map(|variant| {
                        (
                            match parameter.ty {
                                FunctionParameterType::Type => {
                                    let dtype =
                                        format_ident!("{}", variant.to_token_stream().to_string().to_uppercase());
                                    quote! { crate::DataType::#dtype }
                                },
                                FunctionParameterType::Value(_) => quote! { #variant },
                            },
                            quote! { #variant },
                        )
                    })
                })
                .multi_cartesian_product()
                .filter(|variants| {
                    let bindings: Vec<(String, String)> = function_parameters
                        .iter()
                        .enumerate()
                        .map(|(i, p)| (p.name.to_string(), variants[i].1.to_string()))
                        .collect();
                    crate::common::constraints::satisfied(&bindings, &constraint_strs)
                })
                .map(|variants| {
                    let (match_variants, generic_variants): (Vec<TokenStream>, Vec<TokenStream>) =
                        variants.into_iter().unzip();

                    let mut match_variant = match_variants.iter().fold(quote! {}, |a, b| {
                        if !a.is_empty() && !b.is_empty() {
                            quote! {#a , #b}
                        } else {
                            quote! {#a #b}
                        }
                    });

                    if match_variants.len() > 1 {
                        match_variant = quote! { (#match_variant) };
                    }

                    let generic_variant = generic_variants.iter().fold(quote! {}, |a, b| {
                        if !a.is_empty() && !b.is_empty() {
                            quote! {#a , #b}
                        } else {
                            quote! {#a #b}
                        }
                    });

                    let encode = make_encode(generic_variant);

                    quote! { #match_variant => { #encode } }
                })
                .collect::<Vec<_>>();

            quote! {
                match #parameter_idents {
                    #(#match_arms ,)*
                    __dsl_variant => unimplemented!("variant doesn't exist: {__dsl_variant:?}"),
                }
            }
        } else {
            make_encode(quote! {})
        };

        let tokens = quote! {
            #[allow(non_snake_case)]
            pub struct #struct_ident {
                #(#struct_fields_defs ,)*
            }

            impl crate::backends::common::kernel::#trait_ident for #struct_ident {
                type Backend = crate::backends::cpu::Cpu;

                fn new(context: &crate::backends::cpu::context::CpuContext #(, #parameter_args)*) -> Result<Self, crate::backends::cpu::error::CpuError> {
                    Ok(Self {
                        #(#struct_fields_sets ,)*
                    })
                }

                fn encode<#(#encode_generics ,)* 'encoder>(&self, #(#encode_args_defs, )* encoder: &'encoder mut crate::backends::common::Encoder<crate::backends::cpu::Cpu>) {
                    #(#argument_copies)*
                    #encode_body
                }
            }
        };

        let out_path = self.build_dir.join(kernel_ident.to_string()).with_extension("rs");
        write_tokens(tokens, &out_path).context("cannot write bindings")?;

        Ok(Kernel {
            name: kernel_ident.to_string().into_boxed_str(),
            parameters: kernel_parameters,
            arguments: kernel_arguments,
        })
    }

    fn parse_type(ty: Type) -> anyhow::Result<FunctionArgumentType> {
        Ok(match ty {
            Type::Ptr(ty) => FunctionArgumentType::Buffer(if ty.mutability.is_some() {
                KernelBufferAccess::ReadWrite
            } else {
                KernelBufferAccess::Read
            }),
            Type::Reference(ty) => match *ty.elem {
                Type::Slice(ty) => FunctionArgumentType::Constant(*ty.elem, None),
                Type::Array(ty) => FunctionArgumentType::Constant(*ty.elem, Some(ty.len)),
                ty => bail!("unsupported reference type: {} ({:?})", ty.to_token_stream().to_string(), ty),
            },
            Type::Path(ty) => FunctionArgumentType::Scalar(Type::Path(ty)),
            ty => bail!("unsupported type: {} ({:?})", ty.to_token_stream().to_string(), ty),
        })
    }

    fn bindgen<'a>(
        &self,
        objects: impl IntoIterator<Item = &'a (Box<[Box<str>]>, Box<[Kernel]>)> + Clone,
    ) -> anyhow::Result<()> {
        let out_path = self.build_dir.join("dsl.rs");

        let associated_types = objects.into_iter().flat_map(|(file_path, kernels)| {
            kernels.iter().map(|kernel| {
                let file_path: TokenStream = file_path.iter().join("::").parse().unwrap();
                let kernel_trait_name = format_ident!("{}Kernel", kernel.name.as_ref());
                let kernel_struct_name = format_ident!("{}CpuKernel", kernel.name.as_ref());
                quote! { type #kernel_trait_name = #file_path::#kernel_struct_name; }
            })
        });

        let tokens = quote! {
            pub struct CpuKernels;

            impl crate::backends::common::kernel::Kernels for CpuKernels {
                type Backend = crate::backends::cpu::Cpu;

                #(#associated_types)*
            }
        };

        write_tokens(tokens, &out_path).context("cannot write dsl bindings")?;

        Ok(())
    }
}

#[async_trait]
impl Compiler for CpuCompiler {
    async fn build(
        &self,
        _gpu_types: &GpuTypes,
    ) -> anyhow::Result<HashMap<Box<[Box<str>]>, Box<[Kernel]>>> {
        let objects = WalkDir::new(&self.src_dir)
            .into_iter()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_type().is_file() && e.path().extension().and_then(|s| s.to_str()) == Some("rs"))
            .map(|e| self.compile(e.into_path()))
            .collect::<anyhow::Result<Vec<(Box<[Box<str>]>, Box<[Kernel]>)>>>()
            .context("cannot compile cpu sources")?;

        self.bindgen(&objects).context("cannot generate bindings")?;

        Ok(objects.into_iter().collect())
    }
}
