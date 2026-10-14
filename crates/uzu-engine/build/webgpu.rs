//! Generate WebGPU implementations of the existing, backend-independent kernel traits.
//! Signatures and pointer element types come from the CPU reference kernels.
use std::{collections::HashMap, env, fs, path::PathBuf};

use anyhow::Context;
use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{FnArg, GenericArgument, Item, PathArguments, Type};
use walkdir::WalkDir;

use crate::common::{
    codegen::write_tokens,
    identifiers::KernelPath,
    kernel::{Kernel, KernelArgumentType, KernelBufferAccess, KernelParameterType},
};

const SUPPORTED: &[&str] = &[
    "Activation",
    "ActivationTransform",
    "AttentionPrepare",
    "AttentionSinglePass",
    "ContextRingUpdate",
    "GatedActMul",
    "InputEmbeddingLookup",
    "KVCacheUpdate",
    "LogitTransform",
    "Normalization",
    "QKVNorm",
    "RepetitionPenalty",
    "ShortConvDecode",
    "ShortConvPack",
    "ShortConvPrefill",
    "ShortConvTrie",
    "SigmoidGate",
    "Softmax",
    "TensorAddBias",
    "TensorAddScale",
    "UnifiedSampling",
    "DeltaNetConvUpdate",
    "DeltaNetConvScan",
    "DeltaNetNormGate",
    "DeltaNetPrefillPrep",
    "DeltaNetPrefill",
    "DeltaNetUpdate",
    "Conv1dPack",
    "ConvTreeScan",
    "StateAdvance",
];

fn inner(ty: &Type) -> &Type {
    if let Type::Path(p) = ty
        && p.path.segments.last().is_some_and(|s| s.ident == "Option")
        && let PathArguments::AngleBracketed(a) = &p.path.segments.last().unwrap().arguments
        && let Some(GenericArgument::Type(t)) = a.args.first()
    {
        t
    } else {
        ty
    }
}

fn scalar(
    expr: TokenStream,
    ty: &str,
) -> TokenStream {
    let ty = ty.replace(' ', "");
    if let Some(ty) = ty.strip_prefix("Option<").and_then(|s| s.strip_suffix('>')) {
        if ty == "u32" {
            return quote!(#expr.map(Param::u32).unwrap_or_else(|| Param::u32(0)));
        }
        let value = scalar(quote!(value), ty);
        return quote!(#expr.map(|value| #value).unwrap_or_else(|| Param::u32(0)));
    }
    match ty.as_str() {
        "u32" => quote!(Param::u32(#expr)),
        "f32" => quote!(Param::f32(#expr)),
        "bool" => quote!(Param::bool(#expr)),
        "i32" => quote!(Param::i32(#expr)),
        "u64" => quote!(Param::words(vec![#expr as u32, (#expr >> 32) as u32])),
        "crate::backends::common::gpu_types::ring::RingParams" => {
            quote!(Param::words(vec![#expr.ring_offset, #expr.ring_length]))
        },
        "&[crate::backends::common::gpu_types::kv_cache_update::Copy]" => {
            quote!(Param::words(#expr.iter().flat_map(|v| [v.source, v.destination]).collect()))
        },
        _ if ty.starts_with("&[") => quote!(Param::words(#expr.iter().map(|v| *v as u32).collect())),
        _ => quote!(Param::u32(#expr as u32)),
    }
}

pub fn generate(kernels: &HashMap<KernelPath, Box<[Kernel]>>) -> anyhow::Result<()> {
    let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("src/backends/cpu/kernel");
    let mut pointers: HashMap<(String, String), String> = HashMap::new();
    for entry in WalkDir::new(root).into_iter().filter_map(Result::ok) {
        if entry.path().extension().is_none_or(|s| s != "rs") {
            continue;
        }
        for item in syn::parse_file(&fs::read_to_string(entry.path())?)?.items {
            let Item::Fn(f) = item else {
                continue;
            };
            let Some(name) = f.attrs.iter().find(|a| a.path().is_ident("kernel")) else {
                continue;
            };
            let name: syn::Ident = name.parse_args()?;
            for arg in f.sig.inputs {
                let FnArg::Typed(arg) = arg else {
                    continue;
                };
                let syn::Pat::Ident(pat) = *arg.pat else {
                    continue;
                };
                if let Type::Ptr(ptr) = inner(&arg.ty) {
                    let Type::Path(p) = &*ptr.elem else {
                        continue;
                    };
                    pointers.insert(
                        (name.to_string(), pat.ident.to_string()),
                        p.path.segments.last().unwrap().ident.to_string(),
                    );
                }
            }
        }
    }
    let mut implementations = Vec::new();
    let mut associated_types = Vec::new();
    let mut sorted: Vec<_> = kernels.values().flatten().collect();
    sorted.sort_by_key(|k| k.name.as_ref());
    for k in sorted {
        let name = k.name.as_ref();
        let tr = format_ident!("{name}Kernel");
        let ty = format_ident!("{name}WebGpuKernel");
        let supported = SUPPORTED.contains(&name);
        let mut new_args = Vec::new();
        let mut params = Vec::new();
        for p in &k.parameters {
            let ident = format_ident!("{}", p.name.as_ref());
            let key = p.name.as_ref();
            let (ptype, value) = match &p.ty {
                KernelParameterType::Type => (quote!(crate::data_type::DataType), quote!(Param::dtype(#ident))),
                KernelParameterType::Value(t) => {
                    (syn::parse_str::<Type>(t)?.into_token_stream(), scalar(quote!(#ident), t))
                },
            };
            new_args.push(quote!(#ident: #ptype));
            params.push(quote!((#key, #value)));
        }
        let mut args = Vec::new();
        let mut values = Vec::new();
        let mut bindings = Vec::new();
        for a in &k.arguments {
            let ident = format_ident!("{}", a.name.as_ref());
            let key = a.name.as_ref();
            let mut aty = match &a.ty {
                KernelArgumentType::Buffer(access) => {
                    let (trait_ty, access_fn) = match access {
                        KernelBufferAccess::Read => (quote!(BufferRef), quote!(read)),
                        KernelBufferAccess::ReadWrite => (quote!(BufferMut), quote!(write)),
                    };
                    if supported {
                        let elem =
                            pointers.get(&(name.to_owned(), key.to_owned())).context("missing buffer element type")?;
                        let dtype = match elem.as_str() {
                            "f32" => quote!(DataType::F32),
                            "f16" => quote!(DataType::F16),
                            "bf16" => quote!(DataType::BF16),
                            "u8" => quote!(DataType::U8),
                            "i8" => quote!(DataType::I8),
                            "u32" => quote!(DataType::U32),
                            "i32" => quote!(DataType::I32),
                            "u64" => quote!(DataType::U64),
                            "TrieNode" => quote!(DataType::U32),
                            s => quote!(self.0.data_type(#s)),
                        };
                        let dtype = if name == "InputEmbeddingLookup" && key == "values" {
                            quote!(if self.0.flag("table_kind") {
                                DataType::U8
                            } else {
                                self.0.data_type("T")
                            })
                        } else {
                            dtype
                        };
                        let value = quote!(NamedBinding::new(#key, Binding::#access_fn(value), #dtype));
                        if a.conditional {
                            bindings.push(quote!(if let Some(value) = #ident { bindings.push(#value); }));
                        } else {
                            bindings.push(quote!({ let value = #ident; bindings.push(#value); }));
                        }
                    }
                    quote!(impl #trait_ty<Backend = WebGpu>)
                },
                KernelArgumentType::Constant(t) => {
                    if supported {
                        let value = scalar(quote!(value), t);
                        if a.conditional {
                            let default = match t.as_ref() {
                                "f32" => quote!(Param::f32(0.0)),
                                "bool" => quote!(Param::bool(false)),
                                _ => quote!(Param::u32(0)),
                            };
                            values.push(
                                quote!(params.push((#key, if let Some(value) = #ident { #value } else { #default }));),
                            );
                        } else {
                            values.push(quote!({ let value = #ident; params.push((#key, #value)); }));
                        }
                    }
                    syn::parse_str::<Type>(t)?.into_token_stream()
                },
            };
            if a.conditional {
                aty = quote!(Option<#aty>);
            }
            args.push(quote!(#ident: #aty));
        }
        let new_body = if supported {
            quote!(Ok(Self(Kernel::new(#name, vec![#(#params),*], _context)?)))
        } else {
            quote!(Err(WebGpuError::Unsupported(format!("WebGPU kernel {}", #name))))
        };
        let encode_body = if supported {
            quote! {
                let mut bindings = Vec::new(); #(#bindings)*
                let mut params = Vec::new(); #(#values)*
                self.0.encode(bindings, params, command_buffer);
            }
        } else {
            quote!(match *self {})
        };
        let definition = if supported {
            quote!(pub struct #ty(Kernel);)
        } else {
            quote!(pub enum #ty {})
        };
        implementations.push(quote! {
            #definition
            #[allow(non_snake_case, unused_variables, unused_mut)]
            impl crate::backends::common::kernel::#tr for #ty {
                type Backend = WebGpu;
                fn new(_context: &super::context::WebGpuContext, #(#new_args),*) -> Result<Self, WebGpuError> { #new_body }
                fn encode(&self, #(#args,)* command_buffer: &mut WebGpuCommandBufferEncoding) { #encode_body }
            }
        });
        associated_types.push(quote!(type #tr = #ty;));
    }
    let out = quote! {
        #(#implementations)*
        macro_rules! webgpu_kernels { () => { #(#associated_types)* }; }
    };
    write_tokens(out, PathBuf::from(env::var("OUT_DIR")?).join("webgpu.rs"))?;
    Ok(())
}
use quote::ToTokens;
