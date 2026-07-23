use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use serde_derive_internals::{
    Ctxt, Derive,
    ast::{Container as SerdeContainer, Data as SerdeData},
};
use syn::{
    Attribute, Data, DeriveInput, Error, Expr, ExprClosure, ExprLit, Fields, FnArg, GenericArgument, ItemFn, Lit, Meta,
    Pat, PathArguments, ReturnType, Token, Type,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

pub fn uzu_tool_function(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);
    match expand_tool_function(func) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

pub fn uzu_tool_closure(input: TokenStream) -> TokenStream {
    let tool = parse_macro_input!(input as ToolClosure);
    match expand_tool_closure(tool) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

pub fn uzu_derive_tool_schema(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand_tool_schema(input) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

struct Param {
    ident: syn::Ident,
    ty: Type,
    description: String,
    required: bool,
}

fn expand_tool_function(mut func: ItemFn) -> syn::Result<TokenStream2> {
    let nagare = nagare_path()?;

    if !func.sig.generics.params.is_empty() {
        return Err(Error::new_spanned(&func.sig.generics, "#[tool_function] does not support generic functions"));
    }

    let vis = func.vis.clone();
    let name = func.sig.ident.clone();
    let name_str = name.unraw().to_string();
    let description = doc_string(&func.attrs);
    let doc_attrs: Vec<Attribute> = func.attrs.iter().filter(|attr| attr.path().is_ident("doc")).cloned().collect();
    let is_async = func.sig.asyncness.is_some();

    let mut params = Vec::new();
    for input in &mut func.sig.inputs {
        match input {
            FnArg::Receiver(receiver) => {
                return Err(Error::new_spanned(receiver, "#[tool_function] does not support `self` parameters"));
            },
            FnArg::Typed(arg) => {
                let Pat::Ident(pat) = &*arg.pat else {
                    return Err(Error::new_spanned(&arg.pat, "#[tool_function] parameters must be plain identifiers"));
                };
                let description = doc_string(&arg.attrs);
                arg.attrs.retain(|attr| !attr.path().is_ident("doc"));
                params.push(Param {
                    ident: pat.ident.clone(),
                    ty: (*arg.ty).clone(),
                    description,
                    required: !is_option(&arg.ty),
                });
            },
        }
    }

    let (ok_type, is_result) = match &func.sig.output {
        ReturnType::Default => (None, false),
        ReturnType::Type(_, ty) => match result_ok_type(ty) {
            Some(ok) => ((!is_unit(ok)).then(|| ok.clone()), true),
            None => ((!is_unit(ty)).then(|| (**ty).clone()), false),
        },
    };

    let mut call_fn = func.clone();
    call_fn.sig.ident = format_ident!("call");
    call_fn.attrs.retain(|attr| !attr.path().is_ident("doc"));

    let parameters = parameters_tokens(&params, &nagare);
    let return_definition = return_definition_tokens(ok_type.as_ref(), &nagare);
    let arg_parsing = arg_parsing_tokens(&params, &name_str, &nagare);

    let arg_idents = params.iter().map(|param| &param.ident);
    let mut call = quote!(Self::call(#(#arg_idents),*));
    if is_async {
        call = quote!(#call.await);
    }
    let map_err = quote!(.map_err(|error| -> #nagare::tool::func_def::FutureError { error.into() })?);
    let call_stmt = match (&ok_type, is_result) {
        (Some(_), true) => quote!(let result = #call #map_err;),
        (Some(_), false) => quote!(let result = #call;),
        (None, true) => quote!(#call #map_err;),
        (None, false) => quote!(#call;),
    };
    let output_expr = match &ok_type {
        Some(_) => quote! {
            let json = #nagare::__private::serde_json::to_value(&result)?;
            ::core::result::Result::Ok(::core::convert::Into::into(json))
        },
        None => quote! {
            ::core::result::Result::Ok(::core::convert::Into::into(
                #nagare::__private::serde_json::Value::Null,
            ))
        },
    };

    Ok(quote! {
        #(#doc_attrs)*
        #[allow(non_camel_case_types)]
        #[derive(Clone, Copy)]
        #vis struct #name;

        impl #name {
            #call_fn

            #vis fn definition() -> #nagare::tool::func_def::ToolFunctionDefinition {
                #nagare::tool::func_def::ToolFunctionDefinition::new(
                    ::std::string::String::from(#name_str),
                    ::std::string::String::from(#description),
                    #parameters,
                    #return_definition,
                    ::std::boxed::Box::new(|args| ::std::boxed::Box::pin(Self::invoke(args))),
                )
            }

            async fn invoke(
                args: #nagare::tool::func_def::Value,
            ) -> ::core::result::Result<
                #nagare::tool::func_def::Value,
                #nagare::tool::func_def::FutureError,
            >
            {
                let args: #nagare::__private::serde_json::Value =
                    ::core::convert::TryInto::try_into(args)?;
                #(#arg_parsing)*
                #call_stmt
                #output_expr
            }
        }

        impl ::core::convert::From<#name> for #nagare::tool::func_def::ToolFunctionDefinition {
            fn from(_: #name) -> Self {
                #name::definition()
            }
        }
    })
}

/// Input of `uzu_tool_closure!`: optional doc comments, a tool name, a colon, and a closure.
struct ToolClosure {
    attrs: Vec<Attribute>,
    name: syn::Ident,
    closure: ExprClosure,
}

impl Parse for ToolClosure {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let attrs = input.call(Attribute::parse_outer)?;
        let name = input.call(syn::Ident::parse_any)?;
        input.parse::<Token![:]>()?;
        let closure: ExprClosure = input.parse()?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the tool closure"));
        }
        Ok(Self {
            attrs,
            name,
            closure,
        })
    }
}

fn expand_tool_closure(tool: ToolClosure) -> syn::Result<TokenStream2> {
    let nagare = nagare_path()?;
    let ToolClosure {
        attrs,
        name,
        closure,
    } = tool;

    if let Some(constness) = &closure.constness {
        return Err(Error::new_spanned(constness, "uzu_tool_closure! does not support `const` closures"));
    }
    if let Some(movability) = &closure.movability {
        return Err(Error::new_spanned(movability, "uzu_tool_closure! does not support `static` closures"));
    }
    if let Some(lifetimes) = &closure.lifetimes {
        return Err(Error::new_spanned(lifetimes, "uzu_tool_closure! does not support `for<...>` binders"));
    }

    let name_str = name.unraw().to_string();
    let description = doc_string(&attrs);
    let is_async = closure.asyncness.is_some();

    let mut params = Vec::new();
    for input in &closure.inputs {
        let Pat::Type(pat_type) = input else {
            return Err(Error::new_spanned(
                input,
                "uzu_tool_closure! parameters must have explicit types, e.g. `city: String`",
            ));
        };
        let Pat::Ident(pat_ident) = &*pat_type.pat else {
            return Err(Error::new_spanned(&pat_type.pat, "uzu_tool_closure! parameters must be plain identifiers"));
        };
        if pat_ident.subpat.is_some() {
            return Err(Error::new_spanned(pat_ident, "uzu_tool_closure! parameters must be plain identifiers"));
        }
        params.push(Param {
            ident: pat_ident.ident.clone(),
            ty: (*pat_type.ty).clone(),
            description: doc_string(&pat_type.attrs),
            required: !is_option(&pat_type.ty),
        });
    }

    // Without a return type annotation the result is serialized as-is (`()` becomes `null`) and no return
    // schema is published; `Result` unwrapping requires an explicit `-> Result<T, E>` annotation.
    let (return_type, ok_type, is_result) = match &closure.output {
        ReturnType::Default => (None, None, false),
        ReturnType::Type(_, ty) => match result_ok_type(ty) {
            Some(ok) => (Some((**ty).clone()), (!is_unit(ok)).then(|| ok.clone()), true),
            None => (Some((**ty).clone()), (!is_unit(ty)).then(|| (**ty).clone()), false),
        },
    };

    let parameters = parameters_tokens(&params, &nagare);
    let return_definition = return_definition_tokens(ok_type.as_ref(), &nagare);
    let arg_parsing = arg_parsing_tokens(&params, &name_str, &nagare);

    let body = &closure.body;
    let closure_params = params.iter().map(|param| {
        let ident = &param.ident;
        let ty = &param.ty;
        quote!(#ident: #ty)
    });
    // An async body is rewritten as a closure returning an `async move` block, so each invocation produces
    // an owned future; the invoker clones the closure per call, which only requires the captures to be
    // `Clone + Send + Sync + 'static`.
    let func = if is_async {
        quote!(move |#(#closure_params),*| async move { #body })
    } else {
        quote!(move |#(#closure_params),*| { #body })
    };

    let arg_idents = params.iter().map(|param| &param.ident);
    let mut call = quote!(__uzu_tool_func(#(#arg_idents),*));
    if is_async {
        call = quote!(#call.await);
    }
    let result_stmt = match &return_type {
        Some(ty) => quote!(let __uzu_result: #ty = #call;),
        None => quote!(let __uzu_result = #call;),
    };
    let unwrap_stmt = if is_result {
        quote! {
            let __uzu_result =
                __uzu_result.map_err(|error| -> #nagare::tool::func_def::FutureError { error.into() })?;
        }
    } else {
        quote!()
    };
    let serialize_result = return_type.is_none() || ok_type.is_some();
    let output_expr = if serialize_result {
        quote! {
            let json = #nagare::__private::serde_json::to_value(&__uzu_result)?;
            ::core::result::Result::<
                #nagare::tool::func_def::Value,
                #nagare::tool::func_def::FutureError,
            >::Ok(::core::convert::Into::into(json))
        }
    } else {
        quote! {
            let _ = __uzu_result;
            ::core::result::Result::<
                #nagare::tool::func_def::Value,
                #nagare::tool::func_def::FutureError,
            >::Ok(::core::convert::Into::into(#nagare::__private::serde_json::Value::Null))
        }
    };

    Ok(quote! {
        {
            let __uzu_tool_func = #func;
            #nagare::tool::func_def::ToolFunctionDefinition::new(
                ::std::string::String::from(#name_str),
                ::std::string::String::from(#description),
                #parameters,
                #return_definition,
                ::std::boxed::Box::new(move |args| {
                    let __uzu_tool_func = ::core::clone::Clone::clone(&__uzu_tool_func);
                    ::std::boxed::Box::pin(async move {
                        let args: #nagare::__private::serde_json::Value =
                            ::core::convert::TryInto::try_into(args)?;
                        #(#arg_parsing)*
                        #result_stmt
                        #unwrap_stmt
                        #output_expr
                    })
                }),
            )
        }
    })
}

fn parameters_tokens(
    params: &[Param],
    nagare: &TokenStream2,
) -> TokenStream2 {
    if params.is_empty() {
        return quote!(::core::option::Option::None);
    }
    let properties = params.iter().map(|param| {
        let name = param.ident.unraw().to_string();
        let ty = &param.ty;
        let schema = quote!(<#ty as #nagare::tool::schema::ToolSchema>::json_schema());
        if param.description.is_empty() {
            quote!((#name, #schema))
        } else {
            let description = &param.description;
            quote!((#name, #schema.with_description(#description)))
        }
    });
    let required = params.iter().filter(|param| param.required).map(|param| param.ident.unraw().to_string());
    quote! {
        ::core::option::Option::Some(
            #nagare::tool::schema::JsonSchema::object(
                [#(#properties),*],
                ::std::vec::Vec::<&str>::from([#(#required),*]),
            )
            .to_value(),
        )
    }
}

fn return_definition_tokens(
    ok_type: Option<&Type>,
    nagare: &TokenStream2,
) -> TokenStream2 {
    match ok_type {
        Some(ty) => quote! {
            ::core::option::Option::Some(<#ty as #nagare::tool::schema::ToolSchema>::json_schema().to_value())
        },
        None => quote!(::core::option::Option::None),
    }
}

fn arg_parsing_tokens(
    params: &[Param],
    tool_name: &str,
    nagare: &TokenStream2,
) -> Vec<TokenStream2> {
    params
        .iter()
        .map(|param| {
            let ident = &param.ident;
            let ty = &param.ty;
            let arg_name = param.ident.unraw().to_string();
            quote! {
                let #ident: #ty = #nagare::__private::serde_json::from_value(
                    args.get(#arg_name)
                        .cloned()
                        .unwrap_or(#nagare::__private::serde_json::Value::Null),
                )
                .map_err(|error| {
                    ::std::format!("invalid value for parameter `{}` of tool `{}`: {}", #arg_name, #tool_name, error)
                })?;
            }
        })
        .collect()
}

fn expand_tool_schema(input: DeriveInput) -> syn::Result<TokenStream2> {
    let nagare = nagare_path()?;

    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "#[derive(UzuToolSchema)] only supports structs"));
    };
    let Fields::Named(_) = &data.fields else {
        return Err(Error::new_spanned(
            &input.ident,
            "#[derive(UzuToolSchema)] only supports structs with named fields",
        ));
    };

    let serde_context = Ctxt::new();
    let serde_container = SerdeContainer::from_ast(&serde_context, &input, Derive::Deserialize);
    serde_context.check()?;
    let Some(serde_container) = serde_container else {
        return Err(Error::new_spanned(&input.ident, "failed to parse Serde attributes"));
    };
    let SerdeData::Struct(_, serde_fields) = &serde_container.data else {
        unreachable!();
    };

    if serde_container.attrs.transparent()
        || serde_container.attrs.type_from().is_some()
        || serde_container.attrs.type_try_from().is_some()
        || serde_container.attrs.type_into().is_some()
        || serde_container.attrs.remote().is_some()
    {
        return Err(Error::new_spanned(
            &input.ident,
            "UzuToolSchema does not support Serde container attributes that change the serialized representation",
        ));
    }

    let schema_fields = serde_fields
        .iter()
        .filter_map(|field| {
            let attrs = &field.attrs;
            if attrs.skip_serializing() != attrs.skip_deserializing() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema requires fields to be skipped in both Serialize and Deserialize; use #[serde(skip)]",
                )));
            }
            if attrs.skip_serializing() {
                return None;
            }
            if attrs.flatten() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support #[serde(flatten)]",
                )));
            }
            if attrs.skip_serializing_if().is_some() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support #[serde(skip_serializing_if = \"...\")]",
                )));
            }
            if attrs.serialize_with().is_some() || attrs.deserialize_with().is_some() || attrs.getter().is_some() {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema does not support Serde field attributes that change the serialized representation",
                )));
            }

            let serialize_name = attrs.name().serialize_name();
            let deserialize_name = attrs.name().deserialize_name();
            if serialize_name != deserialize_name {
                return Some(Err(Error::new_spanned(
                    field.original,
                    "UzuToolSchema requires matching Serialize and Deserialize field names",
                )));
            }

            let required =
                !is_option(field.ty) && attrs.default().is_none() && serde_container.attrs.default().is_none();
            Some(Ok((field.original, serialize_name.to_string(), required)))
        })
        .collect::<syn::Result<Vec<_>>>()?;

    let ident = &input.ident;
    let mut generics = input.generics.clone();
    for (field, _, _) in &schema_fields {
        let ty = &field.ty;
        generics.make_where_clause().predicates.push(syn::parse_quote!(#ty: #nagare::tool::schema::ToolSchema));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let schema = if schema_fields.is_empty() {
        quote!(#nagare::tool::schema::JsonSchema::empty_object())
    } else {
        let properties = schema_fields.iter().map(|(field, name, _)| {
            let ty = &field.ty;
            let schema = quote!(<#ty as #nagare::tool::schema::ToolSchema>::json_schema());
            let description = doc_string(&field.attrs);
            if description.is_empty() {
                quote!((#name, #schema))
            } else {
                quote!((#name, #schema.with_description(#description)))
            }
        });
        let required = schema_fields.iter().filter(|(_, _, required)| *required).map(|(_, name, _)| name);
        quote! {
            #nagare::tool::schema::JsonSchema::object(
                [#(#properties),*],
                ::std::vec::Vec::<&str>::from([#(#required),*]),
            )
        }
    };

    let struct_description = doc_string(&input.attrs);
    let with_description = if struct_description.is_empty() {
        quote!()
    } else {
        quote!(.with_description(#struct_description))
    };

    Ok(quote! {
        impl #impl_generics #nagare::tool::schema::ToolSchema for #ident #ty_generics #where_clause {
            fn json_schema() -> #nagare::tool::schema::JsonSchema {
                #schema #with_description
            }
        }
    })
}

fn nagare_path() -> syn::Result<TokenStream2> {
    match crate_name("nagare") {
        Ok(FoundCrate::Itself) => Ok(quote!(crate)),
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            Ok(quote!(::#ident))
        },
        Err(error) => Err(Error::new(
            Span::call_site(),
            format!("`nagare` must be a direct dependency to use its tool macros: {error}"),
        )),
    }
}

fn doc_string(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let Meta::NameValue(name_value) = &attr.meta {
                if let Expr::Lit(ExprLit {
                    lit: Lit::Str(literal),
                    ..
                }) = &name_value.value
                {
                    return Some(literal.value().trim().to_string());
                }
            }
            None
        })
        .collect();
    lines.join("\n").trim().to_string()
}

fn is_option(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty {
        if let Some(segment) = type_path.path.segments.last() {
            return segment.ident == "Option";
        }
    }
    false
}

fn result_ok_type(ty: &Type) -> Option<&Type> {
    let Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != "Result" {
        return None;
    }
    let PathArguments::AngleBracketed(arguments) = &segment.arguments else {
        return None;
    };
    arguments.args.iter().find_map(|argument| match argument {
        GenericArgument::Type(ty) => Some(ty),
        _ => None,
    })
}

fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(tuple) if tuple.elems.is_empty())
}
