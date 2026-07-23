use proc_macro::TokenStream;
use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream as TokenStream2};
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, Expr, ExprLit, Fields, FnArg, GenericArgument, ItemFn, Lit, Meta, Pat,
    PathArguments, ReturnType, Type, ext::IdentExt, parse_macro_input,
};

pub fn tool_function(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);
    match expand_tool_function(func) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
    }
}

pub fn derive_tool_schema(input: TokenStream) -> TokenStream {
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

    let parameters = if params.is_empty() {
        quote!(::core::option::Option::None)
    } else {
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
    };

    let return_definition = match &ok_type {
        Some(ty) => quote! {
            ::core::option::Option::Some(<#ty as #nagare::tool::schema::ToolSchema>::json_schema().to_value())
        },
        None => quote!(::core::option::Option::None),
    };

    let arg_parsing = params.iter().map(|param| {
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
                ::std::format!("invalid value for parameter `{}` of tool `{}`: {}", #arg_name, #name_str, error)
            })?;
        }
    });

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

fn expand_tool_schema(input: DeriveInput) -> syn::Result<TokenStream2> {
    let nagare = nagare_path()?;

    let Data::Struct(data) = &input.data else {
        return Err(Error::new_spanned(&input.ident, "#[derive(ToolSchema)] only supports structs"));
    };
    let Fields::Named(fields) = &data.fields else {
        return Err(Error::new_spanned(&input.ident, "#[derive(ToolSchema)] only supports structs with named fields"));
    };

    let ident = &input.ident;
    let mut generics = input.generics.clone();
    for param in generics.type_params_mut() {
        param.bounds.push(syn::parse_quote!(#nagare::tool::schema::ToolSchema));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();

    let schema = if fields.named.is_empty() {
        quote!(#nagare::tool::schema::JsonSchema::empty_object())
    } else {
        let properties = fields.named.iter().map(|field| {
            let name = field.ident.as_ref().expect("named field").unraw().to_string();
            let ty = &field.ty;
            let schema = quote!(<#ty as #nagare::tool::schema::ToolSchema>::json_schema());
            let description = doc_string(&field.attrs);
            if description.is_empty() {
                quote!((#name, #schema))
            } else {
                quote!((#name, #schema.with_description(#description)))
            }
        });
        let required = fields
            .named
            .iter()
            .filter(|field| !is_option(&field.ty))
            .map(|field| field.ident.as_ref().expect("named field").unraw().to_string());
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
