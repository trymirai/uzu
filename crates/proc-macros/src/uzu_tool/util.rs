use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::Span;
use quote::{format_ident, quote};
use syn::{
    __private::TokenStream2, Attribute, Error, Expr, ExprLit, GenericArgument, Lit, Meta, PathArguments, Type,
    ext::IdentExt,
};

pub struct Param {
    pub ident: syn::Ident,
    pub ty: Type,
    pub description: String,
    pub required: bool,
}

pub fn arg_parsing_tokens(
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

pub fn doc_string(attrs: &[Attribute]) -> String {
    let lines: Vec<String> = attrs
        .iter()
        .filter_map(|attr| {
            if !attr.path().is_ident("doc") {
                return None;
            }
            if let Meta::NameValue(name_value) = &attr.meta
                && let Expr::Lit(ExprLit {
                    lit: Lit::Str(literal),
                    ..
                }) = &name_value.value
            {
                return Some(literal.value().trim().to_string());
            }
            None
        })
        .collect();
    lines.join("\n").trim().to_string()
}

pub fn nagare_path() -> syn::Result<TokenStream2> {
    match crate_name("nagare") {
        Ok(FoundCrate::Itself) => Ok(quote!(crate)),
        Ok(FoundCrate::Name(name)) => {
            let ident = format_ident!("{name}");
            Ok(quote!(::#ident))
        },
        Err(nagare_error) => match crate_name("uzu") {
            Ok(FoundCrate::Itself) => Ok(quote!(crate::session)),
            Ok(FoundCrate::Name(name)) => {
                let ident = format_ident!("{name}");
                Ok(quote!(::#ident::session))
            },
            Err(uzu_error) => Err(Error::new(
                Span::call_site(),
                format!(
                    "either `nagare` or `uzu` must be a direct dependency to use the tool macros: \
                     failed to find `nagare` ({nagare_error}); failed to find `uzu` ({uzu_error})"
                ),
            )),
        },
    }
}

pub fn is_option(ty: &Type) -> bool {
    if let Type::Path(type_path) = ty
        && let Some(segment) = type_path.path.segments.last()
    {
        return segment.ident == "Option";
    }
    false
}

pub fn is_unit(ty: &Type) -> bool {
    matches!(ty, Type::Tuple(tuple) if tuple.elems.is_empty())
}

pub fn parameters_tokens(
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

pub fn result_ok_type(ty: &Type) -> Option<&Type> {
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

pub fn return_definition_tokens(
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
