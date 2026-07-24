use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    __private::TokenStream2, Attribute, Error, FnArg, ItemFn, Pat, ReturnType, ext::IdentExt, parse_macro_input,
};

use crate::uzu_tool::util::{
    Param, arg_parsing_tokens, doc_string, is_option, is_unit, nagare_path, parameters_tokens, result_ok_type,
    return_definition_tokens,
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
