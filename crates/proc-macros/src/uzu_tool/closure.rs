use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::quote;
use syn::{
    Attribute, Error, ExprClosure, Pat, ReturnType, Token,
    ext::IdentExt,
    parse::{Parse, ParseStream},
    parse_macro_input,
};

use crate::uzu_tool::util::{
    Param, arg_parsing_tokens, doc_string, is_option, is_unit, nagare_path, parameters_tokens, result_ok_type,
    return_definition_tokens,
};

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

pub fn uzu_tool_closure(input: TokenStream) -> TokenStream {
    let tool = parse_macro_input!(input as ToolClosure);
    match expand_tool_closure(tool) {
        Ok(tokens) => tokens.into(),
        Err(error) => error.to_compile_error().into(),
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
