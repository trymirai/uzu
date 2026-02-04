use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Attribute, Error, Expr, FnArg, Ident, ItemFn, Lit, Meta, Pat, Result,
    ReturnType, Token, Type, TypePath,
    parse::{Parse, ParseStream},
    spanned::Spanned,
};

pub struct ToolArguments {
    pub description: String,
    pub name: Option<String>,
}

impl Parse for ToolArguments {
    fn parse(input: ParseStream) -> Result<Self> {
        let mut description = None;
        let mut name = None;

        while !input.is_empty() {
            let identifier: Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            let literal: Lit = input.parse()?;

            match identifier.to_string().as_str() {
                "description" => {
                    if let Lit::Str(string_literal) = literal {
                        description = Some(string_literal.value());
                    } else {
                        return Err(Error::new(
                            literal.span(),
                            "expected string literal",
                        ));
                    }
                },
                "name" => {
                    if let Lit::Str(string_literal) = literal {
                        name = Some(string_literal.value());
                    } else {
                        return Err(Error::new(
                            literal.span(),
                            "expected string literal",
                        ));
                    }
                },
                other => {
                    return Err(Error::new(
                        identifier.span(),
                        format!("unknown attribute: {}", other),
                    ));
                },
            }

            if input.peek(Token![,]) {
                input.parse::<Token![,]>()?;
            }
        }

        Ok(ToolArguments {
            description: description.unwrap_or_default(),
            name,
        })
    }
}

struct ParameterInfo {
    name: String,
    parameter_type: Type,
    description: Option<String>,
}

fn extract_doc_comment(attributes: &[Attribute]) -> Option<String> {
    let docs: Vec<String> = attributes
        .iter()
        .filter_map(|attribute| {
            if attribute.path().is_ident("doc") {
                if let Meta::NameValue(meta) = &attribute.meta {
                    if let Expr::Lit(expression_literal) = &meta.value {
                        if let Lit::Str(string_literal) =
                            &expression_literal.lit
                        {
                            return Some(
                                string_literal.value().trim().to_string(),
                            );
                        }
                    }
                }
            }
            None
        })
        .collect();

    if docs.is_empty() {
        None
    } else {
        Some(docs.join(" "))
    }
}

fn strip_parameter_docs(mut input_function: ItemFn) -> ItemFn {
    for argument in &mut input_function.sig.inputs {
        if let FnArg::Typed(pattern_type) = argument {
            pattern_type.attrs.retain(|attribute: &Attribute| {
                !attribute.path().is_ident("doc")
            });
        }
    }
    input_function
}

fn to_pascal_case(input: &str) -> String {
    input
        .split('_')
        .map(|part| {
            let mut characters = part.chars();
            match characters.next() {
                Some(first_char) => {
                    first_char.to_uppercase().chain(characters).collect()
                },
                None => String::new(),
            }
        })
        .collect()
}

fn is_result_return_type(return_type: &ReturnType) -> bool {
    if let ReturnType::Type(_, boxed_type) = return_type {
        if let Type::Path(TypePath {
            path,
            ..
        }) = boxed_type.as_ref()
        {
            if let Some(segment) = path.segments.last() {
                return segment.ident == "Result";
            }
        }
    }
    false
}

pub fn expand_tool_attribute(
    arguments: ToolArguments,
    input_function: ItemFn,
) -> Result<TokenStream> {
    let function_name = &input_function.sig.ident;
    let function_name_string =
        arguments.name.unwrap_or_else(|| function_name.to_string());
    let description = &arguments.description;
    let visibility = &input_function.vis;
    let returns_result = is_result_return_type(&input_function.sig.output);

    let mut parameters: Vec<ParameterInfo> = Vec::new();
    for argument in &input_function.sig.inputs {
        if let FnArg::Typed(pattern_type) = argument {
            let parameter_name =
                if let Pat::Ident(pattern_identifier) = &*pattern_type.pat {
                    pattern_identifier.ident.to_string()
                } else {
                    return Err(Error::new(
                        pattern_type.pat.span(),
                        "expected identifier pattern",
                    ));
                };

            let doc_comment = extract_doc_comment(&pattern_type.attrs);

            parameters.push(ParameterInfo {
                name: parameter_name,
                parameter_type: (*pattern_type.ty).clone(),
                description: doc_comment,
            });
        }
    }

    let struct_name = format_ident!(
        "{}ToolImplementation",
        to_pascal_case(&function_name_string)
    );

    let parameter_struct_name = format_ident!("{}Parameters", struct_name);
    let parameter_fields: Vec<_> = parameters
        .iter()
        .map(|parameter_info| {
            let field_name = format_ident!("{}", parameter_info.name);
            let field_type = &parameter_info.parameter_type;
            let doc_attribute = parameter_info
                .description
                .as_ref()
                .map(|description| quote!(#[doc = #description]))
                .unwrap_or_default();
            quote! {
                #doc_attribute
                pub #field_name: #field_type
            }
        })
        .collect();

    let schema_properties: Vec<_> = parameters
        .iter()
        .map(|parameter_info| {
            let property_name = &parameter_info.name;
            let property_type = &parameter_info.parameter_type;
            let property_description = parameter_info.description.clone().unwrap_or_default();
            quote! {
                {
                    let mut schema: serde_json::Value = schema_generator.subschema_for::<#property_type>().to_value();
                    if !#property_description.is_empty() {
                        if let Some(schema_object) = schema.as_object_mut() {
                            schema_object.insert("description".to_string(), serde_json::Value::String(#property_description.to_string()));
                        }
                    }
                    properties.insert(#property_name.to_string(), Value::from(schema));
                    required.push(#property_name.to_string());
                }
            }
        })
        .collect();

    let parameter_extractions: Vec<_> = parameters
        .iter()
        .map(|parameter_info| {
            let field_name = format_ident!("{}", parameter_info.name);
            quote! { parameters.#field_name }
        })
        .collect();

    let clean_function = strip_parameter_docs(input_function.clone());

    let result_handling = if returns_result {
        quote! {
            match #function_name(#(#parameter_extractions),*) {
                Ok(output) => {
                    let json_result = serde_json::to_value(output)
                        .map_err(|error| ToolError::SerializationError {
                            message: error.to_string(),
                        })?;
                    Ok(ToolCallResult::success(
                        tool_call.id.clone(),
                        tool_call.name.clone(),
                        Value::from(json_result),
                    ))
                }
                Err(error) => {
                    Ok(ToolCallResult::failure(
                        tool_call.id.clone(),
                        tool_call.name.clone(),
                        error.to_string(),
                    ))
                }
            }
        }
    } else {
        quote! {
            let result = #function_name(#(#parameter_extractions),*);

            let json_result = serde_json::to_value(result)
                .map_err(|error| ToolError::SerializationError {
                    message: error.to_string(),
                })?;

            Ok(ToolCallResult::success(
                tool_call.id.clone(),
                tool_call.name.clone(),
                Value::from(json_result),
            ))
        }
    };

    let expanded = quote! {
        #clean_function

        #[derive(Debug, Clone, serde::Deserialize, schemars::JsonSchema)]
        #visibility struct #parameter_struct_name {
            #(#parameter_fields),*
        }

        #[derive(Debug, Clone, Default)]
        #visibility struct #struct_name;

        impl uzu::tool_calling::ToolImplementationCallable for #struct_name {
            fn tool(&self) -> uzu::tool_calling::Tool {
                use schemars::generate::SchemaSettings;
                use uzu::tool_calling::{Tool, ToolFunctionParameters, Value};

                let settings = SchemaSettings::default().with(|schema_settings| {
                    schema_settings.inline_subschemas = true;
                });
                let mut schema_generator = settings.into_generator();
                let mut properties = std::collections::HashMap::new();
                let mut required = Vec::new();

                #(#schema_properties)*

                let function_parameters = ToolFunctionParameters::new()
                    .with_properties(properties)
                    .with_required(required);

                Tool::function_with(#function_name_string.to_string(), #description.to_string(), function_parameters)
            }

            fn call(&self, tool_call: &uzu::tool_calling::ToolCall) -> Result<uzu::tool_calling::ToolCallResult, uzu::tool_calling::ToolError> {
                use uzu::tool_calling::{ToolCallResult, ToolError, Value};

                let json_parameters: serde_json::Value = tool_call.arguments.clone().into();

                let parameters: #parameter_struct_name = serde_json::from_value(json_parameters)
                    .map_err(|error| ToolError::InvalidParameters {
                        message: error.to_string(),
                    })?;

                #result_handling
            }
        }
    };

    Ok(expanded)
}
