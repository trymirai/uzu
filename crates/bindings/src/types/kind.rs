use syn::{
    Ident, Token, parenthesized,
    parse::{Parse, ParseStream},
};

use super::{class_flavor::ClassFlavor, method_flavor::MethodFlavor, structure_flavor::StructureFlavor};

pub enum Kind {
    Enumeration,
    Structure(StructureFlavor),
    Class(ClassFlavor),
    Alias,
    Implementation,
    Method(MethodFlavor),
    Error,
}

impl Parse for Kind {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let identifier: Ident = input.parse()?;
        let kind = match identifier.to_string().as_str() {
            "Enumeration" => {
                ensure_no_arguments(input, "Enumeration")?;
                Kind::Enumeration
            },
            "Structure" => {
                let flavor = if input.peek(syn::token::Paren) {
                    let content;
                    parenthesized!(content in input);
                    let flavor_identifier: Ident = content.parse()?;
                    match flavor_identifier.to_string().as_str() {
                        "Plain" => StructureFlavor::Plain,
                        "Class" => StructureFlavor::Class,
                        other => {
                            return Err(syn::Error::new(
                                flavor_identifier.span(),
                                format!("Unknown Structure flavor: {other}, expected `Plain` or `Class`"),
                            ));
                        },
                    }
                } else {
                    StructureFlavor::Plain
                };
                Kind::Structure(flavor)
            },
            "Class" => {
                let flavor = if input.peek(syn::token::Paren) {
                    let content;
                    parenthesized!(content in input);
                    let flavor_identifier: Ident = content.parse()?;
                    match flavor_identifier.to_string().as_str() {
                        "Plain" => ClassFlavor::Plain,
                        "Stream" => ClassFlavor::Stream,
                        other => {
                            return Err(syn::Error::new(
                                flavor_identifier.span(),
                                format!("Unknown Class flavor: {other}, expected `Plain` or `Stream`"),
                            ));
                        },
                    }
                } else {
                    ClassFlavor::Plain
                };
                Kind::Class(flavor)
            },
            "Alias" => {
                ensure_no_arguments(input, "Alias")?;
                Kind::Alias
            },
            "Implementation" => {
                ensure_no_arguments(input, "Implementation")?;
                Kind::Implementation
            },
            "Method" => {
                let flavor = if input.peek(syn::token::Paren) {
                    let content;
                    parenthesized!(content in input);
                    let flavor_identifier: Ident = content.parse()?;
                    match flavor_identifier.to_string().as_str() {
                        "Plain" => MethodFlavor::Plain,
                        "Constructor" => MethodFlavor::Constructor,
                        "Factory" => MethodFlavor::Factory,
                        "FactoryWithCallback" => MethodFlavor::FactoryWithCallback,
                        "Getter" => MethodFlavor::Getter,
                        "Setter" => MethodFlavor::Setter,
                        "StreamNext" => MethodFlavor::StreamNext,
                        other => {
                            return Err(syn::Error::new(
                                flavor_identifier.span(),
                                format!(
                                    "Unknown Method flavor: {other}, expected one of `Plain`, `Constructor`, `Factory`, `FactoryWithCallback`, `Getter`, `Setter`, `StreamNext`"
                                ),
                            ));
                        },
                    }
                } else {
                    MethodFlavor::Plain
                };
                Kind::Method(flavor)
            },
            "Error" => {
                ensure_no_arguments(input, "Error")?;
                Kind::Error
            },
            other => {
                return Err(syn::Error::new(
                    identifier.span(),
                    format!(
                        "Unknown bindings::export kind: {other}, expected one of `Enumeration`, `Structure`, `Class`, `Alias`, `Implementation`, `Method`, `Error`"
                    ),
                ));
            },
        };
        if input.peek(Token![,]) {
            return Err(input.error("Bindings::export accepts a single kind argument"));
        }
        Ok(kind)
    }
}

fn ensure_no_arguments(
    input: ParseStream,
    kind_name: &str,
) -> syn::Result<()> {
    if input.peek(syn::token::Paren) {
        return Err(input.error(format!("Kind `{kind_name}` does not accept arguments")));
    }
    Ok(())
}
