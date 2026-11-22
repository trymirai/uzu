use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use heck::{AsLowerCamelCase, AsSnakeCase};
use ignore::WalkBuilder;

#[derive(Debug, Clone)]
pub struct Factory {
    pub type_name: String,
    pub method_name: String,
    pub is_async: bool,
    pub throws: bool,
    pub optional: bool,
    pub arguments: Vec<FactoryArgument>,
}

#[derive(Debug, Clone)]
pub struct FactoryArgument {
    pub name: String,
    pub foreign_type: String,
}

#[derive(Debug, Clone)]
pub struct CallbackFactory {
    pub type_name: String,
    pub method_name: String,
    pub callback_arg_types: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct StreamBridge {
    pub type_name: String,
    pub next_method_name: String,
    pub item_foreign_type: String,
    pub cancel_getter_name: Option<String>,
}

pub fn collect_all(
    crates_path: &Path,
    spell_type: fn(&syn::Type, &str) -> Option<String>,
) -> Result<(BTreeMap<String, String>, Vec<Factory>, Vec<CallbackFactory>, Vec<StreamBridge>)> {
    let mut aliases: BTreeMap<String, String> = BTreeMap::new();
    let mut factories: Vec<Factory> = Vec::new();
    let mut callback_factories: Vec<CallbackFactory> = Vec::new();
    let mut stream_bridges: Vec<StreamBridge> = Vec::new();

    for entry in WalkBuilder::new(crates_path).build().flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("rs") {
            continue;
        }
        let Ok(source) = fs::read_to_string(path) else {
            continue;
        };
        let Ok(file) = syn::parse_file(&source) else {
            continue;
        };
        collect_from_items(
            &file.items,
            spell_type,
            &mut aliases,
            &mut factories,
            &mut callback_factories,
            &mut stream_bridges,
        );
    }

    factories.sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
    factories.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
    callback_factories
        .sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
    callback_factories.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
    stream_bridges.sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name));
    stream_bridges.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name);

    Ok((aliases, factories, callback_factories, stream_bridges))
}

fn collect_from_items(
    items: &[syn::Item],
    spell_type: fn(&syn::Type, &str) -> Option<String>,
    aliases: &mut BTreeMap<String, String>,
    factories: &mut Vec<Factory>,
    callback_factories: &mut Vec<CallbackFactory>,
    stream_bridges: &mut Vec<StreamBridge>,
) {
    for item in items {
        match item {
            syn::Item::Impl(item_impl) => {
                collect_from_impl(item_impl, spell_type, factories, callback_factories, stream_bridges)
            },
            syn::Item::Type(item_type)
                if item_type.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "Alias")) =>
            {
                if let Some(foreign_type) = spell_type(&item_type.ty, "") {
                    aliases.insert(item_type.ident.to_string(), foreign_type);
                }
            },
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested)) = &item_mod.content {
                    collect_from_items(nested, spell_type, aliases, factories, callback_factories, stream_bridges);
                }
            },
            _ => {},
        }
    }
}

fn collect_from_impl(
    item_impl: &syn::ItemImpl,
    spell_type: fn(&syn::Type, &str) -> Option<String>,
    factories: &mut Vec<Factory>,
    callback_factories: &mut Vec<CallbackFactory>,
    stream_bridges: &mut Vec<StreamBridge>,
) {
    if !impl_is_bindings_implementation(item_impl) {
        return;
    }
    let Some(type_name) = impl_self_type_ident(item_impl) else {
        return;
    };

    let cancel_getter_name = item_impl.items.iter().find_map(|impl_item| {
        let syn::ImplItem::Fn(method) = impl_item else {
            return None;
        };
        if !is_cancel_token_getter(method) {
            return None;
        }
        Some(method.sig.ident.to_string())
    });

    for impl_item in &item_impl.items {
        let syn::ImplItem::Fn(method) = impl_item else {
            continue;
        };
        if method_is_stream_next(method) {
            if let Some(bridge) = extract_stream_bridge(method, &type_name, cancel_getter_name.clone(), spell_type) {
                stream_bridges.push(bridge);
            }
            continue;
        }
        if method_has_receiver(method) {
            continue;
        }
        if method_is_factory_with_callback(method) {
            let callback_arg_types = extract_callback_arg_types(method, &type_name, spell_type).unwrap_or_default();
            callback_factories.push(CallbackFactory {
                type_name: type_name.clone(),
                method_name: method.sig.ident.to_string(),
                callback_arg_types,
            });
            continue;
        }
        if !method_is_factory(method) {
            continue;
        }
        let Some(factory) = extract_factory(method, &type_name, spell_type) else {
            continue;
        };
        factories.push(factory);
    }
}

fn method_is_stream_next(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "StreamNext"))
}

fn extract_stream_bridge(
    method: &syn::ImplItemFn,
    type_name: &str,
    cancel_getter_name: Option<String>,
    spell_type: fn(&syn::Type, &str) -> Option<String>,
) -> Option<StreamBridge> {
    let syn::ReturnType::Type(_, return_type) = &method.sig.output else {
        return None;
    };
    let inner = path_generic(return_type.as_ref(), "Option")?;
    let item_foreign_type = spell_type(inner, type_name)?;
    Some(StreamBridge {
        type_name: type_name.to_string(),
        next_method_name: method.sig.ident.to_string(),
        item_foreign_type,
        cancel_getter_name,
    })
}

fn is_cancel_token_getter(method: &syn::ImplItemFn) -> bool {
    let has_getter_attr = method.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "Getter"));
    if !has_getter_attr {
        return false;
    }
    let syn::ReturnType::Type(_, return_type) = &method.sig.output else {
        return false;
    };
    let syn::Type::Path(type_path) = return_type.as_ref() else {
        return false;
    };
    type_path.path.segments.last().map(|segment| segment.ident == "CancelToken").unwrap_or(false)
}

fn method_is_factory_with_callback(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "FactoryWithCallback"))
}

fn extract_callback_arg_types(
    method: &syn::ImplItemFn,
    self_type_name: &str,
    spell_type: fn(&syn::Type, &str) -> Option<String>,
) -> Option<Vec<String>> {
    for input in &method.sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else {
            continue;
        };
        let Some(inputs) = extract_box_fn_inputs(&pat_type.ty) else {
            continue;
        };
        let foreign_types: Vec<String> =
            inputs.iter().map(|ty| spell_type(ty, self_type_name).unwrap_or_else(|| "Any".to_string())).collect();
        return Some(foreign_types);
    }
    None
}

fn extract_box_fn_inputs(ty: &syn::Type) -> Option<Vec<syn::Type>> {
    let inner = path_generic(ty, "Box")?;
    let syn::Type::TraitObject(trait_object) = inner else {
        return None;
    };
    for bound in &trait_object.bounds {
        let syn::TypeParamBound::Trait(trait_bound) = bound else {
            continue;
        };
        let segment = trait_bound.path.segments.last()?;
        if segment.ident != "Fn" && segment.ident != "FnMut" && segment.ident != "FnOnce" {
            continue;
        }
        let syn::PathArguments::Parenthesized(paren) = &segment.arguments else {
            continue;
        };
        return Some(paren.inputs.iter().map(|input| input.ty.clone()).collect());
    }
    None
}

fn extract_factory(
    method: &syn::ImplItemFn,
    type_name: &str,
    spell_type: fn(&syn::Type, &str) -> Option<String>,
) -> Option<Factory> {
    let (throws, optional, is_self_return) = classify_return(&method.sig.output, type_name)?;
    if !is_self_return {
        return None;
    }

    let mut arguments = Vec::with_capacity(method.sig.inputs.len());
    for input in &method.sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else {
            return None;
        };
        let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() else {
            return None;
        };
        let foreign_type = spell_type(&pat_type.ty, type_name)?;
        arguments.push(FactoryArgument {
            name: pat_ident.ident.to_string(),
            foreign_type,
        });
    }

    Some(Factory {
        type_name: type_name.to_string(),
        method_name: method.sig.ident.to_string(),
        is_async: method.sig.asyncness.is_some(),
        throws,
        optional,
        arguments,
    })
}

fn classify_return(
    output: &syn::ReturnType,
    self_type_name: &str,
) -> Option<(bool, bool, bool)> {
    let syn::ReturnType::Type(_, return_type) = output else {
        return None;
    };
    let mut current = return_type.as_ref();
    let mut throws = false;
    let mut optional = false;

    if let Some(inner) = path_generic(current, "Result") {
        throws = true;
        current = inner;
    }
    if let Some(inner) = path_generic(current, "Option") {
        optional = true;
        current = inner;
    }

    let syn::Type::Path(type_path) = current else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let is_self_return = segment.ident == "Self" || segment.ident == self_type_name;
    if !is_self_return {
        return None;
    }
    Some((throws, optional, true))
}

fn path_generic<'t>(
    ty: &'t syn::Type,
    expected: &str,
) -> Option<&'t syn::Type> {
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    if segment.ident != expected {
        return None;
    }
    let syn::PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    for argument in &args.args {
        if let syn::GenericArgument::Type(inner) = argument {
            return Some(inner);
        }
    }
    None
}

pub fn spell_rust_type(
    ty: &syn::Type,
    self_type_name: &str,
    primitive: fn(&str) -> Option<&'static str>,
    list: fn(String) -> String,
) -> Option<String> {
    if let Some(inner) = path_generic(ty, "Option") {
        let inner = spell_rust_type(inner, self_type_name, primitive, list)?;
        return Some(format!("{}?", inner));
    }
    if let Some(inner) = path_generic(ty, "Vec") {
        return spell_rust_type(inner, self_type_name, primitive, list).map(list);
    }
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let name = segment.ident.to_string();
    if name == "Self" {
        return Some(self_type_name.to_string());
    }
    Some(primitive(&name).map(str::to_string).unwrap_or(name))
}

fn impl_is_bindings_implementation(item_impl: &syn::ItemImpl) -> bool {
    item_impl.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "Implementation"))
}

fn impl_self_type_ident(item_impl: &syn::ItemImpl) -> Option<String> {
    let syn::Type::Path(type_path) = item_impl.self_ty.as_ref() else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    Some(segment.ident.to_string())
}

fn method_is_factory(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "Factory"))
}

fn method_has_receiver(method: &syn::ImplItemFn) -> bool {
    method.sig.inputs.iter().any(|arg| matches!(arg, syn::FnArg::Receiver(_)))
}

fn is_bindings_export_with(
    attribute: &syn::Attribute,
    kind: &str,
) -> bool {
    let path = attribute.path();
    let segments: Vec<String> = path.segments.iter().map(|segment| segment.ident.to_string()).collect();
    let matches_path = segments == ["bindings", "export"] || segments == ["export"];
    if !matches_path {
        return false;
    }
    let syn::Meta::List(list) = &attribute.meta else {
        return false;
    };
    list.tokens
        .to_string()
        .split(|character: char| !character.is_alphanumeric() && character != '_')
        .any(|token| token == kind)
}

pub fn free_function_name(factory: &Factory) -> String {
    format!(
        "{}{}",
        AsLowerCamelCase(AsSnakeCase(&factory.type_name).to_string()),
        capitalize_first(&AsLowerCamelCase(&factory.method_name).to_string()),
    )
}

fn capitalize_first(input: &str) -> String {
    let mut characters = input.chars();
    match characters.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
        None => String::new(),
    }
}
