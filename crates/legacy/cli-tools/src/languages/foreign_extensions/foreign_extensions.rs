use std::{collections::BTreeMap, fs, path::Path};

use anyhow::Result;
use ignore::WalkBuilder;

use super::{
    callback_factory::CallbackFactory, factory::Factory, factory_argument::FactoryArgument, stream_bridge::StreamBridge,
};

/// What the bindings export beyond UniFFI's own surface: type aliases, static factories, closure-taking factories and
/// streams. Each language renders these as idiomatic sugar over the generated free functions and `next()` methods.
pub struct ForeignExtensions {
    pub aliases: BTreeMap<String, syn::Type>,
    pub factories: Vec<Factory>,
    pub callback_factories: Vec<CallbackFactory>,
    pub stream_bridges: Vec<StreamBridge>,
}

impl ForeignExtensions {
    pub fn collect(crates_path: &Path) -> Result<Self> {
        let mut extensions = Self {
            aliases: BTreeMap::new(),
            factories: Vec::new(),
            callback_factories: Vec::new(),
            stream_bridges: Vec::new(),
        };

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
            extensions.collect_from_items(&file.items);
        }

        extensions
            .factories
            .sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
        extensions.factories.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
        extensions
            .callback_factories
            .sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
        extensions
            .callback_factories
            .dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
        extensions.stream_bridges.sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name));
        extensions.stream_bridges.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name);

        Ok(extensions)
    }

    fn collect_from_items(
        &mut self,
        items: &[syn::Item],
    ) {
        for item in items {
            match item {
                syn::Item::Impl(item_impl) => self.collect_from_impl(item_impl),
                syn::Item::Type(item_type)
                    if item_type.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "Alias")) =>
                {
                    self.aliases.insert(item_type.ident.to_string(), item_type.ty.as_ref().clone());
                },
                syn::Item::Mod(item_mod) => {
                    if let Some((_, nested)) = &item_mod.content {
                        self.collect_from_items(nested);
                    }
                },
                _ => {},
            }
        }
    }

    fn collect_from_impl(
        &mut self,
        item_impl: &syn::ItemImpl,
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
                if let Some(bridge) = extract_stream_bridge(method, &type_name, cancel_getter_name.clone()) {
                    self.stream_bridges.push(bridge);
                }
                continue;
            }
            if method_has_receiver(method) {
                continue;
            }
            if method_is_factory_with_callback(method) {
                self.callback_factories.push(CallbackFactory {
                    type_name: type_name.clone(),
                    method_name: method.sig.ident.to_string(),
                    callback_arg_types: extract_callback_arg_types(method).unwrap_or_default(),
                });
                continue;
            }
            if !method_is_factory(method) {
                continue;
            }
            if let Some(factory) = extract_factory(method, &type_name) {
                self.factories.push(factory);
            }
        }
    }
}

/// Spells a Rust type in a foreign language: `primitive` names its scalars, `list` wraps a spelled type, and an
/// optional takes a `?` suffix. Unknown paths keep their Rust name, which the bindings export under the same name.
pub fn foreign_type(
    ty: &syn::Type,
    self_type_name: &str,
    primitive: fn(&str) -> Option<&'static str>,
    list: fn(String) -> String,
) -> Option<String> {
    if let Some(inner) = path_generic(ty, "Option") {
        return foreign_type(inner, self_type_name, primitive, list).map(|inner| format!("{inner}?"));
    }
    if let Some(inner) = path_generic(ty, "Vec") {
        return foreign_type(inner, self_type_name, primitive, list).map(list);
    }
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let name = type_path.path.segments.last()?.ident.to_string();
    if name == "Self" {
        return Some(self_type_name.to_string());
    }
    Some(primitive(&name).map(str::to_string).unwrap_or(name))
}

fn method_is_stream_next(method: &syn::ImplItemFn) -> bool {
    method.attrs.iter().any(|attribute| is_bindings_export_with(attribute, "StreamNext"))
}

fn extract_stream_bridge(
    method: &syn::ImplItemFn,
    type_name: &str,
    cancel_getter_name: Option<String>,
) -> Option<StreamBridge> {
    let syn::ReturnType::Type(_, return_type) = &method.sig.output else {
        return None;
    };
    let inner = path_generic(return_type.as_ref(), "Option")?;
    Some(StreamBridge {
        type_name: type_name.to_string(),
        next_method_name: method.sig.ident.to_string(),
        item_type: inner.clone(),
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

fn extract_callback_arg_types(method: &syn::ImplItemFn) -> Option<Vec<syn::Type>> {
    method.sig.inputs.iter().find_map(|input| {
        let syn::FnArg::Typed(pat_type) = input else {
            return None;
        };
        extract_box_fn_inputs(&pat_type.ty)
    })
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
) -> Option<Factory> {
    let (throws, optional) = classify_return(&method.sig.output, type_name)?;

    let mut arguments = Vec::with_capacity(method.sig.inputs.len());
    for input in &method.sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else {
            return None;
        };
        let syn::Pat::Ident(pat_ident) = pat_type.pat.as_ref() else {
            return None;
        };
        arguments.push(FactoryArgument {
            name: pat_ident.ident.to_string(),
            ty: pat_type.ty.as_ref().clone(),
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

/// Whether a factory returns its own type, and if so whether it throws and whether the value is optional.
fn classify_return(
    output: &syn::ReturnType,
    self_type_name: &str,
) -> Option<(bool, bool)> {
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
    if segment.ident != "Self" && segment.ident != self_type_name {
        return None;
    }
    Some((throws, optional))
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
