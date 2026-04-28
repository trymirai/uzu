use std::{
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;
use heck::{AsLowerCamelCase, AsSnakeCase};
use ignore::WalkBuilder;

pub fn generate_swift_extensions(
    crates_path: PathBuf,
    destination_path: PathBuf,
) -> Result<()> {
    let (factories, callback_factories, stream_bridges) = collect_all(&crates_path)?;
    let swift_source = render_extensions_file(&factories, &callback_factories, &stream_bridges);
    fs::write(&destination_path, swift_source)?;
    Ok(())
}

#[derive(Debug, Clone)]
struct Factory {
    type_name: String,
    method_name: String,
    is_async: bool,
    throws: bool,
    optional: bool,
    arguments: Vec<FactoryArgument>,
}

#[derive(Debug, Clone)]
struct FactoryArgument {
    name: String,
    swift_type: String,
}

#[derive(Debug, Clone)]
struct CallbackFactory {
    type_name: String,
    method_name: String,
    callback_arg_types: Vec<String>,
}

#[derive(Debug, Clone)]
struct StreamBridge {
    type_name: String,
    next_method_name: String,
    item_swift_type: String,
    cancel_getter_name: Option<String>,
}

fn collect_all(crates_path: &Path) -> Result<(Vec<Factory>, Vec<CallbackFactory>, Vec<StreamBridge>)> {
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
        collect_from_items(&file.items, &mut factories, &mut callback_factories, &mut stream_bridges);
    }

    factories.sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
    factories.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
    callback_factories
        .sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name).then_with(|| lhs.method_name.cmp(&rhs.method_name)));
    callback_factories.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name && lhs.method_name == rhs.method_name);
    stream_bridges.sort_by(|lhs, rhs| lhs.type_name.cmp(&rhs.type_name));
    stream_bridges.dedup_by(|lhs, rhs| lhs.type_name == rhs.type_name);

    Ok((factories, callback_factories, stream_bridges))
}

fn collect_from_items(
    items: &[syn::Item],
    factories: &mut Vec<Factory>,
    callback_factories: &mut Vec<CallbackFactory>,
    stream_bridges: &mut Vec<StreamBridge>,
) {
    for item in items {
        match item {
            syn::Item::Impl(item_impl) => collect_from_impl(item_impl, factories, callback_factories, stream_bridges),
            syn::Item::Mod(item_mod) => {
                if let Some((_, nested)) = &item_mod.content {
                    collect_from_items(nested, factories, callback_factories, stream_bridges);
                }
            },
            _ => {},
        }
    }
}

fn collect_from_impl(
    item_impl: &syn::ItemImpl,
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
            if let Some(bridge) = extract_stream_bridge(method, &type_name, cancel_getter_name.clone()) {
                stream_bridges.push(bridge);
            }
            continue;
        }
        if method_has_receiver(method) {
            continue;
        }
        if method_is_factory_with_callback(method) {
            let callback_arg_types = extract_callback_arg_swift_types(method, &type_name).unwrap_or_default();
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
        let Some(factory) = extract_factory(method, &type_name) else {
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
) -> Option<StreamBridge> {
    let syn::ReturnType::Type(_, return_type) = &method.sig.output else {
        return None;
    };
    let inner = path_generic(return_type.as_ref(), "Option")?;
    let swift_type = rust_type_to_swift(inner, type_name)?;
    Some(StreamBridge {
        type_name: type_name.to_string(),
        next_method_name: method.sig.ident.to_string(),
        item_swift_type: swift_type,
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

fn extract_callback_arg_swift_types(
    method: &syn::ImplItemFn,
    self_type_name: &str,
) -> Option<Vec<String>> {
    for input in &method.sig.inputs {
        let syn::FnArg::Typed(pat_type) = input else {
            continue;
        };
        let Some(inputs) = extract_box_fn_inputs(&pat_type.ty) else {
            continue;
        };
        let swift_types: Vec<String> = inputs
            .iter()
            .map(|ty| rust_type_to_swift(ty, self_type_name).unwrap_or_else(|| "Any".to_string()))
            .collect();
        return Some(swift_types);
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
        return Some(paren.inputs.iter().cloned().collect());
    }
    None
}

fn extract_factory(
    method: &syn::ImplItemFn,
    type_name: &str,
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
        let swift_type = rust_type_to_swift(&pat_type.ty, type_name)?;
        arguments.push(FactoryArgument {
            name: pat_ident.ident.to_string(),
            swift_type,
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

fn rust_type_to_swift(
    ty: &syn::Type,
    self_type_name: &str,
) -> Option<String> {
    if let Some(inner) = path_generic(ty, "Option") {
        let inner_swift = rust_type_to_swift(inner, self_type_name)?;
        return Some(format!("{}?", inner_swift));
    }
    if let Some(inner) = path_generic(ty, "Vec") {
        let inner_swift = rust_type_to_swift(inner, self_type_name)?;
        return Some(format!("[{}]", inner_swift));
    }
    let syn::Type::Path(type_path) = ty else {
        return None;
    };
    let segment = type_path.path.segments.last()?;
    let name = segment.ident.to_string();
    Some(match name.as_str() {
        "Self" => self_type_name.to_string(),
        "String" => "String".to_string(),
        "bool" => "Bool".to_string(),
        "i8" => "Int8".to_string(),
        "i16" => "Int16".to_string(),
        "i32" => "Int32".to_string(),
        "i64" => "Int64".to_string(),
        "u8" => "UInt8".to_string(),
        "u16" => "UInt16".to_string(),
        "u32" => "UInt32".to_string(),
        "u64" => "UInt64".to_string(),
        "f32" => "Float".to_string(),
        "f64" => "Double".to_string(),
        _ => name,
    })
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

fn render_extensions_file(
    factories: &[Factory],
    callback_factories: &[CallbackFactory],
    stream_bridges: &[StreamBridge],
) -> String {
    let mut output = String::new();
    output.push_str("// Auto-generated by cli-tools — do not edit.\n\n");
    for factory in factories {
        output.push_str(&render_factory(factory));
        output.push('\n');
    }
    for callback_factory in callback_factories {
        output.push_str(&render_callback_factory(callback_factory));
        output.push('\n');
    }
    for stream_bridge in stream_bridges {
        output.push_str(&render_stream_bridge(stream_bridge));
        output.push('\n');
    }
    output
}

fn render_stream_bridge(bridge: &StreamBridge) -> String {
    let swift_next = AsLowerCamelCase(&bridge.next_method_name).to_string();
    let type_name = &bridge.type_name;
    let item_type = &bridge.item_swift_type;
    let (capture, cancel_hook) = match &bridge.cancel_getter_name {
        Some(getter) => {
            let swift_getter = AsLowerCamelCase(getter).to_string();
            ("[weak self] ", format!("\n                self?.{swift_getter}().cancel()"))
        },
        None => ("", String::new()),
    };
    format!(
        r#"extension {type_name} {{
    public func iterator() -> AsyncThrowingStream<{item_type}, Swift.Error> {{
        AsyncThrowingStream<{item_type}, Swift.Error> {{ continuation in
            let task = Task {{
                while !Task.isCancelled {{
                    let item = await self.{swift_next}()
                    guard let item else {{
                        continuation.finish()
                        break
                    }}
                    continuation.yield(item)
                }}
            }}
            continuation.onTermination = {{ {capture}_ in
                task.cancel(){cancel_hook}
            }}
        }}
    }}
}}
"#,
    )
}

fn render_callback_factory(factory: &CallbackFactory) -> String {
    let type_name = &factory.type_name;
    let swift_method = AsLowerCamelCase(&factory.method_name).to_string();
    let handler_protocol = format!("{}Handler", type_name);
    let wrapper_class = format!("{}HandlerClosureWrapper", type_name);

    let closure_args_signature = factory
        .callback_arg_types
        .iter()
        .enumerate()
        .map(|(index, ty)| {
            let label = format!("arg{index}");
            format!("{label}: {ty}")
        })
        .collect::<Vec<_>>()
        .join(", ");

    let closure_args_forward_positional = factory
        .callback_arg_types
        .iter()
        .enumerate()
        .map(|(index, _)| format!("arg{index}"))
        .collect::<Vec<_>>()
        .join(", ");

    let closure_type_signature = format!("({}) -> Void", factory.callback_arg_types.join(", "),);

    format!(
        r#"extension {type_name} {{
    public typealias Closure = @Sendable {closure_type_signature}

    private final class {wrapper_class}: Sendable, {handler_protocol} {{
        private let closure: Closure

        init(closure: @escaping Closure) {{
            self.closure = closure
        }}

        func onEvent({closure_args_signature}) {{
            closure({closure_args_forward_positional})
        }}
    }}

    public static func {swift_method}(closure: @escaping Closure) -> {type_name} {{
        {type_name}.{swift_method}(handler: {wrapper_class}(closure: closure))
    }}
}}
"#,
        closure_type_signature = closure_type_signature,
        wrapper_class = wrapper_class,
        handler_protocol = handler_protocol,
        closure_args_signature = closure_args_signature,
        closure_args_forward_positional = closure_args_forward_positional,
        type_name = type_name,
        swift_method = swift_method,
    )
}

fn render_factory(factory: &Factory) -> String {
    let swift_method = AsLowerCamelCase(&factory.method_name).to_string();
    let free_function = format!(
        "{}{}",
        AsLowerCamelCase(AsSnakeCase(&factory.type_name).to_string()).to_string(),
        capitalize_first(&swift_method),
    );

    let args_signature = factory
        .arguments
        .iter()
        .map(|arg| format!("{label}: {ty}", label = AsLowerCamelCase(&arg.name).to_string(), ty = arg.swift_type,))
        .collect::<Vec<_>>()
        .join(", ");

    let args_forward = factory
        .arguments
        .iter()
        .map(|arg| {
            let label = AsLowerCamelCase(&arg.name).to_string();
            format!("{label}: {label}")
        })
        .collect::<Vec<_>>()
        .join(", ");

    let mut return_type = factory.type_name.clone();
    if factory.optional {
        return_type.push('?');
    }

    let mut modifiers = String::new();
    if factory.is_async {
        modifiers.push_str("async ");
    }
    if factory.throws {
        modifiers.push_str("throws ");
    }

    let mut invocation = String::new();
    if factory.throws {
        invocation.push_str("try ");
    }
    if factory.is_async {
        invocation.push_str("await ");
    }
    invocation.push_str(&format!("{free_function}({args_forward})"));

    format!(
        "extension {type_name} {{\n    public static func {method}({args}) {modifiers}-> {ret} {{\n        {invocation}\n    }}\n}}\n",
        type_name = factory.type_name,
        method = swift_method,
        args = args_signature,
        modifiers = modifiers,
        ret = return_type,
        invocation = invocation,
    )
}

fn capitalize_first(input: &str) -> String {
    let mut characters = input.chars();
    match characters.next() {
        Some(first) => first.to_ascii_uppercase().to_string() + characters.as_str(),
        None => String::new(),
    }
}
