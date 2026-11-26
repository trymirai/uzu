use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, bail};
use heck::{AsLowerCamelCase, AsUpperCamelCase};
use ignore::WalkBuilder;

// The exported items a language's extension file is generated from.
#[derive(Default)]
pub struct ForeignExtensions {
    pub aliases: BTreeMap<String, ForeignType>,
    pub types: BTreeMap<String, ExportedType>,
}

// What one exported type gains in a language, in source order.
#[derive(Default)]
pub struct ExportedType {
    pub factories: Vec<Factory>,
    pub callback_factories: Vec<CallbackFactory>,
    pub stream: Option<Stream>,
}

pub struct Factory {
    pub name: String,
    pub is_async: bool,
    pub throws: bool,
    pub arguments: Vec<Argument>,
}

pub struct Argument {
    pub name: String,
    pub foreign_type: ForeignType,
}

pub struct CallbackFactory {
    pub name: String,
    pub callback_arguments: Vec<ForeignType>,
}

pub struct Stream {
    pub next_method: String,
    pub item_type: ForeignType,
    pub cancel_getter: Option<String>,
}

// A Rust type as the bindings expose it; each language spells it when it renders.
pub enum ForeignType {
    Optional(Box<ForeignType>),
    List(Box<ForeignType>),
    Named(String),
}

impl ForeignExtensions {
    pub fn collect(crates_path: &Path) -> Result<Self> {
        let mut extensions = Self::default();
        for entry in WalkBuilder::new(crates_path).sort_by_file_name(|left, right| left.cmp(right)).build() {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|extension| extension.to_str()) != Some("rs") {
                continue;
            }
            let source = fs::read_to_string(path).with_context(|| format!("Failed to read {}", path.display()))?;
            let file = syn::parse_file(&source).with_context(|| format!("Failed to parse {}", path.display()))?;
            extensions.collect_items(&file.items).with_context(|| format!("In {}", path.display()))?;
        }
        Ok(extensions)
    }

    fn collect_items(
        &mut self,
        items: &[syn::Item],
    ) -> Result<()> {
        for item in items {
            match item {
                syn::Item::Impl(item_impl) if item_impl.attrs.exports("Implementation") => {
                    self.collect_implementation(item_impl)?
                },
                syn::Item::Type(alias) if alias.attrs.exports("Alias") => {
                    let name = alias.ident.to_string();
                    let foreign_type = ForeignType::from_rust(&alias.ty, "")
                        .with_context(|| format!("Alias {name} has a type the bindings cannot spell"))?;
                    self.aliases.insert(name, foreign_type);
                },
                syn::Item::Mod(module) => {
                    if let Some((_, items)) = &module.content {
                        self.collect_items(items)?;
                    }
                },
                _ => {},
            }
        }
        Ok(())
    }

    fn collect_implementation(
        &mut self,
        item_impl: &syn::ItemImpl,
    ) -> Result<()> {
        let type_name =
            item_impl.self_ty.last_ident().context("A bindings implementation is not for a named type")?.to_string();
        let methods = item_impl.items.iter().filter_map(|item| match item {
            syn::ImplItem::Fn(method) => Some(method),
            _ => None,
        });
        let cancel_getter = methods
            .clone()
            .find(|method| {
                method.attrs.exports("Getter")
                    && method.sig.returned().and_then(TypeExt::last_ident).is_some_and(|ident| ident == "CancelToken")
            })
            .map(|method| method.sig.ident.to_string());

        for method in methods {
            let method_name = &method.sig.ident;
            if method.attrs.exports("StreamNext") {
                let stream = Stream::parse(method, &type_name, cancel_getter.clone())
                    .with_context(|| format!("{type_name}::{method_name} is not a stream next method"))?;
                let exported = self.types.entry(type_name.clone()).or_default();
                if exported.stream.replace(stream).is_some() {
                    bail!("{type_name} has more than one stream next method");
                }
            } else if method.attrs.exports("FactoryWithCallback") {
                let factory = CallbackFactory::parse(method, &type_name)
                    .with_context(|| format!("{type_name}::{method_name} takes no callback the bindings can spell"))?;
                self.types.entry(type_name.clone()).or_default().callback_factories.push(factory);
            } else if method.attrs.exports("Factory") {
                let factory = Factory::parse(method, &type_name)
                    .with_context(|| format!("{type_name}::{method_name} is not a factory the bindings can spell"))?;
                self.types.entry(type_name.clone()).or_default().factories.push(factory);
            }
        }
        Ok(())
    }
}

impl Factory {
    // A static method returning `Self`, or `Result<Self, _>` when it throws.
    fn parse(
        method: &syn::ImplItemFn,
        type_name: &str,
    ) -> Option<Self> {
        let returned = method.sig.returned()?;
        let result = returned.generic_argument("Result");
        let returned = result.unwrap_or(returned).last_ident()?;
        if method.sig.receiver().is_some() || (returned != "Self" && returned != type_name) {
            return None;
        }
        let arguments = method
            .sig
            .inputs
            .iter()
            .map(|input| {
                let syn::FnArg::Typed(argument) = input else {
                    return None;
                };
                let syn::Pat::Ident(pattern) = argument.pat.as_ref() else {
                    return None;
                };
                Some(Argument {
                    name: pattern.ident.to_string(),
                    foreign_type: ForeignType::from_rust(&argument.ty, type_name)?,
                })
            })
            .collect::<Option<_>>()?;
        Some(Self {
            name: method.sig.ident.to_string(),
            is_async: method.sig.asyncness.is_some(),
            throws: result.is_some(),
            arguments,
        })
    }

    // The free function the bindings export the factory as.
    pub fn function_name(
        &self,
        type_name: &str,
    ) -> String {
        format!("{}{}", AsLowerCamelCase(type_name), AsUpperCamelCase(&self.name))
    }
}

impl CallbackFactory {
    // A static method taking a `Box<dyn Fn(..)>`.
    fn parse(
        method: &syn::ImplItemFn,
        type_name: &str,
    ) -> Option<Self> {
        if method.sig.receiver().is_some() {
            return None;
        }
        let callback_arguments = method.sig.inputs.iter().find_map(|input| {
            let syn::FnArg::Typed(argument) = input else {
                return None;
            };
            let syn::Type::TraitObject(trait_object) = argument.ty.generic_argument("Box")? else {
                return None;
            };
            let segment = trait_object.bounds.iter().find_map(|bound| match bound {
                syn::TypeParamBound::Trait(bound) => bound.path.segments.last(),
                _ => None,
            })?;
            let syn::PathArguments::Parenthesized(callback) = &segment.arguments else {
                return None;
            };
            callback.inputs.iter().map(|input| ForeignType::from_rust(&input.ty, type_name)).collect()
        })?;
        Some(Self {
            name: method.sig.ident.to_string(),
            callback_arguments,
        })
    }
}

impl Stream {
    // A method returning `Option<Item>`, `None` once the stream ends.
    fn parse(
        method: &syn::ImplItemFn,
        type_name: &str,
        cancel_getter: Option<String>,
    ) -> Option<Self> {
        let item = method.sig.returned()?.generic_argument("Option")?;
        Some(Self {
            next_method: method.sig.ident.to_string(),
            item_type: ForeignType::from_rust(item, type_name)?,
            cancel_getter,
        })
    }
}

impl ForeignType {
    fn from_rust(
        ty: &syn::Type,
        type_name: &str,
    ) -> Option<Self> {
        if let Some(inner) = ty.generic_argument("Option") {
            return Some(Self::Optional(Box::new(Self::from_rust(inner, type_name)?)));
        }
        if let Some(inner) = ty.generic_argument("Vec") {
            return Some(Self::List(Box::new(Self::from_rust(inner, type_name)?)));
        }
        let name = ty.last_ident()?;
        Some(Self::Named(if name == "Self" {
            type_name.to_string()
        } else {
            name.to_string()
        }))
    }
}

trait AttributesExt {
    // Whether a `#[bindings::export(..)]` attribute names `kind`.
    fn exports(
        &self,
        kind: &str,
    ) -> bool;
}

impl AttributesExt for [syn::Attribute] {
    fn exports(
        &self,
        kind: &str,
    ) -> bool {
        self.iter().any(|attribute| {
            let path = attribute.path();
            let syn::Meta::List(list) = &attribute.meta else {
                return false;
            };
            (path.is_ident("export") || path.segments.iter().map(|segment| &segment.ident).eq(["bindings", "export"]))
                && list
                    .tokens
                    .to_string()
                    .split(|character: char| !character.is_alphanumeric() && character != '_')
                    .any(|token| token == kind)
        })
    }
}

trait SignatureExt {
    fn returned(&self) -> Option<&syn::Type>;
}

impl SignatureExt for syn::Signature {
    fn returned(&self) -> Option<&syn::Type> {
        match &self.output {
            syn::ReturnType::Type(_, returned) => Some(returned),
            syn::ReturnType::Default => None,
        }
    }
}

trait TypeExt {
    fn last_ident(&self) -> Option<&syn::Ident>;

    // `T` when this is `wrapper<T, ..>`.
    fn generic_argument(
        &self,
        wrapper: &str,
    ) -> Option<&syn::Type>;
}

impl TypeExt for syn::Type {
    fn last_ident(&self) -> Option<&syn::Ident> {
        let syn::Type::Path(type_path) = self else {
            return None;
        };
        type_path.path.segments.last().map(|segment| &segment.ident)
    }

    fn generic_argument(
        &self,
        wrapper: &str,
    ) -> Option<&syn::Type> {
        let syn::Type::Path(type_path) = self else {
            return None;
        };
        let segment = type_path.path.segments.last()?;
        let syn::PathArguments::AngleBracketed(arguments) = &segment.arguments else {
            return None;
        };
        if segment.ident != wrapper {
            return None;
        }
        arguments.args.iter().find_map(|argument| match argument {
            syn::GenericArgument::Type(inner) => Some(inner),
            _ => None,
        })
    }
}
