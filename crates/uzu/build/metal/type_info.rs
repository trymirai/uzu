use anyhow::{Context, bail};

use super::{cst::SyntaxNode, token::SyntaxKind};

#[derive(Debug, Clone, PartialEq)]
pub struct ParsedType {
    pub base: BaseType,
    pub qualifiers: Vec<TypeQualifier>,
    pub declarator: Declarator,
    pub namespace: Option<Vec<String>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BaseType {
    Bool,
    Int,
    UInt,
    Float,
    Simd,
    Named(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TypeQualifier {
    Const,
    Device,
    Threadgroup,
    Constant,
    Thread,
    Volatile,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Declarator {
    Value,
    Pointer,
    Reference,
    Array(Option<String>),
}

impl ParsedType {
    pub fn from_cst(node: &SyntaxNode) -> anyhow::Result<Self> {
        if node.kind() != SyntaxKind::TypeRef {
            bail!("Expected TypeRef node, got {:?}", node.kind());
        }

        let mut qualifiers = Vec::new();
        let mut base = None;
        let mut namespace = Vec::new();
        let mut declarator = Declarator::Value;
        let mut array_size = None;

        for element in node.descendants_with_tokens() {
            if let rowan::NodeOrToken::Token(token) = element {
                match token.kind() {
                    SyntaxKind::KwConst => qualifiers.push(TypeQualifier::Const),
                    SyntaxKind::KwDevice => qualifiers.push(TypeQualifier::Device),
                    SyntaxKind::KwThreadgroup => qualifiers.push(TypeQualifier::Threadgroup),
                    SyntaxKind::KwConstant => qualifiers.push(TypeQualifier::Constant),
                    SyntaxKind::KwThread => qualifiers.push(TypeQualifier::Thread),
                    SyntaxKind::KwVolatile => qualifiers.push(TypeQualifier::Volatile),
                    SyntaxKind::Star if matches!(declarator, Declarator::Value) => declarator = Declarator::Pointer,
                    SyntaxKind::Ampersand if matches!(declarator, Declarator::Value) => {
                        declarator = Declarator::Reference
                    },
                    _ => {},
                }
            }
        }

        for child in node.children_with_tokens() {
            match child {
                rowan::NodeOrToken::Token(token) => match token.kind() {
                    SyntaxKind::KwBool if base.is_none() => base = Some(BaseType::Bool),
                    SyntaxKind::KwInt | SyntaxKind::KwInt32T if base.is_none() => base = Some(BaseType::Int),
                    SyntaxKind::KwUInt | SyntaxKind::KwUInt32T if base.is_none() => base = Some(BaseType::UInt),
                    SyntaxKind::KwUnsigned if base.is_none() => base = Some(BaseType::UInt),
                    SyntaxKind::KwFloat if base.is_none() => base = Some(BaseType::Float),
                    SyntaxKind::KwSimd if base.is_none() => base = Some(BaseType::Simd),
                    _ => {},
                },
                rowan::NodeOrToken::Node(node) => match node.kind() {
                    SyntaxKind::QualifiedName => {
                        let mut name_segments = Vec::new();
                        for token in node.descendants_with_tokens().filter_map(|element| element.into_token()) {
                            if token.kind() == SyntaxKind::Ident {
                                name_segments.push(token.text().to_string());
                            }
                        }
                        if let Some(type_name) = name_segments.pop() {
                            namespace = name_segments;
                            base = Some(BaseType::Named(type_name));
                        }
                    },
                    SyntaxKind::ArrayDeclarator => {
                        let mut size = String::new();
                        for token in node.descendants_with_tokens().filter_map(|element| element.into_token()) {
                            match token.kind() {
                                SyntaxKind::LBracket | SyntaxKind::RBracket => {},
                                _ => size.push_str(token.text()),
                            }
                        }
                        if !size.is_empty() {
                            array_size = Some(size);
                        }
                        declarator = Declarator::Array(array_size.clone());
                    },
                    SyntaxKind::BaseType => {
                        let qualified_parts = node
                            .children()
                            .find(|child| child.kind() == SyntaxKind::QualifiedName)
                            .map(|qualified_name| {
                                qualified_name
                                    .descendants_with_tokens()
                                    .filter_map(|element| element.into_token())
                                    .filter(|token| token.kind() == SyntaxKind::Ident)
                                    .map(|token| token.text().to_string())
                                    .collect::<Vec<_>>()
                            })
                            .unwrap_or_default();

                        if !qualified_parts.is_empty() {
                            let mut name_segments = qualified_parts;
                            if let Some(type_name) = name_segments.pop() {
                                namespace = name_segments;
                                base = Some(BaseType::Named(type_name));
                            }
                            continue;
                        }

                        let token_kinds = node
                            .descendants_with_tokens()
                            .filter_map(|element| element.into_token())
                            .map(|token| token.kind())
                            .collect::<Vec<_>>();

                        if token_kinds.contains(&SyntaxKind::KwBool) {
                            base = Some(BaseType::Bool);
                        } else if token_kinds.contains(&SyntaxKind::KwUnsigned)
                            || token_kinds.contains(&SyntaxKind::KwUInt)
                            || token_kinds.contains(&SyntaxKind::KwUInt32T)
                        {
                            base = Some(BaseType::UInt);
                        } else if token_kinds.contains(&SyntaxKind::KwInt)
                            || token_kinds.contains(&SyntaxKind::KwInt32T)
                        {
                            base = Some(BaseType::Int);
                        } else if token_kinds.contains(&SyntaxKind::KwFloat) {
                            base = Some(BaseType::Float);
                        } else if token_kinds.contains(&SyntaxKind::KwSimd) {
                            base = Some(BaseType::Simd);
                        } else if let Some(token) = node
                            .descendants_with_tokens()
                            .filter_map(|element| element.into_token())
                            .find(|token| token.kind() == SyntaxKind::Ident)
                        {
                            base = Some(BaseType::Named(token.text().to_string()));
                        }
                    },
                    _ => {},
                },
            }
        }

        let base = base.context("No base type found in TypeRef")?;
        let namespace = if namespace.is_empty() {
            None
        } else {
            Some(namespace)
        };

        Ok(ParsedType {
            base,
            qualifiers,
            declarator,
            namespace,
        })
    }

    pub fn to_rust_type(&self) -> anyhow::Result<Box<str>> {
        let base_rust_type = match &self.base {
            BaseType::Bool => "bool",
            BaseType::Int => "i32",
            BaseType::UInt => "u32",
            BaseType::Float => "f32",
            BaseType::Simd => bail!("Simd type cannot be converted to Rust scalar type"),
            BaseType::Named(name) => {
                if let Some(namespace_segments) = &self.namespace {
                    if namespace_segments.len() == 1 && namespace_segments[0] == "uzu" {
                        return Ok(format!("crate::backends::common::gpu_types::{}", name).into());
                    }
                }
                return Ok(name.clone().into());
            },
        };

        Ok(base_rust_type.into())
    }

    pub fn is_buffer(&self) -> bool {
        matches!(self.declarator, Declarator::Pointer) && self.qualifiers.contains(&TypeQualifier::Device)
    }

    pub fn is_constant_scalar(&self) -> bool {
        matches!(self.declarator, Declarator::Reference)
            && self.qualifiers.contains(&TypeQualifier::Const)
            && self.qualifiers.contains(&TypeQualifier::Constant)
    }

    pub fn is_constant_array(&self) -> bool {
        matches!(self.declarator, Declarator::Pointer)
            && self.qualifiers.contains(&TypeQualifier::Const)
            && self.qualifiers.contains(&TypeQualifier::Constant)
    }

    pub fn is_threadgroup(&self) -> bool {
        self.qualifiers.contains(&TypeQualifier::Threadgroup)
            && matches!(self.declarator, Declarator::Pointer | Declarator::Reference | Declarator::Array(_))
    }

    pub fn is_simd(&self) -> bool {
        matches!(self.base, BaseType::Simd)
    }
}
