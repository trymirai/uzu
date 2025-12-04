use std::collections::HashMap;

use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

pub type MetalAstNode = clang_ast::Node<MetalAstKind>;

#[derive(Debug, Clone, Deserialize)]
pub struct ParmVarDeclType {
    #[serde(rename = "qualType")]
    qual_type: Box<str>,
    #[serde(rename = "desugaredQualType")]
    desugared_qual_type: Option<Box<str>>,
}

#[derive(Debug, Clone, Deserialize)]
pub enum MetalAstKind {
    TranslationUnitDecl,
    FunctionTemplateDecl,
    TemplateTypeParmDecl,
    FunctionDecl {
        name: Box<str>,
    },
    ParmVarDecl {
        name: Option<Box<str>>,
        range: clang_ast::SourceRange,
        #[serde(rename = "type")]
        ty: ParmVarDeclType,
    },
    AnnotateAttr,
    ConstantExpr,
    ImplicitCastExpr,
    StringLiteral {
        value: Box<str>,
    },
    Other,
}

fn annotation_from_ast_node(
    annotation_node: MetalAstNode
) -> anyhow::Result<Box<[Box<str>]>> {
    if !matches!(annotation_node.kind, MetalAstKind::AnnotateAttr) {
        bail!(
            "unexpected kind of root node: MetalAstKind::AnnotateAttr expected, but {:?} found",
            annotation_node.kind
        );
    }

    annotation_node
        .inner
        .into_iter()
        .map(|mut constant_expr| {
            let MetalAstKind::ConstantExpr = constant_expr.kind else {
                bail!("expected ConstantExpr, found {:?}", constant_expr.kind);
            };

            if constant_expr.inner.len() != 1 {
                bail!(
                    "ConstantExpr must have exactly one child, found {}",
                    constant_expr.inner.len()
                );
            }

            let mut implicit_cast_expr = constant_expr.inner.pop().unwrap();

            let MetalAstKind::ImplicitCastExpr = implicit_cast_expr.kind else {
                bail!(
                    "expected ImplicitCastExpr, found {:?}",
                    implicit_cast_expr.kind
                );
            };

            if implicit_cast_expr.inner.len() != 1 {
                bail!(
                    "ImplicitCastExpr must have exactly one child, found {}",
                    implicit_cast_expr.inner.len()
                );
            }

            let string_literal = implicit_cast_expr.inner.pop().unwrap();

            let MetalAstKind::StringLiteral {
                value,
            } = string_literal.kind
            else {
                bail!(
                    "expected StringLiteral, found {:?}",
                    string_literal.kind
                );
            };

            // NOTE: string literal includes "" (and is probably not parsed?), using json parse here for now
            Ok(serde_json::from_str(&value)
                .context("failed to parse string literal")?)
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalArgumentType {
    Buffer,
    Constant(Box<str>),
    Shared(Box<str>),
    Axis(Box<str>, Box<str>),
    Groups(Box<str>),
    Threads(Box<str>),
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalArgument {
    pub name: Box<str>,
    pub c_type: Box<str>,
    pub annotation: Option<Box<[Box<str>]>>,
    pub source: Box<str>,
}

impl MetalArgument {
    fn from_ast_node_and_source(
        argument_node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Self> {
        let MetalAstKind::ParmVarDecl {
            name,
            range,
            ty:
                ParmVarDeclType {
                    qual_type,
                    desugared_qual_type,
                },
        } = argument_node.kind
        else {
            bail!("argument isn't ParmVarDecl: {:?}", argument_node.kind);
        };

        let name = name.context("ParmVarDecl has no name")?;

        let c_type = desugared_qual_type.unwrap_or(qual_type);

        if argument_node.inner.len() > 1 {
            bail!("more than one annotation on argument ast node");
        }

        let annotation =
            if let Some(annotation_node) = argument_node.inner.first() {
                Some(annotation_from_ast_node(annotation_node.clone())?)
            } else {
                None
            };

        let start_offset = range
            .begin
            .spelling_loc
            .context("no start location in source range")?
            .offset;
        let end_offset = range
            .end
            .spelling_loc
            .context("no end location in source range")?
            .offset;
        let source =
            str::from_utf8(&source.as_bytes()[start_offset..=end_offset])
                .context("source range is not utf-8")?
                .into();

        Ok(Self {
            name,
            c_type,
            annotation,
            source,
        })
    }

    pub fn argument_type(&self) -> anyhow::Result<MetalArgumentType> {
        if let Some(annotation) = self.annotation.as_ref() {
            let mut annotation = annotation.to_vec();

            if annotation.is_empty() {
                bail!("empty annotation");
            }
            let annotation_key = annotation.remove(0);

            match &*annotation_key {
                "dsl.axis" => {
                    if annotation.len() != 2 {
                        bail!(
                            "dsl.axis requires 2 arguments, got {}",
                            annotation.len()
                        );
                    }
                    Ok(MetalArgumentType::Axis(
                        annotation.remove(0),
                        annotation.remove(0),
                    ))
                },
                "dsl.groups" => {
                    if annotation.len() != 1 {
                        bail!(
                            "dsl.groups requires 1 argument, got {}",
                            annotation.len()
                        );
                    }
                    Ok(MetalArgumentType::Groups(annotation.remove(0)))
                },
                "dsl.threads" => {
                    if annotation.len() != 1 {
                        bail!(
                            "dsl.threads requires 1 argument, got {}",
                            annotation.len()
                        );
                    }
                    Ok(MetalArgumentType::Threads(annotation.remove(0)))
                },
                _ => bail!("unknown annotation: {annotation_key}"),
            }
        } else if (self.c_type.contains("device")
            || self.c_type.contains("constant"))
            && self.c_type.contains('*')
            && !self.c_type.contains('&')
        {
            Ok(MetalArgumentType::Buffer)
        } else if let ["const", "constant", c_type_scalar, "&"] =
            self.c_type.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            let rust_type = match *c_type_scalar {
                "uint" | "uint32_t" => "u32",
                "int" | "int32_t" => "i32",
                "float" => "f32",
                _ => {
                    bail!("unknown scalar type: {c_type_scalar}")
                },
            };
            Ok(MetalArgumentType::Constant(rust_type.into()))
        } else if self.c_type.contains("threadgroup")
            && self.c_type.contains('*')
        {
            let lbracket = self
                .source
                .rfind('[')
                .context("threadgroup missing size bracket")?
                + 1;
            let rbracket = self
                .source
                .rfind(']')
                .context("threadgroup missing size bracket")?;
            let size_expr = &self.source[lbracket..rbracket];
            Ok(MetalArgumentType::Shared(size_expr.into()))
        } else {
            bail!("cannot parse c type: {}", self.c_type);
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalKernelInfo {
    pub name: Box<str>,
    pub arguments: Box<[MetalArgument]>,
    pub specializations: Option<(Box<str>, Box<[Box<str>]>)>,
}

impl MetalKernelInfo {
    pub fn has_axis(&self) -> bool {
        self.arguments.iter().any(|a| {
            matches!(a.argument_type(), Ok(MetalArgumentType::Axis(..)))
        })
    }

    pub fn has_groups(&self) -> bool {
        self.arguments.iter().any(|a| {
            matches!(a.argument_type(), Ok(MetalArgumentType::Groups(_)))
        })
    }

    pub fn has_threads(&self) -> bool {
        self.arguments.iter().any(|a| {
            matches!(a.argument_type(), Ok(MetalArgumentType::Threads(_)))
        })
    }
}

impl MetalKernelInfo {
    pub fn from_ast_node_and_source(
        node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Option<Self>> {
        let (is_template, node) =
            if matches!(node.kind, MetalAstKind::FunctionTemplateDecl) {
                let node = node
                .inner
                .into_iter()
                .find(|c| {
                    matches!(
                        c.kind,
                        MetalAstKind::FunctionDecl {
                            name: _
                        }
                    )
                })
                .context(
                    "unexpected kind of root node: template without function",
                )?;

                (true, node)
            } else if matches!(node.kind, MetalAstKind::FunctionDecl { .. }) {
                (false, node)
            } else {
                return Ok(None);
            };

        let MetalAstKind::FunctionDecl {
            name,
        } = node.kind
        else {
            bail!(
                "unexpected kind of root node: function expected, but {:?} found",
                node.kind
            );
        };

        let mut arg_nodes = Vec::new();
        let mut annotations = Vec::new();

        for node in node.inner {
            match node.kind {
                MetalAstKind::ParmVarDecl {
                    name: _,
                    range: _,
                    ty: _,
                } => arg_nodes.push(node),
                MetalAstKind::AnnotateAttr => {
                    annotations.push(annotation_from_ast_node(node)?)
                },
                _ => (),
            }
        }

        let annotations = annotations
            .into_iter()
            .map(|a| {
                if !a.is_empty() {
                    let mut a = a.into_vec();
                    Ok((a.remove(0), a.into_boxed_slice()))
                } else {
                    bail!("zero length annotation");
                }
            })
            .collect::<anyhow::Result<HashMap<_, _>>>()?;

        if !annotations.contains_key("dsl.kernel") {
            return Ok(None);
        }

        if annotations.contains_key("dsl.specialize") != is_template {
            bail!("mismatch between AST nodes and specialization annotation");
        }

        let specializations = if is_template {
            let specialize = annotations
                .get("dsl.specialize")
                .context("missing dsl.specialize annotation")?;
            let [specialization_typename, specialization_variants] =
                specialize.as_ref()
            else {
                bail!("malformed dsl.specialize annotation");
            };

            let specialization_variants = specialization_variants
                .split(',')
                .map(|v| v.trim().into())
                .collect::<Box<[Box<str>]>>();

            Some((specialization_typename.clone(), specialization_variants))
        } else {
            None
        };

        let arguments = arg_nodes
            .into_iter()
            .map(|an| MetalArgument::from_ast_node_and_source(an, source))
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Some(MetalKernelInfo {
            name,
            arguments,
            specializations,
        }))
    }
}
