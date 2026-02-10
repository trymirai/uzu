use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};

use crate::common::kernel::{Kernel, KernelArgument, KernelArgumentType, KernelParameter, KernelParameterType};

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
    TemplateTypeParmDecl {
        name: Option<Box<str>>,
    },
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
    MetalKernelAttr,
    MetalMaxTotalThreadsPerThreadGroupAttr,
    Other,
}

fn annotation_from_ast_node(annotation_node: MetalAstNode) -> anyhow::Result<Box<[Box<str>]>> {
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
                bail!("ConstantExpr must have exactly one child, found {}", constant_expr.inner.len());
            }

            let mut implicit_cast_expr = constant_expr.inner.pop().unwrap();

            let MetalAstKind::ImplicitCastExpr = implicit_cast_expr.kind else {
                bail!("expected ImplicitCastExpr, found {:?}", implicit_cast_expr.kind);
            };

            if implicit_cast_expr.inner.len() != 1 {
                bail!("ImplicitCastExpr must have exactly one child, found {}", implicit_cast_expr.inner.len());
            }

            let string_literal = implicit_cast_expr.inner.pop().unwrap();

            let MetalAstKind::StringLiteral {
                value,
            } = string_literal.kind
            else {
                bail!("expected StringLiteral, found {:?}", string_literal.kind);
            };

            // NOTE: string literal includes "" (and is probably not parsed?), using json parse here for now
            Ok(serde_json::from_str(&value).context("failed to parse string literal")?)
        })
        .collect()
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalConstantType {
    Scalar,
    Array,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalGroupsType {
    Direct(Box<str>),
    Indirect,
}

#[derive(Debug, Serialize, Deserialize)]
pub enum MetalArgumentType {
    Buffer,
    Constant((Box<str>, MetalConstantType)),
    Shared(Option<Box<str>>),
    Specialize(Box<str>),
    Axis(Box<str>, Box<str>),
    Groups(MetalGroupsType),
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
    fn scalar_type_to_rust(c_type: &str) -> anyhow::Result<Box<str>> {
        let mut tokens: Vec<_> = c_type.split_whitespace().collect();
        if tokens.first() == Some(&"const") {
            tokens.remove(0);
        }
        match tokens.as_slice() {
            ["bool"] => Ok("bool".into()),
            ["uint"] | ["uint32_t"] | ["unsigned", "int"] => Ok("u32".into()),
            ["int"] | ["int32_t"] => Ok("i32".into()),
            ["float"] => Ok("f32".into()),
            [vpath] if vpath.starts_with("uzu::") => {
                Ok(vpath.replacen("uzu::", "crate::backends::common::gpu_types::", 1).into())
            },
            _ => bail!("unknown scalar type: {c_type}"),
        }
    }

    fn from_ast_node_and_source(
        argument_node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Self> {
        let MetalAstKind::ParmVarDecl {
            name,
            range,
            ty: ParmVarDeclType {
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

        let annotation = if let Some(annotation_node) = argument_node.inner.first() {
            Some(annotation_from_ast_node(annotation_node.clone())?)
        } else {
            None
        };

        let start_offset = range.begin.spelling_loc.context("no start location in source range")?.offset;
        let end_offset = range.end.spelling_loc.context("no end location in source range")?.offset;
        let source =
            str::from_utf8(&source.as_bytes()[start_offset..=end_offset]).context("source range is not utf-8")?.into();

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
                "dsl.specialize" => {
                    if !annotation.is_empty() {
                        bail!("dsl.specialize takes no arguments, got {}", annotation.len());
                    }
                    let rust_type = Self::scalar_type_to_rust(&self.c_type)?;
                    Ok(MetalArgumentType::Specialize(rust_type.into()))
                },
                "dsl.axis" => {
                    if annotation.len() != 2 {
                        bail!("dsl.axis requires 2 arguments, got {}", annotation.len());
                    }
                    Ok(MetalArgumentType::Axis(annotation.remove(0), annotation.remove(0)))
                },
                "dsl.groups" => {
                    if annotation.len() != 1 {
                        bail!("dsl.groups requires 1 argument, got {}", annotation.len());
                    }
                    let dim = annotation.remove(0);
                    match dim.as_ref() {
                        "INDIRECT" => Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)),
                        _ => Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(dim))),
                    }
                },
                "dsl.threads" => {
                    if annotation.len() != 1 {
                        bail!("dsl.threads requires 1 argument, got {}", annotation.len());
                    }
                    Ok(MetalArgumentType::Threads(annotation.remove(0)))
                },
                _ => bail!("unknown annotation: {annotation_key}"),
            }
        } else if self.c_type.contains("device") && self.c_type.contains('*') && !self.c_type.contains('&') {
            Ok(MetalArgumentType::Buffer)
        } else if let ["const", "constant", c_type_scalar, "&"] =
            self.c_type.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            Ok(MetalArgumentType::Constant((
                Self::scalar_type_to_rust(c_type_scalar)?.into(),
                MetalConstantType::Scalar,
            )))
        } else if let ["const", "constant", c_type_scalar, "*"] =
            self.c_type.split_whitespace().collect::<Vec<_>>().as_slice()
        {
            Ok(MetalArgumentType::Constant((
                Self::scalar_type_to_rust(c_type_scalar)?.into(),
                MetalConstantType::Array,
            )))
        } else if self.c_type.contains("threadgroup") && self.c_type.contains('*') {
            let lbracket = self.source.rfind('[').context("threadgroup missing size bracket")? + 1;
            let rbracket = self.source.rfind(']').context("threadgroup missing size bracket")?;
            let size_expr = &self.source[lbracket..rbracket];
            Ok(MetalArgumentType::Shared(Some(size_expr.into())))
        } else if self.c_type.contains("threadgroup") && self.c_type.contains('&') {
            Ok(MetalArgumentType::Shared(None))
        } else {
            bail!("cannot parse c type: {}", self.c_type);
        }
    }

    fn to_parameter(&self) -> Option<KernelParameter> {
        match self.argument_type() {
            Ok(MetalArgumentType::Specialize(ty)) => Some(KernelParameter {
                name: self.name.clone(),
                ty: KernelParameterType::Specialization(ty),
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalTypeParameter {
    pub name: Box<str>,
    pub variants: Box<[Box<str>]>,
}

impl MetalTypeParameter {
    fn to_parameter(&self) -> KernelParameter {
        KernelParameter {
            name: self.name.clone(),
            ty: KernelParameterType::DType,
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct MetalKernelInfo {
    pub name: Box<str>,
    pub arguments: Box<[MetalArgument]>,
    pub variants: Option<Box<[MetalTypeParameter]>>,
}

impl MetalKernelInfo {
    pub fn has_axis(&self) -> bool {
        self.arguments.iter().any(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Axis(..))))
    }

    pub fn has_groups(&self) -> bool {
        self.arguments.iter().any(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Groups(_))))
    }

    pub fn has_groups_direct(&self) -> bool {
        self.arguments
            .iter()
            .any(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Groups(MetalGroupsType::Direct(_)))))
    }

    pub fn has_groups_indirect(&self) -> bool {
        self.arguments
            .iter()
            .any(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect))))
    }

    pub fn has_threads(&self) -> bool {
        self.arguments.iter().any(|a| matches!(a.argument_type(), Ok(MetalArgumentType::Threads(_))))
    }

    pub fn to_kernel(&self) -> Kernel {
        let mut indirect_flag = false;

        Kernel {
            name: self.name.clone(),
            parameters: self
                .variants
                .as_ref()
                .map(|v| v.iter().map(|p| p.to_parameter()).collect::<Vec<_>>())
                .unwrap_or_default()
                .into_iter()
                .chain(self.arguments.iter().filter_map(|a| a.to_parameter()))
                .collect(),
            arguments: self
                .arguments
                .iter()
                .filter_map(|a| match a.argument_type() {
                    Ok(MetalArgumentType::Buffer) => Some(KernelArgument {
                        name: a.name.clone(),
                        ty: KernelArgumentType::Buffer,
                    }),
                    Ok(MetalArgumentType::Groups(MetalGroupsType::Indirect)) if !indirect_flag => {
                        indirect_flag = true;
                        Some(KernelArgument {
                            name: "__dsl_indirect_dispatch_buffer".into(),
                            ty: KernelArgumentType::Buffer,
                        })
                    },
                    Ok(MetalArgumentType::Constant((ty, MetalConstantType::Scalar))) => Some(KernelArgument {
                        name: a.name.clone(),
                        ty: KernelArgumentType::Scalar(ty),
                    }),
                    Ok(MetalArgumentType::Constant((ty, MetalConstantType::Array))) => Some(KernelArgument {
                        name: a.name.clone(),
                        ty: KernelArgumentType::Constant(ty),
                    }),
                    _ => None,
                })
                .collect(),
        }
    }
}

impl MetalKernelInfo {
    pub fn from_ast_node_and_source(
        node: MetalAstNode,
        source: &str,
    ) -> anyhow::Result<Option<Self>> {
        let (is_template, template_parameters, node) = if matches!(node.kind, MetalAstKind::FunctionTemplateDecl) {
            let mut template_parameters = Vec::new();
            let mut function_node = None;

            for child in node.inner {
                match child.kind {
                    MetalAstKind::TemplateTypeParmDecl {
                        name,
                    } => {
                        let name = name.context("template parameter missing name")?;
                        template_parameters.push(name);
                    },
                    MetalAstKind::FunctionDecl {
                        name: _,
                    } => {
                        function_node = Some(child);
                    },
                    _ => (),
                }
            }

            let node = function_node.context("unexpected kind of root node: template without function")?;

            (true, template_parameters, node)
        } else if matches!(node.kind, MetalAstKind::FunctionDecl { .. }) {
            (false, Vec::new(), node)
        } else {
            return Ok(None);
        };

        let MetalAstKind::FunctionDecl {
            name,
        } = node.kind
        else {
            bail!("unexpected kind of root node: function expected, but {:?} found", node.kind);
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
                MetalAstKind::AnnotateAttr => annotations.push(annotation_from_ast_node(node)?),
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
            .collect::<anyhow::Result<Vec<_>>>()?;

        if !annotations.iter().any(|(k, _)| k.as_ref() == "dsl.kernel") {
            return Ok(None);
        }

        let variants: Box<[MetalTypeParameter]> = annotations
            .iter()
            .filter(|(k, _)| k.as_ref() == "dsl.variants")
            .map(|(_, v)| {
                let [typename, variant_values] = v.as_ref() else {
                    bail!("malformed dsl.variants annotation");
                };

                let variants = variant_values.split(',').map(|v| v.trim().into()).collect::<Box<[Box<str>]>>();

                Ok(MetalTypeParameter {
                    name: typename.clone(),
                    variants,
                })
            })
            .collect::<anyhow::Result<_>>()?;

        if variants.is_empty() != !is_template {
            bail!("mismatch between AST nodes and variants annotation");
        }

        if is_template {
            let template_names = template_parameters.iter().map(|name| name.as_ref()).collect::<Vec<_>>();
            let variant_names = variants.iter().map(|variant| variant.name.as_ref()).collect::<Vec<_>>();
            if template_names != variant_names {
                bail!("template parameters {:?} do not match dsl.variants order {:?}", template_names, variant_names);
            }
        }

        let variants = if variants.is_empty() {
            None
        } else {
            Some(variants)
        };

        let arguments = arg_nodes
            .into_iter()
            .map(|an| MetalArgument::from_ast_node_and_source(an, source))
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Some(MetalKernelInfo {
            name,
            arguments,
            variants,
        }))
    }
}

pub fn validate_raw_kernel(node: &MetalAstNode) -> anyhow::Result<()> {
    let (node, is_template) = if matches!(node.kind, MetalAstKind::FunctionTemplateDecl) {
        let inner_fn = node.inner.iter().find(|c| {
            matches!(
                c.kind,
                MetalAstKind::FunctionDecl {
                    name: _
                }
            )
        });
        match inner_fn {
            Some(n) => (n, true),
            None => return Ok(()),
        }
    } else if matches!(node.kind, MetalAstKind::FunctionDecl { .. }) {
        (node, false)
    } else {
        return Ok(());
    };

    let MetalAstKind::FunctionDecl {
        name,
    } = &node.kind
    else {
        return Ok(());
    };

    let mut has_metal_kernel_attr = false;
    let mut has_max_threads_attr = false;
    let mut has_dsl_kernel = false;

    for child in &node.inner {
        match &child.kind {
            MetalAstKind::MetalKernelAttr => has_metal_kernel_attr = true,
            MetalAstKind::MetalMaxTotalThreadsPerThreadGroupAttr => has_max_threads_attr = true,
            MetalAstKind::AnnotateAttr => {
                if let Ok(annotation) = annotation_from_ast_node(child.clone()) {
                    if annotation.first().map(|s| s.as_ref()) == Some("dsl.kernel") {
                        has_dsl_kernel = true;
                    }
                }
            },
            _ => (),
        }
    }

    if has_metal_kernel_attr && !has_dsl_kernel && !has_max_threads_attr {
        let kind = if is_template {
            "template"
        } else {
            "function"
        };
        bail!(
            "kernel {kind} '{name}' is missing `[[max_total_threads_per_threadgroup(N)]]` attribute! it **MUST** be present and correct!"
        );
    }

    Ok(())
}
