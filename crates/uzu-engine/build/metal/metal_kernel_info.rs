use anyhow::{Context, bail};

use super::{
    Error, MetalArgument, MetalTemplateParameter, annotation_from_ast_node,
    ast::{
        MetalArgumentType, MetalAstKind, MetalAstNode, MetalBufferAccess, MetalConstantType, MetalGroupsType,
        MetalTemplateParameterType,
    },
    integer_object_defines_from_source,
};
use crate::common::{
    identifiers::KernelName,
    kernel::{Kernel, KernelArgument, KernelArgumentType, KernelBufferAccess},
};

#[derive(Debug)]
pub struct MetalKernelInfo {
    pub public: bool,
    pub name: KernelName,
    pub arguments: Box<[MetalArgument]>,
    pub variants: Option<Box<[MetalTemplateParameter]>>,
    pub constraints: Box<[Box<str>]>,
}

impl MetalKernelInfo {
    pub fn has_axis(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Axis(..)))
    }

    pub fn has_groups(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(_)))
    }

    pub fn has_groups_direct(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(MetalGroupsType::Direct(_))))
    }

    pub fn has_groups_indirect(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Groups(MetalGroupsType::Indirect)))
    }

    pub fn has_threads(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::Threads(_)))
    }

    pub fn has_thread_context(&self) -> bool {
        self.arguments.iter().any(|a| matches!(&a.argument_type, MetalArgumentType::ThreadContext))
    }

    pub fn to_kernel(&self) -> Result<Option<Kernel>, Error> {
        if !self.public {
            return Ok(None);
        }

        let mut indirect_flag = false;

        Ok(Some(Kernel {
            name: self.name.clone(),
            parameters: self
                .variants
                .as_ref()
                .map(|v| v.iter().map(|p| p.to_parameter()).collect::<Result<Vec<_>, Error>>())
                .transpose()?
                .unwrap_or_default()
                .into_iter()
                .chain(self.arguments.iter().filter_map(|a| a.to_parameter()))
                .collect(),
            arguments: self
                .arguments
                .iter()
                .filter_map(|a| match &a.argument_type {
                    MetalArgumentType::Buffer(access) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Buffer(match access {
                            MetalBufferAccess::Read => KernelBufferAccess::Read,
                            MetalBufferAccess::ReadWrite => KernelBufferAccess::ReadWrite,
                        }),
                    }),
                    MetalArgumentType::Groups(MetalGroupsType::Indirect) if !indirect_flag => {
                        indirect_flag = true;
                        Some(KernelArgument {
                            name: "__dsl_indirect_dispatch_buffer".into(),
                            conditional: false,
                            ty: KernelArgumentType::Buffer(KernelBufferAccess::Read),
                        })
                    },
                    MetalArgumentType::Constant((ty, MetalConstantType::Scalar)) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(ty.clone()),
                    }),
                    MetalArgumentType::Constant((ty, MetalConstantType::Array(None))) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(format!("&[{ty}]").into_boxed_str()),
                    }),
                    MetalArgumentType::Constant((ty, MetalConstantType::Array(Some(size)))) => Some(KernelArgument {
                        name: a.name.clone(),
                        conditional: a.condition.is_some(),
                        ty: KernelArgumentType::Constant(format!("&[{ty}; {size}]").into_boxed_str()),
                    }),
                    _ => None,
                })
                .collect(),
        }))
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
                        template_parameters.push((name, None));
                    },
                    MetalAstKind::NonTypeTemplateParmDecl {
                        name,
                        ty,
                    } => {
                        let name = name.context("template parameter missing name")?;
                        template_parameters.push((name, Some(ty)));
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

        let public = annotations.iter().any(|(k, _)| k.as_ref() == "dsl.public");

        let variants: Box<[_]> = annotations
            .iter()
            .filter(|(k, _)| k.as_ref() == "dsl.variants")
            .map(|(_, v)| {
                let [variant_name, variant_values] = v.as_ref() else {
                    bail!("malformed dsl.variants annotation");
                };

                let variant_values = variant_values.split(',').map(|v| v.trim().into()).collect::<Box<[Box<str>]>>();

                Ok((variant_name.clone(), variant_values))
            })
            .collect::<anyhow::Result<_>>()?;

        let has_variants = !variants.is_empty();
        if has_variants != is_template {
            bail!("mismatch between AST nodes and variants annotation");
        }

        let variants = if is_template {
            let template_names = template_parameters.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();
            let variant_names = variants.iter().map(|(name, _)| name.as_ref()).collect::<Vec<_>>();
            if template_names != variant_names {
                bail!("template parameters {:?} do not match dsl.variants order {:?}", template_names, variant_names);
            }

            Some(
                template_parameters
                    .into_iter()
                    .zip(variants)
                    .map(|((name, ty), (v_name, variants))| {
                        assert_eq!(name, v_name);

                        Ok(MetalTemplateParameter {
                            name,
                            ty: match ty {
                                None => MetalTemplateParameterType::Type,
                                Some(ntt) => MetalTemplateParameterType::Value(MetalArgument::scalar_type_to_rust(
                                    ntt.desugared_qual_type.unwrap_or(ntt.qual_type).as_ref(),
                                )?),
                            },
                            variants,
                        })
                    })
                    .collect::<anyhow::Result<_>>()?,
            )
        } else {
            None
        };

        let constraints: Box<[_]> = annotations
            .iter()
            .filter(|(k, _)| k.as_ref() == "dsl.constraint")
            .map(|(_, v)| {
                let [constraint_expr] = v.as_ref() else {
                    bail!("malformed dsl.constraint annotation");
                };

                Ok(constraint_expr.clone())
            })
            .collect::<anyhow::Result<_>>()?;

        let integer_defines = integer_object_defines_from_source(source);
        let arguments = arg_nodes
            .into_iter()
            .map(|an| MetalArgument::from_ast_node_and_source(an, source, &integer_defines))
            .collect::<anyhow::Result<Box<[_]>>>()?;

        Ok(Some(MetalKernelInfo {
            public,
            name: KernelName::from(name),
            arguments,
            variants,
            constraints,
        }))
    }
}
