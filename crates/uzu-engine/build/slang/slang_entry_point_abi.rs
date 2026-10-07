use anyhow::{Context, bail, ensure};
use itertools::Itertools;
use shader_slang::{
    ParameterCategory, ScalarType, TypeKind,
    reflection::{EntryPoint, Shader},
};

use super::{Error, SlangFieldAbi};

/// Target ABI of one entry point in the linked wrapper program: its push-constant block and the
/// program's specialization constants. The kernel signature itself is owned by the common `Kernel`.
#[derive(Debug)]
pub struct SlangEntryPointAbi {
    pub name: String,
    pub group_size: [u32; 3],
    pub block_size: usize,
    pub fields: Vec<SlangFieldAbi>,
    /// `(name, constant_id)` of every specialization constant; all are `bool`, passed as `VkBool32`.
    pub specialization_ids: Vec<(String, u32)>,
}

impl SlangEntryPointAbi {
    pub fn from_reflection(
        program: &Shader,
        entry_point: &EntryPoint,
    ) -> Result<Self, Error> {
        let name = entry_point.name().context("Slang entry point has no name")?.to_owned();
        let block = entry_point.type_layout().context("Slang entry point has no layout")?;
        let block_size = if block.categories().any(|category| matches!(category, ParameterCategory::PushConstantBuffer))
        {
            ensure!(matches!(block.kind(), TypeKind::ConstantBuffer), "'{name}': push constants are not a block");
            block
                .element_type_layout()
                .context("push-constant block has no element layout")?
                .size(ParameterCategory::Uniform)
        } else {
            0
        };

        let mut fields = Vec::new();
        for parameter in entry_point.parameters() {
            let parameter_name = parameter.name().context("Slang entry point parameter has no name")?;
            let layout = parameter.type_layout().context("Slang entry point parameter has no layout")?;
            match layout.categories().collect::<Vec<_>>().as_slice() {
                [] if parameter.semantic_name().is_some() => continue,
                [ParameterCategory::Uniform] => {},
                categories => bail!("'{name}': parameter '{parameter_name}' has unsupported categories {categories:?}"),
            }
            let pointee = match layout.kind() {
                TypeKind::Pointer => {
                    let pointee = layout.element_type_layout().context("pointer has no pointee layout")?;
                    let stride = pointee.stride(ParameterCategory::Uniform);
                    let alignment = usize::try_from(pointee.alignment(ParameterCategory::Uniform))?;
                    ensure!(
                        alignment > 0 && stride > 0 && stride.is_multiple_of(alignment),
                        "'{name}': pointee of '{parameter_name}' has stride {stride} and alignment {alignment}"
                    );
                    Some((pointee.name().context("pointee has no name")?.to_owned(), stride, alignment))
                },
                TypeKind::Scalar => None,
                kind => bail!("'{name}': parameter '{parameter_name}' has unsupported kind {kind:?}"),
            };
            fields.push(SlangFieldAbi {
                name: parameter_name.to_owned(),
                offset: parameter.offset(ParameterCategory::Uniform),
                size: layout.size(ParameterCategory::Uniform),
                pointee,
            });
        }
        let mut end = 0;
        for field in fields.iter().sorted_by_key(|field| field.offset) {
            ensure!(field.size > 0 && field.offset >= end, "'{name}': argument '{}' overlaps another", field.name);
            end = field.offset + field.size;
        }
        ensure!(end <= block_size && block_size.is_multiple_of(4), "'{name}': block of {block_size} bytes holds {end}");

        let specialization_ids = program
            .parameters()
            .map(|parameter| {
                let parameter_name = parameter.name().context("Slang global parameter has no name")?;
                let layout = parameter.type_layout().context("Slang global parameter has no layout")?;
                ensure!(
                    matches!(
                        layout.categories().collect::<Vec<_>>().as_slice(),
                        [ParameterCategory::SpecializationConstant]
                    ) && matches!(layout.scalar_type(), Some(ScalarType::Bool)),
                    "'{name}': global '{parameter_name}' is not a bool specialization constant"
                );
                Ok((
                    parameter_name.to_owned(),
                    u32::try_from(parameter.offset(ParameterCategory::SpecializationConstant))?,
                ))
            })
            .collect::<Result<_, Error>>()?;

        let [x, y, z] = entry_point.compute_thread_group_size();
        Ok(Self {
            name,
            group_size: [u32::try_from(x)?, u32::try_from(y)?, u32::try_from(z)?],
            block_size,
            fields,
            specialization_ids,
        })
    }
}
