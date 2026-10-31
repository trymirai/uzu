use anyhow::Context;
use shader_slang::{
    ComponentType, GenericArg, GenericArgType,
    reflection::{Decl, Generic, Shader, Type},
};

pub fn create_specialized_generic<'a>(
    component: &'a ComponentType,
    generic_decl: &Decl,
    concrete_types: &[&str],
) -> anyhow::Result<&'a Generic> {
    let layout: &Shader = component.layout(0)?;
    let generic: &Generic = generic_decl.as_generic().context("declaration has no generic reflection")?;

    let type_ptrs: Vec<&Type> = concrete_types
        .iter()
        .map(|name| layout.find_type_by_name(name).ok_or_else(|| anyhow::anyhow!("type '{}' not found", name)))
        .collect::<anyhow::Result<_>>()?;

    let arg_types: Vec<GenericArgType> =
        std::iter::repeat_n(GenericArgType::SlangGenericArgType, type_ptrs.len()).collect();
    let args: Vec<GenericArg> = type_ptrs
        .iter()
        .map(|t| GenericArg {
            typeVal: *t as *const _ as *mut _,
        })
        .collect();

    layout.specialize_generic(generic, &arg_types, &args).ok_or_else(|| anyhow::anyhow!("failed to specialize generic"))
}
