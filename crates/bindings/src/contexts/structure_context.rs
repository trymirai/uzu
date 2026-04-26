use crate::types::StructureFlavor;

pub struct StructureContext<'item> {
    pub flavor: StructureFlavor,
    pub item: &'item syn::ItemStruct,
}
