use crate::types::ClassFlavor;

pub struct ClassContext<'item> {
    pub flavor: ClassFlavor,
    pub item: &'item syn::ItemStruct,
}
