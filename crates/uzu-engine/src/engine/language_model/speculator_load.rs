use crate::speculators::dflash_tfm::DFlashTfmTreeShape;

#[derive(Debug, Clone, Default)]
pub enum SpeculatorLoad {
    #[default]
    FromShapes,
    Disabled,
    Shape(DFlashTfmTreeShape),
}
