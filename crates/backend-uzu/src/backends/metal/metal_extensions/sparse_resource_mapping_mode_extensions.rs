use metal::MTLSparseTextureMappingMode;

use crate::backends::common::SparseResourceMappingMode;

impl From<SparseResourceMappingMode> for MTLSparseTextureMappingMode {
    fn from(mode: SparseResourceMappingMode) -> Self {
        match mode {
            SparseResourceMappingMode::Map => MTLSparseTextureMappingMode::Map,
            SparseResourceMappingMode::Unmap => MTLSparseTextureMappingMode::Unmap,
        }
    }
}

impl From<MTLSparseTextureMappingMode> for SparseResourceMappingMode {
    fn from(mode: MTLSparseTextureMappingMode) -> Self {
        match mode {
            MTLSparseTextureMappingMode::Map => SparseResourceMappingMode::Map,
            MTLSparseTextureMappingMode::Unmap => SparseResourceMappingMode::Unmap,
        }
    }
}
