/// Backend-agnostic sparse-resource mapping mode for `SparseBufferOperation`
/// (and any future sparse-texture analogue).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum SparseResourceMappingMode {
    /// Allocate physical storage for the affected pages.
    Map,
    /// Release physical storage for the affected pages.
    Unmap,
}
