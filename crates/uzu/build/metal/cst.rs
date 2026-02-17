use rowan::Language;

use super::token::SyntaxKind;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum MetalTypeLanguage {}

impl Language for MetalTypeLanguage {
    type Kind = SyntaxKind;

    fn kind_from_raw(raw: rowan::SyntaxKind) -> Self::Kind {
        let raw_value = raw.0;
        assert!(raw_value <= SyntaxKind::BaseType as u16);
        unsafe { std::mem::transmute(raw_value) }
    }

    fn kind_to_raw(kind: Self::Kind) -> rowan::SyntaxKind {
        kind.into()
    }
}

pub type SyntaxNode = rowan::SyntaxNode<MetalTypeLanguage>;
