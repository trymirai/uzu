use logos::Logos;

#[derive(Logos, Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[logos(error = ())]
#[logos(skip r"[ \t\n\f]+")]
pub enum TokenKind {
    #[token("const")]
    KwConst,
    #[token("device")]
    KwDevice,
    #[token("threadgroup")]
    KwThreadgroup,
    #[token("constant")]
    KwConstant,
    #[token("thread")]
    KwThread,
    #[token("volatile")]
    KwVolatile,
    #[token("static")]
    KwStatic,

    #[token("bool")]
    KwBool,
    #[token("int")]
    KwInt,
    #[token("uint")]
    KwUInt,
    #[token("float")]
    KwFloat,
    #[token("unsigned")]
    KwUnsigned,
    #[token("int32_t")]
    KwInt32T,
    #[token("uint32_t")]
    KwUInt32T,
    #[token("Simd")]
    KwSimd,

    #[token("*")]
    Star,
    #[token("&")]
    Ampersand,
    #[token("::")]
    DoubleColon,
    #[token("[")]
    LBracket,
    #[token("]")]
    RBracket,
    #[token("(")]
    LParen,
    #[token(")")]
    RParen,
    #[token(",")]
    Comma,
    #[token(";")]
    Semicolon,
    #[token("<")]
    Less,
    #[token(">")]
    Greater,

    #[regex(r"[a-zA-Z_][a-zA-Z0-9_]*")]
    Ident,
    #[regex(r"[0-9]+")]
    IntLiteral,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u16)]
pub enum SyntaxKind {
    Error = 0,
    KwConst,
    KwDevice,
    KwThreadgroup,
    KwConstant,
    KwThread,
    KwVolatile,
    KwStatic,
    KwBool,
    KwInt,
    KwUInt,
    KwFloat,
    KwUnsigned,
    KwInt32T,
    KwUInt32T,
    KwSimd,
    Star,
    Ampersand,
    DoubleColon,
    LBracket,
    RBracket,
    LParen,
    RParen,
    Comma,
    Semicolon,
    Less,
    Greater,
    Ident,
    IntLiteral,

    Root,
    TypeRef,
    QualifiedName,
    PointerDeclarator,
    ReferenceDeclarator,
    ArrayDeclarator,
    ArraySize,
    QualifierList,
    BaseType,
}

impl From<TokenKind> for SyntaxKind {
    fn from(token: TokenKind) -> Self {
        match token {
            TokenKind::KwConst => SyntaxKind::KwConst,
            TokenKind::KwDevice => SyntaxKind::KwDevice,
            TokenKind::KwThreadgroup => SyntaxKind::KwThreadgroup,
            TokenKind::KwConstant => SyntaxKind::KwConstant,
            TokenKind::KwThread => SyntaxKind::KwThread,
            TokenKind::KwVolatile => SyntaxKind::KwVolatile,
            TokenKind::KwStatic => SyntaxKind::KwStatic,
            TokenKind::KwBool => SyntaxKind::KwBool,
            TokenKind::KwInt => SyntaxKind::KwInt,
            TokenKind::KwUInt => SyntaxKind::KwUInt,
            TokenKind::KwFloat => SyntaxKind::KwFloat,
            TokenKind::KwUnsigned => SyntaxKind::KwUnsigned,
            TokenKind::KwInt32T => SyntaxKind::KwInt32T,
            TokenKind::KwUInt32T => SyntaxKind::KwUInt32T,
            TokenKind::KwSimd => SyntaxKind::KwSimd,
            TokenKind::Star => SyntaxKind::Star,
            TokenKind::Ampersand => SyntaxKind::Ampersand,
            TokenKind::DoubleColon => SyntaxKind::DoubleColon,
            TokenKind::LBracket => SyntaxKind::LBracket,
            TokenKind::RBracket => SyntaxKind::RBracket,
            TokenKind::LParen => SyntaxKind::LParen,
            TokenKind::RParen => SyntaxKind::RParen,
            TokenKind::Comma => SyntaxKind::Comma,
            TokenKind::Semicolon => SyntaxKind::Semicolon,
            TokenKind::Less => SyntaxKind::Less,
            TokenKind::Greater => SyntaxKind::Greater,
            TokenKind::Ident => SyntaxKind::Ident,
            TokenKind::IntLiteral => SyntaxKind::IntLiteral,
        }
    }
}

impl From<SyntaxKind> for rowan::SyntaxKind {
    fn from(kind: SyntaxKind) -> Self {
        rowan::SyntaxKind(kind as u16)
    }
}
