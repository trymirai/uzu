use anyhow::Context;
use rowan::GreenNodeBuilder;

use super::{cst::SyntaxNode, lexer::Lexer, token::SyntaxKind, type_info::ParsedType};

pub struct TypeParser {
    tokens: Vec<(SyntaxKind, String)>,
    token_position: usize,
    builder: GreenNodeBuilder<'static>,
}

impl TypeParser {
    fn new(input: &str) -> Self {
        let tokens: Vec<_> = Lexer::new(input).map(|(kind, text)| (kind, text.to_string())).collect();
        Self {
            tokens,
            token_position: 0,
            builder: GreenNodeBuilder::new(),
        }
    }

    pub fn parse_type(input: &str) -> anyhow::Result<ParsedType> {
        let parser = Self::new(input);
        let green = parser.parse();
        let root = SyntaxNode::new_root(green);
        let type_ref =
            root.children().find(|node| node.kind() == SyntaxKind::TypeRef).context("No TypeRef node found")?;
        ParsedType::from_cst(&type_ref)
    }

    fn parse(mut self) -> rowan::GreenNode {
        self.start_node(SyntaxKind::Root);
        self.parse_type_ref();
        self.finish_node();
        self.builder.finish()
    }

    fn start_node(
        &mut self,
        kind: SyntaxKind,
    ) {
        self.builder.start_node(kind.into());
    }

    fn finish_node(&mut self) {
        self.builder.finish_node();
    }

    fn is_end_of_tokens(&self) -> bool {
        self.token_position >= self.tokens.len()
    }

    fn peek(&self) -> Option<SyntaxKind> {
        self.tokens.get(self.token_position).map(|(kind, _)| *kind)
    }

    fn is_at_kind(
        &self,
        kind: SyntaxKind,
    ) -> bool {
        self.peek() == Some(kind)
    }

    fn bump(&mut self) {
        if let Some((kind, text)) = self.tokens.get(self.token_position) {
            self.builder.token((*kind).into(), text);
            self.token_position += 1;
        }
    }

    fn parse_type_ref(&mut self) {
        self.start_node(SyntaxKind::TypeRef);
        self.parse_qualifiers();
        self.parse_base_type();
        self.parse_declarator();
        self.finish_node();
    }

    fn parse_qualifiers(&mut self) {
        while let Some(kind) = self.peek() {
            match kind {
                SyntaxKind::KwConst
                | SyntaxKind::KwDevice
                | SyntaxKind::KwThreadgroup
                | SyntaxKind::KwConstant
                | SyntaxKind::KwThread
                | SyntaxKind::KwVolatile
                | SyntaxKind::KwStatic => self.bump(),
                _ => break,
            }
        }
    }

    fn parse_base_type(&mut self) {
        self.start_node(SyntaxKind::BaseType);

        match self.peek() {
            Some(SyntaxKind::KwBool)
            | Some(SyntaxKind::KwInt)
            | Some(SyntaxKind::KwUInt)
            | Some(SyntaxKind::KwFloat)
            | Some(SyntaxKind::KwInt32T)
            | Some(SyntaxKind::KwUInt32T)
            | Some(SyntaxKind::KwSimd) => self.bump(),

            Some(SyntaxKind::KwUnsigned) => {
                self.bump();
                if self.is_at_kind(SyntaxKind::KwInt) {
                    self.bump();
                }
            },

            Some(SyntaxKind::Ident) => self.parse_qualified_name(),

            _ => {
                if !self.is_end_of_tokens() {
                    self.bump();
                }
            },
        }

        self.finish_node();
    }

    fn parse_qualified_name(&mut self) {
        self.start_node(SyntaxKind::QualifiedName);

        if self.is_at_kind(SyntaxKind::Ident) {
            self.bump();
        }

        while self.is_at_kind(SyntaxKind::DoubleColon) {
            self.bump();
            if self.is_at_kind(SyntaxKind::Ident) {
                self.bump();
            } else {
                break;
            }
        }

        self.finish_node();
    }

    fn parse_declarator(&mut self) {
        match self.peek() {
            Some(SyntaxKind::Star) => {
                self.start_node(SyntaxKind::PointerDeclarator);
                self.bump();
                self.finish_node();
            },
            Some(SyntaxKind::Ampersand) => {
                self.start_node(SyntaxKind::ReferenceDeclarator);
                self.bump();
                self.finish_node();
            },
            Some(SyntaxKind::LBracket) => self.parse_array_declarator_internal(),
            _ => {},
        }
    }

    fn parse_array_declarator_internal(&mut self) {
        self.start_node(SyntaxKind::ArrayDeclarator);

        if self.is_at_kind(SyntaxKind::LBracket) {
            self.bump();

            if !self.is_at_kind(SyntaxKind::RBracket) {
                self.start_node(SyntaxKind::ArraySize);
                while !self.is_end_of_tokens() && !self.is_at_kind(SyntaxKind::RBracket) {
                    self.bump();
                }
                self.finish_node();
            }

            if self.is_at_kind(SyntaxKind::RBracket) {
                self.bump();
            }
        }

        self.finish_node();
    }
}
