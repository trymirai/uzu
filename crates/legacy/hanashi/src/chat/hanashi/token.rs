use shoji::types::basic::Token;
use token_stream_parser::types::Token as ParserToken;

pub trait ToParserToken {
    type Output;

    fn to_parser_token(self) -> Self::Output;
}

impl ToParserToken for Token {
    type Output = ParserToken;

    fn to_parser_token(self) -> Self::Output {
        ParserToken {
            id: self.id,
            value: self.value,
            is_special: self.is_special,
        }
    }
}
