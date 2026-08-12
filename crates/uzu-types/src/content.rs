pub struct ContentPartText {
    pub text: String,
}

pub enum Content {
    Text(ContentPartText),
}
