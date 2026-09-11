use minijinja::{Error, ErrorKind, Value};

pub fn raise_exception(message: Value) -> Result<Value, Error> {
    Err(Error::new(ErrorKind::InvalidOperation, message.to_string()))
}
