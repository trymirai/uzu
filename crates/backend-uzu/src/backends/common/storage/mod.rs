mod borrowed;

pub trait Storage {
    type Buffer;
}

pub use borrowed::Borrowed;
