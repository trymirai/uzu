#[derive(Clone, Copy, Eq, Hash, PartialEq)]
pub enum MethodFlavor {
    Plain,
    Constructor,
    Factory,
    FactoryWithCallback,
    Getter,
    Setter,
    StreamNext,
}
