pub trait ResolvableValue<Value> {
    fn resolve(&self) -> Value;
}

pub trait ConfigResolvableValue<Config, Value> {
    fn resolve(
        &self,
        config: &Config,
    ) -> Value;
}
