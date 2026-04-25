use shoji::traits::State as StateTrait;

#[derive(Debug, Clone)]
pub struct State;

impl StateTrait for State {
    fn clone_boxed(&self) -> Box<dyn StateTrait> {
        Box::new(self.clone())
    }
}
