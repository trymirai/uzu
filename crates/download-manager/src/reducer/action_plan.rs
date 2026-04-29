use crate::reducer::Action;

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash)]
pub struct ActionPlan {
    actions: Box<[Action]>,
}

impl ActionPlan {
    pub fn empty() -> Self {
        Self::default()
    }

    pub fn from_ordered_actions(actions: impl IntoIterator<Item = Action>) -> Self {
        let mut ordered_actions = Vec::new();

        for action in actions {
            if !ordered_actions.contains(&action) {
                ordered_actions.push(action);
            }
        }

        Self {
            actions: ordered_actions.into_boxed_slice(),
        }
    }

    pub fn merge_in_order(action_plans: impl IntoIterator<Item = ActionPlan>) -> Self {
        Self::from_ordered_actions(
            action_plans.into_iter().flat_map(|action_plan| Vec::from(action_plan.actions).into_iter()),
        )
    }

    pub fn as_slice(&self) -> &[Action] {
        &self.actions
    }
}
