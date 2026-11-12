use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::trie::TrieNode;

const DYNAMIC: &str = "dynamic";

/// How many tree nodes the verifier checks: a fixed count, or `"dynamic"`, one of the budgets listed in the
/// speculator's `latencies.json`, chosen anew on every step by [`BudgetLatencies::choose`].
#[derive(Serialize, Deserialize, PartialEq, Eq, Debug, Clone, Copy)]
#[serde(try_from = "TreeBudgetJson", into = "TreeBudgetJson")]
pub enum TreeBudget {
    Fixed(u32),
    Dynamic,
}

#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum TreeBudgetJson {
    Fixed(u32),
    Mode(String),
}

impl TryFrom<TreeBudgetJson> for TreeBudget {
    type Error = String;

    fn try_from(json: TreeBudgetJson) -> Result<Self, Self::Error> {
        match json {
            TreeBudgetJson::Fixed(budget) => Ok(Self::Fixed(budget)),
            TreeBudgetJson::Mode(mode) if mode == DYNAMIC => Ok(Self::Dynamic),
            TreeBudgetJson::Mode(mode) => {
                Err(format!("a tree budget is a node count or \"{DYNAMIC}\", not \"{mode}\""))
            },
        }
    }
}

impl From<TreeBudget> for TreeBudgetJson {
    fn from(budget: TreeBudget) -> Self {
        match budget {
            TreeBudget::Fixed(budget) => Self::Fixed(budget),
            TreeBudget::Dynamic => Self::Mode(DYNAMIC.to_string()),
        }
    }
}

/// `latencies.json`: for every device name (or `"default"`, looked up as in `shapes.json`), the duration in
/// milliseconds of a whole decode step (draft, tree construction and verification) at every budget a dynamic tree
/// budget may choose, e.g. `{"Apple M5 Pro": {"16": 68.9, "32": 75.4}}`.
pub type LatenciesFile = HashMap<String, BTreeMap<u32, f32>>;

/// The budgets a dynamic tree budget chooses from, smallest first, with the decode step duration at each.
#[derive(PartialEq, Debug, Clone)]
pub struct BudgetLatencies {
    budgets: Box<[u32]>,
    step_ms: Box<[f32]>,
}

impl BudgetLatencies {
    pub fn new(step_ms: &BTreeMap<u32, f32>) -> Result<Self, String> {
        if step_ms.is_empty() {
            return Err("no budgets to choose from".to_string());
        }
        if let Some((budget, ms)) = step_ms.iter().find(|(budget, ms)| **budget < 2 || !(ms.is_finite() && **ms > 0.0))
        {
            return Err(format!(
                "budget {budget} at {ms} ms: a budget keeps the root and at least one draft token, and a step takes a \
                 positive time"
            ));
        }
        Ok(Self {
            budgets: step_ms.keys().copied().collect(),
            step_ms: step_ms.values().copied().collect(),
        })
    }

    pub fn largest(&self) -> u32 {
        self.budgets[self.budgets.len() - 1]
    }

    /// The budget that gains the most tokens net of its time at the rate the request has had so far: the argmax over
    /// budgets B of E_B - lambda * t_B. E_B is the expected number of accepted draft tokens among the B nodes that
    /// `prune_to_budget(B)` keeps, t_B the step duration, and lambda = tokens_per_pass / t of the smallest budget the
    /// tokens a millisecond of decoding yields. A longer step delays every later token, while the tokens it adds
    /// spare later steps, which yield the request's average rate, not this step's; so the request's rate prices the
    /// time, and this step's tree only enters through E_B. Without a rate yet (before the first decode step) the
    /// smallest budget is taken; ties go to the smaller budget.
    pub fn choose(
        &self,
        trie: &TrieNode,
        tokens_per_pass: Option<f32>,
    ) -> u32 {
        let Some(tokens_per_pass) = tokens_per_pass else {
            return self.budgets[0];
        };
        let tokens_per_ms = tokens_per_pass / self.step_ms[0];
        let expected = trie.expected_accepted_tokens(&self.budgets);
        let mut best = (self.budgets[0], f32::NEG_INFINITY);
        for ((&budget, &ms), gain) in self.budgets.iter().zip(&self.step_ms).zip(expected) {
            let value = gain - tokens_per_ms * ms;
            if value > best.1 {
                best = (budget, value);
            }
        }
        best.0
    }
}

#[cfg(test)]
#[path = "../../unit/speculators/tree_budget_test.rs"]
mod tests;
