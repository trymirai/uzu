#[derive(Clone, Copy)]
pub struct Case {
    pub bits: u32,
    pub group_size: u32,
    pub k: u32,
    pub n: u32,
}

#[derive(Clone, Copy)]
pub struct Cell {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub bits: u32,
    pub group_size: u32,
}

const USAGE: &str =
    "pass bits=<bits> g=<group size> k=<K> n=<N> after the test name, each accepting a comma separated list";

fn values(key: &str) -> Vec<u32> {
    let prefix = format!("{key}=");
    let list = std::env::args()
        .find_map(|argument| argument.strip_prefix(&prefix).map(str::to_owned))
        .unwrap_or_else(|| panic!("missing {key}: {USAGE}"));

    list.split(',')
        .map(|value| value.parse().unwrap_or_else(|_| panic!("{key}: {value} is not a number: {USAGE}")))
        .collect()
}

pub fn cases() -> Vec<Case> {
    let mut cases = Vec::new();
    for bits in values("bits") {
        for group_size in values("g") {
            for k in values("k") {
                for n in values("n") {
                    cases.push(Case {
                        bits,
                        group_size,
                        k,
                        n,
                    });
                }
            }
        }
    }
    cases
}

pub fn batches() -> impl Iterator<Item = u32> {
    (1..=16).chain((18..=32).step_by(2)).chain([64])
}

pub fn cells(case: Case) -> impl Iterator<Item = Cell> {
    batches().map(move |m| Cell {
        m,
        k: case.k,
        n: case.n,
        bits: case.bits,
        group_size: case.group_size,
    })
}
