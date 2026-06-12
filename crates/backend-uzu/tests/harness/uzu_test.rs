extern crate test;

pub enum UzuTest {
    #[allow(dead_code)]
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}
