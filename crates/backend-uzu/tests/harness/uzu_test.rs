extern crate test;

pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(&'static test::TestDescAndFn),
}
