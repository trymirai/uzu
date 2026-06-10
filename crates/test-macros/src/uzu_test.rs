pub enum UzuTest {
    Bench(&'static dyn Fn()),
    Test(test::TestDescAndFn),
}
