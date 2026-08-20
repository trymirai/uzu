use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{ItemFn, Meta, parse_macro_input};

pub fn uzu_test(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let mut func = parse_macro_input!(input as ItemFn);

    let ignore_attr =
        func.attrs.iter().position(|attr| attr.path().is_ident("ignore")).map(|index| func.attrs.remove(index));
    let (ignore, ignore_message) = match ignore_attr.as_ref().map(|attr| &attr.meta) {
        None => (quote! { false }, quote! { ::core::option::Option::None }),
        Some(Meta::Path(_)) => (quote! { true }, quote! { ::core::option::Option::None }),
        Some(Meta::NameValue(name_value)) => {
            let message = &name_value.value;
            (quote! { true }, quote! { ::core::option::Option::Some(#message) })
        },
        Some(meta) => {
            return syn::Error::new_spanned(meta, "unsupported #[ignore] form").to_compile_error().into();
        },
    };

    let name = &func.sig.ident;
    let const_name = format_ident!("__UZU_TEST_CASE_{}", name);

    quote! {
        #func

        #[test_case]
        #[allow(non_upper_case_globals)]
        const #const_name: test_runner::UzuTest =
            test_runner::UzuTest::Test(&test_runner::test::TestDescAndFn {
                desc: test_runner::test::TestDesc {
                    name: test_runner::test::StaticTestName(concat!(
                        module_path!(),
                        "::",
                        stringify!(#name),
                    )),
                    ignore: #ignore,
                    ignore_message: #ignore_message,
                    source_file: file!(),
                    start_line: line!() as usize,
                    start_col: column!() as usize,
                    end_line: line!() as usize,
                    end_col: column!() as usize,
                    should_panic: test_runner::test::ShouldPanic::No,
                    compile_fail: false,
                    no_run: false,
                    test_type: test_runner::test::TestType::Unknown,
                },
                testfn: test_runner::test::StaticTestFn(|| {
                    test_runner::test::assert_test_result(#name())
                }),
            });
    }
    .into()
}

pub fn uzu_bench(
    _args: TokenStream,
    input: TokenStream,
) -> TokenStream {
    let func = parse_macro_input!(input as ItemFn);

    let name = &func.sig.ident;
    let const_name = format_ident!("__UZU_BENCH_CASE_{}", name);

    quote! {
        #func

        #[test_case]
        #[allow(non_upper_case_globals)]
        const #const_name: test_runner::UzuTest = test_runner::UzuTest::Bench(&|| {
            let mut criterion = ::criterion::Criterion::default().configure_from_args();
            #name(&mut criterion);
        });
    }
    .into()
}
