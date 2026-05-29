#[macro_export]
macro_rules! dispatch_dtype {
    (|()| $body:expr) => { $body };
    (|($T:ident : $dt:expr $(, $rest_T:ident : $rest_dt:expr)*)| $body:expr) => {
        match $dt {
            $crate::DataType::F16 => { type $T = half::f16; $crate::dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            $crate::DataType::BF16 => { type $T = half::bf16; $crate::dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            $crate::DataType::F32 => { type $T = f32; $crate::dispatch_dtype!(|($($rest_T : $rest_dt),*)| $body) }
            other => panic!("unsupported dtype in test: {:?}", other),
        }
    };
}
