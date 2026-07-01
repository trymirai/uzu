macro_rules! block_bench {
    (
        name: $name:ident,
        block: $block_type:ty,
        params {
            $($parameter_name:ident : $parameter_type:ty = $parameter_values:expr),* $(,)?
        },
        buffers($buffers_parameters:ident) {
            $($buffer_name:ident : [ $buffer_shape:expr ] : $buffer_data_type:expr),* $(,)?
        },
        build($build_context:ident, $build_parameters:ident) $build_body:block,
        encode($encode_kernel:ident, $encode_buffers:ident, $encode_parameters:ident, $encode_encoder:ident) $encode_body:block $(,)?
    ) => {
        pub const NAME: &str = stringify!($name);

        #[derive(Debug, Clone, Copy)]
        pub struct Parameters {
            $(pub $parameter_name: $parameter_type),*
        }

        pub fn parameter_grid() -> Vec<Parameters> {
            itertools::iproduct!($($parameter_values),*)
                .map(|($($parameter_name),*): ($($parameter_type),*)| Parameters { $($parameter_name),* })
                .collect()
        }

        impl Parameters {
            pub fn csv_header() -> String {
                [$(stringify!($parameter_name)),*].join(",")
            }

            pub fn csv_fields(&self) -> String {
                [$(format!("{:?}", self.$parameter_name)),*].join(",")
            }
        }

        pub struct Buffers {
            $(pub $buffer_name: $crate::backends::common::Allocation<$crate::backends::metal::Metal>),*
        }

        pub fn make_buffers(
            context: &$crate::backends::metal::MetalContext,
            $buffers_parameters: &Parameters,
        ) -> Buffers {
            Buffers {
                $($buffer_name: $crate::tests::helpers::random_buffer(
                    context,
                    &[$buffer_shape],
                    $buffer_data_type,
                    $crate::tests::helpers::seed_from_label(stringify!($buffer_name)),
                )),*
            }
        }

        pub fn build(
            $build_context: &$crate::backends::metal::MetalContext,
            $build_parameters: &Parameters,
        ) -> $block_type $build_body

        pub fn encode(
            $encode_kernel: &$block_type,
            $encode_buffers: &mut Buffers,
            $encode_parameters: &Parameters,
            $encode_encoder: &mut $crate::backends::common::Encoder<$crate::backends::metal::Metal>,
        ) $encode_body
    };
}

pub(crate) use block_bench;
