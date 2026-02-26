use uzu::DataType;

#[derive(Clone)]
pub struct DtypeCombo {
    pub a_dtype: DataType,
    pub b_dtype: DataType,
    pub output_dtype: DataType,
}

impl std::fmt::Display for DtypeCombo {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{:?}*{:?}->{:?}", self.a_dtype, self.b_dtype, self.output_dtype)
    }
}

pub fn test_combos() -> Vec<DtypeCombo> {
    vec![
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::I8,
            output_dtype: DataType::I32,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::F16,
            output_dtype: DataType::F16,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::F32,
            output_dtype: DataType::F32,
        },
        DtypeCombo {
            a_dtype: DataType::BF16,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
        DtypeCombo {
            a_dtype: DataType::I8,
            b_dtype: DataType::BF16,
            output_dtype: DataType::BF16,
        },
    ]
}
