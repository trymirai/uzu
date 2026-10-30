use parquet::data_type::ByteArray;

pub enum Column {
    Int64(Vec<i64>),
    Double(Vec<f64>),
    Utf8(Vec<ByteArray>),
}
