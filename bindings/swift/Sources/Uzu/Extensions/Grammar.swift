import FoundationModels

extension Grammar {
    public static func fromType<T: Generable>(_ type: T.Type) -> Self {
        let schema = T.generationSchema.debugDescription
        let result = Grammar.jsonSchema(schema: schema)
        return result
    }
}
