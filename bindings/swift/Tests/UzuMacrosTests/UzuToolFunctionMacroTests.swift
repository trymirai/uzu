import SwiftSyntaxMacros
import SwiftSyntaxMacrosTestSupport
import XCTest
@testable import UzuMacros

final class UzuToolFunctionMacroTests: XCTestCase {
    private let macros: [String: Macro.Type] = [
        "UzuToolFunction": UzuToolFunctionMacro.self,
    ]

    func testExpansion() {
        assertMacroExpansion(
            """
            /// Look up the temperature.
            /// - Parameter city: City to look up.
            @UzuToolFunction(name: "get_temperature")
            func temperature(city: String, unit: String?) async throws -> Double {
                25
            }
            """,
            expandedSource: """
                /// Look up the temperature.
                /// - Parameter city: City to look up.
                func temperature(city: String, unit: String?) async throws -> Double {
                    25
                }

                @Generable
                fileprivate struct __UzuToolArguments_temperature: Codable, Sendable {
                    @Guide(description: "City to look up.")
                    let city: String

                    let unit: String?
                }

                var temperatureTool: some UzuTool {
                    UzuToolDescriptor<__UzuToolArguments_temperature, Double>(
                        name: "get_temperature",
                        description: "Look up the temperature.",
                        parameters: __UzuToolArguments_temperature.self,
                        returning: (Double).self
                    ) { arguments in
                        try await temperature(city: arguments.city, unit: arguments.unit)
                    }
                }
                """,
            macros: macros
        )
    }

    func testParameterlessVoidExpansion() {
        assertMacroExpansion(
            """
            @UzuToolFunction(description: "Clear stored data")
            func clear() {}
            """,
            expandedSource: """
                func clear() {}

                var clearTool: some UzuTool {
                    UzuToolDescriptor<Void, Void>(
                        name: "clear",
                        description: "Clear stored data"
                    ) {
                        clear()
                    }
                }
                """,
            macros: macros
        )
    }

    func testRejectsDefaultParameters() {
        assertMacroExpansion(
            """
            @UzuToolFunction
            func lookup(limit: Int = 10) -> Int { limit }
            """,
            expandedSource: """
                func lookup(limit: Int = 10) -> Int { limit }
                """,
            diagnostics: [
                DiagnosticSpec(
                    message: "@UzuToolFunction does not support default parameter values",
                    line: 1,
                    column: 1
                )
            ],
            macros: macros
        )
    }
}
