// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Uzu",
    platforms: [
        .iOS("26.4"),
        .macOS("26.4"),
    ],
    products: [
        .library(name: "Uzu", targets: ["Uzu"]),
        .executable(name: "examples", targets: ["Examples"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.6.1")
    ],
    targets: [
        .binaryTarget(
            name: "uzu",
            url: "https://artifacts.trymirai.com/uzu-swift/releases/0.5.23.zip",
            checksum: "228306312fe18a22822102408e3f778bf8b3fcc0d30f5178a30ca8f561274b96"
        ),
        .target(
            name: "Uzu",
            dependencies: ["uzu", "UzuMetalIOSimulatorStubs"],
            path: "crates/legacy/uzu/bindings/swift/Sources/Uzu",
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedFramework("SystemConfiguration"),
                .linkedFramework("Metal"),
                .linkedFramework("MetalPerformanceShadersGraph"),
                .linkedFramework("CoreAudio"),
                .linkedFramework("AudioToolbox"),
            ]
        ),
        .target(
            name: "UzuMetalIOSimulatorStubs",
            path: "Sources/UzuMetalIOSimulatorStubs",
            publicHeadersPath: "include"
        ),
        .executableTarget(
            name: "Examples",
            dependencies: [
                "Uzu",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "crates/legacy/uzu/bindings/swift/Sources/Examples"
        ),
        .testTarget(
            name: "UzuTests",
            dependencies: ["Uzu"],
            path: "crates/legacy/uzu/bindings/swift/Tests/UzuTests",
        ),
    ]
)
