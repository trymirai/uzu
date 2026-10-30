// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "Uzu",
    platforms: [
        .iOS("26.4"),
        .macOS("26.4"),
        .macCatalyst("26.4"),
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
            url: "https://artifacts.trymirai.com/uzu-swift/releases/0.6.14.zip",
            checksum: "c6da893a1926c2ad186b4b4da13b3c5277425c8151914104bacf3c9e855d8288"
        ),
        .target(
            name: "Uzu",
            dependencies: ["uzu", "UzuMetalIOSimulatorStubs"],
            path: "crates/legacy/uzu/bindings/swift/Sources/Uzu",
            linkerSettings: [
                .linkedFramework("SystemConfiguration", .when(platforms: [.macOS])),
                .linkedFramework("Metal"),
                .linkedLibrary("compression"),
                .linkedFramework("AudioToolbox"),
                .linkedFramework("CoreAudio", .when(platforms: [.macOS])),
                .linkedFramework("AVFAudio", .when(platforms: [.iOS, .macCatalyst])),
            ]
        ),
        .target(
            name: "UzuMetalIOSimulatorStubs",
            path: "crates/legacy/uzu/bindings/swift/Sources/UzuMetalIOSimulatorStubs",
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
