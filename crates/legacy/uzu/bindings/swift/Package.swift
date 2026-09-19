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
            path: "uzu.xcframework"
        ),
        .target(
            name: "Uzu",
            dependencies: ["uzu", "UzuMetalIOSimulatorStubs"],
            linkerSettings: [
                .linkedLibrary("c++"),
                .linkedLibrary("compression"),
                .linkedFramework("SystemConfiguration", .when(platforms: [.macOS])),
                .linkedFramework("Metal"),
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
            ]
        ),
        .testTarget(
            name: "UzuTests",
            dependencies: ["Uzu"],
        ),
    ]
)
