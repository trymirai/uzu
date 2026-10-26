import ProjectDescription

let buildPythonPackages = TargetScript.pre(
    script: """
        env -i HOME="$HOME" PATH="$HOME/.cargo/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin" \\
            "$PROJECT_DIR/bootstrap.sh"
        """,
    name: "Build Python Packages",
    basedOnDependencyAnalysis: false
)

let installPython = TargetScript.post(
    script: """
        set -euo pipefail
        FRAMEWORKS="$CODESIGNING_FOLDER_PATH/Frameworks"
        mkdir -p "$FRAMEWORKS"
        rsync -a --delete --exclude Headers \\
            "$PROJECT_DIR/deps/Python.xcframework/ios-arm64/Python.framework" "$FRAMEWORKS/"
        rsync -a --delete "$PROJECT_DIR/app/" "$CODESIGNING_FOLDER_PATH/app/"
        rsync -a --delete "$PROJECT_DIR/deps/app_packages/" "$CODESIGNING_FOLDER_PATH/app_packages/"
        source "$PROJECT_DIR/deps/Python.xcframework/build/utils.sh"
        install_python deps/Python.xcframework app app_packages
        mv "$CODESIGNING_FOLDER_PATH/app_packages/mlx/lib/mlx.metallib" "$FRAMEWORKS/mlx.core.framework/"
        /usr/bin/codesign --force --sign "$EXPANDED_CODE_SIGN_IDENTITY" -o runtime --timestamp=none \\
            --preserve-metadata=identifier,entitlements,flags --generate-entitlement-der \\
            "$FRAMEWORKS/Python.framework" "$FRAMEWORKS/mlx.core.framework"
        """,
    name: "Install Python",
    outputPaths: [
        "$(CODESIGNING_FOLDER_PATH)/Frameworks",
        "$(CODESIGNING_FOLDER_PATH)/app",
        "$(CODESIGNING_FOLDER_PATH)/app_packages",
        "$(CODESIGNING_FOLDER_PATH)/python",
    ],
    basedOnDependencyAnalysis: false
)

let project = Project(
    name: "Benchmarks",
    organizationName: "Mirai",
    targets: [
        .target(
            name: "Benchmarks",
            destinations: [.iPhone, .iPad],
            product: .app,
            bundleId: "com.mirai.tech.benchmarks",
            deploymentTargets: .iOS("26.4"),
            infoPlist: .extendingDefault(with: [
                "LSSupportsOpeningDocumentsInPlace": true,
                "UIFileSharingEnabled": true,
                "UILaunchScreen": [:],
            ]),
            sources: ["Sources/**"],
            entitlements: .dictionary([
                "com.apple.developer.kernel.increased-memory-limit": true
            ]),
            scripts: [buildPythonPackages, installPython],
            settings: .settings(base: [
                "CODE_SIGN_STYLE": "Automatic",
                "DEVELOPMENT_TEAM": "C39GZ239GY",
                "ENABLE_USER_SCRIPT_SANDBOXING": "NO",
                "FRAMEWORK_SEARCH_PATHS": "$(PROJECT_DIR)/deps/Python.xcframework/ios-arm64",
                "OTHER_LDFLAGS": "-framework Python",
                "SUPPORTED_PLATFORMS": "iphoneos",
                "SWIFT_OBJC_BRIDGING_HEADER": "Sources/BridgingHeader.h",
            ])
        )
    ]
)
