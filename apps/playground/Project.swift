import ProjectDescription

let projectName = "Mirai"

let packages: [Package] = [
    .package(path: "../../bindings/swift"),
    .package(url: "https://github.com/Rspoon3/SFSymbols", .exact("2.6.1")),
    .package(url: "https://github.com/firebase/firebase-ios-sdk", .exact("11.15.0")),
]

let settings: Settings = .settings(
    base: [
        "ARCHS": "arm64",
        "ONLY_ACTIVE_ARCH": "YES",
        "PRODUCT_BUNDLE_IDENTIFIER": "com.mirai.tech.playground",
        "PRODUCT_NAME": "Mirai",
        "CODE_SIGN_STYLE": "Automatic",
        "DEVELOPMENT_TEAM": "C39GZ239GY",
        "TARGETED_DEVICE_FAMILY": "1,2",
        "SUPPORTS_MAC_DESIGNED_FOR_IPHONE_IPAD": "NO",
        "MARKETING_VERSION": "2.0.0",
        "CURRENT_PROJECT_VERSION": "1",
    ],
    configurations: [
        .debug(
            name: "Debug",
            settings: [
                "DEBUG_INFORMATION_FORMAT": "dwarf-with-dsym",
                "SWIFT_COMPILATION_MODE": "incremental",
                "GCC_OPTIMIZATION_LEVEL": "0",
            ]),
        .release(
            name: "Release",
            settings: [
                "DEBUG_INFORMATION_FORMAT": "dwarf-with-dsym",
                "SWIFT_COMPILATION_MODE": "wholemodule",
                "GCC_OPTIMIZATION_LEVEL": "s",
            ]),
    ]
)

let crashlyticsScript = TargetScript.post(
    script: """
        # Upload dSYM files to Crashlytics for crash symbolication
        SCRIPT="${BUILD_DIR%/Build/*}/SourcePackages/checkouts/firebase-ios-sdk/Crashlytics/run"

        # Only run for Release builds or if explicitly requested
        if [ "$CONFIGURATION" = "Release" ] || [ "${UPLOAD_SYMBOLS_TO_CRASHLYTICS:-NO}" = "YES" ]; then
            if [ -f "$SCRIPT" ]; then
                echo "Uploading dSYM to Crashlytics..."
                "$SCRIPT" || echo "Warning: Crashlytics upload script failed"
            else
                echo "Crashlytics run script not found at: $SCRIPT"
            fi
        else
            echo "Skipping Crashlytics dSYM upload for $CONFIGURATION build"
        fi
        """,
    name: "Crashlytics dSYM Upload",
    inputPaths: [
        "$(DWARF_DSYM_FOLDER_PATH)/$(DWARF_DSYM_FILE_NAME)/Contents/Resources/DWARF/$(TARGET_NAME)",
        "$(BUILT_PRODUCTS_DIR)/$(INFOPLIST_PATH)",
    ],
    basedOnDependencyAnalysis: false
)

let appTarget: Target = .target(
    name: projectName,
    destinations: [.iPhone, .iPad, .mac],
    product: .app,
    bundleId: "com.mirai.tech.playground",
    deploymentTargets: .multiplatform(iOS: "26.0", macOS: "26.0"),
    infoPlist: .extendingDefault(with: [
        "CFBundleShortVersionString": "$(MARKETING_VERSION)",
        "CFBundleVersion": "$(CURRENT_PROJECT_VERSION)",
        "ITSAppUsesNonExemptEncryption": false,
        "LSApplicationCategoryType": "public.app-category.utilities",
        "LSRequiresIPhoneOS": true,
        "NSSupportsLiveActivities": true,
        "UIApplicationSupportsIndirectInputEvents": true,
        "UILaunchStoryboardName": "LaunchScreen",
        "UIRequiresFullScreen": true,
        "UIRequiredDeviceCapabilities": ["arm64"],
        "UISupportedInterfaceOrientations": [
            "UIInterfaceOrientationPortrait"
        ],
    ]),
    sources: ["Sources/**"],
    resources: [.glob(pattern: "Resources/**", excluding: ["Resources/Playground.entitlements"])],
    scripts: [crashlyticsScript],
    dependencies: [
        .package(product: "Uzu"),
        .package(product: "SFSymbols"),
        .package(product: "FirebaseCrashlytics"),
        .package(product: "FirebaseAnalytics"),
        .sdk(name: "Metal", type: .framework),
    ],
    settings: .settings(base: [
        "CODE_SIGN_ENTITLEMENTS": "Resources/Playground.entitlements",
        "DERIVED_SOURCES_DIR": "$(SRCROOT)/Generated",
        // Generate debug symbols for both the stub and the debug dylib
        "COPY_PHASE_STRIP": "NO",
        "STRIP_INSTALLED_PRODUCT": "NO",
    ])
)

let appScheme: Scheme = .scheme(
    name: projectName,
    buildAction: .buildAction(
        targets: [.init(stringLiteral: projectName)],
        preActions: [
            .executionAction(
                title: "Update Uzu Package",
                scriptText:
                    "/bin/bash \"$PROJECT_DIR/../../scripts/prepare_bindings.sh\" swift",
                target: .init(stringLiteral: projectName)
            )
        ]
    ),
    runAction: .runAction(configuration: .release),
    archiveAction: .archiveAction(configuration: .release),
    profileAction: .profileAction(configuration: .release),
    analyzeAction: .analyzeAction(configuration: .debug)
)

let project = Project(
    name: projectName,
    organizationName: "Mirai",
    packages: packages,
    settings: settings,
    targets: [appTarget],
    schemes: [appScheme],
    resourceSynthesizers: [
        .assets(),
        .plists(),
        .fonts(),
        .strings(),
        .files(extensions: ["json", "txt", "md"]),
    ]
)
