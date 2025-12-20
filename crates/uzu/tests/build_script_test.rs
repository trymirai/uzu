mod build_script {
    include!("../build.rs");

    #[test]
    fn parse_env_flag_value_accepts_truthy_values() {
        for value in ["1", "true", "TRUE", " yes ", "On"] {
            assert!(parse_env_flag_value(Some(value)));
        }
    }

    #[test]
    fn parse_env_flag_value_rejects_falsey_values() {
        for value in ["0", "false", "no", "off", ""] {
            assert!(!parse_env_flag_value(Some(value)));
        }
        assert!(!parse_env_flag_value(None));
    }

    #[test]
    fn resolve_metal_sdk_for_platforms() {
        assert_eq!(
            resolve_metal_sdk("macos", "aarch64-apple-darwin", ""),
            Some("macosx")
        );
        assert_eq!(
            resolve_metal_sdk("ios", "aarch64-apple-ios", ""),
            Some("iphoneos")
        );
        assert_eq!(
            resolve_metal_sdk("ios", "x86_64-apple-ios", "sim"),
            Some("iphonesimulator")
        );
        assert_eq!(resolve_metal_sdk("linux", "x86_64-unknown-linux-gnu", ""), None);
    }

    #[test]
    fn detects_missing_toolchain_messages() {
        assert!(is_toolchain_missing(
            "error: cannot execute tool 'metal' due to missing Metal Toolchain"
        ));
        assert!(is_toolchain_missing(
            "xcrun: error: unable to find utility \"metal\", not a developer tool or in PATH"
        ));
        assert!(!is_toolchain_missing("some other error"));
    }
}
