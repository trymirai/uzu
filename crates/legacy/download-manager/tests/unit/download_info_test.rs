use super::DownloadInfo;

#[test]
fn constructors_keep_the_legacy_three_field_shape() {
    assert_eq!(
        DownloadInfo::new("https://example.test/model", "/models/model"),
        DownloadInfo {
            source_url: "https://example.test/model".to_string(),
            destination_path: "/models/model".to_string(),
            crc32c: None,
        }
    );
    assert_eq!(
        DownloadInfo::with_crc("https://example.test/model", "/models/model", "AAAAAA=="),
        DownloadInfo {
            source_url: "https://example.test/model".to_string(),
            destination_path: "/models/model".to_string(),
            crc32c: Some("AAAAAA==".to_string()),
        }
    );
}

#[test]
fn json_round_trip_uses_the_legacy_field_names() {
    let info = DownloadInfo::with_crc("https://example.test/model", "/models/model", "AAAAAA==");
    let json = info.to_json().unwrap();

    assert_eq!(
        json,
        r#"{"source_url":"https://example.test/model","destination_path":"/models/model","crc32c":"AAAAAA=="}"#
    );
    assert_eq!(DownloadInfo::from_json(&json).unwrap(), info);
}
