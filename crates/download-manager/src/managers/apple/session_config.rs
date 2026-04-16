use crate::prelude::*;

#[allow(unused)]
#[derive(Debug, Clone)]
pub enum BackgroundSessionID {
    Default,
    Custom(String),
}

#[allow(unused)]
#[derive(Debug, Clone)]
pub enum SessionConfig {
    Foreground,
    Background(BackgroundSessionID),
    /// Background in apps, Foreground in CLIs
    Automatic,
}

impl Default for SessionConfig {
    fn default() -> Self {
        Self::Automatic
    }
}

impl SessionConfig {
    pub fn ns_url_session_configuration(&self) -> Retained<NSURLSessionConfiguration> {
        let bundle_id = NSBundle::mainBundle().bundleIdentifier().unwrap_or_default().to_string();

        let create_ephemeral_config = || NSURLSessionConfiguration::ephemeralSessionConfiguration();

        let create_background_config = |background_session_id: &BackgroundSessionID| {
            let session_id = NSString::from_str(&match background_session_id {
                BackgroundSessionID::Default => {
                    if bundle_id.is_empty() {
                        "trymirai.download-manager".to_string()
                    } else {
                        format!("{}.trymirai.download-manager", bundle_id)
                    }
                },
                BackgroundSessionID::Custom(id) => id.clone(),
            });

            let config = NSURLSessionConfiguration::backgroundSessionConfigurationWithIdentifier(&session_id);
            config.setSessionSendsLaunchEvents(true);
            config.setDiscretionary(false);
            config.setWaitsForConnectivity(true);
            config
        };

        match self {
            SessionConfig::Foreground => create_ephemeral_config(),
            SessionConfig::Background(background_session_id) => create_background_config(background_session_id),
            SessionConfig::Automatic => {
                if bundle_id.is_empty() {
                    create_ephemeral_config()
                } else {
                    create_background_config(&BackgroundSessionID::Default)
                }
            },
        }
    }
}
