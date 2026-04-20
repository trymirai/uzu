use reqwest::Client as ReqwestClient;
use serde::de::DeserializeOwned;

use crate::api::{Config, Endpoint, Error};

pub struct Client {
    config: Config,
    client: ReqwestClient,
}

impl Client {
    pub fn new(config: Config) -> Result<Self, Error> {
        #[allow(unused_mut)]
        let mut client_builder = reqwest::Client::builder();
        #[cfg(not(target_family = "wasm"))]
        {
            client_builder = client_builder.timeout(config.timeout);
        }
        let client = client_builder.build().map_err(Error::from)?;

        Ok(Self {
            config,
            client,
        })
    }

    pub async fn response<E: Endpoint, T: DeserializeOwned>(
        &self,
        endpoint: &E,
    ) -> Result<T, Error> {
        let url = self.url(endpoint);
        let method = endpoint.method();
        let payload = endpoint.payload(&self.config);

        let mut request = self.client.request(method, url);
        if let Some(query) = payload.query {
            request = request.query(
                &query.into_iter().map(|(key, value)| (key, value.to_string())).collect::<Vec<(String, String)>>(),
            );
        }
        if let Some(body) = payload.body {
            request = request.json(&body);
        }
        for (key, value) in self.config.headers.iter() {
            request = request.header(key, value);
        }
        for (key, value) in endpoint.headers().iter() {
            request = request.header(key, value);
        }

        let response = request.send().await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Http {
                code: status.as_u16(),
                body,
            });
        }
        response.json::<T>().await.map_err(|error| Error::Decode(error.to_string()))
    }
}

impl Client {
    fn url<E: Endpoint>(
        &self,
        endpoint: &E,
    ) -> String {
        format!("{}/{}", self.config.base_url, endpoint.path())
    }
}
