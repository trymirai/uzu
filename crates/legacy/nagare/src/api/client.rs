use std::time::Duration;

use bon::bon;
use reqwest::{Client as ReqwestClient, RequestBuilder, Response};
use serde::Serialize;

use crate::api::{Endpoint, Error, RetryConfig};

pub struct Client {
    client: ReqwestClient,
    base_url: String,
    bearer_token: Option<String>,
    retry: RetryConfig,
}

#[bon]
impl Client {
    #[builder]
    pub fn new(
        #[builder(into)] base_url: String,
        #[builder(default = Duration::from_secs(10))] timeout: Duration,
        #[builder(into)] bearer_token: Option<String>,
        #[builder(default)] retry: RetryConfig,
    ) -> Result<Self, Error> {
        let client = ReqwestClient::builder().timeout(timeout).build().map_err(Error::from)?;

        Ok(Self {
            client,
            base_url,
            bearer_token,
            retry,
        })
    }

    /// POST `request` as JSON and decode `E::Response`.
    pub async fn call<E: Endpoint>(
        &self,
        request: &E::Request,
    ) -> Result<E::Response, Error> {
        let response = self.checked(E::PATH, request).await?;
        response.json::<E::Response>().await.map_err(|error| Error::Decode(error.to_string()))
    }

    /// POST `request` as JSON and discard the response, checking only the status.
    pub async fn send<E: Endpoint>(
        &self,
        request: &E::Request,
    ) -> Result<(), Error> {
        self.checked(E::PATH, request).await?;
        Ok(())
    }
}

impl Client {
    async fn checked(
        &self,
        path: &str,
        body: &impl Serialize,
    ) -> Result<Response, Error> {
        let response = self.retry.send(|| self.request(path).json(body).send()).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().await.unwrap_or_default();
            return Err(Error::Http {
                code: status,
                body,
            });
        }
        Ok(response)
    }

    fn request(
        &self,
        path: &str,
    ) -> RequestBuilder {
        let request = self.client.post(format!("{}/{}", self.base_url, path));
        match &self.bearer_token {
            Some(token) => request.bearer_auth(token),
            None => request,
        }
    }
}
