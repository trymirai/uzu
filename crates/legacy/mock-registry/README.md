# mock-registry

`mock-registry` provides a local model registry for tests that should not depend on the internet. It returns realistic `shoji::Model` data from a registry endpoint and serves deterministic model files from local HTTP URLs.

## Internals

- `MockRegistry` starts two local servers: a `wiremock` server for `POST /fetch/models` and a lightweight TCP file server for model file downloads.
- The registry response is built from `models`, so tests can model a remote registry with more than one model.
- Files are generated in memory with deterministic bytes, real sizes, and CRC32C hashes. No fixture files are checked into the repository.
- `FileServer` supports `GET`, `HEAD`, range requests, and close-delimited HTTP responses used by the download managers.
- `Behavior` is a bitflag set for test scenarios such as corrupted bodies and throttled streaming.
- Public APIs return the crate-local `Result<T>` and `Error` types from `error.rs`.

The crate is intended for integration-style tests in `download-manager` and `uzu`, especially tests that need real HTTP URLs without calling external services.
