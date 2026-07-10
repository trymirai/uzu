This example demonstrates the browser-compatible functionality of the `download-manager` crate.

The crate uses [OPFS](https://developer.mozilla.org/en-US/docs/Web/API/File_System_API/Origin_private_file_system)
for storage. For this reason, an OPFS browser extension can be useful for debugging — for example,
[OPFS Explorer](https://chromewebstore.google.com/detail/opfs-explorer/hhegfidnlemidclkkldeekjamkfcamic) for Chrome.

This example uses [Trunk](https://github.com/trunk-rs/trunk) as the web application bundler, so it must be installed
first.

Trunk compiles the application for the `wasm32-unknown-unknown` target:

```shell
rustup target add wasm32-unknown-unknown --toolchain nightly
```

To run the web application, execute the following:

```shell
cd www
trunk serve
```
