{
  lib,
  rustPlatform,
  fetchFromGitHub,
  pkg-config,
  cmake,
  curl,
  openssl,
  zlib,
}:
rustPlatform.buildRustPackage (finalAttrs: {
  pname = "cargo-instruments";
  version = "0.4.18";

  src = fetchFromGitHub {
    owner = "cmyr";
    repo = "cargo-instruments";
    tag = "v${finalAttrs.version}";
    hash = "sha256-rLGsrzxQJRYJ/2goTy7Pa78Y22Rw9APD9YAECS9sbiE=";
  };

  cargoHash = "sha256-qcihtXcPNLbB/pG6JrA43z0dUQ6/JsBuabZFADQtziA=";

  # The demangling test inspects a symbol optimized out in release builds.
  checkType = "debug";

  nativeBuildInputs = [pkg-config cmake];
  buildInputs = [curl openssl zlib];

  meta = {
    description = "Profile Rust binaries on macOS using Xcode Instruments";
    homepage = "https://github.com/cmyr/cargo-instruments";
    license = lib.licenses.mit;
    mainProgram = "cargo-instruments";
    platforms = lib.platforms.darwin;
  };
})
