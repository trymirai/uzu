{
  lib,
  rustPlatform,
  fetchFromGitHub,
}:
rustPlatform.buildRustPackage (finalAttrs: {
  pname = "safetensors-explorer";
  version = "0.2.0";

  src = fetchFromGitHub {
    owner = "EricLBuehler";
    repo = "safetensors_explorer";
    tag = "v${finalAttrs.version}";
    hash = "sha256-VHWaCxVDy7yVLLYDyQvjUzNnO3wmUcqdQMvM6CXRIH4=";
  };

  cargoHash = "sha256-KtJd5X5qG2ZAwMeI6cayUleF4otESk+3dJz+oe4mkx4=";

  meta = {
    description = "Terminal explorer for safetensors and GGUF files";
    homepage = "https://github.com/EricLBuehler/safetensors_explorer";
    license = lib.licenses.mit;
    mainProgram = "safetensors_explorer";
    platforms = lib.platforms.unix;
  };
})
