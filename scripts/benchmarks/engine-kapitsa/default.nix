{
  pkgs,
  native,
  sources,
  python,
}: let
  kapitsa = native {
    name = "kapitsa";
    src = sources.kapitsa;
    patches = [./patches/sdk-result.patch];
    nativeBuildInputs = [pkgs.cmake];
    # The build names its commit from Git, which the source lacks.
    cmakeFlags = ["-DKAPITSA_GIT=${sources.kapitsa.rev}" "-DKAPITSA_BUILD_SDK=ON"];
    buildFlags = ["kapitsa_sdk"];
    installPhase = ''install -Dm755 libkapitsa_sdk.dylib "$out/lib/libkapitsa_sdk.dylib"'';
  };
in
  (python {
    projectRoot = ./.;
    name = "engine-kapitsa";
  }).overrideAttrs (old: {
    nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.makeWrapper];
    postFixup = ''
      wrapProgram "$out/bin/engine-kapitsa" --set BENCHMARK_KAPITSA_LIBRARY ${kapitsa}/lib/libkapitsa_sdk.dylib
    '';
  })
