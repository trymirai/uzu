{
  pkgs,
  native,
  sources,
  python,
}: let
  splash = native {
    name = "splash";
    src = sources.splash;
    patches = [./patches/in-process.patch];
    nativeBuildInputs = [pkgs.python3 pkgs.perl];
    buildFlags = ["build/splash"];
    enableParallelBuilding = true;
    # WORKAROUND: Nixpkgs' Clang warns where Xcode's does not, and upstream builds with -Werror.
    env.NIX_CFLAGS_COMPILE = "-Wno-missing-designated-field-initializers";
    installPhase = ''
      mkdir -p "$out/splash/engine"
      cp build/splash build/*.metallib "$out/splash/engine/"
      cp -r install server "$out/splash/"
      echo '{"version": "${sources.splash.rev}"}' > "$out/splash/release.json"
    '';
  };
in
  (python {
    projectRoot = ./.;
    name = "engine-splash";
  }).overrideAttrs (old: {
    nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.makeWrapper];
    postFixup = ''wrapProgram "$out/bin/engine-splash" --set BENCHMARK_SPLASH_ROOT ${splash}/splash'';
  })
