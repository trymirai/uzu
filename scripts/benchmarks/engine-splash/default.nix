{
  pkgs,
  native,
  sources,
  python,
  server,
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
      mkdir -p "$out/engine"
      cp build/splash build/*.metallib "$out/engine/"
      cp -r install server "$out/"
      echo '{"version": "${sources.splash.rev}"}' > "$out/release.json"
    '';
  };
in
  (python {
    projectRoot = ./.;
    name = "bench-splash";
  }).overrideAttrs (final: old: {
    nativeBuildInputs = old.nativeBuildInputs ++ [pkgs.makeWrapper];
    postFixup = ''wrapProgram "$out/bin/bench-splash" --set BENCHMARK_SPLASH_ROOT ${splash}'';
    passthru.server = server "splash" {
      BENCHMARK_SPLASH_ROOT = "${splash}";
      BENCHMARK_SPLASH_PYTHON = "${final.finalPackage}/bin/python";
    };
  })
