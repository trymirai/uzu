{
  pkgs,
  pyproject-nix,
  uv2nix,
  pyproject-build-systems,
}: {
  projectRoot,
  name,
  overrides ? (_: _: {}),
}: let
  inherit (pkgs) lib;
  version = lib.versions.majorMinor (lib.trim (builtins.readFile ../.python-version));
  python = pkgs.${"python" + builtins.replaceStrings ["."] [""] version};
  workspace = uv2nix.lib.workspace.loadWorkspace {workspaceRoot = projectRoot;};
  overlay = lib.composeManyExtensions [
    pyproject-build-systems.overlays.wheel
    (workspace.mkPyprojectOverlay {sourcePreference = "wheel";})
    (final: prev:
      {
        benchmarks-common = prev.benchmarks-common.overrideAttrs {
          postPatch = ''
            mkdir -p ../common-cpp
            cp -r ${../common-cpp/src} ../common-cpp/src
          '';
        };
      }
      // lib.optionalAttrs (prev ? mlx) {
        # WORKAROUND: The wheel's libmlx belongs to a separate distribution.
        mlx = prev.mlx.overrideAttrs (old: {
          postFixup =
            (old.postFixup or "")
            + ''
              for module in "$out/${python.sitePackages}/mlx/"*.so; do
                install_name_tool -change @rpath/libmlx.dylib ${final.mlx-metal}/${python.sitePackages}/mlx/lib/libmlx.dylib "$module"
              done
            '';
        });
      })
    overrides
  ];
  pythonSet = (pkgs.callPackage pyproject-nix.build.packages {inherit python;}).overrideScope (final: prev:
    # WORKAROUND: uv2nix adds sandbox exceptions rejected by sandbox=true.
      lib.mapAttrs (
        _: package:
          if lib.isDerivation package
          then package.overrideAttrs {sandboxProfile = "";}
          else package
      ) (overlay final prev));
in
  pythonSet.mkVirtualEnv name workspace.deps.default
