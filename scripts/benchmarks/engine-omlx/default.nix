{
  lib,
  python,
}:
python {
  projectRoot = ./.;
  name = "bench-omlx";
  overrides = final: prev:
    lib.genAttrs ["omlx" "mlx-vlm" "mlx-audio" "mlx-embeddings"] (name:
      prev.${name}.overrideAttrs (old: {
        nativeBuildInputs = old.nativeBuildInputs ++ final.resolveBuildSystem {setuptools = [];};
      }))
    // {
      mlx-lm = prev.mlx-lm.overrideAttrs (old: {
        nativeBuildInputs =
          old.nativeBuildInputs
          ++ final.resolveBuildSystem {
            setuptools = [];
            setuptools-scm = [];
          };
        env = (old.env or {}) // {SETUPTOOLS_SCM_PRETEND_VERSION = old.version;};
      });
    };
}
