{
  description = "A high-performance inference engine for AI models";
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    rust-overlay = {
      url = "github:oxalica/rust-overlay";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    crane.url = "github:ipetkov/crane";
    flake-utils.url = "github:numtide/flake-utils";
  };
  outputs = {
    nixpkgs,
    rust-overlay,
    crane,
    flake-utils,
    ...
  }:
    flake-utils.lib.eachSystem ["aarch64-darwin" "aarch64-linux" "x86_64-linux"] (
      system: let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [(import rust-overlay)];
        };

        rustToolchainNixCC = pkgs.rust-bin.fromRustupToolchainFile ./rust-toolchain.toml;
        rustToolchainStableNixCC = pkgs.rust-bin.fromRustupToolchain (
          (builtins.fromTOML (builtins.readFile ./rust-toolchain.toml)).toolchain // {channel = "stable";}
        );
        rustToolchainsNixCC = {
          nightly = rustToolchainNixCC;
          stable = rustToolchainStableNixCC;
        };
        withoutNixCC = toolchain:
          toolchain.overrideAttrs (_: {
            depsHostHostPropagated = [];
            propagatedBuildInputs = [];
            depsTargetTargetPropagated = [];
          });
        rustupShim = import ./scripts/rustup-shim.nix pkgs;
        rustShimNixCC = rustupShim (rustToolchainsNixCC // {default = "nightly";});
        rustShimHostCC = rustupShim ((pkgs.lib.mapAttrs (_: withoutNixCC) rustToolchainsNixCC) // {default = "nightly";});
        craneLibNixCC = (crane.mkLib pkgs).overrideToolchain rustToolchainNixCC;

        metal-toolchain = import ./scripts/metal-toolchain.nix pkgs {
          build = "27A266a";
          hash = "sha256-vpSn2ZVUo5tNRgJ89KMBCmPE4JFefP70DfCarVagniU=";
        };

        shader-slang-env = with pkgs; let
          libenv =
            {
              darwin = "DYLD_FALLBACK_LIBRARY_PATH";
              linux = "LD_LIBRARY_PATH";
            }.${
              stdenv.hostPlatform.parsed.kernel.name
            };
        in
          makeSetupHook {
            name = "shader-slang-env";
            propagatedBuildInputs = [shader-slang];
          }
          (writeText "shader-slang-env.sh" ''
            export SLANG_INCLUDE_DIR="${lib.getDev shader-slang}/include"
            export SLANG_LIB_DIR="${lib.getLib shader-slang}/lib"
            addToSearchPath ${libenv} "${lib.getLib shader-slang}/lib"
          '');

        bindgen-env = with pkgs;
          makeSetupHook {name = "bindgen-env";}
          (writeText "bindgen-env.sh" ''
            export LIBCLANG_PATH="${lib.getLib libclang}/lib"
            export BINDGEN_EXTRA_CLANG_ARGS="-isystem ${lib.getDev glibc}/include"
          '');

        vulkan-env = with pkgs;
          makeSetupHook {name = "vulkan-env";}
          (writeText "vulkan-env.sh" ''
            addToSearchPath LD_LIBRARY_PATH "${lib.getLib vulkan-loader}/lib"
            export VK_ADD_LAYER_PATH="${vulkan-validation-layers}/share/vulkan/explicit_layer.d"
          '');

        playwright-env = with pkgs;
          makeSetupHook {
            name = "playwright-env";
            propagatedBuildInputs = [playwright-driver playwright-driver.browsers-chromium];
          }
          (writeText "playwright-env.sh" ''
            export PLAYWRIGHT_BROWSERS_PATH="${playwright-driver.browsers-chromium}"
            export PLAYWRIGHT_DRIVER_PATH="${playwright-driver}"
            export PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD="1"
          '');

        hostToolchainBuildInputs = with pkgs; (lib.optionals stdenv.hostPlatform.isLinux [alsa-lib]);
        nixToolchainBuildInputs = hostToolchainBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [apple-sdk_26]);

        hostToolchainNativeBuildInputs = with pkgs; [cmake] ++ (lib.optionals stdenv.hostPlatform.isLinux [pkg-config]);
        nixToolchainNativeBuildInputs = hostToolchainNativeBuildInputs ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [metal-toolchain]);

        commonDevshellPackages = with pkgs;
          [
            nil
            shader-slang-env
            playwright-env
            uv
            maturin
            nodejs_26
            pnpm_12
            wasm-bindgen-cli_0_2_129 # Keep in sync with Cargo.toml.
            evcxr
            cargo-deny
            cargo-nextest
            cargo-hack
            cargo-expand
            samply
            cargo-flamegraph
            cargo-show-asm
            critcmp
          ]
          ++ (with pkgs; lib.optionals stdenv.hostPlatform.isLinux [bindgen-env vulkan-env vulkan-tools shaderc spirv-tools lldb])
          ++ (with pkgs; lib.optionals stdenv.hostPlatform.isDarwin [cargo-swift tuist]);
        hostToolchainDevshellPackages = commonDevshellPackages ++ [rustShimHostCC];
        nixToolchainDevshellPackages = commonDevshellPackages ++ [rustShimNixCC];

        mirai = craneLibNixCC.buildPackage {
          pname = "mirai";
          src = ./.;

          cargoExtraArgs = "-p cli";
          installPhaseCommand = ''
            mkdir -p $out/bin
            install -Dm755 target/release/cli $out/bin/mirai
          '';

          buildInputs = nixToolchainBuildInputs;
          nativeBuildInputs = nixToolchainNativeBuildInputs;

          doCheck = false;
        };
      in {
        formatter = pkgs.alejandra;

        packages = {
          inherit mirai;
          default = mirai;
        };

        devShells = {
          default = pkgs.mkShell {
            buildInputs = nixToolchainBuildInputs;
            nativeBuildInputs = nixToolchainNativeBuildInputs;

            packages = nixToolchainDevshellPackages;
          };

          host-xcode = pkgs.mkShellNoCC {
            buildInputs = hostToolchainBuildInputs;
            nativeBuildInputs = hostToolchainNativeBuildInputs;

            packages = hostToolchainDevshellPackages;

            TUIST_GENERATE_OPEN = "false";
          };
        };
      }
    );
}
