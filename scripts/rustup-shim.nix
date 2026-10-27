pkgs: {default, ...} @ args: let
  inherit (pkgs) lib;
  toolchains = builtins.removeAttrs args ["default"];
  channels = builtins.attrNames toolchains;
  propagatedDependencies =
    lib.genAttrs [
      "depsHostHostPropagated"
      "propagatedBuildInputs"
      "depsTargetTargetPropagated"
    ] (name: let
      dependencies = toolchains.${default}.${name} or [];
    in
      assert lib.assertMsg
      (lib.all (toolchain: (toolchain.${name} or []) == dependencies) (builtins.attrValues toolchains))
      "rustup-shim: toolchains disagree on ${name}"; dependencies);
  shim = ''
    #!${pkgs.runtimeShell}
    set -euo pipefail

    tool="''${0##*/}"

    if [[ "''${1-}" == +* ]]; then
      selection="''${1#+}"
      shift
    else
      selection="''${RUSTUP_TOOLCHAIN:-${default}}"
    fi

    if [[ "$tool" == rustup ]]; then
      kind="''${1-}"
      if [[ ( "$kind" != target && "$kind" != component ) || "''${2-}" != list ]]; then
        echo 'rustup-shim: supported commands: target list, component list' >&2
        exit 1
      fi
      shift 2
      installed_only=false
      while (( $# )); do
        case "$1" in
          --toolchain)
            if (( $# < 2 )); then
              echo 'rustup-shim: --toolchain requires a value' >&2
              exit 1
            fi
            selection="$2"
            shift 2
            ;;
          --toolchain=*) selection="''${1#*=}"; shift ;;
          --installed) installed_only=true; shift ;;
          *) printf "rustup-shim: unsupported argument '%s'\n" "$1" >&2; exit 1 ;;
        esac
      done
    fi

    case "$selection" in
      ${lib.concatMapStringsSep "\n" (channel: ''
        ${lib.escapeShellArg channel}|${lib.escapeShellArg (toString toolchains.${channel})})
          channel=${lib.escapeShellArg channel}
          toolchain=${lib.escapeShellArg (toString toolchains.${channel})}
          ;;
      '')
      channels}
      *)
        printf "rustup-shim: unknown toolchain '%s'; available: %s\n" "$selection" ${lib.escapeShellArg (lib.concatStringsSep ", " channels)} >&2
        exit 1
        ;;
    esac

    if [[ "$tool" == rustup ]]; then
      suffix=' (installed)'
      $installed_only && suffix=""
      if [[ "$kind" == target ]]; then
        shopt -s nullglob
        for target in "$toolchain"/lib/rustlib/*; do
          libraries=("$target"/lib/libcore-*.rlib)
          if (( ''${#libraries[@]} )); then
            printf '%s%s\n' "''${target##*/}" "$suffix"
          fi
        done
      elif [[ -f "$toolchain/lib/rustlib/src/rust/library/Cargo.toml" ]]; then
        printf 'rust-src%s\n' "$suffix"
      fi
      exit 0
    fi

    if [[ ! -f "$toolchain/bin/$tool" || ! -x "$toolchain/bin/$tool" ]]; then
      printf "rustup-shim: '%s' is not available in toolchain '%s'\n" "$tool" "$channel" >&2
      exit 127
    fi

    bin="${builtins.placeholder "out"}/bin"
    if [[ "''${PATH-}" != "$bin" && "''${PATH-}" != "$bin":* ]]; then
      export PATH="$bin''${PATH:+:$PATH}"
    fi

    cargo_home="''${CARGO_HOME:-''${HOME:+$HOME/.cargo}}"
    if [[ -n "$cargo_home" ]]; then
      [[ "$cargo_home" == /* ]] || cargo_home="$PWD/$cargo_home"
      case ":$PATH:" in
        *":$cargo_home/bin:"*) ;;
        *) export PATH="$PATH:$cargo_home/bin" ;;
      esac
    fi

    export RUSTUP_HOME="''${RUSTUP_HOME-}"
    export RUSTUP_TOOLCHAIN="$toolchain"

    exec "$toolchain/bin/$tool" "$@"
  '';
in
  pkgs.stdenvNoCC.mkDerivation (propagatedDependencies
    // {
      name = "rustup-shim";
      passthru = {inherit default toolchains;};
      inherit shim;
      passAsFile = ["shim"];
      dontUnpack = true;
      dontConfigure = true;
      dontBuild = true;
      installPhase = ''
        runHook preInstall

        mkdir -p "$out/bin"
        install -m755 "$shimPath" "$out/bin/rustup"

        for toolchain in ${lib.escapeShellArgs (map toString (builtins.attrValues toolchains))}; do
          for binary in "$toolchain"/bin/*; do
            ln -sfn "$out/bin/rustup" "$out/bin/''${binary##*/}"
          done
        done

        runHook postInstall
      '';
    })
