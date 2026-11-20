#compdef benchman

_benchman() {
  eval "$(env _TYPER_COMPLETE_ARGS="${words[1,$CURRENT]}" _BENCHMAN_COMPLETE=complete_zsh benchman)"
}

# Loaded from the function path, this file is the first completion itself; sourced, it only registers.
if [[ $zsh_eval_context[-1] == loadautofunc ]]; then
  _benchman "$@"
else
  compdef _benchman benchman
fi
