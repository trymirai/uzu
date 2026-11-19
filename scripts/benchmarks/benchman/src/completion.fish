function __benchman_complete
    set --local found (env _BENCHMAN_COMPLETE=complete_fish _TYPER_COMPLETE_ARGS=(commandline -cp) benchman)
    or __fish_complete_path (commandline -ct)
    string join \n -- $found
end

complete --command benchman --no-files --keep-order --arguments "(__benchman_complete)"
