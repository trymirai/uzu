import os


def forbid_local_server() -> None:
    # Prefect's default profile lets it start a local server; only the environment, read on import, overrides it.
    os.environ["PREFECT_SERVER_EPHEMERAL_ENABLED"] = "false"
