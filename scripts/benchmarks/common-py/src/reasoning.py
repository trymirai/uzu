LEVELS = ("low", "medium", "high", "xhigh")


def map_reasoning_efforts(supported, default="default"):
    """Maps every effort to one of the supported, as close in meaning as they allow; `default` is what none asks for."""
    levels = [level for level in LEVELS if level in supported]

    def resolve(effort):
        if effort == "default":
            return default
        if effort in supported:
            return effort
        if effort == "disabled":
            # The lowest level if reasoning can't be turned off.
            return levels[0] if levels else default
        # The next level up, else the highest, else the default.
        higher = [level for level in levels if LEVELS.index(level) > LEVELS.index(effort)]
        return (higher or levels or [default])[0 if higher else -1]

    return {effort: resolve(effort) for effort in ("disabled", "default", *LEVELS)}
