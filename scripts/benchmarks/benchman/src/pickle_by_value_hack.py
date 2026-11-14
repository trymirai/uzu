import sys

import cloudpickle


def pickle_benchman_by_value() -> None:
    cloudpickle.register_pickle_by_value(sys.modules["benchman"])
