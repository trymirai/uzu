from collections.abc import Awaitable, Callable
from functools import partial

from benchman.datasets import aime25, gsm8k, humaneval, math500, mbpp, mirai_mixture, mt_bench, oneoff
from benchman.datasets.dataset import Dataset

LOADERS: dict[str, Callable[[], Awaitable[Dataset]]] = {
    "oneoff": oneoff.load,
    "mirai-mixture-fast": partial(mirai_mixture.load, "fast"),
    "mirai-mixture-slow": partial(mirai_mixture.load, "slow"),
    "mt-bench": mt_bench.load,
    "humaneval": humaneval.load,
    "math500": math500.load,
    "gsm8k": gsm8k.load,
    "aime25": aime25.load,
    "mbpp": mbpp.load,
}


async def load_dataset(name: str) -> Dataset:
    return await LOADERS[name]()
