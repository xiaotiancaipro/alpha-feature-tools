import string

import random
from typing import Any

import numpy as np
import pandas as pd


class Secret(object):

    def __init__(self):
        self.__label_0 = list("bcdefghi0123")
        self.__label_1 = list("lmnopqr456")
        self.__label_ = list("uvwxyz789")

    def encode(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        df[label] = df[label].map(self.__encode_map_func)
        return df

    def decode(self, df: pd.DataFrame, label: str) -> pd.DataFrame:
        df[label] = df[label].map(self.__decode_map_func)
        return df

    def __encode_map_func(self, x: Any) -> str:
        sample = list(string.ascii_lowercase) + list(string.digits)
        _begin = random.choices(sample, k=5)
        _end = random.choices(sample, k=26)
        try:
            x = int(x)
        except Exception:
            return "".join([*_begin, random.choice(self.__label_), *_end])
        if x not in [0, 1]:
            raise "unknown"
        _random = None
        if x == 0:
            _random = random.choice(self.__label_0)
        if x == 1:
            _random = random.choice(self.__label_1)
        return "".join([*_begin, _random, *_end])

    def __decode_map_func(self, x: str):
        _random = x[5]
        if _random in self.__label_0:
            return 0
        if _random in self.__label_1:
            return 1
        return np.nan
