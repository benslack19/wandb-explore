"""Common data science utility functions"""

from typing import List, Union


import pandas as pd


import numpy as np


RANDOM_SEED = 12345


def standardize(x: Union[List, np.array]):
    x = (x - np.mean(x)) / np.std(x)
    return x


# df with continuous, ordinal, and binary variables
nrows, ncols = 100, 5
rng = np.random.default_rng(RANDOM_SEED)

df_example = pd.DataFrame(
    rng.rand(nrows, ncols), columns=["a", "b", "c", "d", "e"]
).assign(
    f_ordinal=rng.choice([0, 1, 2, 3], nrows),
    g_binary=rng.choice([True, False], nrows),
)
