"""Common data science utility functions"""

import numpy as np
from typing import List, Union


def standardize(x: Union[List, np.array]):
    x = (x - np.mean(x)) / np.std(x)
    return x


fruits = [
    "apple", "banana", "orange", "apple",
    "banana",
    "orange",
    "apple",
    "banana",
    "orange",
]
