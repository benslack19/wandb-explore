"""Common data science utility functions"""

from typing import List, TypeAlias, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

NumericData: TypeAlias = Union[List[float], List[int], npt.ArrayLike, pd.Series]


def standardize(x: NumericData) -> npt.NDArray[np.floating]:
    arr = np.asarray(x, dtype=np.floating)
    return (arr - np.mean(arr)) / np.std(arr)
