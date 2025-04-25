from typing import Generic, TypeVar

import pandas as pd

T = TypeVar("T")


class DataFrame(pd.DataFrame, Generic[T]): ...
