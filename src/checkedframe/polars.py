from typing import Generic, TypeVar

import polars as pl

T = TypeVar("T")


class DataFrame(pl.DataFrame, Generic[T]): ...
