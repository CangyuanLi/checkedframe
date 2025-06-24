from typing import Generic, TypeVar

import pyarrow as pa

T = TypeVar("T")


class Table(pa.Table, Generic[T]): ...
