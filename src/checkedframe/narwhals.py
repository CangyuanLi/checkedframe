from typing import Generic, TypeVar

import narwhals as nw

T = TypeVar("T")


class DataFrame(nw.DataFrame, Generic[T]): ...
