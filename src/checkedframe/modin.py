from typing import Generic, TypeVar

import modin.pandas as md

T = TypeVar("T")


class DataFrame(md.DataFrame, Generic[T]): ...
