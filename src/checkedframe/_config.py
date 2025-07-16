from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict

from .selectors import Selector, by_name

if TYPE_CHECKING:
    from typing_extensions import Unpack


class _PossibleConfigs(TypedDict):
    nullable: bool
    required: bool
    cast: bool
    allow_nan: bool
    allow_inf: bool


class Config:
    """See :func:`apply_configs` for how this class is used.

    Parameters
    ----------
    columns : str | Iterable[str] | Selector
        The columns to apply the configuration to
    **kwargs : Unpack[_PossibleConfigs]
        Valid keyword arguments are:

        nullable : bool
            Whether the column is nullable or not
        required : bool
            Whether the column is required to exist or not
        cast : bool
            Whether to cast to the specified datatype
        allow_nan : bool
            Whether the column can have NaNs or not
        allow_inf : bool
            Whether the column can have inf/-inf or not
    """

    def __init__(
        self,
        columns: str | Iterable[str] | Selector,
        **kwargs: Unpack[_PossibleConfigs],
    ):
        if not isinstance(columns, Selector):
            actual_selector = by_name(columns)
        else:
            actual_selector = columns

        self.selector = actual_selector
        self.dct = kwargs


# This just makes it easier to do isinstance checks
class ConfigList:
    def __init__(self, *args: Config):
        self.args = args


def apply_configs(*args: Config):
    """Change the configuration options of columns.

    .. note::
        Config changes done by `apply_configs` will overwrite per-column configurations.

    Examples
    --------
    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        @cf.apply_configs(
            cf.Config(cfs.float(), allow_nan=True)
        )
        class S(cf.Schema):
            check_bal = cf.Float64()
            sav_bal = cf.Float64(allow_nan=False)


        df = pl.DataFrame({"check_bal": [100.4, float("nan")], "sav_bal": [88.99, float("nan")]})
        S.validate(df)  # passes!
    """

    def decorator(cls):
        cls.__private_checkedframe_config = ConfigList(*args)

        return cls

    return decorator
