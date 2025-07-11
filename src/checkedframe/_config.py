from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, TypedDict, Union

from .selectors import Selector, by_name

if TYPE_CHECKING:
    from typing_extensions import Unpack

SelectorLike = Union[str, Iterable[str], Selector]


class _PossibleConfigs(TypedDict):
    nullable: bool
    required: bool
    cast: bool
    allow_nan: bool
    allow_inf: bool


class Config:
    def __init__(
        self,
        selector: SelectorLike,
        **kwargs: Unpack[_PossibleConfigs],
    ):
        if not isinstance(selector, Selector):
            actual_selector = by_name(selector)
        else:
            actual_selector = selector

        self.selector = actual_selector
        self.dct = kwargs


# This just makes it easier to do isinstance checks
class ConfigList:
    def __init__(self, *args: Config):
        self.args = args


def apply_configs(*args: Config):
    def decorator(cls):
        cls.__private_checkedframe_config = ConfigList(*args)

        return cls

    return decorator
