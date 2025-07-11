from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Callable, Union

from ._dtypes import Boolean, Categorical, Date, Datetime, NarwhalsDType, String

TypeOrInstance = Union[NarwhalsDType, type[NarwhalsDType]]


class Selector:
    def __init__(self, condition: Callable[[str, TypeOrInstance], bool]):
        self.condition = condition

    def __call__(self, schema: dict[str, TypeOrInstance]) -> list[str]:
        return [col for col, dtype in schema.items() if self.condition(col, dtype)]

    def __invert__(self) -> Selector:
        return Selector(lambda col, dtype: not self.condition(col, dtype))

    def __sub__(self, other: Selector) -> Selector:
        return Selector(
            lambda col, dtype: self.condition(col, dtype)
            and not other.condition(col, dtype)
        )

    def __and__(self, other: Selector) -> Selector:
        return Selector(
            lambda col, dtype: self.condition(col, dtype)
            and other.condition(col, dtype)
        )

    def __or__(self, other: Selector) -> Selector:
        return Selector(
            lambda col, dtype: self.condition(col, dtype) or other.condition(col, dtype)
        )

    def __xor__(self, other: Selector) -> Selector:
        return Selector(
            lambda col, dtype: self.condition(col, dtype) != other.condition(col, dtype)
        )

    def exclude(self, other: str | Iterable[str] | Selector) -> Selector:
        if not isinstance(other, Selector):
            selector_other = by_name(other)
        else:
            selector_other = other

        return self.__sub__(selector_other)


def _flatten_str_iterable(lst: Iterable[str | Iterable[str]]) -> list[str]:
    res = []
    for str_or_iterable in lst:
        if isinstance(str_or_iterable, str):
            res.append(str_or_iterable)
        else:
            for str_ in str_or_iterable:
                res.append(str_)

    return res


def all() -> Selector:
    return Selector(lambda _1, _2: True)


def by_name(*names: str | Iterable[str]):
    all_names = _flatten_str_iterable(names)

    return Selector(lambda actual_name, _: actual_name in all_names)


def matches(pattern: str | re.Pattern[str]) -> Selector:
    return Selector(lambda name, _: re.search(pattern, name) is not None)


def contains(*substrings: str | Iterable[str]) -> Selector:
    all_substrings = _flatten_str_iterable(substrings)

    return Selector(lambda name, _: any(s in name for s in all_substrings))


def starts_with(*prefixes: str | Iterable[str]):
    all_prefixes = tuple(_flatten_str_iterable(prefixes))

    return Selector(lambda name, _: name.startswith(all_prefixes))


def ends_with(*suffixes: str | Iterable[str]):
    all_suffixes = tuple(_flatten_str_iterable(suffixes))

    return Selector(lambda name, _: name.endswith(all_suffixes))


def by_dtype(*dtypes: TypeOrInstance | Iterable[TypeOrInstance]) -> Selector:
    all_dtypes = []
    for d in dtypes:
        if isinstance(d, Iterable):
            for x in d:
                all_dtypes.append(x)
        else:
            all_dtypes.append(d)

    return Selector(lambda _, actual_dtype: actual_dtype in all_dtypes)


def boolean() -> Selector:
    return by_dtype(Boolean)


def categorical() -> Selector:
    return by_dtype(Categorical)


def date() -> Selector:
    return by_dtype(Date)


def datetime() -> Selector:
    return by_dtype(Datetime)


def decimal() -> Selector:
    return Selector(lambda _, dtype: dtype.is_decimal())


def float() -> Selector:
    return Selector(lambda _, dtype: dtype.is_float())


def integer() -> Selector:
    return Selector(lambda _, dtype: dtype.is_integer())


def signed_integer() -> Selector:
    return Selector(lambda _, dtype: dtype.is_signed_integer())


def unsigned_integer() -> Selector:
    return Selector(lambda _, dtype: dtype.is_unsigned_integer())


def numeric() -> Selector:
    return Selector(lambda _, dtype: dtype.is_numeric())


def string() -> Selector:
    return by_dtype(String)


def temporal() -> Selector:
    return Selector(lambda _, dtype: dtype.is_temporal())
