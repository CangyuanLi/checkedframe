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
    """Select all columns.

    Returns
    -------
    Selector

    Examples
    --------
    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            balances = cf.Float64()


        df = pl.DataFrame({
            "customer_id": ["a", "b", None],
            "balances": [1.0, None, 3.0],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 2 error(s)
          customer_id: 1 error(s)
            - `nullable=False` failed for 1 / 3 (33.33%) rows: Must not be null
          balances: 1 error(s)
            - `nullable=False` failed for 1 / 3 (33.33%) rows: Must not be null

    Make all columns nullable

    .. code-block:: python

        @cf.apply_configs(cf.Config(cfs.all(), nullable=True))
        class S(cf.Schema):
            customer_id = cf.String()
            balances = cf.Float64()


        S.validate(df)
    """
    return Selector(lambda _1, _2: True)


def by_name(*names: str | Iterable[str]) -> Selector:
    """Select all columns matching the given names.

    Returns
    -------
    Selector

    Examples
    --------
    Make all float columns and `customer_id` nullable

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs


        @cf.apply_configs(cf.Config(cfs.float() | cfs.by_name("customer_id"), nullable=True))
        class S(cf.Schema):
            customer_id = cf.String()
            checking_balances = cf.Float64()
            savings_balances = cf.Float64()
    """
    all_names = _flatten_str_iterable(names)

    return Selector(lambda actual_name, _: actual_name in all_names)


def matches(pattern: str | re.Pattern[str]) -> Selector:
    '''Select all columns that match the given regex pattern.

    Parameters
    ----------
    pattern : str | re.Pattern[str]
        A Python regex

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all columns ending in "L" followed by a number

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            age = cf.Int64()
            age_L1 = cf.Int64()
            age_L2 = cf.Int64()

            @cf.Check(columns=cfs.matches(r"L\d+$"))
            def check_lags(name: str) -> pl.Expr:
                """Age lags should be less than current age"""
                return pl.col(name).lt(pl.col("age"))


        df = pl.DataFrame({
            "age": [60, 62],
            "age_L1": [100, 61],
            "age_L2": [58, 60],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          age_L1: 1 error(s)
            - check_lags failed for 1 / 2 (50.00%) rows: Age lags should be less than current age
    '''
    return Selector(lambda name, _: re.search(pattern, name) is not None)


def contains(*substrings: str | Iterable[str]) -> Selector:
    '''Select columns whose names contain the given literal substring(s).

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all columns that contain "balances"

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import pandas as pd


        class S(cf.Schema):
            balances_L1 = cf.Float64()
            balances_L2 = cf.Float64()

            @cf.Check(columns=cfs.starts_with("balances"))
            def check_balances_range(s: pd.Series) -> pd.Series:
                """Balances must be in range [0, 1_000_000]"""
                return (s >= 0) & (s <= 1_000_000)


        df = pd.DataFrame(
            {"balances_L1": [-1.0, 500.56, 300.12], "balances_L2": [500.29, 600.99, 700.42]}
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          balances_L1: 1 error(s)
            - check_balances_range failed for 1 / 3 (33.33%) rows: Balances must be in range [0, 1_000_000]
    '''
    all_substrings = _flatten_str_iterable(substrings)

    return Selector(lambda name, _: any(s in name for s in all_substrings))


def starts_with(*prefixes: str | Iterable[str]) -> Selector:
    '''Select columns whose names start with the given literal substring(s).

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all columns that start with "balances"

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import pandas as pd


        class S(cf.Schema):
            balances_L1 = cf.Float64()
            balances_L2 = cf.Float64()

            @cf.Check(columns=cfs.starts_with("balances"))
            def check_balances_range(s: pd.Series) -> pd.Series:
                """Balances must be in range [0, 1_000_000]"""
                return (s >= 0) & (s <= 1_000_000)


        df = pd.DataFrame(
            {"balances_L1": [-1.0, 500.56, 300.12], "balances_L2": [500.29, 600.99, 700.42]}
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          balances_L1: 1 error(s)
            - check_balances_range failed for 1 / 3 (33.33%) rows: Balances must be in range [0, 1_000_000]
    '''
    all_prefixes = tuple(_flatten_str_iterable(prefixes))

    return Selector(lambda name, _: name.startswith(all_prefixes))


def ends_with(*suffixes: str | Iterable[str]) -> Selector:
    '''Select columns whose names end with the given literal substring(s).

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all columns ending in "balances"

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import pandas as pd


        class S(cf.Schema):
            L1_balances = cf.Float64()
            L2_balances = cf.Float64()

            @cf.Check(columns=cfs.starts_with("balances"))
            def check_balances_range(s: pd.Series) -> pd.Series:
                """Balances must be in range [0, 1_000_000]"""
                return (s >= 0) & (s <= 1_000_000)


        df = pd.DataFrame(
            {"L1_balances": [-1.0, 500.56, 300.12], "L2_balances": [500.29, 600.99, 700.42]}
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          L1_balances: 1 error(s)
            - check_balances_range failed for 1 / 3 (33.33%) rows: Balances must be in range [0, 1_000_000]
    '''
    all_suffixes = tuple(_flatten_str_iterable(suffixes))

    return Selector(lambda name, _: name.endswith(all_suffixes))


def by_dtype(*dtypes: TypeOrInstance | Iterable[TypeOrInstance]) -> Selector:
    '''Select columns whose dtypes match the given dtypes.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all List(String) columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            payment_pattern = cf.List(cf.String)
            payment_pattern_enum = cf.List(cf.Int64)

            @cf.Check(columns=cfs.by_dtype(cf.List(cf.String)))
            def check_list_length(name: str) -> pl.Expr:
                """Each payment pattern must have at least one element."""
                return pl.col(name).list.len() > 0


        df = pl.DataFrame({
            "customer_id": ["a", "b", "c"],
            "payment_pattern": [["a"], [], ["c"]],
            "payment_pattern_enum": [[1], [2], [3]],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          payment_pattern: 1 error(s)
            - check_list_length failed for 1 / 3 (33.33%) rows: Each payment pattern must have at least one element.
    '''
    all_dtypes = []
    for d in dtypes:
        if isinstance(d, Iterable):
            for x in d:
                all_dtypes.append(x)
        else:
            all_dtypes.append(d)

    return Selector(lambda _, actual_dtype: actual_dtype in all_dtypes)


def boolean() -> Selector:
    '''Select all boolean columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all boolean columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            is_new_customer = cf.Boolean()
            is_high_balances = cf.Boolean()

            @cf.Check(columns=cfs.boolean())
            def check_cardinality(s: pl.Series) -> bool:
                """Should not be all True/False"""
                return not (s.all() or s.not_().all())


        df = pl.DataFrame(
            {
                "is_new_customer": [False, False, False],
                "is_high_balances": [True, True, False],
            }
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          is_new_customer: 1 error(s)
            - check_cardinality failed for 3 / 3 (100.00%) rows: Should not be all True/False
    '''
    return by_dtype(Boolean)


def categorical() -> Selector:
    '''Select all categorical columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all categorical columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import pandas as pd


        class S(cf.Schema):
            customer_id = cf.String()
            business_type = cf.Categorical()

            @cf.Check(columns=cfs.categorical())
            def check_categories(s: pd.Series) -> bool:
                """Number of categories must be <= 2."""
                return len(s.cat.categories) <= 2


        df = pd.DataFrame(
            {
                "customer_id": ["a23", "c39", "b88"],
                "business_type": ["tech", "finance", "non-profit"],
            }
        ).assign(business_type=lambda df: df["business_type"].astype("category"))

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          business_type: 1 error(s)
            - check_categories failed for 3 / 3 (100.00%) rows: Number of categories must be <= 2.
    '''
    return by_dtype(Categorical)


def date() -> Selector:
    '''Select all date columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all date columns

    .. code-block:: python

        import datetime

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            birth_date = cf.Date()

            @cf.Check(columns=cfs.date())
            def check_date(name: str) -> pl.Expr:
                """Date must be after 1900-01-01"""
                return pl.col(name) > datetime.date(1900, 1, 1)


        df = pl.DataFrame(
            {
                "customer_id": ["a", "b", "c"],
                "birth_date": [
                    datetime.date(1899, 1, 1),
                    datetime.date(2000, 1, 1),
                    datetime.date(2010, 1, 1),
                ],
            }
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          birth_date: 1 error(s)
            - check_date failed for 1 / 3 (33.33%) rows: Date must be after 1900-01-01
    '''
    return by_dtype(Date)


def datetime() -> Selector:
    '''Select all datetime columns.

    Returns
    -------
    Selector

    Examples
    --------
    Datetimes must be after 1900-01-01.

    .. code-block:: python

        import datetime

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            birth_date = cf.Datetime()

            @cf.Check(columns=cfs.datetime())
            def check_date(name: str) -> pl.Expr:
                """Date must be after 1900-01-01"""
                return pl.col(name) > datetime.datetime(1900, 1, 1)


        df = pl.DataFrame(
            {
                "customer_id": ["a", "b", "c"],
                "birth_date": [
                    datetime.datetime(1899, 1, 1),
                    datetime.datetime(2000, 1, 1),
                    datetime.datetime(2010, 1, 1),
                ],
            }
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          birth_date: 1 error(s)
            - check_date failed for 1 / 3 (33.33%) rows: Date must be after 1900-01-01
    '''
    return by_dtype(Datetime)


def decimal() -> Selector:
    '''Select all decimal columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all decimal columns

    .. code-block:: python

        from decimal import Decimal

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            income = cf.Decimal()

            @cf.Check(columns=cfs.decimal())
            def check_positive(name: str) -> pl.Expr:
                """Decimal must be positive"""
                return pl.col(name) > 0


        df = pl.DataFrame({
            "customer_id": ["a", "b", "c"],
            "income": [Decimal("100.0"), Decimal("-50.0"), Decimal("200.0")],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          income: 1 error(s)
            - check_positive failed for 1 / 3 (33.33%) rows: Decimal must be positive
    '''
    return Selector(lambda _, dtype: dtype.is_decimal())


def float() -> Selector:
    """Select all float columns.

    Returns
    -------
    Selector

    Examples
    --------
    Allow NaNs in all float columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs

        @cf.apply_configs(cf.Config(cfs.float(), allow_nan=True))
        class S(cf.Schema):
            customer_id = cf.String()
            balances = cf.Float64()
            income = cf.Float32()
    """
    return Selector(lambda _, dtype: dtype.is_float())


def integer() -> Selector:
    '''Select all integer columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all integer columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            tenure = cf.Int64()
            age = cf.UInt32()

            @cf.Check(columns=cfs.unsigned_integer())
            def check_positive(name: str) -> pl.Expr:
                """Value must be positive"""
                return pl.col(name) > 0


        df = pl.DataFrame({
            "customer_id": ["a", "b", "c"],
            "tenure": [1, 2, 3],
            "age": [0, 2, 3],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          age: 1 error(s)
            - check_positive failed for 1 / 3 (33.33%) rows: Value must be positive
    '''
    return Selector(lambda _, dtype: dtype.is_integer())


def signed_integer() -> Selector:
    '''Select all signed integer columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all signed integer columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            tenure = cf.Int64()
            age = cf.UInt32()

            @cf.Check(columns=cfs.signed_integer())
            def check_positive(name: str) -> pl.Expr:
                """Value must be positive"""
                return pl.col(name) > 0


        df = pl.DataFrame({
            "customer_id": ["a", "b", "c"],
            "tenure": [1, 2, 3],
            "age": [0, 2, 3],
        })

        S.validate(df)
    '''
    return Selector(lambda _, dtype: dtype.is_signed_integer())


def unsigned_integer() -> Selector:
    '''Select all unsigned integer columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all unsigned integer columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            tenure = cf.Int64()
            age = cf.UInt32()

            @cf.Check(columns=cfs.unsigned_integer())
            def check_positive(name: str) -> pl.Expr:
                """Value must be positive"""
                return pl.col(name) > 0


        df = pl.DataFrame({
            "customer_id": ["a", "b", "c"],
            "tenure": [1, 2, 3],
            "age": [0, 2, 3],
        })

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          age: 1 error(s)
            - check_positive failed for 1 / 3 (33.33%) rows: Value must be positive
    '''
    return Selector(lambda _, dtype: dtype.is_unsigned_integer())


def numeric() -> Selector:
    '''Select all numeric columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all numeric columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            customer_id = cf.String()
            tenure = cf.Int64()
            age = cf.Float64()

            @cf.Check(columns=cfs.numeric())
            def check_numeric_max(name: str) -> pl.Expr:
                """Numeric cols must be less than 100"""
                return pl.col(name).lt(100)


        df = pl.DataFrame(
            {
                "customer_id": ["a1", "b1", "c1"],
                "tenure": [10, 5, 40],
                "age": [30.2, 25.6, 150.1],
            }
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          age: 1 error(s)
            - check_numeric_max failed for 1 / 3 (33.33%) rows: Numeric cols must be less than 100
    '''
    return Selector(lambda _, dtype: dtype.is_numeric())


def string() -> Selector:
    '''Select all string columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all string columns

    .. code-block:: python

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl


        class S(cf.Schema):
            marital_status = cf.String()
            employment_status = cf.String()

            @cf.Check(columns=cfs.string())
            def check_cardinality(s: pl.Series) -> bool:
                """String column must have <= 2 unique values"""
                return s.n_unique() <= 2


        df = pl.DataFrame(
            {
                "marital_status": ["married", "unmarried", "unknown"],
                "employment_status": ["employed", "unemployed", "employed"],
            }
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 1 error(s)
          marital_status: 1 error(s)
            - check_cardinality failed for 3 / 3 (100.00%) rows: String column must have <= 2 unique values
    '''
    return by_dtype(String)


def temporal() -> Selector:
    """Select all temporal columns.

    Returns
    -------
    Selector

    Examples
    --------
    Apply a check to all temporal columns

    .. code-block:: python

        import datetime

        import checkedframe as cf
        import checkedframe.selectors as cfs
        import polars as pl

        class S(cf.Schema):
            join_time = cf.Datetime()
            birth_date = cf.Date()

            @cf.Check(columns=cfs.temporal())
            def check_no_invalid_dates(name: str) -> pl.Expr:
                return pl.col(name).dt.date() >= pl.date(1900, 1, 1)

        df = pl.DataFrame(
            {
                "join_time": [
                    datetime.datetime(1899, 1, 1, 1, 1),
                    datetime.datetime(2010, 2, 20, 6, 3, 9),
                ],
                "birth_date": [datetime.date(1899, 1, 1), datetime.date(2000, 5, 19)],
            },
        )

        S.validate(df)

    Output:

    .. code-block:: text

        SchemaError: Found 2 error(s)
          join_time: 1 error(s)
          - check_no_invalid_dates failed for 1 / 2 (50.00%) rows:
          birth_date: 1 error(s)
          - check_no_invalid_dates failed for 1 / 2 (50.00%) rows:
    """
    return Selector(lambda _, dtype: dtype.is_temporal())
