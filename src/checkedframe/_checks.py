from __future__ import annotations

import functools
import inspect
from collections.abc import Collection, Iterable, Sequence
from typing import Any, Callable, Literal, Optional, get_args, get_type_hints

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt
from narwhals.stable.v1.dependencies import (
    get_cudf,
    get_modin,
    get_pandas,
    get_polars,
    get_pyarrow,
)

from .selectors import Selector

INF = float("inf")
NEG_INF = float("-inf")


def _is_polars_series(ser: Any) -> bool:
    return (pl := get_polars()) is not None and issubclass(ser, pl.Series)


def _is_polars_expr(expr: Any) -> bool:
    return (pl := get_polars()) is not None and issubclass(expr, pl.Expr)


def _is_polars_dataframe(df: Any) -> bool:
    return (pl := get_polars()) is not None and issubclass(df, pl.DataFrame)


def _is_pandas_series(ser: Any) -> bool:
    return (pd := get_pandas()) is not None and issubclass(ser, pd.Series)


def _is_pandas_dataframe(df: Any) -> bool:
    return (pd := get_pandas()) is not None and issubclass(df, pd.DataFrame)


def _is_modin_dataframe(df: Any) -> bool:
    return (mpd := get_modin()) is not None and issubclass(df, mpd.DataFrame)


def _is_modin_series(ser: Any) -> bool:
    return (mpd := get_modin()) is not None and issubclass(ser, mpd.Series)


def _is_cudf_dataframe(df: Any) -> bool:
    return (cudf := get_cudf()) is not None and issubclass(df, cudf.DataFrame)


def _is_cudf_series(ser: Any) -> bool:
    return (cudf := get_cudf()) is not None and issubclass(ser, cudf.Series)


def _is_pyarrow_expr(expr: Any) -> bool:
    return (pa := get_pyarrow()) is not None and issubclass(expr, pa.compute.Expression)


def _is_pyarrow_chunked_array(ser: Any) -> bool:
    return (pa := get_pyarrow()) is not None and issubclass(ser, pa.ChunkedArray)


def _is_pyarrow_table(df: Any) -> bool:
    return (pa := get_pyarrow()) is not None and issubclass(df, pa.Table)


def _is_series(x: Any) -> bool:
    return (
        issubclass(x, nw.Series)
        or _is_pandas_series(x)
        or _is_modin_series(x)
        or _is_cudf_series(x)
        or _is_polars_series(x)
        or _is_pyarrow_chunked_array(x)
    )


def _is_expr(x: Any) -> bool:
    return issubclass(x, nw.Expr) or _is_polars_expr(x)


def _is_dataframe(x: Any) -> bool:
    return (
        isinstance(x, nw.DataFrame)
        or _is_polars_dataframe(x)
        or _is_pandas_dataframe(x)
        or _is_modin_dataframe(x)
        or _is_cudf_dataframe(x)
        or _is_pyarrow_table(x)
    )


def _infer_input_type(
    type_hints: dict[str, Any], signature: inspect.Signature
) -> CheckInputType:
    params = signature.parameters
    if len(params) == 0:
        return None

    first_param_name = list(params.keys())[0]
    try:
        type_hint = type_hints[first_param_name]
    except KeyError:
        return "auto"

    if issubclass(type_hint, str):
        return "str"
    elif _is_dataframe(type_hint):
        return "Frame"
    elif _is_series(type_hint):
        return "Series"

    return "auto"


def _infer_return_type(
    type_hints: dict[str, Any], input_type: CheckInputType
) -> CheckReturnType:
    try:
        # Try to get it from the type hints first
        type_hint = type_hints["return"]

        if issubclass(type_hint, bool):
            return "bool"
        elif _is_expr(type_hint):
            return "Expr"
        elif _is_series(type_hint):
            return "Series"
    except KeyError:
        # If type hints don't exist, we try to infer from the input_type
        pass

    if input_type == "str" or input_type is None:
        return "Expr"

    return "auto"


def _infer_narwhals(type_hints: dict[str, Any]) -> bool | Literal["auto"]:
    if len(type_hints) == 0:
        return "auto"

    return any(
        issubclass(v, nw.Expr)
        or issubclass(v, nw.Series)
        or issubclass(v, nw.DataFrame)
        for v in type_hints.values()
    )


def _numeric_to_expr(expr: str | float | nw.Expr) -> nw.Expr:
    if isinstance(expr, str):
        return nw.col(expr)
    elif isinstance(expr, nw.Expr):
        return expr
    else:
        return nw.lit(expr)


def _get_repr(x: Any) -> str:
    # Unfortunately the representation of an expression is not very good
    if isinstance(x, nw.Expr):
        return "Expr"

    return f"{x}"


ClosedInterval = Literal["left", "right", "none", "both"]


def _is_not_null(name: str) -> nw.Expr:
    return nw.col(name).is_null().__invert__()


def _is_not_nan(name: str) -> nw.Expr:
    return nw.col(name).is_nan().__invert__()


def _is_not_inf(name: str) -> nw.Expr:
    return nw.col(name).is_in((INF, NEG_INF)).__invert__()


def _is_between(
    name: str,
    lower_bound,
    upper_bound,
    closed: ClosedInterval,
) -> nw.Expr:
    return nw.col(name).is_between(lower_bound, upper_bound, closed=closed)


def _lt(name: str, other) -> nw.Expr:
    return nw.col(name) < other


def _le(name: str, other) -> nw.Expr:
    return nw.col(name) <= other


def _gt(name: str, other) -> nw.Expr:
    return nw.col(name) > other


def _ge(name: str, other) -> nw.Expr:
    return nw.col(name) >= other


def _eq(name: str, other) -> nw.Expr:
    return nw.col(name) == other


def _private_approx_eq(
    left: nw.Expr,
    right: nw.Expr,
    rtol: float,
    atol: float,
    nan_equal: bool,
) -> nw.Expr:
    res = (
        left.__sub__(right)
        .abs()
        .__le__(nw.lit(atol).__add__(rtol).__mul__(right.abs()))
    )

    if nan_equal:
        res = res.__or__(left.is_nan().__and__(right.is_nan()))

    return res


def _approx_eq(name: str, other, rtol, atol, nan_equal) -> nw.Expr:
    return _private_approx_eq(nw.col(name), other, rtol, atol, nan_equal)


def _series_lit_approx_eq(
    left: nw.Series, right: float, rtol: float, atol: float, nan_equal: bool
) -> nw.Series:
    name = "__checkedframe_approx_eq__"
    return left.to_frame().select(
        _private_approx_eq(
            nw.col(left.name), nw.lit(right), rtol=rtol, atol=atol, nan_equal=nan_equal
        ).alias(name)
    )[name]


def _is_in(name: str, other: Collection) -> nw.Expr:
    return nw.col(name).is_in(other)


def _is_finite(name: str) -> nw.Expr:
    return nw.col(name).is_finite()


def _is_sorted(s: nw.Series, descending: bool) -> bool:
    return s.is_sorted(descending=descending)


def _is_id(df: nw.DataFrame, subset: str | list[str]) -> bool:
    n_rows = df.shape[0]

    # n_unique on dataframes is not available on narhwals, so if we have only one
    # column specified as the subset, take a potential fast path, otherwise fallback to
    # a generic version
    if isinstance(subset, str):
        n_unique_rows = df[subset].n_unique()
    else:
        n_unique_rows = df.select(subset).unique().shape[0]

    return n_rows == n_unique_rows


def _series_equals(
    left: nw.Series,
    right: nw.Series,
    check_dtypes: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    if check_dtypes:
        if left.dtype != right.dtype:
            return False

    if left.dtype.is_float() and not check_exact:
        return (
            left.to_frame()
            .with_columns(right)
            .select(
                _private_approx_eq(
                    nw.col(left.name),
                    nw.col(right.name),
                    rtol=rtol,
                    atol=atol,
                    nan_equal=True,
                ).all()
            )
            .item()
        )
    else:
        return (left == right).all()


def _frame_equals(
    left: nw.DataFrame,
    right: nw.DataFrame,
    check_column_order: bool = True,
    check_dtypes: bool = True,
    check_exact: bool = False,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> bool:
    l_cols = left.columns
    r_cols = right.columns
    if check_column_order:
        if l_cols != r_cols:
            return False
    else:
        if set(l_cols) != set(r_cols):
            return False

    results = []
    for c in l_cols:
        results.append(
            _series_equals(
                left[c],
                right[c],
                check_dtypes=check_dtypes,
                check_exact=check_exact,
                rtol=rtol,
                atol=atol,
            )
        )

    return all(results)


def _frame_is_sorted(
    df: nw.DataFrame,
    by: str | Sequence[str],
    descending: bool | Sequence[bool],
    compare_all: bool,
) -> bool:
    if compare_all:
        df_sorted = df.sort(by=by, descending=descending)

        return _frame_equals(df, df_sorted, check_exact=True)
    else:
        if isinstance(by, str):
            assert isinstance(descending, bool)

            return df[by].is_sorted(descending=descending)
        else:
            return _frame_equals(
                df.select(by), df.select(by).sort(by=by, descending=descending)
            )


def _str_ends_with(name: str, suffix: str) -> nw.Expr:
    return nw.col(name).str.ends_with(suffix)


def _str_starts_with(name: str, prefix: str) -> nw.Expr:
    return nw.col(name).str.starts_with(prefix)


def _str_contains(name: str, pattern: str, literal: bool = False) -> nw.Expr:
    return nw.col(name).str.contains(pattern, literal=literal)


CardinalityRatio = Literal["1:1", "1:m", "m:1"]


def _cardinality_ratio(
    df: nw.DataFrame,
    left: str,
    right: str,
    cardinality: CardinalityRatio,
    by: str | list[str] | None = None,
    allow_duplicates: bool = False,
):
    index_col = "__checkedframe_temp_cardinality_ratio_private_index__"
    result_col = left

    original_lf = df.with_row_index(index_col).lazy()

    if by is None:
        by = "__checkedframe_temp_cardinality_ratio_private_by__"
        original_lf = original_lf.with_columns(nw.lit(1).alias(by))

    if isinstance(by, str):
        by = [by]

    lf = original_lf.select(left, right, *by)

    if allow_duplicates:
        lf = lf.unique()

    if cardinality == "1:1":
        result_lf = (
            lf.group_by(by)
            .agg(
                nw.col(left).n_unique().__eq__(nw.len()),
                nw.col(right).n_unique().__eq__(nw.len()),
            )
            .select(*by, nw.col(left).__and__(nw.col(right)).alias(result_col))
        )
    elif cardinality == "1:m":
        result_lf = (
            lf.group_by(by)
            .agg(nw.col(left).n_unique().__eq__(nw.len()).alias(result_col))
            .select(*by, result_col)
        )
    elif cardinality == "m:1":
        result_lf = (
            lf.group_by(by)
            .agg(nw.col(right).n_unique().__eq__(nw.len()).alias(result_col))
            .select(*by, result_col)
        )
    else:
        raise ValueError(
            f"Invalid cardinality `{cardinality}`, must be one of `{get_args(CardinalityRatio)}`"
        )

    return (
        original_lf.select(index_col, *by)
        .join(result_lf, on=by, how="left")
        .sort(index_col)  # joins are not guaranteed to preserve order
        .select(result_col)
        .collect()[result_col]
    )


CheckInputType = Optional[Literal["auto", "Frame", "str", "Series"]]
CheckReturnType = Literal["auto", "bool", "Expr", "Series"]


class Check:
    """Represents a check to run.

    Parameters
    ----------
    func : Optional[Callable], optional
        The check to run, by default None
    columns : Optional[str | list[str] | Selector], optional
        The columns associated with the check, by default None
    input_type : Optional[Literal["auto", "Frame", "str", "Series"]], optional
        The input to the check function. If "auto", attempts to determine via the
        context, by default "auto"
    return_type : Literal["auto", "bool", "Expr", "Series"], optional
        The return type of the check function. If "auto", attempts to determine via the
        context, by default "auto"
    native : bool | Literal["auto"], optional
        Whether to run the check on the native DataFrame or the Narwhals DataFrame. If
        "auto", attempts to determine via the context, by default "auto"
    name : Optional[str], optional
        The name of the check, by default None
    description : Optional[str], optional
        The description of the check. If None, attempts to read from the __doc__
        attribute, by default None
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        columns: Optional[str | list[str] | Selector] = None,
        input_type: CheckInputType = "auto",
        return_type: CheckReturnType = "auto",
        native: bool | Literal["auto"] = "auto",
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        self.func = func
        self.input_type = input_type
        self.return_type = return_type
        self.native = native
        self.name = name
        self.description = description
        self.columns = [columns] if isinstance(columns, str) else columns

        if self.func is not None:
            self._set_params()

    def _set_params(self) -> None:
        assert self.func is not None
        auto_input_type = self.input_type == "auto"
        auto_return_type = self.return_type == "auto"
        auto_native = self.native == "auto"

        if auto_input_type or auto_return_type or auto_native:
            signature = inspect.signature(self.func)
            type_hints = get_type_hints(self.func)

        if auto_native:
            self.native = not _infer_narwhals(type_hints)

        if auto_input_type:
            self.input_type = _infer_input_type(type_hints, signature)

        if auto_return_type:
            self.return_type = _infer_return_type(
                type_hints,
                self.input_type,
            )

        if self.native == "auto":
            raise ValueError(
                f"Whether `{self.name}` expects to be run natively or via narwhals could not be automatically determined from context"
            )

        if self.input_type == "auto":
            raise ValueError(
                f"Input type of `{self.name}` could not be automatically determined from context"
            )

        if self.return_type == "auto":
            raise ValueError(
                f"Return type of `{self.name}` could not be automatically determined from context"
            )

        if self.name is None:
            self.name = None if self.func.__name__ == "<lambda>" else self.func.__name__

        if self.description is None:
            self.description = "" if self.func.__doc__ is None else self.func.__doc__

    def __call__(self, func: Callable):
        return Check(
            func=func,
            columns=self.columns,
            input_type=self.input_type,  # type: ignore
            return_type=self.return_type,  # type: ignore
            native=self.native,
            name=self.name,
            description=self.description,
        )

    @staticmethod
    def is_not_null() -> Check:
        """Tests whether values are not null.

        .. note::
            This method is mainly here for completeness. Columns are by default not
            nullable.

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                customer_id = cf.String(checks=[cf.Check.is_not_null()])


            df = pl.DataFrame({"customer_id": ["a23", None]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 2 error(s)
              customer_id: 2 error(s)
                - `nullable=False` failed for 1 / 2 (50.00%) rows: Must not be null
                - is_not_null failed for 1 / 2 (50.00%) rows: Must not be null

        """
        return Check(
            func=_is_not_null,
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_not_null",
            description="Must not be null",
        )

    @staticmethod
    def is_not_nan() -> Check:
        """Tests whether values are not NaN.

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                balances = cf.Float64(checks=[cf.Check.is_not_nan()])


            df = pl.DataFrame({"balances": [1, 2, float("nan")]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              balances: 1 error(s)
                - is_not_nan failed for 1 / 3 (33.33%) rows: Must not be NaN

        """
        return Check(
            func=_is_not_nan,
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_not_nan",
            description="Must not be NaN",
        )

    @staticmethod
    def is_not_inf() -> Check:
        """Tests whether values are not infinite.

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                balances = cf.Float64(checks=[cf.Check.is_not_inf()])


            df = pl.DataFrame({"balances": [1, 2, float("inf")]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              balances: 1 error(s)
                - is_not_inf failed for 1 / 3 (33.33%) rows: Must not be inf/-inf

        """
        return Check(
            func=_is_not_inf,
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_not_inf",
            description="Must not be inf/-inf",
        )

    @staticmethod
    def is_between(
        lower_bound: Any, upper_bound: Any, closed: ClosedInterval = "both"
    ) -> Check:
        """Tests whether values are between `lower_bound` and `upper_bound`. Strings are
        interpreted as column names.

        Parameters
        ----------
        lower_bound : Any
            The lower bound
        upper_bound : Any
            The upper bound
        closed : ClosedInterval, optional
            Defines which sides of the interval are closed, by default "both"

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                age = cf.Int64(checks=[cf.Check.is_between(0, 128)])
                min_balance = cf.Int64()
                med_balance = cf.Int64(checks=[cf.Check.is_between("min_balance", "max_balance")])
                max_balance = cf.Int64()


            df = pl.DataFrame(
                {
                    "age": [5, 10, 150],
                    "min_balance": [1, 100, 500],
                    "med_balance": [0, 83, 525],
                    "max_balance": [788, 82, 550],
                }
            )
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 2 error(s)
              age: 1 error(s)
                - is_between failed for 1 / 3 (33.33%) rows: Must be in range [0, 128]
              med_balance: 1 error(s)
                - is_between failed for 2 / 3 (66.67%) rows: Must be in range [min_balance, max_balance]

        """
        if closed == "both":
            l_paren, r_paren = ("[", "]")
        elif closed == "left":
            l_paren, r_paren = ("[", ")")
        elif closed == "right":
            l_paren, r_paren = ("(", "]")
        elif closed == "none":
            l_paren, r_paren = ("(", ")")

        return Check(
            func=functools.partial(
                _is_between,
                lower_bound=lower_bound,
                upper_bound=upper_bound,
                closed=closed,
            ),
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_between",
            description=f"Must be in range {l_paren}{lower_bound}, {upper_bound}{r_paren}",
        )

    @staticmethod
    def lt(other: Any) -> Check:
        """Tests whether values are less than `other`. Strings are interpreted as
        column names.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                age = cf.Int64(
                    checks=[
                        cf.Check.lt(10),
                        cf.Check.lt("max_age"),
                        cf.Check.lt(cf.col("max_age") - 10),
                    ]
                )


            df = pl.DataFrame(
                {
                    "age": [5, 10, 11],
                    "max_age": [10, 5, 8],
                }
            )
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 2 error(s)
              age: 2 error(s)
                - less_than failed for 2 / 3 (66.67%) rows: Must be < 10
                - less_than failed for 1 / 3 (33.33%) rows: Must be < max_age

        """
        return Check(
            func=functools.partial(_lt, other=_numeric_to_expr(other)),
            input_type="str",
            return_type="Expr",
            native=False,
            name="less_than",
            description=f"Must be < {_get_repr(other)}",
        )

    @staticmethod
    def le(other: Any) -> Check:
        """Tests whether values are less than or equal to `other`. Strings are
        interpreted as column names.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                age = cf.Int64(
                    checks=[
                        cf.Check.le(10),
                        cf.Check.le("max_age"),
                        cf.Check.le(cf.col("max_age") - 10),
                    ]
                )


            df = pl.DataFrame(
                {
                    "age": [5, 10, 11],
                    "max_age": [10, 5, 8],
                }
            )
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              age: 1 error(s)
                - less_than_or_equal_to failed for 1 / 3 (33.33%) rows: Must be <= 10

        """
        return Check(
            func=functools.partial(_le, other=_numeric_to_expr(other)),
            input_type="str",
            return_type="Expr",
            native=False,
            name="less_than_or_equal_to",
            description=f"Must be <= {_get_repr(other)}",
        )

    @staticmethod
    def gt(other: Any) -> Check:
        """Tests whether values are greater than `other`. Strings are interpreted as
        column names.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                age = cf.Int64(
                    checks=[
                        cf.Check.gt(10),
                        cf.Check.gt("min_age"),
                        cf.Check.gt(cf.col("min_age") - 100),
                    ]
                )


            df = pl.DataFrame(
                {
                    "age": [5, 10, 11],
                    "min_age": [10, 5, 8],
                }
            )
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 2 error(s)
              age: 2 error(s)
                - greater_than failed for 2 / 3 (66.67%) rows: Must be > 10
                - greater_than failed for 1 / 3 (33.33%) rows: Must be > min_age

        """
        return Check(
            func=functools.partial(_gt, other=_numeric_to_expr(other)),
            input_type="str",
            return_type="Expr",
            native=False,
            name="greater_than",
            description=f"Must be > {_get_repr(other)}",
        )

    @staticmethod
    def ge(other: Any) -> Check:
        """Tests whether values are greater than or equal to `other`. Strings are
        interpreted as column names.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                age = cf.Int64(
                    checks=[
                        cf.Check.ge(10),
                        cf.Check.ge("min_age"),
                        cf.Check.ge(cf.col("min_age") - 10),
                    ]
                )


            df = pl.DataFrame(
                {
                    "age": [5, 10, 11],
                    "min_age": [10, 5, 8],
                }
            )
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 2 error(s)
              age: 2 error(s)
                - greater_than_or_equal_to failed for 1 / 3 (33.33%) rows: Must be >= 10
                - greater_than_or_equal_to failed for 1 / 3 (33.33%) rows: Must be >= min_age

        """
        return Check(
            func=functools.partial(_ge, other=_numeric_to_expr(other)),
            input_type="str",
            return_type="Expr",
            native=False,
            name="greater_than_or_equal_to",
            description=f"Must be >= {_get_repr(other)}",
        )

    @staticmethod
    def eq(other: Any) -> Check:
        """Tests whether values are equal to `other`. Strings are interpreted as column
        names.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                group = cf.String(checks=[cf.Check.eq("A")])


            df = pl.DataFrame({"group": ["A", "B", "A"]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              group: 1 error(s)
                - equal_to failed for 1 / 3 (33.33%) rows: Must be = A

        """
        return Check(
            func=functools.partial(_eq, other=_numeric_to_expr(other)),
            input_type="str",
            return_type="Expr",
            native=False,
            name="equal_to",
            description=f"Must be = {_get_repr(other)}",
        )

    @staticmethod
    def approx_eq(
        other: Any,
        rtol: float = 1e-5,
        atol: float = 1e-8,
        nan_equal: bool = False,
    ) -> Check:
        """Tests whether values are approximately equal to `other`. Strings are
        interpreted as column names.

        Parameters
        ----------
        other : Any
        rtol : float, optional
            Relative tolerance, by default 1e-5
        atol : float, optional
            Absolute tolerance, by default 1e-8
        nan_equal : bool, optional
            Whether to consider NaN values equal, by default False

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                prob = cf.Float64(checks=[cf.Check.approx_eq(0.5)])


            df = pl.DataFrame({"prob": [0.5, 0.50000001, 0.6]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              prob: 1 error(s)
                - approximately_equal_to failed for 1 / 3 (33.33%) rows: Must be approximately equal to 0.5 (rtol=1e-05, atol=1e-08, nan_equal=False)

        """
        return Check(
            func=functools.partial(
                _approx_eq,
                other=_numeric_to_expr(other),
                rtol=rtol,
                atol=atol,
                nan_equal=nan_equal,
            ),
            input_type="str",
            return_type="Expr",
            native=False,
            name="approximately_equal_to",
            description=f"Must be approximately equal to {_get_repr(other)} ({rtol=}, {atol=}, {nan_equal=})",
        )

    @staticmethod
    def is_in(other: Collection) -> Check:
        """Tests whether all values of the Series are in the given collection.

        Parameters
        ----------
        other : Collection
            The collection

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                business_type = cf.String(checks=[cf.Check.is_in(["tech", "finance"])])


            df = pl.DataFrame({"business_type": ["x", "tech", "finance"]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              business_type: 1 error(s)
                - is_in failed for 1 / 3 (33.33%) rows: Must be in allowed values ['tech', 'finance']

        """
        return Check(
            func=functools.partial(_is_in, other=other),
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_in",
            description=f"Must be in allowed values {other}",
        )

    @staticmethod
    def is_finite() -> Check:
        """Tests whether values are finite.

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                balances = cf.Float64(checks=[cf.Check.is_finite()])


            df = pl.DataFrame({"balances": [1, 2, float("inf")]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              balances: 1 error(s)
                - is_finite failed for 1 / 3 (33.33%) rows: All values must be finite

        """
        return Check(
            func=_is_finite,
            input_type="str",
            return_type="Expr",
            native=False,
            name="is_finite",
            description="All values must be finite",
        )

    @staticmethod
    def str_ends_with(suffix: str) -> Check:
        """Tests whether string values end with the given suffix.

        Parameters
        ----------
        suffix : str
            The suffix to check for

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                emails = cf.String(checks=[cf.Check.str_ends_with("@gmail.com")])


            df = pl.DataFrame({"emails": ["a@gmail.com", "b@yahoo.com"]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              emails: 1 error(s)
                - ends_with failed for 1 / 2 (50.00%) rows: Must end with @gmail.com

        """
        return Check(
            func=functools.partial(_str_ends_with, suffix=suffix),
            input_type="str",
            return_type="Expr",
            native=False,
            name="ends_with",
            description=f"Must end with {suffix}",
        )

    @staticmethod
    def str_starts_with(prefix: str) -> Check:
        """Tests whether string values start with the given prefix.

        Parameters
        ----------
        prefix : str
            The prefix to check for

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                ids = cf.String(checks=[cf.Check.str_starts_with("user_")])


            df = pl.DataFrame({"ids": ["user_a", "admin_b"]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              ids: 1 error(s)
                - starts_with failed for 1 / 2 (50.00%) rows: Must start with user_

        """
        return Check(
            func=functools.partial(_str_starts_with, prefix=prefix),
            input_type="str",
            return_type="Expr",
            native=False,
            name="starts_with",
            description=f"Must start with {prefix}",
        )

    @staticmethod
    def str_contains(pattern: str, literal: bool = False) -> Check:
        """Tests whether string values contain the given pattern.

        Parameters
        ----------
        pattern : str
            The pattern to check for
        literal : bool, optional
            Whether to interpret the pattern as a literal string or a regex, by default False

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                domains = cf.String(checks=[cf.Check.str_contains(r"\.com$", literal=False)])


            df = pl.DataFrame({"domains": ["a.com", "b.org"]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              domains: 1 error(s)
                - contains failed for 1 / 2 (50.00%) rows: Must contain \.com$

        """
        return Check(
            func=functools.partial(_str_contains, pattern=pattern, literal=literal),
            input_type="str",
            return_type="Expr",
            native=False,
            name="contains",
            description=f"Must contain {pattern}",
        )

    @staticmethod
    def is_sorted(descending: bool = False) -> Check:
        """Tests whether a Series is sorted.

        Parameters
        ----------
        descending : bool, optional
            Whether to check for descending order, by default False

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class S(cf.Schema):
                timestamps = cf.Int64(checks=[cf.Check.is_sorted()])


            df = pl.DataFrame({"timestamps": [1, 2, 1]})
            S.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              timestamps: 1 error(s)
                - is_sorted failed: Must be sorted in ascending order

        """
        order = "descending" if descending else "ascending"

        return Check(
            func=functools.partial(_is_sorted, descending=descending),
            input_type="Series",
            return_type="bool",
            native=False,
            name="is_sorted",
            description=f"Must be sorted in {order} order",
        )

    @staticmethod
    def is_sorted_by(
        by: str | Sequence[str],
        descending: bool | Sequence[bool] = False,
        compare_all: bool = True,
    ) -> Check:
        """Tests whether a DataFrame is sorted by the given columns.

        Parameters
        ----------
        by : str | Sequence[str]
            The column(s) to sort by
        descending : bool | Sequence[bool], optional
            Whether to sort in descending order, by default False
        compare_all : bool, optional
            Whether to compare all columns or just the sorting columns, by default True

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class MySchema(cf.Schema):
                timestamps = cf.Int64()
                values = cf.Int64()

                _sorted_check = cf.Check.is_sorted_by("timestamps")


            df = pl.DataFrame({"timestamps": [1, 2, 1], "values": [1, 2, 3]})
            MySchema.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              * is_sorted_by failed for 3 / 3 (100.00%) rows: Must be sorted by timestamps, where descending is False

        """
        return Check(
            func=functools.partial(
                _frame_is_sorted, by=by, descending=descending, compare_all=compare_all
            ),
            input_type="Frame",
            return_type="bool",
            native=False,
            name="is_sorted_by",
            description=f"Must be sorted by {by}, where descending is {descending}",
        )

    @staticmethod
    def is_id(subset: str | list[str]) -> Check:
        """Tests whether the given column(s) identify the DataFrame.

        Parameters
        ----------
        subset : str | list[str]
            The columns that identify the DataFrame

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class MySchema(cf.Schema):
                group = cf.String()

                _id_check = cf.Check.is_id("group")


            df = pl.DataFrame({"group": ["A", "B", "A"]})
            MySchema.validate(df)

        Output:

        .. code-block:: text

            SchemaError: Found 1 error(s)
              * is_id failed for 3 / 3 (100.00%) rows: group must uniquely identify the DataFrame

        """
        return Check(
            func=functools.partial(_is_id, subset=subset),
            input_type="Frame",
            return_type="bool",
            native=False,
            name="is_id",
            description=f"{subset} must uniquely identify the DataFrame",
        )

    @staticmethod
    def cardinality_ratio(
        left: str,
        right: str,
        cardinality: CardinalityRatio,
        by: str | list[str] | None = None,
        allow_duplicates: bool = False,
    ) -> Check:
        """Tests whether the `left` and `right` columns have the specified cardinality
        ratio. The three possible cardinality ratios are '1:1' (each entity in `left` is
        related to exactly one entity in `right`), '1:m' (each entity in `left` can be
        related to many entities in `right`, but each entity in `right` can only be
        related to one entity in `left`), and 'm:1', which is the same '1:m', except
        with `left` and `right` swapped.

        Parameters
        ----------
        left : str
            The left column
        right : str
            The right column
        cardinality : CardinalityRatio
            The cardinality
        by : str | list[str] | None, optional
            Variables to group by. If specified, the cardinality ratio is checked within
            each group, by default None
        allow_duplicates : bool, optional
            Whether to allow duplicates by `left` and `right`. For example, if
            duplicates are allowed, [1, 1, 1], [2, 2, 2] is considered a '1:1'
            relationship, by default False

        Returns
        -------
        Check

        Examples
        --------
        .. code-block:: python

            import checkedframe as cf
            import polars as pl


            class MySchema(cf.Schema):
                feature = cf.String()
                special_value = cf.Int64()
                imputed = cf.String(nullable=True)
                reason = cf.String()

                _cardinality_check = cf.Check.cardinality_ratio(
                    "imputed",
                    "reason",
                    cardinality="m:1",
                    by="feature",
                    allow_duplicates=True,
                )


            df = pl.DataFrame(
                {
                    "feature": ["f1", "f1", "f1", "f2"],
                    "special_value": [-1, -6, -4, -7],
                    "imputed": [None, None, "MAX_WIN_P1", None],
                    "reason": ["o1", "o1", "o2", "o3"],
                }
            )

            MySchema.validate(df)

        Output::

        ..code-block:: text

            SchemaError: Found 1 error(s)
              * cardinality_ratio failed for 3 / 4 (75.00%) rows: The relationship between imputed and reason must be m:1 (by=feature, allow_duplicates=True)
        """
        return Check(
            func=functools.partial(
                _cardinality_ratio,
                left=left,
                right=right,
                cardinality=cardinality,
                by=by,
                allow_duplicates=allow_duplicates,
            ),
            input_type="Frame",
            return_type="Series",
            native=False,
            name="cardinality_ratio",
            description=f"The relationship between {left} and {right} must be {cardinality} (by={by}, allow_duplicates={allow_duplicates})",
        )
