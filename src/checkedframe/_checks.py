from __future__ import annotations

import functools
import inspect
from collections.abc import Collection, Sequence
from typing import Any, Callable, Literal, Optional

import narwhals.stable.v1 as nw

col = nw.col
lit = nw.lit


class staticproperty:
    """
    A decorator that allows defining a read-only, class-level attribute
    that is computed by a function which takes no arguments (like a static method).
    """

    def __init__(self, func):
        if not callable(func):
            raise TypeError("staticproperty can only decorate callables")
        self.func = func
        self.__doc__ = getattr(func, "__doc__")
        self.__name__ = getattr(func, "__name__")

    def __get__(self, obj, objtype=None):
        # The decorated function (self.func) is called directly with NO arguments.
        # It MUST be defined to accept zero arguments.
        return self.func()

    def __set__(self, obj, value):
        raise AttributeError(f"can't set attribute '{self.__name__}'")

    def __delete__(self, obj):
        raise AttributeError(f"can't delete attribute '{self.__name__}'")


def _resolve_return_type_from_annotation(func: Callable):
    try:
        dtype = str(func.__annotations__["return"])
    except KeyError:
        return "auto"

    if dtype == "bool":
        return "bool"

    if len(inspect.signature(func).parameters) == 0:
        return "Expr"

    if "Series" in dtype:
        return "Series"
    elif "Expr" in dtype:
        return "Expr"

    return "auto"


ClosedInterval = Literal["left", "right", "none", "both"]


def _is_between(
    s: nw.Series,
    lower_bound,
    upper_bound,
    closed: ClosedInterval,
) -> nw.Series:
    return s.is_between(lower_bound, upper_bound, closed=closed)


def _lt(s: nw.Series, other) -> nw.Series:
    return s < other


def _le(s: nw.Series, other) -> nw.Series:
    return s <= other


def _gt(s: nw.Series, other) -> nw.Series:
    return s > other


def _ge(s: nw.Series, other) -> nw.Series:
    return s >= other


def _eq(s: nw.Series, other) -> nw.Series:
    return s == other


def _approx_eq(
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


def _series_lit_approx_eq(
    left: nw.Series, right: float, rtol: float, atol: float, nan_equal: bool
) -> nw.Series:
    name = "__checkedframe_approx_eq__"
    return left.to_frame().select(
        _approx_eq(
            nw.col(left.name), nw.lit(right), rtol=rtol, atol=atol, nan_equal=nan_equal
        ).alias(name)
    )[name]


def _is_in(s: nw.Series, other: Collection) -> nw.Series:
    return s.is_in(other)


def _is_finite(s: nw.Series) -> nw.Series:
    return s.is_finite()


def _is_sorted(s: nw.Series, descending: bool) -> bool:
    return s.is_sorted(descending=descending)


def _is_id(df: nw.DataFrame, subset: str | list[str]) -> bool:
    n_rows = df.shape[0]
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
                _approx_eq(
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


def _str_ends_with(s: nw.Series, suffix: str) -> nw.Series:
    return s.str.ends_with(suffix)


def _str_starts_with(s: nw.Series, prefix: str) -> nw.Series:
    return s.str.starts_with(prefix)


def _str_contains(s: nw.Series, pattern: str, literal: bool = False) -> nw.Series:
    return s.str.contains(pattern, literal=literal)


class _BuiltinStringMethods:
    @staticmethod
    def ends_with(suffix: str) -> Check:
        return Check(
            func=functools.partial(_str_ends_with, suffix=suffix),
            input_type="Series",
            return_type="Series",
            native=False,
            name="ends_with",
            description=f"Must end with {suffix}",
        )

    @staticmethod
    def starts_with(prefix: str) -> Check:
        return Check(
            func=functools.partial(_str_starts_with, prefix=prefix),
            input_type="Series",
            return_type="Series",
            native=False,
            name="starts_with",
            description=f"Must start with {prefix}",
        )

    @staticmethod
    def contains(pattern: str, literal: bool = False) -> Check:
        return Check(
            func=functools.partial(_str_contains, pattern=pattern, literal=literal),
            input_type="Series",
            return_type="Series",
            native=False,
            name="contains",
            description=f"Must contain {pattern}",
        )


CheckInputType = Optional[Literal["auto", "Frame", "Expr", "Series"]]
CheckReturnType = Literal["auto", "bool", "Expr", "Series"]


class Check:
    """Represents a check to run.

    Parameters
    ----------
    func : Optional[Callable], optional
        The check to run, by default None
    columns : Optional[str | list[str]], optional
        The columns associated with the check, by default None
    input_type : Optional[Literal["auto", "Frame", "Series"]], optional
        The input to the check function. If "auto", attempts to determine via the
        context, by default "auto"
    return_type : Literal["auto", "bool", "Expr", "Series"], optional
        The return type of the check function. If "auto", attempts to determine via the
        return type annotation and number of arguments, by default "auto"
    native : bool, optional
        Whether to run the check on the native DataFrame or the Narwhals DataFrame, by
        default True
    name : Optional[str], optional
        The name of the check, by default None
    description : Optional[str], optional
        The description of the check. If None, attempts to read from the __doc__
        attribute, by default None
    """

    def __init__(
        self,
        func: Optional[Callable] = None,
        columns: Optional[str | list[str]] = None,
        input_type: CheckInputType = "auto",
        return_type: CheckReturnType = "auto",
        native: bool = True,
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
        self._func_n_params = len(inspect.signature(self.func).parameters)

        if self.input_type == "auto":
            if self._func_n_params == 0:
                self.input_type = None

        if self.return_type == "auto" and self.func is not None:
            if self.input_type is None:
                self.return_type = "Expr"
            else:
                self.return_type = _resolve_return_type_from_annotation(
                    self.func,
                )

        if self.return_type == "Expr":
            self.input_type = None

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
    def is_between(lower_bound, upper_bound, closed: ClosedInterval = "both") -> Check:
        """Tests whether all values of the Series are in the given range.

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
            input_type="Series",
            return_type="Series",
            native=False,
            name="is_between",
            description=f"Must be in range {l_paren}{lower_bound}, {upper_bound}{r_paren}",
        )

    @staticmethod
    def lt(other: Any) -> Check:
        """Tests whether all values in the Series are less than the given value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(_lt, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="less_than",
            description=f"Must be < {other}",
        )

    @staticmethod
    def le(other: Any) -> Check:
        """Tests whether all values in the Series are less than or equal to the given
        value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(_le, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="less_than_or_equal_to",
            description=f"Must be <= {other}",
        )

    @staticmethod
    def gt(other: Any) -> Check:
        """Tests whether all values in the Series are greater than the given value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(_gt, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="greater_than",
            description=f"Must be > {other}",
        )

    @staticmethod
    def ge(other: Any) -> Check:
        """Tests whether all values in the Series are greater than or equal to the given
        value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(_ge, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="greater_than_or_equal_to",
            description=f"Must be >= {other}",
        )

    @staticmethod
    def eq(other: Any) -> Check:
        """Tests whether all values in the Series are equal to the given value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(_eq, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="equal_to",
            description=f"Must be = {other}",
        )

    @staticmethod
    def approx_eq(
        other: Any, rtol: float = 1e-5, atol: float = 1e-8, nan_equal: bool = False
    ) -> Check:
        """Tests whether all values in the Series are approximately equal to the given
        value.

        Parameters
        ----------
        other : Any

        Returns
        -------
        Check
        """
        return Check(
            func=functools.partial(
                _series_lit_approx_eq,
                right=other,
                rtol=rtol,
                atol=atol,
                nan_equal=nan_equal,
            ),
            input_type="Series",
            return_type="Series",
            native=False,
            name="approximately_equal_to",
            description=f"Must be approximately equal to {other} ({rtol=}, {atol=}, {nan_equal=})",
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
        """
        return Check(
            func=functools.partial(_is_in, other=other),
            input_type="Series",
            return_type="Series",
            native=False,
            name="is_in",
            description=f"Must be in allowed values {other}",
        )

    @staticmethod
    def is_finite() -> Check:
        return Check(
            func=_is_finite,
            input_type="Series",
            return_type="Series",
            native=False,
            name="is_finite",
            description="All values must be finite",
        )

    @staticmethod
    def is_sorted(descending: bool = False) -> Check:
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
        """
        return Check(
            func=functools.partial(_is_id, subset=subset),
            input_type="Frame",
            return_type="bool",
            native=False,
            name="is_id",
            description=f"{subset} must uniquely identify the DataFrame",
        )

    @staticproperty
    @staticmethod
    def str():
        return _BuiltinStringMethods
