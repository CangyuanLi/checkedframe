from __future__ import annotations

import functools
import inspect
from collections.abc import Collection
from typing import Any, Callable, Literal, Optional

import narwhals.stable.v1 as nw


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


ClosedInterval = Optional[Literal["both", "left", "right"]]


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


def _is_in(s: nw.Series, other: Collection) -> nw.Series:
    return s.is_in(other)


def _is_finite(s: nw.Series) -> nw.Series:
    return s.is_finite()


def _is_id(df: nw.DataFrame, subset: str | list[str]) -> bool:
    n_rows = df.shape[0]
    n_unique_rows = df.select(subset).unique().shape[0]

    return n_rows == n_unique_rows


class Check:
    """Represents a check to run.

    Parameters
    ----------
    func : Optional[Callable], optional
        The check to run, by default None
    column : Optional[str], optional
        The column associated with the check, by default None
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
        column: Optional[str] = None,
        input_type: Optional[Literal["auto", "Frame", "Series"]] = "auto",
        return_type: Literal["auto", "bool", "Expr", "Series"] = "auto",
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
        self.column = column

        if self.func is not None:
            self._set_params()

    def _set_params(self):
        self._func_n_params = len(inspect.signature(self.func).parameters)

        if self.input_type == "auto":
            if self._func_n_params == 0:
                self.input_type = None

        if self.return_type == "auto" and self.func is not None:
            if self.input_type is None:
                self.return_type == "Expr"
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

    def __call__(self, func):
        return Check(
            func=func,
            column=self.column,
            input_type=self.input_type,
            return_type=self.return_type,
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
            l_paren, r_paren = "[]"
        elif closed == "left":
            l_paren, r_paren = "[)"
        elif closed == "right":
            l_paren, r_paren = "(]"
        elif closed == "neither":
            l_paren, r_paren = "()"

        return Check(
            func=functools.partial(
                _is_between, lower_bound, upper_bound, closed=closed
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
