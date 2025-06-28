from __future__ import annotations

import copy
from collections import defaultdict
from typing import Optional

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt

from ._checks import Check, CheckInputType
from ._dtypes import _Column, _nw_type_to_cf_type, _TypedColumn
from ._utils import get_class_members
from .exceptions import ColumnNotFoundError, SchemaError, ValidationError, _ErrorStore
from .selectors import Selector

INFINITY = float("inf")
NEGATIVE_INFINITY = float("-inf")


class _NullValueCheck:
    def __init__(self):
        self.name = "NullValueCheck"
        self.description = "Nulls found in non-nullable column"


class _NanValueCheck:
    def __init__(self):
        self.name = "NaNValueCheck"
        self.description = "NaNs found but not allowed"


class _InfValueCheck:
    def __init__(self):
        self.name = "InfValueCheck"
        self.description = "-inf/inf found but not allowed"


def _run_check(
    check: Check,
    nw_df: nw.DataFrame,
    check_name: str,
    check_input_type: CheckInputType,
    series_name: Optional[str] = None,
) -> tuple[bool, Optional[int]]:
    """_summary_

    Parameters
    ----------
    check : Check
        _description_
    nw_df : nw.DataFrame
        _description_
    check_name : str
        _description_
    check_input_type : CheckInputType
        _description_
    series_name : Optional[str], optional
        , by default None

    Returns
    -------
    bool | int
        Returns either a boolean that is True when the check passes or an integer
        representing the number of rows that fail the check.

    Raises
    ------
    ValueError
        _description_
    ValueError
        _description_
    """
    assert check.func is not None
    if check_input_type is None or check.return_type == "Expr":
        if check.native:
            frame = nw_df.to_native()
        else:
            frame = nw_df

        n_failed = (
            frame.select(check.func().alias(check_name))[check_name].__invert__().sum()
        )

        return (n_failed == 0, n_failed)
    else:
        if check_input_type in ("auto", "Series"):
            if series_name is None:
                raise ValueError(
                    "Series cannot be automatically determined in this context"
                )

            input_ = nw_df[series_name]
        elif check_input_type == "Frame":
            # mypy complains here that the input type is Series, not DataFrame, but it
            # is only a Series if the above branch is hit, which means this branch is
            # not

            input_ = nw_df  # type: ignore[assignment]
        else:
            raise ValueError("Invalid input type")

        if check.native:
            input_ = input_.to_native()

        passed_check = check.func(input_)

        if isinstance(passed_check, bool):
            return (passed_check, None)
        else:
            n_failed = nw.from_native(passed_check, series_only=True).__invert__().sum()

        return (n_failed == 0, n_failed)


def _build_check_err(check: Check, n_failed: Optional[int], n_rows: int):
    if n_failed is not None:
        failed_pct = n_failed / n_rows
        err_msg = f" for {n_failed} / {n_rows} rows ({failed_pct:.2%})"
    else:
        err_msg = ""

    return ValidationError(check, err_msg)


def _validate(schema: Schema, df: nwt.IntoDataFrameT, cast: bool) -> nwt.IntoDataFrameT:
    nw_df = nw.from_native(df, eager_only=True)
    df_schema = nw_df.collect_schema()  # type: ignore[attribute]

    n_rows = nw_df.shape[0]
    errors: dict[str, _ErrorStore] = defaultdict(_ErrorStore)

    for expected_name, expected_col in schema.expected_schema.items():
        error_store = errors[expected_name]

        # check existence
        try:
            _ = df_schema[expected_name]
        except KeyError:
            if expected_col.required:
                error_store.missing_column = ColumnNotFoundError(
                    "Column marked as required but not found"
                )
            continue

        # check nullability
        if not expected_col.nullable:
            null_count = nw_df[expected_name].is_null().sum()
            if null_count > 0:
                null_pct = null_count / n_rows
                error_store.invalid_nulls = ValidationError(
                    _NullValueCheck(),
                    f" for {null_count} / {n_rows} rows ({null_pct:.2%})",
                )

        # check nan-ability
        if hasattr(expected_col, "allow_nan"):
            if not expected_col.allow_nan:
                nan_count = nw_df[expected_name].is_nan().sum()
                if nan_count > 0:
                    nan_pct = nan_count / n_rows
                    error_store.failed_checks.append(
                        ValidationError(
                            _NanValueCheck(),
                            f" for {nan_count} / {n_rows} rows ({nan_pct:.2%})",
                        )
                    )

        # check inf-ability
        if hasattr(expected_col, "allow_inf"):
            if not expected_col.allow_inf:
                nan_count = (
                    nw_df[expected_name].is_in((INFINITY, NEGATIVE_INFINITY)).sum()
                )
                if nan_count > 0:
                    nan_pct = nan_count / n_rows
                    error_store.failed_checks.append(
                        ValidationError(
                            _InfValueCheck(),
                            f" for {nan_count} / {n_rows} rows ({nan_pct:.2%})",
                        )
                    )
        # check data types
        actual_dtype = df_schema[expected_name]
        if actual_dtype == expected_col.to_narwhals():
            pass
        else:
            if expected_col.cast or cast:
                try:
                    nw_df = nw_df.with_columns(
                        _nw_type_to_cf_type(actual_dtype)._safe_cast(
                            nw_df[expected_name], expected_col
                        )
                    )
                except TypeError as e:
                    error_store.invalid_dtype = e
                    continue
            else:
                error_store.invalid_dtype = TypeError(
                    f"Expected {expected_col}, got {actual_dtype}"
                )
                continue

        # user checks
        for i, check in enumerate(expected_col.checks):
            check_name = f"check_{i}" if check.name is None else check.name

            passed_check = _run_check(
                check,
                nw_df,
                check_name=check_name,
                check_input_type=check.input_type,  # type: ignore[assignment]
                series_name=expected_name,
            )

            if not passed_check[0]:
                error_store.failed_checks.append(
                    _build_check_err(check, passed_check[1], n_rows)
                )

    failed_checks: list[ValidationError] = []
    for i, check in enumerate(schema.checks):
        check_name = f"frame_check_{i}" if check.name is None else check.name

        check_input_type: CheckInputType
        if check.input_type == "auto":
            check_input_type = "Expr" if check._func_n_params == 0 else "Frame"  # type: ignore[assignment]
        else:
            check_input_type = check.input_type  # type: ignore[assignment]

        passed_check = _run_check(
            check, nw_df, check_name=check_name, check_input_type=check_input_type
        )

        if not passed_check[0]:
            failed_checks.append(_build_check_err(check, passed_check[1], n_rows))

    schema_error = SchemaError(errors, failed_checks)

    if not schema_error.is_empty():
        raise schema_error

    return nw_df.to_native()


class _SchemaCacheMeta(type):
    def __new__(cls, name, bases, namespace):
        new_class = super().__new__(cls, name, bases, namespace)
        new_class._schema = None

        return new_class


class Schema(metaclass=_SchemaCacheMeta):
    """A lightweight schema representing a DataFrame. Briefly, a schema consists of
    columns and their associated data types. In addition, the schema stores checks that
    can be run either on a specific column or the entire DataFrame. Since `checkedframe`
    leverages `narwhals`, any Narwhals-compatible DataFrame (Pandas, Polars, Modin,
    PyArrow, cuDF) is valid.

    A Schema can be used in two ways. It can either be initialized directly from a
    dictionary or inherited from in a class.

    Parameters
    ----------
    expected_schema : dict[str, Column]
        A dictionary of column names and data types
    checks : Optional[Sequence[Check]], optional
        A list of checks to run, by default None

    Examples
    --------
    Let's say we have a Polars DataFrame we want to validate. We have one column, a
    string, that should be 3 characters.

    .. code-block:: python

        import polars as pl

        df = pl.DataFrame({"col1": ["abc", "ef"]})

    Via inheritance:

    .. code-block:: python

        import checkedframe as cf

        class MySchema(cf.Schema):
            col1 = cf.String()

            @cf.Check(column="col1")
            def check_length(s: pl.Series) -> pl.Series:
                return s.str.len_bytes() == 3

        MySchema.validate(df)

    Via explicit construction:

    .. code-block:: python

        import checkedframe as cf

        MySchema = cf.Schema({
            "col1": cf.String(
                checks=[cf.Check(lambda s: s.str.len_bytes() == 3)]
            )
        })

        MySchema.validate(df)
    """

    _schema: Optional[Schema]

    def __init__(
        self,
        expected_schema: dict[str, _TypedColumn],
        checks: Optional[list[Check]] = None,
    ):
        self.expected_schema = expected_schema
        self.checks = [] if checks is None else checks
        self.validate = self.__validate  # type: ignore
        self.columns = self.__columns  # type: ignore

    @classmethod
    def columns(cls) -> list[str]:
        if cls._schema is None:
            cls._schema = cls._parse_into_schema()

        return list(cls._schema.expected_schema.keys())

    def __columns(self) -> list[str]:
        return list(self.expected_schema.keys())

    @classmethod
    def _parse_into_schema(cls) -> Schema:
        if cls._schema is not None:
            return cls._schema

        schema_dict: dict[str, _TypedColumn] = {}
        checks = []

        attr_list = get_class_members(cls)

        for attr, val in attr_list:
            if isinstance(val, _Column):
                new_val = copy.copy(val)
                # We may modify checks, which is a list, so we need to copy it
                new_val.checks = list(val.checks)

                # TODO: A TypedColumn is a Column and a DType, but the isinstance check
                # above does not work on TypedColumn, only Column.
                schema_dict[attr] = new_val  # type: ignore[assignment]

        for attr, val in attr_list:
            if isinstance(val, Check):
                if (cols_or_selector := val.columns) is not None:
                    if isinstance(cols_or_selector, Selector):
                        cols = cols_or_selector(schema_dict)
                    else:
                        cols = cols_or_selector

                    for c in cols:
                        if c in schema_dict:
                            schema_dict[c].checks.append(val)
                else:
                    checks.append(val)

        res = Schema(expected_schema=schema_dict, checks=checks)
        cls._schema = res

        return res

    @classmethod
    def validate(cls, df: nwt.IntoDataFrameT, cast: bool = False) -> nwt.IntoDataFrameT:
        """Validate the given DataFrame

        Parameters
        ----------
        df : nwt.IntoDataFrameT
            Any Narwhals-compatible DataFrame, see https://narwhals-dev.github.io/narwhals/
            for more information
        cast : bool, optional
            Whether to cast columns, by default False

        Returns
        -------
        nwt.IntoDataFrameT
            Your original DataFrame

        Raises
        ------
        SchemaError
            If validation fails
        """
        return _validate(cls._parse_into_schema(), df, cast)

    def __validate(
        self, df: nwt.IntoDataFrameT, cast: bool = False
    ) -> nwt.IntoDataFrameT:
        return _validate(self, df, cast)
