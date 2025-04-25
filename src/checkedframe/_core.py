from __future__ import annotations

import inspect
from collections import defaultdict
from collections.abc import Sequence
from typing import Callable, Optional

import narwhals.stable.v1 as nw
import narwhals.stable.v1.typing as nwt

from ._checks import Check
from .exceptions import ColumnNotFoundError, SchemaError, ValidationError, _ErrorStore


def _run_check(
    check: Check, nw_df: nw.DataFrame, series_name: Optional[str] = None
) -> bool:
    if check.input_type is None or check.return_type == "Expr":
        if check.native:
            frame = nw_df.to_native()
        else:
            frame = nw_df

        return frame.select(check.func().alias(check.name))[check.name].all()
    else:
        if check.input_type in ("auto", "Series"):
            if series_name is None:
                raise ValueError(
                    "Series cannot be automatically determined in this context"
                )

            input_ = nw_df[series_name]
        elif check.input_type == "Frame":
            input_ = nw_df
        else:
            raise ValueError("Invalid input type")

        if check.native:
            input_ = input_.to_native()

        passed_check = check.func(input_)

        if isinstance(passed_check, bool):
            return passed_check
        else:
            passed_check = nw.from_native(passed_check, series_only=True).all()

        return passed_check


class Column:
    def __init__(
        self,
        dtype: nw.dtypes.DType,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        self.dtype = dtype
        self.nullable = nullable
        self.cast = cast
        self.required = required
        self.checks = [] if checks is None else checks


def _validate(self: Schema, df: nwt.IntoDataFrameT, cast: bool) -> nwt.IntoDataFrameT:
    nw_df = nw.from_native(df, eager_only=True)
    df_schema = nw_df.collect_schema()

    errors = defaultdict(_ErrorStore)

    for expected_name, expected_col in self.expected_schema.items():
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
            if nw_df[expected_name].is_null().any():
                error_store.invalid_nulls = ValueError(
                    "Null values in non-nullable column"
                )

        # check data types
        actual_dtype = df_schema[expected_name]
        expected_dtype = expected_col.dtype
        if actual_dtype == expected_col.dtype:
            pass
        else:
            if expected_col.cast or cast:
                if hasattr(actual_dtype, "_safe_cast"):
                    try:
                        nw_df = nw_df.with_columns(
                            actual_dtype._safe_cast(
                                nw_df[expected_name], expected_dtype
                            )
                        )
                    except TypeError as e:
                        error_store.invalid_dtype = e
                        continue
                else:
                    try:
                        nw_df = nw_df.with_columns(
                            nw.col(expected_name).cast(expected_dtype)
                        )
                    except Exception as e:
                        error_store.invalid_dtype = e
                        continue
            else:
                error_store.invalid_dtype = TypeError(
                    f"Expected {expected_dtype.__name__}, got {actual_dtype}"
                )
                continue

        # user checks
        for i, check in enumerate(expected_col.checks):
            if check.name is None:
                check.name = f"check_{i}"

            passed_check = _run_check(check, nw_df, expected_name)

            if not passed_check:
                error_store.failed_checks.append(ValidationError(check))

    failed_checks: list[ValidationError] = []
    for i, check in enumerate(self.checks):
        if check.name is None:
            check.name = f"frame_check_{i}"

        if check.input_type == "auto":
            check.input_type = "Expr" if check._func_n_params == 0 else "Frame"

        passed_check = _run_check(check, nw_df)

        if not passed_check:
            failed_checks.append(ValidationError(check))

    schema_error = SchemaError(errors, failed_checks)

    if not schema_error.is_empty():
        raise schema_error

    return nw_df.to_native()


class Schema:
    def __init__(
        self,
        expected_schema: dict[str, Column],
        checks: Optional[Sequence[Check]] = None,
    ):
        self.expected_schema = expected_schema
        self.checks = [] if checks is None else checks
        self.validate = self.__validate

    @classmethod
    def _parse_into_schema(cls) -> Schema:
        schema_dict = {}
        checks = []
        for attr, val in cls.__dict__.items():
            if isinstance(val, Column):
                schema_dict[attr] = val

        for attr, val in cls.__dict__.items():
            if isinstance(val, Check):
                if val.column is not None:
                    if val.column in schema_dict:
                        schema_dict[val.column].checks.append(val)
                else:
                    checks.append(val)

        return Schema(expected_schema=schema_dict, checks=checks)

    @classmethod
    def validate(cls, df: nwt.IntoDataFrameT, cast: bool = False) -> nwt.IntoDataFrameT:
        return _validate(cls._parse_into_schema(), df, cast)

    def __validate(
        self, df: nwt.IntoDataFrameT, cast: bool = False
    ) -> nwt.IntoDataFrameT:
        return _validate(self, df, cast)
