from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Optional, Union

import narwhals.stable.v1 as nw
from narwhals.stable.v1.dtypes import DType as NarwhalsDType

from ._checks import Check


class Column:
    """Represents a column in a DataFrame.

    Parameters
    ----------
    dtype : nw.dtypes.DType
        The type of the column
    nullable : bool, optional
        Whether to allow nulls, by default False
    required : bool, optional
        Whether the column is required to be present, by default True
    cast : bool, optional
        Whether to automatically try to cast the column to the expected data type, by
        default False
    checks : Optional[Sequence[Check]], optional
        Checks to run on the column, by default None
    """

    def __init__(
        self,
        dtype: DType,
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


def _checked_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    # We need this because narwhals will just silently not cast if the datatype isn't
    # supported by the physical backend. E.g., casting a float to UInt128 with Polars
    # backend works because Polars doesn't have a UInt128 type and the original column
    # is just returned.

    to_nw_dtype = to_dtype.to_narwhals()
    try:
        s_cast = s.cast(to_nw_dtype)
    except Exception:
        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype}")

    if s_cast.dtype != to_nw_dtype:
        raise TypeError(
            f"Cannot cast {s.dtype} to {to_dtype}; {to_dtype} not supported by your DataFrame library"
        )

    return s_cast


def _int_to_uint_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    s_min = s.min()
    if s_min >= 0:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; actual min {s_min} < allowed min 0"
    )


def _allowed_max_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    allowed_max = to_dtype._max
    s_max = s.max()
    if s_max <= allowed_max:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; actual max {s_max} > allowed max {allowed_max}"
    )


def _allowed_range_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    allowed_min = to_dtype._min
    allowed_max = to_dtype._max
    s_min = s.min()
    s_max = s.max()

    if s_min >= allowed_min and s_max <= allowed_max:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; invalid range [{s_min}, {s_max}], expected range [{allowed_min:,}, {allowed_max:,}]"
    )


def _numeric_to_boolean_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    if s.__eq__(1).__or__(s.__eq__(0)).all():
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; all values must be either 1 or 0"
    )


def _fallback_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
    s_cast = _checked_cast(s, to_dtype)

    if s_cast.__eq__(s).all():
        return s_cast

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype}; casting resulted in different series"
    )


class Int8(Column, nw.Int8):
    _min = -128
    _max = 127

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Int8.__init__(self)
        Column.__init__(
            self,
            dtype=Int8,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals() -> nw.Int8:
        return nw.Int8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype in (Int8, Int16, Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt8, UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Int16(nw.Int16, Column):
    _min = -32_768
    _max = 32_767

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Int16.__init__(self)
        Column.__init__(
            self,
            dtype=Int16,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is Int16:
            return s
        elif to_dtype in (Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, UInt8):
            return _allowed_range_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Int32(nw.Int32, Column):
    _min = -2_147_483_648
    _max = 2_147_483_647

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Int32.__init__(self)
        Column.__init__(
            self,
            dtype=Int32,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is Int32:
            return s
        elif to_dtype in (Int64, Int128, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, UInt8, UInt16, Float32):
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Int64(nw.Int64, Column):
    _min = -9_223_372_036_854_775_808
    _max = 9_223_372_036_854_775_807

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Int64.__init__(self)
        Column.__init__(
            self,
            dtype=Int64,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is Int64:
            return s
        elif to_dtype is Int128:
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, Int32, UInt8, UInt16, UInt32, Float32, Float64):
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Int128(nw.Int128, Column):
    _min = -170141183460469231731687303715884105728
    _max = 170141183460469231731687303715884105727

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Int128.__init__(self)
        Column.__init__(
            self,
            dtype=Int128,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype == Int128:
            return s
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
        ):
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class UInt8(nw.UInt8, Column):
    _min = 0
    _max = 255

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.UInt8.__init__(self)
        Column.__init__(
            self,
            dtype=UInt8,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype == UInt8:
            return s
        elif to_dtype in (
            Int16,
            Int32,
            Int64,
            Int128,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
            Float32,
            Float64,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype is Int8:
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class UInt16(nw.UInt16, Column):
    _min = 0
    _max = 65_535

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.UInt16.__init__(self)
        Column.__init__(
            self,
            dtype=UInt16,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is UInt16:
            return s
        elif to_dtype in (
            Int32,
            Int64,
            Int128,
            UInt32,
            UInt64,
            UInt128,
            Float32,
            Float64,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, UInt8):
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class UInt32(nw.UInt32, Column):
    _min = 0
    _max = 4_294_967_295

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.UInt32.__init__(self)
        Column.__init__(
            self,
            dtype=UInt32,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is UInt32:
            return s
        elif to_dtype in (
            Int64,
            Int128,
            UInt64,
            UInt128,
            Float64,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, Int32, UInt8, UInt16, Float32):
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class UInt64(nw.UInt64, Column):
    _min = 0
    _max = 18446744073709551615

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.UInt64.__init__(self)
        Column.__init__(
            self,
            dtype=UInt64,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is UInt64:
            return s
        elif to_dtype in (
            Int128,
            UInt128,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            UInt8,
            UInt16,
            UInt32,
            Float32,
            Float64,
        ):
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class UInt128(nw.UInt128, Column):
    _min = 0
    _max = 340_282_366_920_938_463_463_374_607_431_768_211_455

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.UInt128.__init__(self)
        Column.__init__(
            self,
            dtype=UInt128,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype == UInt128:
            return s
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            Float32,
            Float64,
        ):
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Float32(nw.Float32, Column):
    # min and max represent min/max representible int that can be converted without loss
    # of precision
    _min = -16_777_216
    _max = 16_777_216

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Float32.__init__(self)
        Column.__init__(
            self,
            dtype=Float32,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Float32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype == Float32:
            return s
        elif to_dtype is Float64:
            return _checked_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
        ):
            return _fallback_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Float64(nw.Float64, Column):
    _min = -9_007_199_254_740_991
    _max = 9_007_199_254_740_991

    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Float64.__init__(self)
        Column.__init__(
            self,
            dtype=Float64,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Float64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype == Float64:
            return s
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (
            Int8,
            Int16,
            Int32,
            Int64,
            Int128,
            UInt8,
            UInt16,
            UInt32,
            UInt64,
            UInt128,
            Float32,
        ):
            return _fallback_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)


class Decimal(nw.Decimal, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Decimal.__init__(self)
        Column.__init__(
            self,
            dtype=Decimal,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Decimal

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Binary(nw.Binary, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Binary.__init__(self)
        Column.__init__(
            self,
            dtype=Binary,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Binary

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Boolean(nw.Boolean, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Boolean.__init__(self)
        Column.__init__(
            self,
            dtype=Boolean,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Boolean

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Categorical(nw.Categorical, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Categorical.__init__(self)
        Column.__init__(
            self,
            dtype=Categorical,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Categorical

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Enum(nw.Enum, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Enum.__init__(self)
        Column.__init__(
            self,
            dtype=Enum,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Enum

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Date(nw.Date, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Date.__init__(self)
        Column.__init__(
            self,
            dtype=Date,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Date

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Datetime(nw.Datetime, Column):
    def __init__(
        self,
        time_unit="us",
        time_zone=None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Datetime.__init__(self, time_unit=time_unit, time_zone=time_zone)
        Column.__init__(
            self,
            dtype=Datetime,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.to_narwhals = self.__to_narwhals

    @staticmethod
    def to_narwhals():
        return nw.Datetime

    def __to_narwhals(self):
        return nw.Datetime(time_unit=self.time_unit, time_zone=self.time_zone)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Datetime) -> Datetime:
        if hasattr(nw_dtype, "time_unit"):
            return Datetime(time_unit=nw_dtype.time_unit, time_zone=nw_dtype.time_zone)

        return Datetime

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Duration(nw.Duration, Column):
    def __init__(
        self,
        time_unit="us",
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Duration.__init__(self, time_unit=time_unit)
        Column.__init__(
            self,
            dtype=Duration,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.to_narwhals = self.__to_narwhals

    @staticmethod
    def to_narwhals():
        return nw.Duration

    def __to_narwhals(self):
        return nw.Duration(time_unit=self.time_unit)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Duration):
        if hasattr(nw_dtype, "time_unit"):
            return Duration(nw_dtype.time_unit)

        return Duration

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class String(nw.String, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.String.__init__(self)
        Column.__init__(
            self,
            dtype=String,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.String

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Object(nw.Object, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Object.__init__(self)
        Column.__init__(
            self,
            dtype=Object,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Object

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Unknown(nw.Unknown, Column):
    def __init__(
        self,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Unknown.__init__(self)
        Column.__init__(
            self,
            dtype=Unknown,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Unknown

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Array(nw.Array, Column):
    def __init__(
        self,
        inner: DType,
        shape: int | tuple[int, ...],
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Array.__init__(self, inner=inner, shape=shape)
        Column.__init__(
            self,
            dtype=Array,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.inner: DType

    def to_narwhals(self):
        return nw.Array(self.inner.to_narwhals(), self.shape)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Array) -> Array:
        return Array(_nw_type_to_cf_type(nw_dtype.inner), shape=nw_dtype.shape)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class List(nw.List, Column):
    def __init__(
        self,
        inner: DType,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.List.__init__(self, inner=inner)
        Column.__init__(
            self,
            dtype=List,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.inner: DType

    def to_narwhals(self):
        return nw.List(self.inner.to_narwhals())

    @staticmethod
    def from_narwhals(nw_dtype: nw.List) -> list:
        return List(_nw_type_to_cf_type(nw_dtype.inner))

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


class Struct(nw.Struct, Column):
    def __init__(
        self,
        fields: Mapping[str, DType],
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Sequence[Check]] = None,
    ):
        nw.Struct.__init__(self, fields=fields)
        Column.__init__(
            self,
            dtype=Struct,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    def to_narwhals(self) -> nw.Struct:
        dct = {}
        for field in self.fields:
            dct[field.name] = field.dtype.to_narwhals()

        return nw.Struct(dct)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Struct) -> Struct:
        dct = {}
        for field in nw_dtype.fields:
            dct[field.name] = _nw_type_to_cf_type(field.dtype)

        return Struct(dct)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        return _checked_cast(s, to_dtype)


DType = Union[
    Array,
    Binary,
    Boolean,
    Date,
    Datetime,
    Decimal,
    Duration,
    Enum,
    Float32,
    Float64,
    Int8,
    Int16,
    Int32,
    Int64,
    Int128,
    List,
    Object,
    String,
    Struct,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    UInt128,
    Unknown,
]

_NARWHALS_DTYPE_TO_CHECKEDFRAME_DTYPE_MAPPER = {
    nw.Binary: Binary,
    nw.Boolean: Boolean,
    nw.Date: Date,
    nw.Datetime: Datetime,
    nw.Decimal: Decimal,
    nw.Enum: Enum,
    nw.Float32: Float32,
    nw.Float64: Float64,
    nw.Int8: Int8,
    nw.Int16: Int16,
    nw.Int32: Int32,
    nw.Int64: Int64,
    nw.Int128: Int128,
    nw.Object: Object,
    nw.String: String,
    nw.UInt8: UInt8,
    nw.UInt16: UInt16,
    nw.UInt32: UInt32,
    nw.UInt64: UInt64,
    nw.UInt128: UInt128,
    nw.Unknown: Unknown,
}


def _nw_type_to_cf_type(nw_dtype: NarwhalsDType) -> DType:
    if isinstance(nw_dtype, nw.Array):
        return Array.from_narwhals(nw_dtype)
    elif isinstance(nw_dtype, nw.List):
        return List.from_narwhals(nw_dtype)
    elif isinstance(nw_dtype, nw.Struct):
        return Struct.from_narwhals(nw_dtype)
    elif isinstance(nw_dtype, nw.Datetime):
        return Datetime.from_narwhals(nw_dtype)
    elif isinstance(nw_dtype, nw.Duration):
        return Duration.from_narwhals(nw_dtype)

    return _NARWHALS_DTYPE_TO_CHECKEDFRAME_DTYPE_MAPPER[nw_dtype]
