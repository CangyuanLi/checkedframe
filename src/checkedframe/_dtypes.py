from __future__ import annotations

import string
from abc import abstractmethod
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING

import narwhals.stable.v1 as nw
from narwhals.stable.v1.dtypes import DType as NarwhalsDType

from ._utils import _all_equal, _parse_args_into_iterable
from .exceptions import CastError

if TYPE_CHECKING:
    from datetime import timezone
    from typing import Literal, Optional, Union

    from ._checks import Check

    TimeUnit = Literal["ns", "us", "ms", "s"]


class _DType:
    @staticmethod
    @abstractmethod
    def to_narwhals() -> NarwhalsDType | type[NarwhalsDType]: ...

    @staticmethod
    @abstractmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType) -> nw.Series: ...

    @staticmethod
    @abstractmethod
    def _to_repr(prefix: str = "") -> str: ...


class _BoundedDType(_DType):
    _min: int | float
    _max: int | float


class _Column:
    """Represents a column in a DataFrame.

    Parameters
    ----------
    name : Optional[str], optional
        The name of the column, by default None
    nullable : bool, optional
        Whether to allow nulls, by default False
    required : bool, optional
        Whether the column is required to be present, by default True
    cast : bool, optional
        Whether to automatically try to cast the column to the expected data type, by
        default False
    checks : Optional[Iterable[Check]], optional
        Checks to run on the column, by default None
    """

    def __init__(
        self,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        self.name = name
        self.nullable = nullable
        self.cast = cast
        self.required = required
        self.checks = [] if checks is None else list(checks)


class TypedColumn(NarwhalsDType, _DType, _Column): ...


def _checked_cast(s: nw.Series, to_dtype: _DType) -> nw.Series:
    # We need this because narwhals will just silently not cast if the datatype isn't
    # supported by the physical backend. E.g., casting a float to UInt128 with Polars
    # backend works because Polars doesn't have a UInt128 type and the original column
    # is just returned.

    to_nw_dtype = to_dtype.to_narwhals()
    try:
        s_cast = s.cast(to_nw_dtype)  # type: ignore
    except Exception:
        raise CastError(
            string.Template("Cannot cast ${from_dtype} to ${to_dtype}").safe_substitute(
                {"from_dtype": s.dtype, "to_dtype": to_dtype}
            ),
            nw.lit(False),
        )

    if s_cast.dtype != to_nw_dtype:
        raise CastError(
            string.Template(
                "Cannot cast ${from_dtype} to ${to_dtype}; failed for {summary} rows ${to_dtype} not supported by your DataFrame library"
            ).safe_substitute({"from_dtype": s.dtype, "to_dtype": to_dtype}),
            element_passes=nw.lit(False),
        )

    return s_cast


def _int_to_uint_cast(s: nw.Series, to_dtype: _DType) -> nw.Series:
    passes = s >= 0

    if passes.all():
        return _checked_cast(s, to_dtype)

    raise CastError(
        string.Template(
            "Cannot safely cast ${from_dtype} to ${to_dtype}; {summary} rows < allowed min 0"
        ).safe_substitute({"from_dtype": s.dtype, "to_dtype": to_dtype}),
        element_passes=passes,
    )


def _allowed_max_cast(s: nw.Series, to_dtype: _BoundedDType) -> nw.Series:
    allowed_max = to_dtype._max
    passes = s <= allowed_max

    if passes.all():
        return _checked_cast(s, to_dtype)

    raise CastError(
        string.Template(
            "Cannot safely cast ${from_dtype} to ${to_dtype}; {summary} rows > allowed max ${allowed_max}"
        ).safe_substitute(
            {
                "from_dtype": s.dtype,
                "to_dtype": to_dtype,
                "allowed_max": f"{allowed_max:,}",
            }
        ),
        passes,
    )


def _allowed_range_cast(s: nw.Series, to_dtype: _BoundedDType) -> nw.Series:
    allowed_min = to_dtype._min
    allowed_max = to_dtype._max
    passes = s.is_between(
        lower_bound=allowed_min, upper_bound=allowed_max, closed="both"
    )

    if passes.all():
        return _checked_cast(s, to_dtype)

    raise CastError(
        string.Template(
            "Cannot safely cast ${from_dtype} to ${to_dtype}; {summary} rows outside of expected range [${allowed_min}, ${allowed_max}]"
        ).safe_substitute(
            {
                "from_dtype": s.dtype,
                "to_dtype": to_dtype,
                "allowed_min": f"{allowed_min:,}",
                "allowed_max": f"{allowed_max:,}",
            }
        ),
        passes,
    )


def _numeric_to_boolean_cast(s: nw.Series, to_dtype: _DType) -> nw.Series:
    passes = (s == 1) | (s == 0)

    if passes.all():
        return _checked_cast(s, to_dtype)

    raise CastError(
        string.Template(
            "Cannot safely cast ${from_dtype} to ${to_dtype}; {summary} rows are not either 1 or 0"
        ).safe_substitute({"from_dtype": s.dtype, "to_dtype": to_dtype}),
        passes,
    )


def _fallback_cast(s: nw.Series, to_dtype: _DType) -> nw.Series:
    s_cast = _checked_cast(s, to_dtype)
    passes = s_cast == s

    if passes.all():
        return s_cast

    raise CastError(
        string.Template(
            "Cannot safely cast ${from_dtype} to ${to_dtype}; different results for {summary} rows"
        ).safe_substitute({"from_dtype": s.dtype, "to_dtype": to_dtype}),
        passes,
    )


def _union_cast(cls: type[TypedColumn], s: nw.Series, union: CfUnion) -> nw.Series:
    error = None

    for to in union.columns:
        try:
            return cls._safe_cast(s, to)
        except CastError as e:
            # Keep the first error instead of the last
            if error is None:
                error = e
            continue

    # This is fine because we will either return or raise
    assert error is not None
    raise error


def _fmt_optional_string(s: str | None) -> str | None:
    return f'"{s}"' if s is not None else None


def _fmt_dict_string(d: dict) -> str:
    items = []
    for k, v in d.items():
        # Format the key with double quotes
        key = f'"{k}"'

        # Check if the value is a dictionary and recurse
        if isinstance(v, dict):
            value = _fmt_dict_string(v)
        # Check if the value is a string and remove quotes
        elif isinstance(v, str):
            value = v
        # Otherwise, use json.dumps for standard formatting (e.g., numbers, lists)
        else:
            raise ValueError(
                "All keys must be strings and all values must either be strings or dicts"
            )

        items.append(f"{key}: {value}")

    return "{" + ", ".join(items) + "}"


class Int8(nw.Int8, _BoundedDType, TypedColumn):
    _min = -128
    _max = 127

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Int8.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Int8, s, to_dtype)
        if to_dtype in (Int8, Int16, Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt8, UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Int8"


class Int16(nw.Int16, _BoundedDType, TypedColumn):
    _min = -32_768
    _max = 32_767

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Int16.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Int16, s, to_dtype)

        if to_dtype == Int16:
            return s
        elif to_dtype in (Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, UInt8):
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_range_cast(s, to_dtype)
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Int16"


class Int32(nw.Int32, _BoundedDType, TypedColumn):
    _min = -2_147_483_648
    _max = 2_147_483_647

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Int32.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Int32, s, to_dtype)

        if to_dtype == Int32:
            return s
        elif to_dtype in (Int64, Int128, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, UInt8, UInt16, Float32):
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Int32"


class Int64(nw.Int64, TypedColumn, _BoundedDType):
    _min = -9_223_372_036_854_775_808
    _max = 9_223_372_036_854_775_807

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Int64.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Int64, s, to_dtype)

        if to_dtype == Int64:
            return s
        elif to_dtype == Int128:
            return _checked_cast(s, to_dtype)
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, Int32, UInt8, UInt16, UInt32, Float32, Float64):
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Int64"


class Int128(nw.Int128, TypedColumn, _BoundedDType):
    _min = -170141183460469231731687303715884105728
    _max = 170141183460469231731687303715884105727

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Int128.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Int128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Int128, s, to_dtype)

        if to_dtype == Int128:
            return s
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype == UInt128:
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
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_range_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Int128"


class UInt8(nw.UInt8, TypedColumn, _BoundedDType):
    _min = 0
    _max = 255

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.UInt8.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(UInt8, s, to_dtype)

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
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype == Int8:
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}UInt8"


class UInt16(nw.UInt16, TypedColumn, _BoundedDType):
    _min = 0
    _max = 65_535

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.UInt16.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(UInt16, s, to_dtype)

        if to_dtype == UInt16:
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
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, UInt8):
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}UInt16"


class UInt32(nw.UInt32, TypedColumn, _BoundedDType):
    _min = 0
    _max = 4_294_967_295

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.UInt32.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(UInt32, s, to_dtype)

        if to_dtype == UInt32:
            return s
        elif to_dtype in (
            Int64,
            Int128,
            UInt64,
            UInt128,
            Float64,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype == Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)
        elif to_dtype in (Int8, Int16, Int32, UInt8, UInt16, Float32):
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}UInt32"


class UInt64(nw.UInt64, TypedColumn, _BoundedDType):
    _min = 0
    _max = 18_446_744_073_709_551_615

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.UInt64.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(UInt64, s, to_dtype)

        if to_dtype == UInt64:
            return s
        elif to_dtype in (
            Int128,
            UInt128,
        ):
            return _checked_cast(s, to_dtype)
        elif to_dtype == Boolean:
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
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}UInt64"


class UInt128(nw.UInt128, TypedColumn, _BoundedDType):
    _min = 0
    _max = 340_282_366_920_938_463_463_374_607_431_768_211_455

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.UInt128.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.UInt128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(UInt128, s, to_dtype)

        if to_dtype == UInt128:
            return s
        elif to_dtype == Boolean:
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
            assert isinstance(to_dtype, _BoundedDType)
            return _allowed_max_cast(s, to_dtype)

        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}UInt128"


class Float32(nw.Float32, TypedColumn, _BoundedDType):
    # min and max represent min/max representible int that can be converted without loss
    # of precision
    _min = -16_777_216
    _max = 16_777_216

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        allow_nan: bool = False,
        allow_inf: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Float32.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

    @staticmethod
    def to_narwhals():
        return nw.Float32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Float32, s, to_dtype)

        if to_dtype == Float32:
            return s
        elif to_dtype == Float64:
            return _checked_cast(s, to_dtype)
        elif to_dtype == Boolean:
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

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Float32"


class Float64(nw.Float64, TypedColumn, _BoundedDType):
    _min = -9_007_199_254_740_991
    _max = 9_007_199_254_740_991

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        allow_nan: bool = False,
        allow_inf: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Float64.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.allow_nan = allow_nan
        self.allow_inf = allow_inf

    @staticmethod
    def to_narwhals():
        return nw.Float64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Float64, s, to_dtype)

        if to_dtype == Float64:
            return s
        elif to_dtype == Boolean:
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

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Float64"


class Decimal(nw.Decimal, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Decimal.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Decimal

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Decimal, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Decimal"


class Binary(nw.Binary, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Binary.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Binary

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Binary, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Binary"


class Boolean(nw.Boolean, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Boolean.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Boolean

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Boolean, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Boolean"


class Categorical(nw.Categorical, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Categorical.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Categorical

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Categorical, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Categorical"


class Enum(nw.Enum, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Enum.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Enum

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Enum, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Enum"


class Date(nw.Date, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Date.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Date

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Date, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Date"


class Datetime(nw.Datetime, TypedColumn, _DType):
    def __init__(
        self,
        time_unit: TimeUnit = "us",
        time_zone: Union[str, timezone, None] = None,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Datetime.__init__(self, time_unit=time_unit, time_zone=time_zone)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.to_narwhals = self.__to_narwhals  # type: ignore
        self._to_repr = self.__to_repr  # type: ignore

    @staticmethod
    def to_narwhals():
        return nw.Datetime

    def __to_narwhals(self):
        return nw.Datetime(time_unit=self.time_unit, time_zone=self.time_zone)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Datetime, **column_kwargs) -> Datetime:
        if hasattr(nw_dtype, "time_unit"):
            return Datetime(
                time_unit=nw_dtype.time_unit,
                time_zone=nw_dtype.time_zone,
                **column_kwargs,
            )

        return Datetime(**column_kwargs)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Datetime, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Datetime"

    def __to_repr(self, prefix: str = "") -> str:
        return f"{prefix}Datetime(time_unit={_fmt_optional_string(self.time_unit)}, time_zone={_fmt_optional_string(self.time_zone)})"


class Duration(nw.Duration, TypedColumn, _DType):
    def __init__(
        self,
        time_unit: TimeUnit = "us",
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Duration.__init__(self, time_unit=time_unit)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.to_narwhals = self.__to_narwhals  # type: ignore
        self._to_repr = self.__to_repr  # type: ignore

    @staticmethod
    def to_narwhals():
        return nw.Duration

    def __to_narwhals(self):
        return nw.Duration(time_unit=self.time_unit)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Duration, **column_kwargs) -> Duration:
        if hasattr(nw_dtype, "time_unit"):
            return Duration(nw_dtype.time_unit, **column_kwargs)

        return Duration(**column_kwargs)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Duration, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Duration"

    def __to_repr(self, prefix: str = "") -> str:
        return f"{prefix}Duration(time_unit={_fmt_optional_string(self.time_unit)})"


class String(nw.String, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.String.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.String

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(String, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}String"


class Object(nw.Object, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Object.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Object

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Object, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Object"


class Unknown(nw.Unknown, TypedColumn, _DType):
    def __init__(
        self,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Unknown.__init__(self)
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

    @staticmethod
    def to_narwhals():
        return nw.Unknown

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Unknown, s, to_dtype)
        return _checked_cast(s, to_dtype)

    @staticmethod
    def _to_repr(prefix: str = "") -> str:
        return f"{prefix}Unknown"


class Array(nw.Array, TypedColumn, _DType):
    def __init__(
        self,
        inner: _DType,
        shape: int | tuple[int, ...],
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Array.__init__(self, inner=inner, shape=shape)  # type: ignore
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.inner: _DType  # type: ignore

    def to_narwhals(self):  # type: ignore
        return nw.Array(self.inner.to_narwhals(), self.shape)

    @staticmethod
    def from_narwhals(nw_dtype: nw.Array, **column_kwargs) -> Array:
        return Array(
            _nw_type_to_cf_type(nw_dtype.inner), shape=nw_dtype.shape, **column_kwargs
        )

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Array, s, to_dtype)
        return _checked_cast(s, to_dtype)

    def _to_repr(self, prefix="") -> str:  # type: ignore
        return f"{prefix}Array({self.inner._to_repr(prefix)})"


class List(nw.List, TypedColumn, _DType):
    def __init__(
        self,
        inner: _DType,
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.List.__init__(self, inner=inner)  # type: ignore
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.inner: _DType  # type: ignore

    def to_narwhals(self):  # type: ignore[override]
        return nw.List(self.inner.to_narwhals())

    @staticmethod
    def from_narwhals(nw_dtype: nw.List, **column_kwargs) -> List:
        return List(_nw_type_to_cf_type(nw_dtype.inner), **column_kwargs)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(List, s, to_dtype)
        return _checked_cast(s, to_dtype)

    def _to_repr(self, prefix="") -> str:  # type: ignore
        return f"{prefix}List({self.inner._to_repr(prefix)})"


class _Field:
    name: str
    dtype: _DType

    def __init__(self, name: str, dtype: _DType) -> None:
        self.name = name
        self.dtype = dtype


class Struct(nw.Struct, TypedColumn, _DType):
    def __init__(
        self,
        fields: Mapping[str, _DType],
        *,
        name: Optional[str] = None,
        nullable: bool = False,
        required: bool = True,
        cast: bool = False,
        checks: Optional[Iterable[Check]] = None,
    ):
        nw.Struct.__init__(self, fields=fields)  # type: ignore
        _Column.__init__(
            self,
            name=name,
            nullable=nullable,
            required=required,
            cast=cast,
            checks=checks,
        )

        self.fields: list[_Field]  # type: ignore

    def to_narwhals(self) -> nw.Struct:  # type: ignore
        dct = {}
        for field in self.fields:
            dct[field.name] = field.dtype.to_narwhals()

        return nw.Struct(dct)  # type: ignore

    @staticmethod
    def from_narwhals(nw_dtype: nw.Struct, **column_kwargs) -> Struct:
        dct = {}
        for field in nw_dtype.fields:
            dct[field.name] = _nw_type_to_cf_type(field.dtype)

        return Struct(dct, **column_kwargs)

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: _DType | CfUnion) -> nw.Series:
        if isinstance(to_dtype, CfUnion):
            return _union_cast(Struct, s, to_dtype)
        return _checked_cast(s, to_dtype)

    def _to_repr(self, prefix="") -> str:  # type: ignore
        dct = {}
        for field in self.fields:
            dct[field.name] = field.dtype._to_repr(prefix)

        return f"{prefix}Struct({_fmt_dict_string(dct)})"


class CfUnion:
    """Union type.

    Examples
    --------
    .. code-block:: python

        import checkedframe as cf
        import polars as pl


        class S(cf.Schema):
            by_var = cf.Union(cf.Boolean(), cf.Int64(cast=True))


        df = pl.DataFrame({"by_var": [1.0, 5.0]})
        S.validate(df)

    Output:

    .. code-block:: text



    """

    def __init__(self, *columns: TypedColumn | Iterable[TypedColumn]):
        self.columns: Iterable[TypedColumn] = _parse_args_into_iterable(columns)
        self.required = True

        assert _all_equal(c.name for c in self.columns)
        if not all(c.required for c in self.columns):
            raise NotImplementedError(
                "Currently, non-required Unions are not supported."
            )

    @staticmethod
    def is_decimal():
        return False

    @staticmethod
    def is_float():
        return False

    @staticmethod
    def is_integer():
        return False

    @staticmethod
    def is_signed_integer():
        return False

    @staticmethod
    def is_unsigned_integer():
        return False

    @staticmethod
    def is_numeric():
        return False

    @staticmethod
    def is_temporal():
        return False

    def _resolve(self, s: nw.Series, actual_cf_type: TypedColumn):
        cast_error = None
        type_error = None
        for c in self.columns:
            if c == actual_cf_type:
                return c, None
            else:
                if c.cast:
                    try:
                        s_cast = actual_cf_type._safe_cast(s, c)
                        return c, s_cast
                    except CastError as e:
                        if cast_error is None:
                            cast_error = e
                        continue
                else:
                    if type_error is None:
                        type_error = TypeError()

        if type_error is not None:
            raise TypeError()

        if cast_error is not None:
            raise cast_error


_NARWHALS_DTYPE_TO_CHECKEDFRAME_DTYPE_MAPPER: dict[
    type[NarwhalsDType], type[_DType]
] = {
    nw.Binary: Binary,
    nw.Boolean: Boolean,
    nw.Categorical: Categorical,
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


def _nw_type_to_cf_type(
    nw_dtype: Union[NarwhalsDType, type[NarwhalsDType]], **column_kwargs
) -> TypedColumn:
    if isinstance(nw_dtype, nw.Array):
        return Array.from_narwhals(nw_dtype, **column_kwargs)
    elif isinstance(nw_dtype, nw.List):
        return List.from_narwhals(nw_dtype, **column_kwargs)
    elif isinstance(nw_dtype, nw.Struct):
        return Struct.from_narwhals(nw_dtype, **column_kwargs)
    elif isinstance(nw_dtype, nw.Datetime):
        return Datetime.from_narwhals(nw_dtype, **column_kwargs)
    elif isinstance(nw_dtype, nw.Duration):
        return Duration.from_narwhals(nw_dtype, **column_kwargs)

    return _NARWHALS_DTYPE_TO_CHECKEDFRAME_DTYPE_MAPPER[nw_dtype](**column_kwargs)  # type: ignore
