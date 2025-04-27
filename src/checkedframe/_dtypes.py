from __future__ import annotations

import narwhals.stable.v1 as nw
from narwhals.stable.v1.dtypes import DType


def _checked_cast(s: nw.Series, to_dtype) -> nw.Series:
    # We need this because narwhals will just silently not cast if the datatype isn't
    # supported by the physical backend. E.g., casting a float to UInt128 with Polars
    # backend works because Polars doesn't have a UInt128 type and the original column
    # is just returned.

    to_nw_dtype = to_dtype.to_narwhals()
    s_cast = s.cast(to_nw_dtype)

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
    err = TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; casting resulted in different series"
    )

    try:
        s_cast = _checked_cast(s, to_dtype)
    except Exception as e:
        # There are a couple of different failure modes. One is that the native library
        # cannot do this particular cast--these all return different exceptions, so we
        # just catch Exception. However, the cast could "succeed" but fail due to the
        # issue described in _checked_cast.
        if type(e) is TypeError:
            raise e

        raise err

    if s_cast.__eq__(s).all():
        return s_cast

    raise err


class Int8(nw.Int8):
    _min = -128
    _max = 127

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals() -> nw.Int8:
        return nw.Int8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype):
        if to_dtype in (Int8, Int16, Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt8, UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Int16(nw.Int16):
    _min = -32_768
    _max = 32_767

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Int16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype):
        if to_dtype in (Int16, Int32, Int64, Int128, Float32, Float64, String):
            return _checked_cast(s, to_dtype)
        elif to_dtype in (UInt16, UInt32, UInt64, UInt128):
            return _int_to_uint_cast(s, to_dtype)
        elif to_dtype in (Int8, UInt8):
            return _allowed_range_cast(s, to_dtype)
        elif to_dtype is Boolean:
            return _numeric_to_boolean_cast(s, to_dtype)

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Int32(nw.Int32):
    _min = -2_147_483_648
    _max = 2_147_483_647

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Int32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Int64(nw.Int64):
    _min = -9_223_372_036_854_775_808
    _max = 9_223_372_036_854_775_807

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Int64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Int128(nw.Int128):
    _min = -170141183460469231731687303715884105728
    _max = 170141183460469231731687303715884105727

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Int128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
        if to_dtype is Int128:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class UInt8(nw.UInt8):
    _min = 0
    _max = 255

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.UInt8

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:

        if to_dtype in (
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype}")


class UInt16(nw.UInt16):
    _min = 0
    _max = 65_535

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.UInt16

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:

        if to_dtype in (
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class UInt32(nw.UInt32):
    _min = 0
    _max = 4_294_967_295

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.UInt32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:

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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class UInt64(nw.UInt64):
    _min = 0
    _max = 18446744073709551615

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.UInt64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class UInt128(nw.UInt128):
    _min = 0
    _max = 340_282_366_920_938_463_463_374_607_431_768_211_455

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.UInt128

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
        if to_dtype is Boolean:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Float32(nw.Float32):
    # min and max represent min/max representible int that can be converted without loss
    # of precision
    _min = -16_777_216
    _max = 16_777_216

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Float32

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype) -> nw.Series:
        if to_dtype is Float64:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Float64(nw.Float64):
    _min = -9_007_199_254_740_991
    _max = 9_007_199_254_740_991

    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Float64

    @staticmethod
    def _safe_cast(s: nw.Series, to_dtype: DType) -> nw.Series:
        if to_dtype is Boolean:
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

        raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


class Decimal(nw.Decimal):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Decimal


class Binary(nw.Binary):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Binary


class Boolean(nw.Boolean):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Boolean


class Categorical(nw.Categorical):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Categorical


class Date(nw.Date):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Date


class Datetime(nw.Datetime):
    def __init__(self, time_unit="us", time_zone=None):
        super().__init__(time_unit, time_zone)

        self.to_narwhals = self.__to_narwhals

    @staticmethod
    def to_narwhals():
        return nw.Datetime

    def __to_narwhals(self):
        return nw.Datetime(time_unit=self.time_unit, time_zone=self.time_zone)


class Duration(nw.Duration):
    def __init__(self, time_unit="us"):
        super().__init__(time_unit)

        self.to_narwhals = self.__to_narwhals

    @staticmethod
    def to_narwhals():
        return nw.Duration

    def __to_narwhals(self):
        return nw.Duration(time_unit=self.time_unit)


class String(nw.String):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.String


class Object(nw.Object):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Object


class Unknown(nw.Unknown):
    def __init__(self):
        super().__init__()

    @staticmethod
    def to_narwhals():
        return nw.Unknown


class Array(nw.Array):
    def __init__(self, inner, shape: int | tuple[int, ...]):
        super().__init__(inner, shape)

    def to_narwhals(self):
        return nw.Array(self.inner.to_narwhals(), self.shape)


class List(nw.List):
    def __init__(self, inner):
        super().__init__(inner)

    def to_narwhals(self):
        return nw.List(self.inner.to_narwhals())


class Struct(nw.Struct):
    def __init__(self, fields):
        super().__init__(fields)

    def to_narwhals(self):
        dct = {}
        for field in self.fields:
            dct[field.name] = field.dtype.to_narwhals()

        return nw.Struct(dct)
