from __future__ import annotations

import narwhals.stable.v1 as nw


def _checked_cast(s: nw.Series, to_dtype) -> nw.Series:
    # We need this because narwhals will just silently not cast if the datatype isn't
    # supported by the physical backend. E.g., casting a float to UInt128 with Polars
    # backend works because Polars doesn't have a UInt128 type and the original column
    # is just returned.

    s_cast = s.cast(to_dtype)

    if s_cast.dtype != to_dtype:
        dtype_name = to_dtype.__name__

        raise TypeError(
            f"Cannot cast {s.dtype} to {dtype_name}; {dtype_name} not supported by your DataFrame library"
        )

    return s_cast


def _int_to_uint_cast(s: nw.Series, to_dtype) -> nw.Series:
    s_min = s.min()
    if s_min >= 0:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; actual min {s_min} < allowed min 0"
    )


def _allowed_max_cast(s: nw.Series, to_dtype) -> nw.Series:
    allowed_max = to_dtype._max
    s_max = s.max()
    if s_max <= allowed_max:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; actual max {s_max} > allowed max {allowed_max}"
    )


def _allowed_range_cast(s: nw.Series, to_dtype) -> nw.Series:
    allowed_min = to_dtype._min
    allowed_max = to_dtype._max
    s_min = s.min()
    s_max = s.max()

    if s_min >= allowed_min and s_max <= allowed_max:
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; invalid range [{s_min}, {s_max}], expected range [{allowed_min:,}, {allowed_max:,}]"
    )


def _numeric_to_boolean_cast(s: nw.Series, to_dtype) -> nw.Series:
    if s.__eq__(1).__or__(s.__eq__(0)).all():
        return _checked_cast(s, to_dtype)

    raise TypeError(
        f"Cannot safely cast {s.dtype} to {to_dtype.__name__}; all values must be either 1 or 0"
    )


def _fallback_cast(s: nw.Series, to_dtype) -> nw.Series:
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


def _monkeypatch(cls, _min, _max, _safe_cast):
    cls._nw_dtype = cls
    cls._min = _min
    cls._max = _max
    cls._safe_cast = staticmethod(_safe_cast)

    return cls


def _int8_safe_cast(s: nw.Series, to_dtype):
    if to_dtype in (Int8, Int16, Int32, Int64, Int128, Float32, Float64, String):
        return _checked_cast(s, to_dtype)
    elif to_dtype in (UInt8, UInt16, UInt32, UInt64, UInt128):
        return _int_to_uint_cast(s, to_dtype)
    elif to_dtype is Boolean:
        return _numeric_to_boolean_cast(s, to_dtype)

    raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


Int8 = _monkeypatch(nw.Int8, -128, 127, _int8_safe_cast)


def _int16_safe_cast(s: nw.Series, to_dtype):
    if to_dtype in (Int16, Int32, Int64, Int128, Float32, Float64, String):
        return _checked_cast(s, to_dtype)
    elif to_dtype in (UInt16, UInt32, UInt64, UInt128):
        return _int_to_uint_cast(s, to_dtype)
    elif to_dtype in (Int8, UInt8):
        return _allowed_range_cast(s, to_dtype)
    elif to_dtype is Boolean:
        return _numeric_to_boolean_cast(s, to_dtype)

    raise TypeError(f"Cannot cast {s.dtype} to {to_dtype.__name__}")


Int16 = _monkeypatch(nw.Int16, _min=-32_768, _max=32_767, _safe_cast=_int16_safe_cast)


def _int32_safe_cast(s: nw.Series, to_dtype):
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


Int32 = _monkeypatch(
    nw.Int32, _min=-2_147_483_648, _max=2_147_483_647, _safe_cast=_int32_safe_cast
)


def _int64_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


Int64 = _monkeypatch(
    nw.Int64,
    _min=-9_223_372_036_854_775_808,
    _max=9_223_372_036_854_775_807,
    _safe_cast=_int64_safe_cast,
)


def _int28_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


Int128 = _monkeypatch(
    nw.Int128,
    _min=-170141183460469231731687303715884105728,
    _max=170141183460469231731687303715884105727,
    _safe_cast=_int28_safe_cast,
)


def _uint8_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


UInt8 = _monkeypatch(nw.UInt8, _min=0, _max=255, _safe_cast=_uint8_safe_cast)


def _uint16_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


UInt16 = _monkeypatch(nw.UInt16, _min=0, _max=65_535, _safe_cast=_uint16_safe_cast)


def _uint32_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


UInt32 = _monkeypatch(
    nw.UInt32, _min=0, _max=4_294_967_295, _safe_cast=_uint32_safe_cast
)


def _uint64_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


UInt64 = _monkeypatch(
    nw.UInt64, _min=0, _max=18446744073709551615, _safe_cast=_uint64_safe_cast
)


def _uint128_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


UInt128 = _monkeypatch(
    nw.UInt128,
    _min=0,
    _max=340_282_366_920_938_463_463_374_607_431_768_211_455,
    _safe_cast=_uint128_safe_cast,
)


def _float32_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


# min and max represent min/max representible int that can be converted without loss
# of precision
Float32 = _monkeypatch(
    nw.Float32, _min=-16_777_216, _max=16_777_216, _safe_cast=_float32_safe_cast
)


def _float64_safe_cast(s: nw.Series, to_dtype) -> nw.Series:
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


Float64 = _monkeypatch(
    nw.Float64,
    _min=-9_007_199_254_740_991,
    _max=9_007_199_254_740_991,
    _safe_cast=_float64_safe_cast,
)


Decimal = nw.Decimal
Array = nw.Array
List = nw.List
Binary = nw.Binary
Boolean = nw.Boolean
Categorical = nw.Categorical
Date = nw.Date
Datetime = nw.Datetime
Duration = nw.Duration
String = nw.String
Struct = nw.Struct
Object = nw.Object
Uknown = nw.Unknown
