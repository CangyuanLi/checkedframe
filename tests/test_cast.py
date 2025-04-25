import narwhals as nw
import polars as pl
import pytest

import checkedframe as cf

INT_TYPES = [
    cf.Int8,
    cf.Int16,
    cf.Int32,
    cf.Int64,
    cf.Int128,
]
UINT_TYPES = [cf.UInt8, cf.UInt16, cf.UInt32, cf.UInt64, cf.UInt128]
FLOAT_TYPES = [cf.Float32, cf.Float64]


@pytest.mark.parametrize("to_dtype", INT_TYPES + UINT_TYPES)
def test_float_to_int(to_dtype):
    s = nw.from_native(
        pl.Series(
            [
                1.0,
                2.0,
                3.0,
                None,
                float("nan"),
            ]
        ),
        series_only=True,
    )

    with pytest.raises(TypeError):
        cf.Float64._safe_cast(s, to_dtype)
        cf.Float32._safe_cast(s, to_dtype)

    s = s.filter(~s.is_nan())

    if to_dtype is cf.UInt128:
        with pytest.raises(TypeError):
            cf.Float64._safe_cast(s, to_dtype)
            cf.Float32._safe_cast(s, to_dtype)
    else:
        cf.Float64._safe_cast(s, to_dtype)
        cf.Float32._safe_cast(s, to_dtype)


# def test_float64_to_float32(to_dtype):
#     s = nw.from_native(
#         pl.Series(
#             [
#                 1.1234567890123456789,
#                 1.0,
#                 2.0,
#                 3.0,
#                 None,
#                 float("nan"),
#             ]
#         ),
#         series_only=True,
#     )

#     with pytest.raises(TypeError):
#         cf.Float64._safe_cast(s, cf.Float32)

#     s = nw.from_native(
#         pl.Series(
#             [
#                 1.0,
#                 2.0,
#                 3.0,
#                 None,
#                 float("nan"),
#             ]
#         ),
#         series_only=True,
#     )

#     cf.Float64._safe_cast(s, cf.Float32)
