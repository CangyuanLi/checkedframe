import narwhals as nw
import polars as pl
import polars.testing
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


@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [(a, b) for a in [cf.Float32, cf.Float64] for b in (INT_TYPES + UINT_TYPES)],
)
def test_float_to_int(from_dtype, to_dtype):
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
        from_dtype._safe_cast(s, to_dtype)

    s = s.filter(~s.is_nan())

    if to_dtype is cf.UInt128:
        with pytest.raises(TypeError):
            from_dtype._safe_cast(s, to_dtype)
    else:
        from_dtype._safe_cast(s, to_dtype)


@pytest.mark.parametrize(
    "from_dtype,to_dtype",
    [(a, cf.Boolean) for a in [cf.Float32, cf.Float64]],
)
def test_float_to_bool(from_dtype, to_dtype):
    s = nw.from_native(
        pl.Series(
            [
                1.0,
                2.0,
                3.0,
                0.0,
                None,
                float("nan"),
            ]
        ),
        series_only=True,
    )

    with pytest.raises(TypeError):
        from_dtype._safe_cast(s, to_dtype)

    s = s.filter(s.is_in([1.0, 0.0, None]))

    from_dtype._safe_cast(s, to_dtype)


def test_float64_to_float32():
    s = nw.from_native(
        pl.Series(
            [
                1.1234567890123456789,
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
        cf.Float64._safe_cast(s, cf.Float32)

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

    cf.Float64._safe_cast(s, cf.Float32)


def test_float32_to_float64():
    s = nw.from_native(
        pl.Series(
            [
                1.12345678,
                1.0,
                2.0,
                3.0,
                None,
                float("nan"),
            ]
        ).cast(pl.Float32),
        series_only=True,
    )

    s_cast = cf.Float32._safe_cast(s, cf.Float64)

    polars.testing.assert_series_equal(
        s_cast.to_native(), s.to_native(), check_dtypes=False
    )
