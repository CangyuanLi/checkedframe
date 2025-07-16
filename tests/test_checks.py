import narwhals.stable.v1 as nw
import pandas as pd
import polars as pl
import pytest

import checkedframe as cf

ENGINES = [pd.DataFrame, pl.DataFrame]


def test_type_inference_polars():
    class A(cf.Schema):
        a = cf.String()

        @cf.Check
        def frame_check(df: pl.DataFrame) -> bool:
            return df.height == 2

        @cf.Check(columns="a")
        def a_check1(name: str) -> pl.Expr:
            return pl.col(name).is_not_null()

        @cf.Check(columns="a")
        def a_check2(s: pl.Series) -> pl.Series:
            return s.is_not_null()

        @cf.Check(columns="a")
        def a_check3(s: pl.Series) -> bool:
            return s.is_not_null().all()

        @cf.Check(columns="a")
        def a_check4() -> pl.Expr:
            return pl.col("a").is_not_null()

    schema = A._parse_into_schema()

    frame_checks = {check.name: check for check in schema.checks}
    col_checks = {}
    for k, v in schema.expected_schema.items():
        for c in v.checks:
            col_checks[c.name] = c

    assert frame_checks["frame_check"].input_type == "Frame"
    assert frame_checks["frame_check"].return_type == "bool"

    assert col_checks["a_check1"].input_type == "str"
    assert col_checks["a_check1"].return_type == "Expr"

    assert col_checks["a_check2"].input_type == "Series"
    assert col_checks["a_check2"].return_type == "Series"

    assert col_checks["a_check3"].input_type == "Series"
    assert col_checks["a_check3"].return_type == "bool"

    assert col_checks["a_check4"].input_type is None
    assert col_checks["a_check4"].return_type == "Expr"


def test_type_inference_pandas():
    class A(cf.Schema):
        a = cf.String()

        @cf.Check
        def frame_check(df: pd.DataFrame) -> bool:
            return df.shape[0] == 2

        @cf.Check(columns="a")
        def a_check2(s: pd.Series) -> pd.Series:
            return s.notnull()

        @cf.Check(columns="a")
        def a_check3(s: pd.Series) -> bool:
            return s.notnull().all()

    schema = A._parse_into_schema()

    frame_checks = {check.name: check for check in schema.checks}
    col_checks = {}
    for k, v in schema.expected_schema.items():
        for c in v.checks:
            col_checks[c.name] = c

    assert frame_checks["frame_check"].input_type == "Frame"
    assert frame_checks["frame_check"].return_type == "bool"

    assert col_checks["a_check2"].input_type == "Series"
    assert col_checks["a_check2"].return_type == "Series"

    assert col_checks["a_check3"].input_type == "Series"
    assert col_checks["a_check3"].return_type == "bool"


@pytest.mark.parametrize("engine", ENGINES)
def test_is_between(engine):
    df = engine({"a": [1, 2, 3]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.is_between(1, 3)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [7, "b", cf.lit(7), cf.col("b")])
def test_lt(engine, val):
    df = engine({"a": [1, 4, 5, 6], "b": [7, 7, 7, 7]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.lt(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [0, "b", cf.lit(0), cf.col("b")])
def test_lt_raises(engine, val):
    df = engine({"a": [1, 4, 5, 6], "b": [0, 0, 0, 0]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.lt(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [1, "b", cf.lit(1), cf.col("b")])
def test_gt(engine, val):
    df = engine({"a": [4, 5, 6], "b": [1, 1, 1]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.gt(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [5, "b", cf.lit(5), cf.col("b")])
def test_gt_raises(engine, val):
    df = engine({"a": [4, 5, 6], "b": [5, 5, 5]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.gt(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [7, "b", cf.lit(7), cf.col("b")])
def test_le(engine, val):
    df = engine({"a": [1, 4, 5, 6], "b": [7, 7, 7, 7]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.le(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [0, "b", cf.lit(0), cf.col("b")])
def test_le_raises(engine, val):
    df = engine({"a": [1, 4, 5, 6], "b": [0, 0, 0, 0]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.le(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [1, "b", cf.lit(1), cf.col("b")])
def test_ge(engine, val):
    df = engine({"a": [4, 5, 6], "b": [1, 1, 1]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.ge(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [5, "b", cf.lit(5), cf.col("b")])
def test_ge_raises(engine, val):
    df = engine({"a": [4, 5, 6], "b": [5, 5, 5]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.ge(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [4, "b", cf.lit(4), cf.col("b")])
def test_eq(engine, val):
    df = engine({"a": [4, 4, 4], "b": [4, 4, 4]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.eq(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [5, "b", cf.lit(5), cf.col("b")])
def test_eq_raises(engine, val):
    df = engine({"a": [4, 5, 6], "b": [5, 5, 5]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.eq(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [0.01, "b", cf.lit(0.01), cf.col("b")])
def test_approx_eq(engine, val):
    df = engine({"a": [0.010000000001], "b": [0.01]})

    class S(cf.Schema):
        a = cf.Float64(checks=[cf.Check.approx_eq(val)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
@pytest.mark.parametrize("val", [0.01, "b", cf.lit(0.01), cf.col("b")])
def test_approx_eq_raises(engine, val):
    df = engine({"a": [0.011], "b": [0.01]})

    class S(cf.Schema):
        a = cf.Float64(checks=[cf.Check.approx_eq(val)])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_is_not_nan(engine):
    df = engine({"a": [1.0, float("nan")]})

    class S(cf.Schema):
        a = cf.Float64(checks=[cf.Check.is_not_nan()])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_str_starts_with(engine):
    df = engine({"a": ["x", "xx", "xy"]})

    class S(cf.Schema):
        a = cf.String(checks=[cf.Check.str_starts_with("x")])

    S.validate(df)

    df = engine({"a": ["yx", "x_a"]})

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_str_ends_with(engine):
    df = engine({"a": ["x", "xx", "yx"]})

    class S(cf.Schema):
        a = cf.String(checks=[cf.Check.str_ends_with("x")])

    S.validate(df)

    df = engine({"a": ["yx", "x_a"]})

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_str_contains(engine):
    df = engine({"a": ["x", "xx", "xy"]})

    class S(cf.Schema):
        a = cf.String(checks=[cf.Check.str_contains("x")])

    S.validate(df)

    df = engine({"a": ["yx", "a_a"]})

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_is_sorted(engine):
    df = engine({"a": [1, 2, 3]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.is_sorted()])

    S.validate(df)

    df = engine({"a": [3, 2, 1]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.is_sorted()])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.is_sorted(descending=True)])

    S.validate(df)


@pytest.mark.parametrize("engine", ENGINES)
def test_is_sorted_by(engine):
    df = engine({"a": [1, 3, 2], "b": [1, 1, 2]})

    class S(cf.Schema):
        _c = cf.Check.is_sorted_by(["a", "b"])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)

    S.validate(nw.from_native(df).sort(["a", "b"]).to_native())


@pytest.mark.parametrize("engine", ENGINES)
def test_is_id(engine):
    df = engine({"a": [1, 1, 2], "b": [1, 2, 1]})

    class S(cf.Schema):
        _c = cf.Check.is_id(["a", "b"])

    S.validate(df)
