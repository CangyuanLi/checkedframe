import pandas as pd
import polars as pl
import pytest

import checkedframe as cf


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


def test_is_between():
    df = pl.DataFrame({"a": [1, 2, 3]})

    class S(cf.Schema):
        a = cf.Int64(checks=[cf.Check.is_between(1, 3)])

    S.validate(df)


def test_lt():
    df = pl.DataFrame({"a": [1, None, 4, 5, 6]})

    class S(cf.Schema):
        a = cf.Int64(nullable=True, checks=[cf.Check.lt(7)])

    S.validate(df)


def test_is_sorted_by():
    df = pl.DataFrame({"a": [1, 3, 2], "b": [1, 1, 2]})

    class S(cf.Schema):
        _c = cf.Check.is_sorted_by(["a", "b"])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)

    S.validate(df.sort(["a", "b"]))
