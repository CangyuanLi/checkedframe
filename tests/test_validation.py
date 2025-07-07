import polars as pl
import pytest

import checkedframe as cf


def test_readme_example():
    import polars as pl

    import checkedframe as cf
    from checkedframe.polars import DataFrame

    class AASchema(cf.Schema):
        reason_code = cf.String()
        reason_code_description = cf.String(nullable=True)
        features = cf.List(cf.String)
        shap = cf.Float64(cast=True)
        rank = cf.UInt8(cast=True)

        @cf.Check(columns="reason_code")
        def check_reason_code_length(s: pl.Series) -> pl.Series:
            """Reason codes must be exactly 3 chars"""
            return s.str.len_bytes() == 3

        @cf.Check(columns="shap")
        def check_shap_is_reasonable() -> pl.Expr:
            """Shap values must be reasonable"""
            return pl.col("shap").lt(5).and_(pl.col("shap").gt(0.01))

        @cf.Check
        def check_row_height(df: pl.DataFrame) -> bool:
            """DataFrame must have 2 rows"""
            return df.height == 2

        _id_check = cf.Check.is_id("reason_code")

    df = pl.DataFrame(
        {
            "reason_code": ["R23", "R23", "R9"],
            "reason_code_description": ["Credit score too low", "Income too low", None],
            "shap": [1, 2, 3],
            "rank": [-1, 2, 1],
        }
    )

    with pytest.raises(cf.exceptions.SchemaError):
        x: DataFrame[AASchema] = AASchema.validate(df)

    res = AASchema.interrogate(df)

    errors = res.summary.filter(pl.col("n_failed").gt(0))

    assert errors.filter(pl.col("column").eq("__dataframe__")).height == 2
    assert errors.filter(pl.col("column").eq("reason_code")).height == 1
    assert "reason_code_description" not in errors["column"]
    assert errors.filter(pl.col("column").eq("features")).height == 1
    assert "shap" not in errors["column"]
    assert errors.filter(pl.col("column").eq("rank")).height == 1


def test_mutation():
    # Check that we aren't accidentally mutating columns / checks
    class BaseSchema(cf.Schema):
        is_true = cf.Boolean()

    class Schema1(BaseSchema):
        @cf.Check(columns="is_true")
        def check_is_all_true(s: pl.Series) -> bool:
            return s.all()

    class Schema2(BaseSchema):
        x = cf.Int64()

    df = pl.DataFrame({"is_true": [True, False], "x": [1, 1]})

    with pytest.raises(cf.exceptions.SchemaError):
        Schema1.validate(df)

    Schema2.validate(df)


def test_columns():
    class BaseSchema(cf.Schema):
        x = cf.Int64()
        y = cf.Int64()

    assert BaseSchema.columns() == ["x", "y"]

    class Schema1(BaseSchema):
        z = cf.Int64()

    assert Schema1.columns() == ["x", "y", "z"]


def test_null_in_non_nullable_raises():
    class MySchema(cf.Schema):
        x = cf.Int64()

    df = pl.DataFrame({"x": [None, 1]})

    with pytest.raises(cf.exceptions.SchemaError):
        MySchema.validate(df)


def test_nan_inf_raises_if_disallowed():
    class MySchema(cf.Schema):
        x = cf.Float64()

    df = pl.DataFrame({"x": [float("nan"), float("inf"), float("-inf")]})

    with pytest.raises(cf.exceptions.SchemaError):
        MySchema.validate(df)


def test_nan_inf_passes_if_allowed():
    class MySchema(cf.Schema):
        x = cf.Float64(allow_nan=True, allow_inf=True)

    df = pl.DataFrame({"x": [float("nan"), float("inf"), float("-inf")]})

    MySchema.validate(df)
