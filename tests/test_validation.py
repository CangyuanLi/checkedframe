import polars as pl
import pytest

import checkedframe as cf


def test_readme_example():
    class AASchema(cf.Schema):
        reason_code = cf.Column(cf.String)
        reason_code_description = cf.Column(cf.String, nullable=True)
        shap = cf.Column(cf.Float64, cast=True)
        rank = cf.Column(cf.UInt8, cast=True)

        @cf.Check(column="reason_code")
        def check_reason_code_length(s: pl.Series) -> pl.Series:
            """Reason codes must be exactly 3 chars"""
            return s.str.len_bytes() == 3

        @cf.Check(column="reason_code")
        def check_is_id(s: pl.Series) -> bool:
            """Reason code must uniquely identify dataset"""
            return s.n_unique() == s.len()

        @cf.Check
        def check_row_height(df: pl.DataFrame) -> bool:
            """DataFrame must have 2 rows"""
            return df.height == 2

    df = pl.DataFrame(
        {
            "reason_code": ["abc", "abc", "o9"],
            "reason_code_description": ["a desc here", "another desc", None],
            "shap": [1, 2, 3],
            "rank": [-1, 2, 1],
        }
    )

    with pytest.raises(cf.exceptions.SchemaError):
        AASchema.validate(df)


def test_mutation():
    # Check that we aren't accidentally mutating columns / checks
    class BaseSchema(cf.Schema):
        is_true = cf.Boolean()

    class Schema1(BaseSchema):
        @cf.Check(column="is_true")
        def check_is_all_true(s: pl.Series) -> pl.Series:
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
