import datetime

import polars as pl
import pytest

import checkedframe as cf
import checkedframe.selectors as cfs


class MySchema(cf.Schema):
    string = cf.String()
    int8 = cf.Int8()
    float32 = cf.Float32()
    boolean = cf.Boolean()
    list_string = cf.List(cf.String())
    list_list_string = cf.List(cf.List(cf.String()))


SCHEMA = MySchema._parse_into_schema().expected_schema


def test_all():
    assert set(cfs.all()(SCHEMA)) == set(SCHEMA.keys())


def test_by_name():
    assert set(cfs.by_name("string", "int8")(SCHEMA)) == set(["string", "int8"])


def test_matches():
    assert set(cfs.matches(".*string.*")(SCHEMA)) == set(
        ["string", "list_string", "list_list_string"]
    )


def test_starts_with():
    assert set(cfs.starts_with("string")(SCHEMA)) == set(["string"])


def test_ends_with():
    assert set(cfs.ends_with("_list_string")(SCHEMA)) == set(["list_list_string"])


def test_contains():
    assert set(cfs.contains("string")(SCHEMA)) == set(
        ["string", "list_string", "list_list_string"]
    )
    assert set(cfs.contains("string", "int8")(SCHEMA)) == set(
        ["string", "list_string", "list_list_string", "int8"]
    )


def test_by_dtype():
    assert set(cfs.by_dtype(cf.List(cf.String()))(SCHEMA)) == set(["list_string"])
    assert set(cfs.by_dtype(cf.List(cf.List(cf.String())))(SCHEMA)) == set(
        ["list_list_string"]
    )


def test_string():
    assert set(cfs.string()(SCHEMA)) == set(["string"])


def test_int():
    assert set(cfs.integer()(SCHEMA)) == set(["int8"])


def test_float():
    assert set(cfs.float()(SCHEMA)) == set(["float32"])


def test_boolean():
    assert set(cfs.boolean()(SCHEMA)) == set(["boolean"])


def test_numeric():
    assert set(cfs.numeric()(SCHEMA)) == set(["int8", "float32"])


def test_exclude():
    assert cfs.numeric().exclude("int8")(SCHEMA) == ["float32"]
    assert cfs.numeric().exclude(cfs.by_name("int8"))(SCHEMA) == ["float32"]
    assert cfs.numeric().exclude(["int8"])(SCHEMA) == ["float32"]


def test_temporal_example():
    class S(cf.Schema):
        join_time = cf.Datetime()
        birth_date = cf.Date()

        @cf.Check(columns=cfs.temporal())
        def check_no_invalid_dates(name: str) -> pl.Expr:
            return pl.col(name).dt.date() >= pl.date(1900, 1, 1)

    df = pl.DataFrame(
        {
            "join_time": [
                datetime.datetime(1899, 1, 1, 1, 1),
                datetime.datetime(2010, 2, 20, 6, 3, 9),
            ],
            "birth_date": [datetime.date(1899, 1, 1), datetime.date(2000, 5, 19)],
        },
    )

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)


def test_union():
    class S(cf.Schema):
        date = cf.Date()
        union = cf.Union(cf.Float64(), cf.Float32())

    assert cfs.temporal()(S._parse_into_schema().expected_schema) == ["date"]
