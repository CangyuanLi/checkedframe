import datetime

import polars as pl
import pytest

import checkedframe as cf

TYPES = [
    cf.Array,
    cf.Binary,
    cf.Boolean,
    cf.Categorical,
    cf.Date,
    cf.Datetime,
    cf.Decimal,
    cf.Duration,
    cf.Enum,
    cf.Float32,
    cf.Float64,
    cf.Int8,
    cf.Int16,
    cf.Int32,
    cf.Int64,
    cf.Int128,
    cf.List,
    cf.Object,
    cf.String,
    cf.Struct,
    cf.UInt8,
    cf.UInt16,
    cf.UInt32,
    cf.UInt64,
    cf.UInt128,
    cf.Unknown,
]


def test_schema_caching():
    class A(cf.Schema):
        a = cf.String()

        @cf.Check(columns="a")
        def a_check() -> bool:
            return True

    a_schema = A._parse_into_schema()

    assert set(a_schema.expected_schema.keys()) == set(["a"])

    class B(A):
        b = cf.Float64()

    b_schema = B._parse_into_schema()

    assert set(b_schema.expected_schema.keys()) == set(["a", "b"])

    assert set(a_schema.expected_schema.keys()) == set(["a"])


def test_ordered_parsing():
    class A(cf.Schema):
        x = cf.String()
        b = cf.String()

    class B(A):
        a = cf.String()

    assert A.columns() == ["x", "b"]
    assert B.columns() == ["x", "b", "a"]


@pytest.mark.parametrize("dtype", TYPES)
def test_name_override(dtype):
    name = "Reason Code"
    if dtype == cf.List:
        dtype = dtype(cf.String(), name=name)
    elif dtype == cf.Array:
        dtype = dtype(cf.String(), 2, name=name)
    elif dtype == cf.Struct:
        dtype = cf.Struct({"x": cf.String()}, name=name)
    else:
        dtype = dtype(name=name)

    class A(cf.Schema):
        reason_code = dtype

        @cf.Check(columns=name)
        def a_check() -> bool:
            return False

    schema_dict = A._parse_into_schema().expected_schema

    assert "reason_code" not in schema_dict
    assert name in schema_dict

    assert len(schema_dict[name].checks) == 1


def test_schema_generation():
    correct_repr = (
        "import checkedframe as cf\n"
        "\n"
        "class AASchema(cf.Schema):\n"
        '    column_0 = cf.Float64(name="reason code", nullable=True, allow_nan=True)\n'
        "    y = cf.List(cf.Int64, nullable=True)\n"
        "    z = cf.List(cf.List(cf.Int64), nullable=True)\n"
        '    datetime = cf.Datetime(time_unit="us", time_zone=None)\n'
        '    struct = cf.Struct({"a": cf.Int64, "b": cf.Struct({"d": cf.List(cf.Int64)})}, nullable=True)'
    )

    df = pl.DataFrame(
        {
            "reason code": [1.0, float("nan"), None],
            "y": [[1], [2], None],
            "z": [[[1]], None, [[3]]],
            "datetime": datetime.datetime(2016, 1, 22),
            "struct": [
                {"a": 1, "b": {"d": [1, 2, 3]}},
                {"a": 2, "b": {"d": [1, 2, 3]}},
                None,
            ],
        }
    )

    schema_repr = cf.generate_schema_repr(
        df,
        class_name="AASchema",
    )

    assert schema_repr.schema_repr == correct_repr


def test_schema_generation_lazy():
    correct_repr = (
        "import checkedframe as cf\n"
        "\n"
        "class AASchema(cf.Schema):\n"
        '    column_0 = cf.Float64(name="reason code")\n'
        "    y = cf.List(cf.Int64)\n"
        "    z = cf.List(cf.List(cf.Int64))\n"
        '    datetime = cf.Datetime(time_unit="us", time_zone=None)\n'
        '    struct = cf.Struct({"a": cf.Int64, "b": cf.Struct({"d": cf.List(cf.Int64)})})'
    )

    df = pl.DataFrame(
        {
            "reason code": [1.0, float("nan"), None],
            "y": [[1], [2], None],
            "z": [[[1]], None, [[3]]],
            "datetime": datetime.datetime(2016, 1, 22),
            "struct": [
                {"a": 1, "b": {"d": [1, 2, 3]}},
                {"a": 2, "b": {"d": [1, 2, 3]}},
                None,
            ],
        }
    )

    schema_repr = cf.generate_schema_repr(
        df,
        lazy=True,
        class_name="AASchema",
    )

    assert schema_repr.schema_repr == correct_repr

    schema_repr = cf.generate_schema_repr(df.lazy(), class_name="AASchema")

    assert schema_repr.schema_repr == correct_repr
