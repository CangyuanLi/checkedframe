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
