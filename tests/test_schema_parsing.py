import checkedframe as cf


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
