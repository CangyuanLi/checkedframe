import checkedframe as cf


def test_inheritance():
    class A(cf.Schema):
        col1 = cf.Column(cf.String)

        @cf.Check(column="col1")
        def a_check() -> bool:
            return True

        @cf.Check(input_type="Frame")
        def df_check(df) -> bool:
            return df.height == 2

    class B(A):
        col2 = cf.Column(cf.Int64)

    schema = B._parse_into_schema()

    assert "col1" in schema.expected_schema.keys()
    assert "col2" in schema.expected_schema.keys()

    assert schema.expected_schema["col1"].checks[0].name == "a_check"
    assert schema.checks[0].name == "df_check"
