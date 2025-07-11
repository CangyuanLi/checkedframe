import checkedframe as cf


def test_config():
    @cf.apply_configs(cf.Config(cf.selectors.float(), nullable=True))
    class S(cf.Schema):
        float1 = cf.Float64()
        float2 = cf.Float64()
        int1 = cf.Int64()

    s = S._parse_into_schema().expected_schema

    assert s["float1"].nullable
    assert s["float2"].nullable
    assert not s["int1"].nullable

    class S2(S):
        float3 = cf.Float64()

    s2 = S2._parse_into_schema().expected_schema

    assert s["float1"].nullable
    assert s["float2"].nullable
    assert s2["float3"].nullable
