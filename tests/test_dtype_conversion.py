import narwhals.stable.v1 as nw

import checkedframe as cf
from checkedframe._dtypes import _nw_type_to_cf_type


def test_simple_conversion():
    assert _nw_type_to_cf_type(nw.Float64) == cf.Float64
    assert _nw_type_to_cf_type(nw.String) == cf.String


def test_nested_conversion():
    assert _nw_type_to_cf_type(nw.List(nw.Float64)) == cf.List(cf.Float64)
    assert _nw_type_to_cf_type(nw.List(nw.List(nw.Float64))) == cf.List(
        cf.List(cf.Float64)
    )

    assert _nw_type_to_cf_type(
        nw.Struct({"a": nw.String, "b": nw.List(nw.List(nw.Struct({"d": nw.Boolean})))})
    ) == cf.Struct(
        {"a": cf.String, "b": cf.List(cf.List(cf.Struct({"d": cf.Boolean})))}
    )

    assert _nw_type_to_cf_type(nw.Datetime(time_unit="ns")) == cf.Datetime(
        time_unit="ns"
    )
    assert _nw_type_to_cf_type(nw.Duration(time_unit="s")) == cf.Duration(time_unit="s")
