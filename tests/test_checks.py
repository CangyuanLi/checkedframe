import polars as pl
import pytest

import checkedframe as cf


def test_is_sorted_by():
    df = pl.DataFrame({"a": [1, 3, 2], "b": [1, 1, 2]})

    class S(cf.Schema):
        _c = cf.Check.is_sorted_by(["a", "b"])

    with pytest.raises(cf.exceptions.SchemaError):
        S.validate(df)

    S.validate(df.sort(["a", "b"]))
