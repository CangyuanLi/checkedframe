# checkedframe:
[![PyPI version](https://badge.fury.io/py/checkedframe.svg)](https://badge.fury.io/py/checkedframe)
![PyPI - Downloads](https://img.shields.io/pypi/dm/checkedframe)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Tests](https://github.com/CangyuanLi/checkedframe/actions/workflows/tests.yaml/badge.svg)

<p align="center">
  <a href="https://cangyuanli.github.io/checkedframe/">Documentation</a>
<br>
</p>

## What is it?

**checkedframe** is a lightweight library for DataFrame validation built on top of **narwhals**.

## Why use checkedframe?

# Usage:

## Installing

The easiest way is to install **checkedframe** is from PyPI using pip:

```sh
pip install checkedframe
```

## Examples

```python
import checkedframe as cf
import polars as pl
from checkedframe.polars import DataFrame


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



df = pl.DataFrame({
    "reason_code": ["abc", "abc", "o9"], 
    "reason_code_description": ["a desc here", "another desc", None],
    "shap": [1, 2, 3],
    "rank": [-1, 2, 1]
})

df: DataFrame[AASchema] = AASchema.validate(df)
```