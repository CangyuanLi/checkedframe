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

**checkedframe** is a lightweight library for DataFrame validation built on top of **narwhals**. This means it has first-class support for all the engines that **narwhals** supports (primarily Pandas, Polars, cuDF, Modin, and PyArrow). 

## Why use checkedframe?


|                        | [checkedframe](https://github.com/cangyuanli/checkedframe) | [pandera](https://pandera.readthedocs.io/) | [patito](https://patito.readthedocs.io/) | [dataframely](https://github.com/...) |
| ---------------------- | ---------------------------------------------------------- | ------------------------------------------ | ---------------------------------------- | ------------------------------------- |
| **DataFrame agnostic** | ✅                                                          | 🟡 (1.)                                     | 🟡 (polars + duckdb-only)                 | ❌ (polars-only)                       |
| **Lightweight**        | ✅                                                          | ❌ (pydantic)                               | ❌ (pydantic)                             | ✅                                     |
| **Custom checks**      | ✅ (2.)                                                     | ✅                                          | ✅                                        | 🟡                                     |
| **Nested types**       | ✅ (2.)                                                     | ✅                                          | ✅                                        | 🟡                                     |
| **Battle-tested**      | ❌ (You can help!)                                          | ✅                                          | 🟡                                        | ✅                                     |

- ✅ = Fully supported  
- 🟡 = Partial/limited support  
- ❌ = Not supported  

1. While **pandera** does support multiple libraries, it requires code changes to switch between them. Feature completeness also varies across different engines.
2. 2. 

### Notes:  

- *DataFrameLy appears to be a less-maintained/niche library (couldn’t find official docs).  

Would you like me to refine any features or add more libraries (e.g., `great-expectations`)?

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
    reason_code = cf.String()
    reason_code_description = cf.String(nullable=True)
    features = cf.List(cf.String)
    shap = cf.Float64(cast=True)
    rank = cf.UInt8(cast=True)

    @cf.Check(columns="reason_code")
    def check_reason_code_length(s: pl.Series) -> pl.Series:
        """Reason codes must be exactly 3 chars"""
        return s.str.len_bytes() == 3

    @cf.Check
    def check_row_height(df: pl.DataFrame) -> bool:
        """DataFrame must have 2 rows"""
        return df.height == 2

    _id_check = cf.Check.is_id("reason_code")


df = pl.DataFrame(
    {
        "reason_code": ["R23", "R23", "R9"],
        "reason_code_description": ["Credit score too low", "Income too low", None],
        "shap": [1, 2, 3],
        "rank": [-1, 2, 1],
    }
)

df: DataFrame[AASchema] = AASchema.validate(df)
```

```
checkedframe.exceptions.SchemaError: Found 5 error(s)
  features: 1 error(s)
    - Column marked as required but not found
  rank: 1 error(s)
    - Cannot safely cast Int64 to UInt8; invalid range [-1, 2], expected range [0, 255]
  reason_code: 1 error(s)
    - check_reason_code_length failed for 1 / 3 rows (33.33%): Reason codes must be exactly 3 chars
  * is_id failed: reason_code must uniquely identify the DataFrame
  * check_row_height failed: DataFrame must have 2 rows
```