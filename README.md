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

**checkedframe** is a lightweight and flexible library for DataFrame validation built on top of **narwhals**. This means it has first-class support for both **narwhals** itself and all the engines that **narwhals** supports (primarily Pandas, Polars, cuDF, Modin, and PyArrow). 

## Why use checkedframe?

The key advantages of **checkedframe** are DataFrame agnosticism (validate Pandas, Polars, Modin, etc. with a single unified API), separation from the **pydantic** ecosystem (which is fantastic but not suited for columnar data and relies too heavily on brittle type annotation magic), and a flexible, intuitive API for user-defined functions. Below is a (subjective) comparison of **checkedframe** and several other popular DataFrame validation libraries. If there are any errors / you want your library to be added, please send a PR!

|                            | [checkedframe](https://github.com/cangyuanli/checkedframe) | [pandera](https://pandera.readthedocs.io/) | [patito](https://patito.readthedocs.io/) | [dataframely](https://dataframely.readthedocs.io/en/latest/index.html) | [great-expectations](https://docs.greatexpectations.io) | [pointblank](https://posit-dev.github.io/pointblank/) |
| -------------------------- | ---------------------------------------------------------- | ------------------------------------------ | ---------------------------------------- | ---------------------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------- |
| **DataFrame agnostic**     | ✅                                                          | 🟡 (1.)                                     | ❌ (polars-only)                          | ❌ (polars-only)                                                        | ❌ (pandas < 2.2-only)                                   | ✅                                                     |
| **Lightweight**            | ✅                                                          | ❌ (pydantic)                               | ❌ (pydantic)                             | ✅                                                                      | ❌                                                       | 🟡                                                     |
| **Custom checks**          | ✅                                                          | 🟡 (2.)                                     | ❌                                        | 🟡 (3.)                                                                 | ❌                                                       | 🟡                                                     |
| **Static typing**          | 🟡                                                          | ✅                                          | ✅                                        | ✅                                                                      | ❌                                                       | ❌                                                     |
| **Nested types**           | ✅                                                          | ✅                                          | ✅                                        | ✅                                                                      | ❌                                                       | ✅                                                     |
| **Safe casting**           | ✅                                                          | ✅                                          | ❌                                        | 🟡 (4.)                                                                 | ❌                                                       | ❌                                                     |
| **Filtering**              | ✅                                                          | ❌                                          | ❌                                        | ✅                                                                      | ❌                                                       | ❌                                                     |
| **Schema generation**      | ✅                                                          | ❌                                          | ❌                                        | ❌                                                                      | ❌                                                       | ❌                                                     |
| **Union types**            | 🟡                                                          | ❌                                          | ❌                                        | ❌                                                                      | ❌                                                       | ❌                                                     |
| **Python version support** | ✅ (3.9+)                                                   | 🟡 (<= 3.12)                                | ✅                                        | ❌ (3.11+)                                                              | ✅                                                       | 🟡  (3.10+)                                            |
| **Battle-tested**          | ❌ (You can help!)                                          | ✅                                          | 🟡                                        | 🟡                                                                      | ✅                                                       | 🟡                                                     |

- ✅ = Fully supported  
- 🟡 = Partial/limited support  
- ❌ = Not supported  

1. While **pandera** does support multiple libraries, it requires code changes to switch between them. Feature completeness also varies across different engines.
2. This is quite subjective, but I find writing non-trivial checks (e.g. those requiring multiple columns, group-by, etc.) non-intuitive and difficult
3. Checks must return an expr, which hampers boolean checks, such as a t-test between two columns
4. Either all columns are cast or none are


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

    @cf.Check(columns="shap")
    def check_shap_is_reasonable() -> pl.Expr:
        """Shap values must be reasonable"""
        return pl.col("shap").lt(5).and_(pl.col("shap").gt(0.01))

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
  reason_code: 1 error(s)
    - check_reason_code_length failed for 1 / 3 (33.33%) rows: Reason codes must be exactly 3 chars
  features: 1 error(s)
    - Column marked as required but not found
  rank: 1 error(s)
    - Cannot safely cast Int64 to UInt8; 1 / 3 (33.33%) rows outside of expected range [0, 255]
  * check_row_height failed for 3 / 3 (100.00%) rows: DataFrame must have 2 rows
  * is_id failed for 3 / 3 (100.00%) rows: reason_code must uniquely identify the DataFrame
```

Let's walk through the code step by step. We declare a schema (note that we inherit from `cf.Schema`) that represents a dataframe with 5 columns called `reason_code`, `reason_code_description`, `features`, `shap`, and `rank`. We declare the data type of each column, e.g. `String`, `Float64`, and so on. In addition, we declare certain properties about the columns. For example, we are OK with nulls in `reason_code_description` (by default, columns are not assumed to be nullable), so we set `nullable=True`. For `shap` and `rank`, we expect the specified data type but don't error if the column is not exactly that data type. Instead, since `cast=True`, we try to (safely) cast the column to the specified data type if possible. 

Next, we use checks to assert different properties about our data. For example, we expect that all reason codes are exactly 3 characters long. Note the flexibility in how we perform checks. In the first example, we operate on the series. In the second example, we use expressions. In the third, we operate on the dataframe. In the fourth, we also operate on the dataframe but use a built-in check for convenience. All of these constructs are perfectly valid, with no need to switch between different decorators or remember complex arguments. In this example, the inputs and outputs of the checks are automatically determined from the type hints, but they can also be specified manually in case this fails.

```python
  @cf.Check(columns="reason_code", input_type="Series", return_type="Series")
  def check_reason_code_length(s):
      """Reason codes must be exactly 3 chars"""
      return s.str.len_bytes() == 3
```

Finally, when calling `AASchema.validate` on our bad data, we get a nice error message, including clear descriptions of why casting failed, why checks failed (and for what number of rows, if applicable), and so on.

For more advanced usage, please see the [documentation]("https://cangyuanli.github.io/checkedframe/").

### Mypy Plugin

The example code as-is will actually throw some type errors, as type checkers will complain that the user-defined checks do not take a "self" parameter. This is because there is currently no way to mark a function as a `staticmethod` without using the `staticmethod` decorator. You can simply add this decorator to make the errors go away. If that's annoying, **checkedframe** also provides a **mypy** plugin that marks all methods decorated with `cf.Check` as staticmethods. Just add

```
[tool.mypy]
plugins = ["checkedframe.mypy"]
```

to your pyproject.toml. Unfortunately, no other type checker provides plugin capabilities.

## Typing

**checkedframe** is also meant to integrate with static typing. When validation is successful, the returned dataframe can be parametrized by the schema. For example,

```python
import checkedframe as cf
import polars as pl
from checkedframe.polars import DataFrame


class MySchema(cf.Schema):
    x = cf.String()


df = pl.DataFrame({
    "x": ["a", "b", "c"]
})

def func_that_requires_cleaned_data(df: DataFrame[MySchema]): ...

func_that_requires_cleaned_data(df)  # type error

validated_df: DataFrame[MySchema] = MySchema.validate(df)
func_that_requires_cleaned_data(validated_df)  # passes!
```

# Roadmap:

1. Better static typing. `MySchema.validate` should automatically return a DataFrame of your input type parametrized by `MySchema`.