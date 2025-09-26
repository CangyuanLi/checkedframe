Checks
======

Custom Checks
-------------

**checkedframe** aims to maximize your control over the checks you write. Checks take 4 types of inputs: a Series, a string, a DataFrame, or nothing. Checks then allow 3 types of outputs: a Series, a boolean, or an Expression. In addition, checks can be "native", which means they are written using the DataFrame library you are using, or they can be written in a DataFrame-agnostic way using **narwhals**. Let's take a look at some examples.

.. code-block:: python

   import checkedframe as cf
   import polars as pl


   class S(cf.Schema):
      customer_id = cf.String()
      checking_balance = cf.Float64()
      savings_balance = cf.Float64()


   df = pl.DataFrame({
      "x": 
   })

In this example, we have a check that operates on a Series and returns a Series.

.. code-block:: python

   @cf.Check(columns="checking_balance")
   def series_series_check(s: pl.Series) -> pl.Series:
      return s <= 100

The ``columns`` argument tells **checkedframe** apply this check to the "checking_balance" Series. Since our return type is a Series, **checkedframe** is able to tell us exactly which rows fail this check as well.


.. code-block:: python

   @cf.Check(columns="checking_balance")
   def str_expr_check(name: str) -> pl.Expr:
      return pl.col(name) <= 100


.. code-block:: python

   @cf.Check
   def expr_check() -> pl.Expr:
      return pl.col("checking_balance") <= 100


.. code-block:: python

   @cf.Check
   def df_check(df: pl.DataFrame) -> pl.Series:
      return df["checking_balance"] <= 100


Functions
---------

.. automodule:: checkedframe._checks
   :members: