Checks
======

Custom Checks
-------------

**checkedframe** aims to maximize your control over the checks you write. Checks take 4 types of inputs: a Series, a string, a DataFrame, or nothing. Checks then allow 3 types of outputs: a Series, a boolean, or an Expression. In addition, checks can be "native", which means they are written using the DataFrame library you are using, or they can be written in a DataFrame-agnostic way. Let's take a look at some examples using Polars.

.. code-block:: python

   import checkedframe as cf
   import polars as pl


   class S(cf.Schema):
      customer_id = cf.String()
      checking_balance = cf.Float64()
      savings_balance = cf.Float64()


   df = pl.DataFrame(
      {
         "customer_id": ["TV09", "DTG9"],
         "checking_balance": [400.4, 20.2],
         "savings_balance": [1_000.1, 89.91],
      }
   )

In this example, we have a check that operates on a Series and returns a Series. The ``columns`` argument tells **checkedframe** apply this check to the "checking_balance" Series. Since our return type is a Series, **checkedframe** is able to tell us exactly which rows fail this check as well. In addition, in error reporting, this check will be attached to the "checking_balance" column.


.. tabs::

   .. group-tab:: Polars

      .. code-block:: python

         @cf.Check(columns="checking_balance")
         def series_series_check(s: pl.Series) -> pl.Series:
            return s <= 100

   .. group-tab:: Agnostic

      .. code-block:: python

         @cf.Check(columns="checking_balance")
         def series_series_check(s: cf.Series) -> cf.Series:
            return s <= 100

Similarly, we can use expressions. Expression checks usually take a string and return an expression. Since expressions are series-like, we similarly get detailed error reporting. Furthermore, expressions are run in parallel (if your engine runs expressions in parallel). To get the best performance, use expressions.

.. tabs::

   .. group-tab:: Polars

      .. code-block:: python

         @cf.Check(columns="checking_balance")
         def str_expr_check(name: str) -> pl.Expr:
            return pl.col(name) <= 100

   .. group-tab:: Agnostic

      .. code-block:: python

            @cf.Check(columns="checking_balance")
            def str_expr_check(name: str) -> cf.Expr:
               return pl.col(name) <= 100

We also can operate at the DataFrame level using expressions.

.. tabs::

   .. group-tab:: Polars

      .. code-block:: python

         @cf.Check
         def expr_check() -> pl.Expr:
            return pl.col("checking_balance") <= 100

   .. group-tab:: Agnostic

      .. code-block:: python

         @cf.Check
         def expr_check() -> cf.Expr:
            return cf.col("checking_balance") <= 100

We can also just take the DataFrame in directly.

.. tabs::

   .. group-tab:: Polars

      .. code-block:: python

         @cf.Check
         def df_check(df: pl.DataFrame) -> pl.Series:
            return df["checking_balance"] <= 100

   .. group-tab:: Agnostic

      .. code-block:: python

         @cf.Check
         def df_check(df: cf.DataFrame) -> cf.Series:
            return df["checking_balance"] <= 100


While not particularly useful when operating on a single column, DataFrame-level checks shine when multi-column checks are needed.


.. note::
   The above examples use type hints to indicate the input and return types. A check without type hints would look like this:


   .. tabs::

      .. group-tab:: Polars

         .. code-block:: python

            @cf.Check(columns="checking_balance", input_type="Series", return_type="Series", native=True)
            def series_series_check(s):
               return s <= 100 

      .. group-tab:: Agnostic

         .. code-block:: python

            @cf.Check(columns="checking_balance", input_type="Series", return_type="Series", native=False)
            def series_series_check(s):
               return s <= 100

   In addition, it is possible to omit the *return* type hint or the *return_type* parameter, in which case **checkedframe** will inspect the resulting object to try and infer the return type. However, it is required to type hint the *input* or specify the *input_type* parameter.


Built-in Checks
---------------

**checkedframe** also comes with a large suite of built-in checks. Built-in checks are always agnostic, so you can use them the same way regardless of your DataFrame library.

.. automodule:: checkedframe._checks
   :members: