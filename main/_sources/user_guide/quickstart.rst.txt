Quickstart
==========

Install the *checkedframe* library from PyPI:

.. code-block:: shell

    pip install checkedframe


**checkedframe** is agnostic to your DataFrame library. This means that validation, built-in checks, etc. work with every library supported by **narwhals**, primarily Pandas, Polars, PyArrow, cuDF, and Modin. At the same time, it allows you to write user-defined checks using whatever engine you prefer. In the example below, there will be tabs for Pandas and Polars, as well as an "agnostic" tab that has a completely engine-agnostic schema.

Let's say you have a very simple DataFrame containing a customer id, their balances, and whether or not they have overdraft protection.


.. tabs::
    
    .. group-tab:: Pandas

        .. code-block:: python

            import pandas as pd

            df = pd.DataFrame(
                {
                    "customer_id": ["TYX89", "F38J0M"],
                    "balance": [198, -56],
                    "overdraft_protection": [True, False],
                }
            )    

    .. group-tab:: Polars

        .. code-block:: python

            import polars as pl

            df = pl.DataFrame(
                {
                    "customer_id": ["TYX89", "F38J0M"],
                    "balance": [198, -56],
                    "overdraft_protection": [True, False],
                }
            )



Let's start by defining a very basic schema. 

.. code-block:: python

    import checkedframe as cf


    class MySchema(cf.Schema):
        customer_id = cf.String()
        balance = cf.Float64()
        overdraft_protection = cf.Boolean()


This schema represents a DataFrame with a single string column and two float columns. Schemas are typically created by inheriting from the ``Schema`` class. Already, we can catch some errors.

.. code-block:: python

    MySchema.validate(df)

Output:

.. code-block:: text

    Found 1 error(s)
      balance: 1 error(s)
        - Expected Float64, got Int64

**checkedframe** complains because we declared ``balance`` to be a Float64, but we got an Int64. We could fix this ourselves, or we could tell **checkedframe** to. Casting in **checkedframe** comes with some extra safety (casting must be value-preserving) and convenience (ablity to pinpoint which rows fail). Let's edit our schema to reflect this.


.. code-block:: python

    class MySchema(cf.Schema):
        customer_id = cf.String()
        balance = cf.Float64(cast=True)
        overdraft_protection = cf.Boolean()


Now our DataFrame passes validation! **checkedframe** was able to inspect ``balance`` and safely cast to our desired data type, Float64.

.. tabs::

    .. group-tab:: Pandas

        .. code-block:: text

              customer_id  balance  overdraft_protection
            0       TYX89    198.0                  True
            1      F38J0M    -56.0                 False

        
    .. group-tab:: Polars

        .. code-block:: text

            shape: (2, 3)
            ┌─────────────┬─────────┬──────────────────────┐
            │ customer_id ┆ balance ┆ overdraft_protection │
            │ ---         ┆ ---     ┆ ---                  │
            │ str         ┆ f64     ┆ bool                 │
            ╞═════════════╪═════════╪══════════════════════╡
            │ TYX89       ┆ 198.0   ┆ true                 │
            │ F38J0M      ┆ -56.0   ┆ false                │
            └─────────────┴─────────┴──────────────────────┘


Beyond these core concepts, you may also want to run arbitrary checks against your DataFrame. This is handled by the ``Check`` class. Checks are user-defined (or built-in) functions that assert some property of your custom Schema class. For example, let's say ``customer_id`` is always 6 characters. There are actually a couple of ways to do this, but for now let's assume you want to write your own function (perhaps your real check is much more complex). Let's add the below function to our schema class.

.. tabs::

    .. group-tab:: Pandas

        .. code-block:: python
        
            class MySchema(cf.Schema):
                customer_id = cf.String()
                balance = cf.Float64(cast=True)
                overdraft_protection = cf.Boolean()

                @cf.Check(columns="customer_id")
                def check_id_length(s: pd.Series) -> pd.Series:
                    """customer_id must be of length 6"""
                    return s.str.len() == 6

    .. group-tab:: Polars

        .. code-block:: python

            class MySchema(cf.Schema):
                customer_id = cf.String()
                balance = cf.Float64(cast=True)
                overdraft_protection = cf.Boolean()

                @cf.Check(columns="customer_id")
                def check_id_length(name: str) -> pl.Expr:
                    """customer_id must be of length 6"""
                    return pl.col(name).str.len_chars() == 6

    .. group-tab:: Agnostic

        .. code-block:: python

            class MySchema(cf.Schema):
                customer_id = cf.String()
                balance = cf.Float64(cast=True)
                overdraft_protection = cf.Boolean()

                @cf.Check(columns="customer_id")
                def check_id_length(name: str) -> cf.Expr:
                    """customer_id must be of length 6"""
                    return cf.col(name).str.len_chars() == 6


.. note::
    Activate the mypy plugin to allow mypy to recognize ``Check`` as a staticmethod.

    .. code-block:: text

        [tool.mypy]
        plugins = ["checkedframe.mypy"]

    Alternatively, manually add ``@staticmethod``. This doesn't do anything but silence the type checker.

    .. code-block:: python

        @cf.Check
        @staticmethod
        def my_check(): ...

Let's try validating again.

.. code-block:: python

    MySchema.validate(df)

Output:

.. code-block:: text

    SchemaError: Found 1 error(s)
      customer_id: 1 error(s)
        - check_id_length failed: customer_id must be of length 6


Here, we wrote a custom function that we identified as a check with the ``Check`` decorator. In addition, via the ``column`` argument, we attached the check to the ``customer_id`` column.

Now, let's re-visit ``balances``. Notice that we have a negative here. It may make sense to have negative balances (in case of overdraft), but if overdraft protection is on, the transaction simply wouldn't go through, meaning that it isn't possible to have a negative balance. Let's write a check for this.


.. tabs::

    .. group-tab:: Pandas

        .. code-block:: python

            @cf.Check
            def check_balances_pos_if_protected(df: pd.DataFrame) -> pd.Series:
                """Balances can only be negative if there is no overdraft protection"""
                return (df["balance"] >= 0) & df["overdraft_protection"]

    .. group-tab:: Polars

        .. code-block:: python

            @cf.Check
            def check_balances_pos_if_protected() -> pl.Expr:
                """Balances can only be negative if there is no overdraft protection"""
                return (pl.col("balance") >= 0) & pl.col("overdraft_protection")

    .. group-tab:: Agnostic

        .. code-block:: python

            @cf.Check
            def check_balances_pos_if_protected() -> cf.Expr:
                """Balances can only be negative if there is no overdraft protection"""
                return (cf.col("balance") >= 0) & cf.col("overdraft_protection")

Output:

.. code-block:: text

    Found 2 error(s)
      customer_id: 1 error(s)
        - check_id_length failed for 1 / 2 (50.00%) rows: customer_id must be of length 6
      * check_balances_pos_if_protected failed for 1 / 2 (50.00%) rows: Balances can only be negative if there is no overdraft protection


.. note::
    **checkedframe** will not stop on the first error; rather, it will try to find all errors before raising.


You've created your first schema, but there is a lot more to **checkedframe**, including filtering (returning only rows that pass validation), detailed error inspection, static type-checking, and more. See the relevant sections in the User Guide and API Reference for more information.