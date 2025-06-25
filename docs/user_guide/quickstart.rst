Quickstart
==========

Install the *checkedframe* library from PyPI:

.. code-block:: shell

    pip install checkedframe


Let's say you have a very simple DataFrame containing a customer id and their checking and savings account balance. **checkedframe** is agnostic to your DataFrame library--validation, built-in checks, etc. work with every library supported by **narwhals**. For the sake of this particular example, let's use Polars.

.. code-block:: python

    import polars as pl

    df = pl.DataFrame({
        "customer_id": ["TYX890", "ABC98"],
        "checking_balance": [198.28, None],
        "savings_balance": [100, None]
    })


Let's start by defining a very basic schema. 

.. code-block:: python

    import checkedframe as cf

    class MySchema(cf.Schema):
        customer_id = cf.String()
        checking_balance = cf.Float64()
        savings_balance = cf.Float64()


This schema represents a DataFrame with a single string column and two float columns. Schemas are typically created by inheriting from the ``Schema`` class. Already, we can catch some errors.

.. code-block:: python

    MySchema.validate(df)


Output:

::

    SchemaError: Found 3 error(s)
      checking_balance: 1 error(s)
        - NullValueCheck failed for 1 / 2 rows (50.00%): Nulls found in non-nullable column
      savings_balance: 2 error(s)
        - Expected Float64, got Int64
        - NullValueCheck failed for 1 / 2 rows (50.00%): Nulls found in non-nullable column


By default, **checkedframe** assumes that all columns are non-nullable, so it tells us we have an errant null in ``checking_balance``. We also declared ``savings_balance`` to be a Float64, so it complains that we got an Int64. Now, these behaviors may not necessarily be desireable in this case. It probably makes sense that balances can be null. In addition, we probably don't want to be so strict in the data types we accept--as long as it is castable to float. Let's edit our schema to reflect this.


.. code-block:: python

    class MySchema(cf.Schema):
        customer_id = cf.String()
        checking_balance = cf.Float64(nullable=True, cast=True)
        savings_balance = cf.Float64(nullable=True, cast=True)


Now our DataFrame passes validation! **checkedframe** was able to inspect ``savings_column`` and determine that it could be safely cast to our desired data type, Float64.

Beyond these core concepts, you may also want to run arbitrary checks against your DataFrame. This is handled by the ``Check`` class. Checks are user-defined (or built-in) functions that assert some property of your custom Schema class. For example, let's say ``customer_id`` is always 6 characters. There are actually a couple of ways to do this, but for now let's assume you want to write your own function (perhaps your real check is much more complex).

.. code-block:: python

    class MySchema(cf.Schema):
        customer_id = cf.String()
        checking_balance = cf.Float64(nullable=True, cast=True)
        savings_balance = cf.Float64(nullable=True, cast=True)

        @cf.Check(columns="customer_id")
        def check_id_length(s: pl.Series) -> pl.Series:
            """customer_id must be of length 6"""
            return s.str.len_bytes() == 6


Output:

::

    SchemaError: Found 1 error(s)
      customer_id: 1 error(s)
        - check_id_length failed: customer_id must be of length 6


Here, we wrote a custom function that we identified as a check with the ``Check`` decorator. In addition, via the ``column`` argument, we attached the check to the ``customer_id`` column.

Now, let's re-visit our assumption that balances can be nullable. This may be true for each individual balance (savings account but no checking account or vice versa), but it may not make sense for both to be null at the same time. Let's write a check for this.

.. code-block:: python

    class MySchema(cf.Schema):
        customer_id = cf.String()
        checking_balance = cf.Float64(nullable=True, cast=True)
        savings_balance = cf.Float64(nullable=True, cast=True)

        @cf.Check(columns="customer_id")
        def check_id_length(s: pl.Series) -> pl.Series:
            """customer_id must be of length 6"""
            return s.str.len_bytes() == 6

        @cf.Check
        def check_both_balances_not_null() -> pl.Expr:
            """either checkings or savings must be present"""
            return (
                pl.col("checking_balance")
                .is_not_null()
                .or_(pl.col("savings_balance").is_not_null())
            )

Output:

::
    SchemaError: Found 2 error(s)
      customer_id: 1 error(s)
        - check_id_length failed for 1 / 2 rows (50.00%): customer_id must be of length 6
      * check_both_balances_not_null failed for 1 / 2 rows (50.00%): either checkings or savings must be present


Here, we wrote a multi-column check that returned an Expr instead of a Series. In the Checks section, we will go over all the ways to write a Check.