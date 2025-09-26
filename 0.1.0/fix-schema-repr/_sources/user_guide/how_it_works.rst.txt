How it Works
============

**checkedframe** achieves its DataFrame agnosticism by making **narwhals** do all the hard work. When you call ``Schema.validate``, **checkedframe** first converts your DataFrame into a **narwhals** DataFrame. All of your native datatypes are then mapped to narwhals datatypes. This is how, for example, a checkedframe String (which is a wrapper around a narwhals String) understands Pandas strings, Polars strings, PyArrow strings, and so on.

Similarly, when you use ``cf.col`` to define a column in your schema, you are actually using an alias for ``nw.col``. Where you see ``cf.col`` or ``cf.lit`` in examples, you can usually replace this directly with the corresponding **narwhals** function.

.. note::
    To be exact, checkedframe leverages narwhals.stable.v1. For consistency, you should use the re-exported narwhals functions (or import from narwhals.stable.v1 yourself), as there may be differences between the stable API and the top-level narwhals API.

Briefly, narwhals is a lightweight library that provides a unified API for various dataframe libraries. It acts as a translation layer, converting its own API calls into the native API calls of the underlying dataframe library. For example:

- A narwhals expression like ``nw_df.with_columns(nw.col("a") + 1)`` would be translated to ``pl_df.with_columns(pl.col("a") + 1)`` if you are using Polars.
- If you are using Pandas, the same expression would effectively become ``pd_df["a"] = pd_df["a"] + 1``.

This powerful abstraction allows you to write code once that works seamlessly across different DataFrame implementations, and it's the core reason why **checkedframe** can support multiple backends. You can read more about **narwhals** here: https://narwhals-dev.github.io/narwhals/. 
