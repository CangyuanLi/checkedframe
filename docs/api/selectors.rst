Selectors
=========

Inspired (or rather directly lifted from) Polars, selectors are a convenient way to interact with the schema.

Importing
=========
=========

Selectors are available from the top-level namespace

.. code-block:: python

   import checkedframe as cf

   cf.selectors.all()

For a shorter name, you can use, by convention

.. code-block:: python

   import checkedframe.selectors as cfs

   cfs.all()


Usage
=====
=====

Selectors support set operations and resolve to a list of column names. For example, given the below schema,

.. code-block:: python

   import checkedframe as cf
   import checkedframe.selectors as cfs

   
   class S(cf.Schema):
      float1 = cf.Float64()
      float2 = cf.Float64()
      string1 = cf.String()

   
+----------------------+---------------------------------+---------------------------------+
| Operation            | Expression                      | Result                          |
+======================+=================================+=================================+
| UNION                | cfs.float() | cfs.string()      | ["float1", "float2", "string1"] |
+----------------------+---------------------------------+---------------------------------+
| INTERSECTION         | cfs.float() & cfs.contains("1") | ["float1"]                      |
+----------------------+---------------------------------+---------------------------------+
| DIFFERENCE           | cfs.float() - cfs.contains("1") | ["float2"]                      |
+----------------------+---------------------------------+---------------------------------+
| SYMMETRIC DIFFERENCE | cfs.float() ^ cfs.contains("1") | ["float2", "string1"]           |
+----------------------+---------------------------------+---------------------------------+
| COMPLEMENT           | ~cfs.float()                    | ["string1"]                     |
+----------------------+---------------------------------+---------------------------------+

.. note::
   Selectors operate on the *schema*, not the DataFrame.


.. automodule:: checkedframe.selectors
   :members: