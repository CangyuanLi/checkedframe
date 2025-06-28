from __future__ import annotations

import keyword
from typing import Optional

import narwhals.stable.v1 as nw
import narwhals.stable.v1.selectors as nws
import narwhals.stable.v1.typing as nwt

from ._dtypes import _nw_type_to_cf_type

INF = float("inf")
NEG_INF = float("-inf")


def generate_schema_repr(
    df: nwt.IntoDataFrame,
    class_name: str = "MySchema",
    header: Optional[str] = "import checkedframe as cf",
    import_alias: str = "cf.",
) -> str:
    nw_df = nw.from_native(df, eager_only=True)

    null_df = nw_df.select(nws.all().is_null().any())

    float_selector = nws.by_dtype(nw.Float32, nw.Float64)
    nan_df = nw_df.select(float_selector.is_nan().any())
    inf_df = nw_df.select(float_selector.is_in((INF, NEG_INF)).any())
    float_cols = set(nan_df.columns)

    # Build string representation of schema
    columns = []
    i = 0
    for col, nw_dtype in nw_df.schema.items():
        column_kwargs: dict[str, bool | str] = {}
        if not col.isidentifier() or keyword.iskeyword(col):
            column_kwargs["name"] = f'"{col}"'

            sanitized_col = f"column_{i}"
            i += 1
        else:
            sanitized_col = col

        if null_df[col].item():
            column_kwargs["nullable"] = True

        if col in float_cols:
            if nan_df[col].item():
                column_kwargs["allow_nan"] = True

            if inf_df[col].item():
                column_kwargs["allow_inf"] = True

        cf_dtype = _nw_type_to_cf_type(nw_dtype, **column_kwargs)

        kwargs_to_show = []
        for k, v in column_kwargs.items():
            kwargs_to_show.append(f"{k}={v}")

        display_kwargs = ", ".join(kwargs_to_show)
        display_dtype = str(cf_dtype).replace("(", f"({import_alias}")

        columns.append(
            f"    {sanitized_col} = {import_alias}{display_dtype}({display_kwargs})".replace(
                ")(", ", " if len(kwargs_to_show) > 0 else ""
            )
        )

    if header is not None:
        header = f"{header}\n\n"
    else:
        header = ""

    col_repr = "\n".join(columns)

    return f"{header}class {class_name}({import_alias}Schema):\n{col_repr}"
