import checkedframe as cf


class HugeSchema(cf.Schema):
    col1 = cf.String()


for i in range(2, 15001):
    setattr(HugeSchema, f"col{i}", cf.Float64())


def bench_parse_into_schema():
    HugeSchema._parse_into_schema()


def bench_columns():
    HugeSchema.columns()
