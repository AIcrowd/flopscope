from scripts.cost_sheet.schema import CostRow, COLUMNS, LEGEND

def test_schema_has_20_columns_matching_dataclass():
    import dataclasses
    fields = [f.name for f in dataclasses.fields(CostRow)]
    assert len(fields) == 20
    assert COLUMNS == fields              # CSV header order == field order
    assert set(LEGEND) == set(fields)     # every column documented
