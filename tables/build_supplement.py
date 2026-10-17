"""Validate aggregate sources and export the fifteen supplementary tables."""
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'tables/generated'
tables = []
for path in sorted(OUT.glob('tables_*.json')):
    tables.extend(json.loads(path.read_text()))
tables.sort(key=lambda table: int(table['id'][1:]))
assert [table['id'] for table in tables] == [f'S{i}' for i in range(1, 16)]
sources, provenance, source_checks = {}, [], 0
for table in tables:
    if table['id'] == 'S10':
        table['columns'] = [column.replace('×1000', '×1,000') for column in table['columns']]
    rows = []
    for row_number, row in enumerate(table['rows'], 1):
        assert len(row) == len(table['columns']), (table['id'], row_number)
        values = []
        for column, cell in zip(table['columns'], row):
            value = cell['text'] if isinstance(cell, dict) else cell
            values.append(value)
            if not re.search(r'\d', value):
                continue
            assert isinstance(cell, dict) and cell.get('sources'), (table['id'], value)
            for source in cell['sources']:
                path = source['path']
                assert path.startswith('results/') and path.endswith('.csv'), path
                if path not in sources:
                    with (ROOT / path).open(newline='') as handle:
                        sources[path] = list(csv.DictReader(handle))
                actual = sources[path][int(source['row']) - 2][source['field']]
                assert actual == source['raw'], (source, actual)
                source_checks += 1
            provenance.append({
                'table': table['id'], 'row': row_number, 'column': column, 'value': value,
                'source path:row': '; '.join(f"{s['path']}:{s['row']}" for s in cell['sources']),
                'source fields': '; '.join(s['field'] for s in cell['sources']),
                'source raw values': json.dumps([s['raw'] for s in cell['sources']], ensure_ascii=False),
            })
        rows.append(values)
    with (OUT / f"{table['id']}.csv").open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(table['columns'])
        writer.writerows(rows)
with (OUT / 'supplement_provenance.csv').open('w', newline='') as handle:
    writer = csv.DictWriter(handle, fieldnames=list(provenance[0]))
    writer.writeheader()
    writer.writerows(provenance)
print(f'Tables: {len(tables)}; numeric cells: {len(provenance)}; exact source-field checks: {source_checks}; orphan values: 0')
