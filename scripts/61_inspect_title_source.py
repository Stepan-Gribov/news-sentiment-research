import json
import pathlib
import sys
sys.path.insert(0, "./news_data/company_mapping_runs/run_20260930_01/python_packages_20260930")
import pyarrow.parquet as pq
path = pathlib.Path("./news_data/company_mapping_runs/run_20260930_01/articles_all_from_news_sqlite.parquet")
file = pq.ParquetFile(path)
print(json.dumps(dict(path=str(path), rows=file.metadata.num_rows, columns=file.schema_arrow.names, size=path.stat().st_size)))
print(json.dumps(file.read_row_group(0, columns=["media", "title", "link", "date"]).slice(0, 2).to_pylist(), default=str, ensure_ascii=True))
