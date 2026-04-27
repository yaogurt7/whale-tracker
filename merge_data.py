import duckdb
import os

DATA_PATH = "data/filtered_4_ml/*/*.csv"
OUTPUT_FILE = "data/master_trades.parquet"

print("Merging date-based CSVs into a single Parquet file")

con = duckdb.connect()

# market_id: Extracts the folder name.
# trade_date: Extracts the YYYY-MM-DD from the filename.
con.execute(f"""
    COPY (
        SELECT 
            *, 
            regexp_extract(filename, 'filtered_4_ml/([^/]+)', 1) AS market_id,
            regexp_extract(filename, '([^/]+)\.csv$', 1) AS trade_date
        FROM read_csv_auto('{DATA_PATH}', filename=True, union_by_name=True)
    ) TO '{OUTPUT_FILE}' (FORMAT PARQUET);
""")

print(f"Master file created at: {OUTPUT_FILE}")