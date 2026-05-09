import pandas as pd

pd.set_option('display.max_columns', None)
data = pd.read_parquet("data/master_trades.parquet")
print(data.head())