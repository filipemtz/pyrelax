from pyrelax.engine import execute_query
from src.pyrelax.data import load_databases

databases = load_databases("data/ufes.txt")
tables = databases["UFES - Banking database"]
print(execute_query("pi customer_name borrower"))
