from glob import glob

from src.pyrelax.data import databases

for db, tables in databases.items():
    print(db)

    for table, content in tables.items():

        print(f"\t{table}")
        for column in content.columns:
            print(f"\t\t{column}")

    print()
