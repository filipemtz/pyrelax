import duckdb
import pandas as pd
import streamlit as st

from pyrelax.data import databases
from pyrelax.engine import execute_query


def execute_sql(query: str, tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Execute a SQL query over the supplied pandas DataFrames."""
    with duckdb.connect(database=":memory:") as db_conn:
        for name, df in tables.items():
            db_conn.register(name, df)

        return db_conn.sql(query).df()


def display_result(result):
    """Display a query result."""
    if not isinstance(result, pd.DataFrame):
        st.error(
            "The interpreter returned an object that is not " "a pandas DataFrame."
        )
        return

    st.write(f"{len(result)} items returned.")

    st.dataframe(
        result,
        width="stretch",
        hide_index=True,
    )


def main():
    st.set_page_config(
        page_title="Relational Algebra Interpreter",
        page_icon="",
        layout="wide",
    )

    st.title("Relational Algebra Interpreter")

    st.write(
        "Write a relational algebra or SQL expression and execute it "
        "over the database shown below."
    )

    # ------------------------------------------------------------------
    # Database selection
    # ------------------------------------------------------------------

    names = sorted(databases.keys())
    default = names.index("Mutz - University")

    database = st.selectbox(
        "Select a database",
        names,
        index=default,
    )

    tables = databases[database]

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    relational_tab, sql_tab = st.tabs(["Relational Algebra", "SQL"])

    with relational_tab:
        st.subheader("Relational algebra expression")

        relational_query = st.text_area(
            "Relational algebra query",
            height=120,
            placeholder="Write a relational algebra expression here...",
            label_visibility="collapsed",
            key="relational_query",
        )

        if st.button(
            "Execute",
            type="primary",
            key="execute_relational",
        ):
            if not relational_query.strip():
                st.warning("Enter a relational algebra expression.")
            else:
                try:
                    result = execute_query(
                        relational_query,
                        tables,
                    )

                    st.subheader("Result")
                    display_result(result)

                except Exception as e:
                    print("Relational algebra error:", str(e))
                    st.subheader("Error")
                    st.error(str(e))

    with sql_tab:
        st.subheader("SQL query")

        sql_query = st.text_area(
            "SQL query",
            height=120,
            placeholder="SELECT * FROM Cliente;",
            label_visibility="collapsed",
            key="sql_query",
        )

        if st.button(
            "Execute",
            type="primary",
            key="execute_sql",
        ):
            if not sql_query.strip():
                st.warning("Enter a SQL query.")
            else:
                try:
                    result = execute_sql(
                        sql_query,
                        tables,
                    )

                    st.subheader("Result")
                    display_result(result)

                except Exception as e:
                    print("SQL error:", str(e))
                    st.subheader("Error")
                    st.error(str(e))

    # ------------------------------------------------------------------
    # Database
    # ------------------------------------------------------------------

    st.subheader("Database")

    for table_name, table in tables.items():
        st.write(f"**{table_name}**")

        st.dataframe(
            table,
            width="stretch",
            hide_index=True,
        )


if __name__ == "__main__":
    main()
