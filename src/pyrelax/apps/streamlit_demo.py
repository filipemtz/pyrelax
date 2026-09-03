import pandas as pd
import streamlit as st

from pyrelax.data import databases
from pyrelax.engine import execute_query


def main():
    tables = databases["Mutz - University"]

    st.set_page_config(
        page_title="Relational Algebra Interpreter",
        page_icon="",
        layout="wide",
    )

    st.title("Relational Algebra Interpreter")

    st.write(
        "Write a relational algebra expression and execute it over the "
        "database shown below."
    )

    st.subheader("Relational algebra expression")

    query = st.text_area(
        "Query",
        height=120,
        placeholder="Write a relational algebra expression here...",
        label_visibility="collapsed",
    )

    if st.button("Execute", type="primary"):
        if not query.strip():
            st.warning("Enter a relational algebra expression.")
        else:
            try:
                result = execute_query(query, tables)

                st.subheader("Result")

                if not isinstance(result, pd.DataFrame):
                    st.error(
                        "The interpreter returned an object that is not "
                        "a pandas DataFrame."
                    )
                else:
                    st.dataframe(
                        result,
                        width="stretch",
                        hide_index=True,
                    )

            except Exception as e:
                print("Error:", str(e))
                st.subheader("Error")
                st.error(str(e))

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
