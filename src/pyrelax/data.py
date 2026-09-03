import re
from glob import glob
from importlib.resources import files
from pathlib import Path

import pandas as pd


def load_databases(path):
    """
    Load databases from a file in the database format.

    Returns
    -------
    dict
        {
            database_name: {
                "description": str,
                "courtesy": str,
                "category": str,
                "tables": {
                    table_name: pandas.DataFrame,
                    ...
                }
            },
            ...
        }
    """

    text = Path(path).read_text(encoding="utf-8")

    # Remove C-style block comments:
    #     /* comment */
    #
    # DOTALL is required because comments may span multiple lines.
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.DOTALL)

    # Split into database blocks.
    blocks = re.split(
        r"(?=^\s*group\s*:)",
        text,
        flags=re.MULTILINE,
    )

    databases = {}

    for block in blocks:
        block = block.strip()

        if not block:
            continue

        # ------------------------------------------------------------
        # Database metadata
        # ------------------------------------------------------------

        group_match = re.search(
            r"^\s*group\s*:\s*(.*?)\s*$",
            block,
            flags=re.MULTILINE,
        )

        if not group_match:
            continue

        database_name = group_match.group(1).strip()

        description_match = re.search(
            r"^\s*description\s*:\s*(.*?)\s*$",
            block,
            flags=re.MULTILINE,
        )

        category_match = re.search(
            r"^\s*category@en\s*:\s*(.*?)\s*$",
            block,
            flags=re.MULTILINE,
        )

        description = description_match.group(1).strip() if description_match else ""

        category = category_match.group(1).strip() if category_match else ""

        # Extract "Courtesy: ..." from the description.
        courtesy_match = re.search(
            r"\bCourtesy\s*:\s*(.*?)(?=\s+Credits\s*:|$)",
            description,
            flags=re.IGNORECASE,
        )

        courtesy = courtesy_match.group(1).strip() if courtesy_match else ""

        # ------------------------------------------------------------
        # Tables
        # ------------------------------------------------------------

        tables = {}

        table_pattern = re.compile(
            r"""
            (?P<name>[A-Za-z_][A-Za-z0-9_]*)
            \s*=\s*\{
            (?P<body>.*?)
            \}
            """,
            flags=re.DOTALL | re.VERBOSE,
        )

        for table_match in table_pattern.finditer(block):
            table_name = table_match.group("name")
            body = table_match.group("body")

            lines = [line.strip() for line in body.splitlines() if line.strip()]

            if not lines:
                continue

            # --------------------------------------------------------
            # Header
            # --------------------------------------------------------

            schema = lines[0]

            columns = []

            for column in schema.split(","):
                column = column.strip()

                if not column:
                    continue

                # column_name:type
                column_name = column.split(":", 1)[0].strip()

                if column_name:
                    columns.append(column_name)

            # --------------------------------------------------------
            # Rows
            # --------------------------------------------------------

            rows = []

            for line in lines[1:]:
                line = line.strip()

                if not line:
                    continue

                line = line.rstrip(",")

                values = _split_values(line)

                if not values:
                    continue

                values = [_parse_value(value) for value in values]

                if len(values) != len(columns):
                    raise ValueError(
                        f"Invalid row in table '{table_name}': "
                        f"expected {len(columns)} values, "
                        f"got {len(values)}\n{line}"
                    )

                rows.append(values)

            tables[table_name] = pd.DataFrame(
                rows,
                columns=columns,
            )

        databases[database_name] = {
            "description": description,
            "courtesy": courtesy,
            "category": category,
            "tables": tables,
        }

    return databases


def _split_values(line):
    """
    Split a row on commas while ignoring commas inside
    single-quoted or double-quoted strings.
    """

    values = []
    current = []
    quote = None

    for char in line:
        if char in ("'", '"'):
            if quote is None:
                quote = char
            elif quote == char:
                quote = None

            current.append(char)

        elif char == "," and quote is None:
            values.append("".join(current).strip())
            current = []

        else:
            current.append(char)

    if current:
        values.append("".join(current).strip())

    return values


def _parse_value(value):
    """
    Convert a value from the database format into a Python value.
    """

    value = value.strip()

    # NULL
    if value.upper() == "NULL":
        return None

    # Quoted string
    if len(value) >= 2 and value[0] == value[-1]:
        if value[0] in ("'", '"'):
            return value[1:-1]

    # Integer
    try:
        return int(value)
    except ValueError:
        pass

    # Float
    try:
        return float(value)
    except ValueError:
        pass

    # Fallback
    return value


def _build_simple_school_db():
    students = pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "name": ["Alice", "Bob", "Carol", "David", "Eva"],
            "city": ["Vitória", "Rio", "Vitória", "São Paulo", "Rio"],
        }
    )

    courses = pd.DataFrame(
        {
            "id": [10, 20, 30],
            "name": ["Databases", "Algorithms", "AI"],
            "credits": [4, 4, 6],
        }
    )

    enrollments = pd.DataFrame(
        {
            "student_id": [1, 1, 2, 3, 4],
            "course_id": [10, 30, 10, 20, 30],
        }
    )

    db = {
        "Student": students,
        "Course": courses,
        "Enrollment": enrollments,
    }

    return db


def _build_simple_library_db():
    Livro = pd.DataFrame(
        [
            (1, "Dom Casmurro", 1899, 10),
            (2, "O Cortiço", 1890, 20),
            (3, "Memórias Póstumas", 1881, 10),
            (4, "Iracema", 1865, 30),
            (5, "A Hora da Estrela", 1977, 40),
        ],
        columns=["id_livro", "titulo", "ano", "id_autor"],
    )

    Autor = pd.DataFrame(
        [
            (10, "Machado de Assis", "BR"),
            (20, "Aluísio Azevedo", "BR"),
            (30, "José de Alencar", "BR"),
            (50, "Jorge Luis Borges", "AR"),
        ],
        columns=["id_autor", "nome", "pais"],
    )

    Usuario = pd.DataFrame(
        [
            (100, "Ana"),
            (200, "Bruno"),
        ],
        columns=["id_usuario", "nome_usuario"],
    )

    Emprestimo = pd.DataFrame(
        [
            (1000, 100, 1, 5),
            (1001, 100, 2, 3),
            (1002, 200, 1, 5),
        ],
        columns=["id_emprestimo", "id_usuario", "id_livro", "dias"],
    )

    db = {
        "Livro": Livro,
        "Autor": Autor,
        "Usuario": Usuario,
        "Emprestimo": Emprestimo,
    }

    return db


def _load_all_databases():
    dbs = {}

    for file in glob(str(files("pyrelax").joinpath("data/*.txt"))):
        print(f"loading database '{file}'")
        dbs.update(
            {
                name: contents["tables"]
                for name, contents in load_databases(file).items()
            }
        )

    dbs.update(
        {
            "simple-school": _build_simple_school_db(),
            "simple-library": _build_simple_library_db(),
        }
    )
    return dbs


databases = _load_all_databases()
