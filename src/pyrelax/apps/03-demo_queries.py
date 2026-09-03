import pandas as pd
from relalg import RelAlgEngine, RelAlgError

from src.pyrelax.data import databases

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)


engine = RelAlgEngine(databases["simple-library"])


def run(title, q, **kw):
    print("=" * 70)
    print(title)
    print("-" * 70)
    print(q.strip())
    print("-" * 70)
    try:
        print(engine.query(q, **kw))
    except RelAlgError as e:
        print("ERROR:", e)
    print()


def run_tests():

    run("projection + selection (unicode)", "π titulo, ano (σ ano > 1880 (Livro))")

    run(
        "projection + selection (ascii keywords)",
        "pi titulo, ano (sigma ano > 1880 (Livro))",
    )

    run(
        "rename relation + column, natural join",
        """
        L = ρ L (Livro)
        A = ρ A (ρ nome_autor <- nome (Autor))
        π L.titulo, nome_autor (L ⨝ L.id_autor = A.id_autor A)
    """,
    )

    run("natural join (shared column name)", "Livro ⨝ Autor")

    run(
        "theta join with explicit condition",
        "Livro ⨝ Livro.id_autor = Autor.id_autor Autor",
    )

    run("left outer join (natural)", "Livro ⟕ Autor")

    run(
        "anti join: books whose author has no other work in the catalogue?",
        "π nome (Autor ▷ Autor.id_autor = Livro.id_autor Livro)",
    )

    run(
        "cross join + selection == theta join",
        "σ Livro.id_autor = Autor.id_autor (Livro ⨯ Autor)",
    )

    run(
        "group by + aggregate",
        "γ id_autor; COUNT(id_livro) -> qtd_livros, AVG(ano) -> ano_medio (Livro)",
    )

    run(
        "group by with no grouping columns (whole-table aggregate)",
        "γ COUNT(*) -> total, SUM(id_autor) -> soma (Livro)",
    )

    run("order by", "τ ano desc (π titulo, ano (Livro))")

    run(
        "union / intersect / difference",
        "(π id_livro (Livro)) ∪ (π id_livro (Emprestimo))",
        eliminate_duplicates=True,
    )

    run("difference", "(π id_autor (Autor)) - (π id_autor (Livro))")

    run(
        "CASE WHEN",
        "π titulo, (case when ano < 1880 then 'antigo' when ano < 1950 then 'clássico' else 'moderno' end) -> era (Livro)",
    )

    run("LIKE + string function", "π titulo (σ titulo like 'A%' (Livro))")

    run(
        "BETWEEN + arithmetic expr",
        "π titulo, ano, (ano + 100) -> ano_seculo_seguinte (σ ano between 1880 and 1900 (Livro))",
    )

    run("semi join: authors who have at least one book", "π nome (Autor ⋉ Livro)")

    run(
        "division: users who borrowed EVERY book with id_livro in {1,2}",
        """
        Alvo = (π id_livro (σ id_livro = 1 (Livro))) ∪ (π id_livro (σ id_livro = 2 (Livro)))
        π id_usuario, id_livro (Emprestimo) ÷ Alvo
        """,
    )

    run(
        "IS NULL via outer join",
        "π Autor.nome (σ Livro.id_livro = null (Autor ⟕ Livro))",
    )


if __name__ == "__main__":
    run_tests()
