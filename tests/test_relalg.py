import pandas as pd
import pytest

from relalg import RelAlgEngine, RelAlgExecutionError, RelAlgSyntaxError


@pytest.fixture
def engine():
    livro = pd.DataFrame([
        (1, "Dom Casmurro", 1899, 10),
        (2, "O Corti\u00e7o", 1890, 20),
        (3, "Mem\u00f3rias P\u00f3stumas", 1881, 10),
        (4, "Iracema", 1865, 30),
        (5, "A Hora da Estrela", 1977, 40),
    ], columns=["id_livro", "titulo", "ano", "id_autor"])

    autor = pd.DataFrame([
        (10, "Machado de Assis", "BR"),
        (20, "Alu\u00edsio Azevedo", "BR"),
        (30, "Jos\u00e9 de Alencar", "BR"),
        (50, "Jorge Luis Borges", "AR"),
    ], columns=["id_autor", "nome", "pais"])

    usuario = pd.DataFrame([
        (100, "Ana"),
        (200, "Bruno"),
    ], columns=["id_usuario", "nome_usuario"])

    emprestimo = pd.DataFrame([
        (1000, 100, 1, 5),
        (1001, 100, 2, 3),
        (1002, 200, 1, 5),
    ], columns=["id_emprestimo", "id_usuario", "id_livro", "dias"])

    return RelAlgEngine({
        "Livro": livro, "Autor": autor, "Usuario": usuario, "Emprestimo": emprestimo,
    })


# ---- projection / selection ------------------------------------------------

def test_projection_and_selection_unicode(engine):
    out = engine.query("\u03c0 titulo, ano (\u03c3 ano > 1880 (Livro))")
    assert set(out["titulo"]) == {"Dom Casmurro", "O Corti\u00e7o", "Mem\u00f3rias P\u00f3stumas", "A Hora da Estrela"}


def test_projection_and_selection_ascii(engine):
    out = engine.query("pi titulo, ano (sigma ano > 1880 (Livro))")
    assert set(out["titulo"]) == {"Dom Casmurro", "O Corti\u00e7o", "Mem\u00f3rias P\u00f3stumas", "A Hora da Estrela"}


def test_star_projection(engine):
    out = engine.query("\u03c0 * (Livro)")
    assert list(out.columns) == ["id_livro", "titulo", "ano", "id_autor"]
    assert len(out) == 5


# ---- rename -------------------------------------------------------------

def test_rename_relation_and_columns_then_theta_join(engine):
    out = engine.query("""
        L = \u03c1 L (Livro)
        A = \u03c1 A (\u03c1 nome_autor <- nome (Autor))
        \u03c0 L.titulo, nome_autor (L \u2a1d L.id_autor = A.id_autor A)
    """)
    row = out[out["titulo"] == "Dom Casmurro"].iloc[0]
    assert row["nome_autor"] == "Machado de Assis"


# ---- joins ----------------------------------------------------------------

def test_natural_join_unicode(engine):
    out = engine.query("Livro \u2a1d Autor")
    assert "nome" in out.columns and "titulo" in out.columns
    # "A Hora da Estrela" has id_autor=40, which has no matching Autor row,
    # so an *inner* natural join drops it (4 of the 5 books remain).
    assert len(out) == 4


def test_natural_join_ascii_keyword(engine):
    out = engine.query("Livro natural join Autor")
    assert len(out) == 4


def test_theta_join_explicit_condition(engine):
    out = engine.query("Livro \u2a1d Livro.id_autor = Autor.id_autor Autor")
    assert len(out) == 4


def test_cross_join_plus_selection_equals_theta_join(engine):
    a = engine.query("\u03c3 Livro.id_autor = Autor.id_autor (Livro \u2a2f Autor)")
    b = engine.query("Livro \u2a1d Livro.id_autor = Autor.id_autor Autor")
    assert len(a) == len(b) == 4


def test_inner_join_ascii_keyword(engine):
    out = engine.query("Livro join Livro.id_autor = Autor.id_autor Autor")
    assert len(out) == 4


def test_left_outer_join_natural(engine):
    out = engine.query("Autor \u27d5 Livro")
    # Borges (id_autor=50) has no books -> appears once with NULL book columns
    borges = out[out["nome"] == "Jorge Luis Borges"]
    assert len(borges) == 1
    assert pd.isna(borges.iloc[0]["titulo"])


def test_left_outer_join_ascii_keyword(engine):
    out = engine.query("Autor left outer join Livro")
    borges = out[out["nome"] == "Jorge Luis Borges"]
    assert len(borges) == 1
    assert pd.isna(borges.iloc[0]["titulo"])


def test_left_semi_join(engine):
    out = engine.query("\u03c0 nome (Autor \u22c9 Livro)")
    assert set(out["nome"]) == {"Machado de Assis", "Alu\u00edsio Azevedo", "Jos\u00e9 de Alencar"}


def test_anti_join(engine):
    out = engine.query("\u03c0 nome (Autor \u25b7 Autor.id_autor = Livro.id_autor Livro)")
    assert list(out["nome"]) == ["Jorge Luis Borges"]


def test_anti_join_ascii_natural(engine):
    out = engine.query("\u03c0 nome (Autor anti join Livro)")
    assert list(out["nome"]) == ["Jorge Luis Borges"]


# ---- group by / order by ---------------------------------------------------

def test_group_by_aggregate(engine):
    out = engine.query(
        "\u03b3 id_autor; COUNT(id_livro) -> qtd_livros, AVG(ano) -> ano_medio (Livro)"
    ).sort_values("id_autor").reset_index(drop=True)
    assert list(out["qtd_livros"]) == [2, 1, 1, 1]


def test_group_by_no_grouping_columns(engine):
    out = engine.query("\u03b3 COUNT(*) -> total, SUM(id_autor) -> soma (Livro)")
    assert out.iloc[0]["total"] == 5
    assert out.iloc[0]["soma"] == 10 + 20 + 10 + 30 + 40


def test_order_by(engine):
    out = engine.query("\u03c4 ano desc (\u03c0 titulo, ano (Livro))")
    assert list(out["ano"]) == sorted(out["ano"], reverse=True)


# ---- set operators ----------------------------------------------------------

def test_union(engine):
    out = engine.query("(\u03c0 id_livro (Livro)) \u222a (\u03c0 id_livro (Emprestimo))")
    assert set(out["id_livro"]) == {1, 2, 3, 4, 5}


def test_difference(engine):
    out = engine.query("(\u03c0 id_autor (Autor)) - (\u03c0 id_autor (Livro))")
    assert list(out["id_autor"]) == [50]


def test_intersect(engine):
    out = engine.query("(\u03c0 id_livro (Livro)) \u2229 (\u03c0 id_livro (Emprestimo))")
    assert set(out["id_livro"]) == {1, 2}


# ---- value expressions -------------------------------------------------------

def test_case_when(engine):
    out = engine.query(
        "\u03c0 titulo, "
        "(case when ano < 1880 then 'antigo' when ano < 1950 then 'cl\u00e1ssico' else 'moderno' end) -> era "
        "(Livro)"
    )
    row = out[out["titulo"] == "Iracema"].iloc[0]
    assert row["era"] == "antigo"
    row = out[out["titulo"] == "A Hora da Estrela"].iloc[0]
    assert row["era"] == "moderno"


def test_like(engine):
    out = engine.query("\u03c0 titulo (\u03c3 titulo like 'A%' (Livro))")
    assert set(out["titulo"]) == {"A Hora da Estrela"}


def test_between_and_arithmetic(engine):
    out = engine.query(
        "\u03c0 titulo, ano, (ano + 100) -> ano_seculo_seguinte "
        "(\u03c3 ano between 1880 and 1900 (Livro))"
    )
    assert set(out["titulo"]) == {"Dom Casmurro", "O Corti\u00e7o", "Mem\u00f3rias P\u00f3stumas"}
    row = out[out["titulo"] == "Dom Casmurro"].iloc[0]
    assert row["ano_seculo_seguinte"] == 1999


def test_is_null_via_outer_join(engine):
    out = engine.query(
        "\u03c0 Autor.nome (\u03c3 Livro.id_livro = null (Autor \u27d5 Livro))"
    )
    assert list(out["nome"]) == ["Jorge Luis Borges"]


# ---- division -----------------------------------------------------------

def test_division(engine):
    out = engine.query("""
        Alvo = (\u03c0 id_livro (\u03c3 id_livro = 1 (Livro))) \u222a (\u03c0 id_livro (\u03c3 id_livro = 2 (Livro)))
        \u03c0 id_usuario, id_livro (Emprestimo) \u00f7 Alvo
    """)
    # Ana (100) borrowed both book 1 and 2; Bruno (200) only borrowed book 1
    assert list(out["id_usuario"]) == [100]


# ---- multi-statement scripts --------------------------------------------------

def test_multiline_assignment_walrus_style(engine):
    out = engine.query("R := pi titulo (Livro)\nR")
    assert "titulo" in out.columns
    assert len(out) == 5


# ---- error handling ----------------------------------------------------------

def test_unknown_relation_raises(engine):
    with pytest.raises(RelAlgExecutionError):
        engine.query("\u03c0 * (NaoExiste)")


def test_ambiguous_column_raises(engine):
    with pytest.raises((RelAlgExecutionError, Exception)):
        engine.query("\u03c0 id_autor (Livro \u2a2f Autor)")


def test_syntax_error_raises(engine):
    with pytest.raises(RelAlgSyntaxError):
        engine.query("\u03c0 titulo (((Livro)")
