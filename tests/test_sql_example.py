"""Offline checks for the SQL example's in-memory verifier."""
import importlib.util
from pathlib import Path
import sqlite3

import pytest


@pytest.fixture
def example(monkeypatch):
    directory = Path(__file__).resolve().parents[1] / "examples"
    monkeypatch.syspath_prepend(str(directory))
    spec = importlib.util.spec_from_file_location("sql_example", directory / "04_sql_synth.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def database(example):
    with sqlite3.connect(":memory:") as conn:
        conn.executescript(example.SCHEMA)
        yield conn


@pytest.mark.parametrize("sql", [
    "SELECT name FROM employees",
    "WITH names AS (SELECT name FROM employees) SELECT name FROM names",
    "WITH RECURSIVE nums(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM nums WHERE n<3) SELECT n FROM nums",
])
def test_read_queries_remain_supported(example, database, sql):
    error, message = example.run_query(database, sql)
    assert error >= 0
    assert "sql error" not in message
    assert "rows;" in message


def test_expected_window_query_still_scores_zero(example, database):
    sql = """WITH ranked AS (
        SELECT d.name AS department_name, e.name AS employee_name, e.salary,
               ROW_NUMBER() OVER (PARTITION BY d.id ORDER BY e.salary DESC) AS rank
        FROM employees e JOIN departments d ON e.department_id = d.id
    ) SELECT department_name, employee_name, salary FROM ranked WHERE rank <= 2
      ORDER BY department_name, salary DESC"""
    assert example.run_query(database, sql)[0] == 0
    assert example.run_query(database, sql)[0] == 0


@pytest.mark.parametrize("sql", [
    "WITH marker AS (SELECT 1) DELETE FROM employees",
    "WITH marker AS (SELECT 1) UPDATE employees SET salary = 0",
    "WITH marker AS (SELECT 1) INSERT INTO employees VALUES (9, 'Synthetic', 1, 0)",
])
def test_mutations_cannot_change_fixture(example, database, sql):
    before = database.execute("SELECT * FROM employees ORDER BY id").fetchall()
    error, message = example.run_query(database, sql)
    assert error > 0
    assert "sql error" in message
    assert database.execute("SELECT * FROM employees ORDER BY id").fetchall() == before
    assert "rows;" in example.run_query(database, "SELECT name FROM employees")[1]


def test_runtime_guard_rejects_nonread_operations(example, database):
    example.run_query(database, "SELECT name FROM employees")
    for statement in ["DELETE FROM employees", "CREATE TABLE extra(id)",
                      "ATTACH DATABASE ':memory:' AS extra", "PRAGMA query_only=OFF"]:
        with pytest.raises(sqlite3.DatabaseError):
            database.execute(statement)


def test_extension_loading_is_denied_by_authorizer(example, database):
    # The authorizer denies the function before the extension loader is invoked.
    error, message = example.run_query(database, "SELECT load_extension('synthetic-not-a-file')")
    assert error > 0
    assert "not authorized" in message
