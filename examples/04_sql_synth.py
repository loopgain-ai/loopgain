"""Example 04 — Text-to-SQL with execution-based verification.

Real verify-revise loop. SQLite is the verifier:
  1. Claude is given a natural-language query against a small in-memory
     schema (employees, departments).
  2. Claude returns a single SELECT statement.
  3. We refuse anything that isn't a single SELECT, execute it against
     the fixture, and compare its result set to the expected set.
  4. Error = size of the symmetric difference of result rows. 0 fires
     TARGET_MET.

Expected band:  Mixed. Top-N-per-group is a classic case where Opus
                usually one-shots, but lesser models can hit STALLING
                or OSCILLATING before resolving. Documented as "any
                band, deterministic verifier."
Loop type:      tool_use_retry (SQLite is the tool).
"""

from __future__ import annotations

import re
import sqlite3

from loopgain import LoopGain

from _common import (
    call_claude,
    get_client,
    print_comparison,
    print_iteration,
    print_result,
    send_telemetry,
)

WORKLOAD_ID = "example-04-sql-synth"
FIXED_CAP = 5

SCHEMA = """\
CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL
);
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    department_id INTEGER NOT NULL REFERENCES departments(id),
    salary INTEGER NOT NULL
);
INSERT INTO departments(id, name) VALUES
    (1, 'Engineering'), (2, 'Sales'), (3, 'Marketing');
INSERT INTO employees(id, name, department_id, salary) VALUES
    (1, 'Alice',   1, 180000),
    (2, 'Bob',     1, 150000),
    (3, 'Carol',   1, 140000),
    (4, 'Dan',     2, 120000),
    (5, 'Eve',     2, 110000),
    (6, 'Frank',   2,  95000),
    (7, 'Grace',   3,  90000),
    (8, 'Hank',    3,  85000);
"""

QUESTION = (
    "Return the top 2 highest-paid employees in each department. "
    "Columns, in order: department_name, employee_name, salary. "
    "Sort by department_name ascending, then salary descending."
)

EXPECTED = [
    ("Engineering", "Alice", 180000),
    ("Engineering", "Bob",   150000),
    ("Marketing",   "Grace",  90000),
    ("Marketing",   "Hank",   85000),
    ("Sales",       "Dan",   120000),
    ("Sales",       "Eve",   110000),
]


def extract_sql(text: str) -> str:
    fence = re.search(r"```(?:sql)?\s*(.*?)\s*```", text, re.DOTALL)
    return (fence.group(1) if fence else text).strip().rstrip(";").strip()


def run_query(conn: sqlite3.Connection, sql: str):
    if not sql:
        return len(EXPECTED) + 1, "empty query"
    if not re.match(r"(?is)^\s*(with\b|select\b)", sql):
        return len(EXPECTED) + 1, "not a SELECT/WITH statement"
    if ";" in sql:
        return len(EXPECTED) + 1, "multi-statement input rejected"
    try:
        rows = conn.execute(sql).fetchall()
    except sqlite3.Error as exc:
        return len(EXPECTED) + 1, f"sql error: {exc}"
    got = [tuple(r) for r in rows]
    diff = set(got).symmetric_difference(set(EXPECTED))
    return len(diff), f"{len(got)} rows; {len(diff)} mismatched"


def one_iteration(client, conn, prev_sql: str, prev_msg: str):
    if not prev_sql:
        prompt = (
            f"Schema:\n{SCHEMA}\n\nQuestion: {QUESTION}\n\n"
            "Return ONLY a single SELECT (or WITH ... SELECT) statement. "
            "No prose, no fences, no trailing semicolon."
        )
    else:
        prompt = (
            f"Schema:\n{SCHEMA}\n\nQuestion: {QUESTION}\n\n"
            f"Your previous SQL was:\n{prev_sql}\n\n"
            f"It produced: {prev_msg}\n\nReturn a corrected single SELECT. "
            "SQL only, no prose."
        )
    sql = extract_sql(call_claude(client, prompt))
    err, msg = run_query(conn, sql)
    return err, sql, msg


def baseline_run(client, conn):
    print(f"─── BASELINE: no LoopGain, fixed cap = {FIXED_CAP} ───")
    sql, msg, err = "", "", -1
    for i in range(FIXED_CAP):
        err, sql, msg = one_iteration(client, conn, sql, msg)
        print(f"  iter {i+1:>2}  error={err:>3}  ({msg})")
    print(f"  → kept LAST output. final error={err}\n")
    return err, FIXED_CAP


def loopgain_run(client, conn):
    print(f"─── WITH LOOPGAIN: target_error=0, max_iterations={FIXED_CAP} ───")
    lg = LoopGain(target_error=0, max_iterations=FIXED_CAP)
    sql, msg = "", ""
    while lg.should_continue():
        err, sql, msg = one_iteration(client, conn, sql, msg)
        first_line = sql.splitlines()[0] if sql else "[no sql]"
        preview = f"{msg}; {first_line}"
        state = lg.observe(err, output=sql)
        print_iteration(lg.result.iterations_used, err, state, lg.eta, preview)
    print_result(lg)
    return lg


def main() -> None:
    client = get_client()
    conn = sqlite3.connect(":memory:")
    conn.executescript(SCHEMA)
    print("Spec: top-2-per-department over an 8-row fixture.\n")
    baseline_err, baseline_iters = baseline_run(client, conn)
    lg = loopgain_run(client, conn)
    print_comparison(baseline_iters, baseline_err, lg)
    send_telemetry(lg, workload_id=WORKLOAD_ID, loop_type="tool_use_retry")


if __name__ == "__main__":
    main()
