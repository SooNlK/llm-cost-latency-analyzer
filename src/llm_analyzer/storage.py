import sqlite3
from contextlib import closing
from pathlib import Path

from llm_analyzer.models import LLMResponse, RunRecord, RunStatus

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id       TEXT    NOT NULL,
    timestamp           TEXT    NOT NULL,
    provider            TEXT    NOT NULL,
    model               TEXT    NOT NULL,
    prompt_id           TEXT    NOT NULL,
    repetition          INTEGER NOT NULL,
    status              TEXT    NOT NULL,
    text                TEXT,
    input_tokens        INTEGER,
    output_tokens       INTEGER,
    cached_input_tokens INTEGER,
    ttft_ms             REAL,
    total_latency_ms    REAL,
    error_message       TEXT
);
"""

INSERT = """
INSERT INTO runs (
    experiment_id, timestamp, provider, model, prompt_id, repetition,
    status, text, input_tokens, output_tokens, cached_input_tokens,
    ttft_ms, total_latency_ms, error_message
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""


def init_db(path: str | Path) -> None:
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            conn.execute(SCHEMA)


def save_run(path: str | Path, record: RunRecord) -> int:
    r = record.response
    values = (
        record.experiment_id,
        record.timestamp,
        record.provider,
        record.model,
        record.prompt_id,
        record.repetition,
        r.status.value,
        r.text,
        r.input_tokens,
        r.output_tokens,
        r.cached_input_tokens,
        r.ttft_ms,
        r.total_latency_ms,
        r.error_message,
    )
    with closing(sqlite3.connect(path)) as conn:
        with conn:
            cursor = conn.execute(INSERT, values)
            run_id = cursor.lastrowid
    return run_id


def load_runs(
    path: str | Path, experiment_id: str | None = None
) -> list[RunRecord]:
    query = "SELECT * FROM runs"
    params: tuple = ()
    if experiment_id is not None:
        query += " WHERE experiment_id = ?"
        params = (experiment_id,)
    query += " ORDER BY run_id"

    with closing(sqlite3.connect(path)) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(query, params).fetchall()

    return [_row_to_record(row) for row in rows]


def _row_to_record(row: sqlite3.Row) -> RunRecord:
    response = LLMResponse(
        status=RunStatus(row["status"]),
        text=row["text"],
        input_tokens=row["input_tokens"],
        output_tokens=row["output_tokens"],
        cached_input_tokens=row["cached_input_tokens"],
        ttft_ms=row["ttft_ms"],
        total_latency_ms=row["total_latency_ms"],
        error_message=row["error_message"],
    )
    return RunRecord(
        run_id=row["run_id"],
        experiment_id=row["experiment_id"],
        timestamp=row["timestamp"],
        provider=row["provider"],
        model=row["model"],
        prompt_id=row["prompt_id"],
        repetition=row["repetition"],
        response=response,
    )