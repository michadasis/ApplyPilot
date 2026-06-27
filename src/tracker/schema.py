"""
schema.py — SQLite database schema and connection manager.

All application state lives here. This is the single source of truth for
every job the bot has seen, attempted, or completed.
"""

import sqlite3
import logging
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime
from typing import Generator

logger = logging.getLogger(__name__)

# Canonical path for the SQLite database file
DB_PATH = Path(__file__).parent.parent.parent / "data" / "jobs.db"


# DDL — Database schema (run once on startup via init_db())

SCHEMA_SQL = """
-- ============================================================
-- jobs: every discovered job posting, de-duplicated by URL
-- ============================================================
CREATE TABLE IF NOT EXISTS jobs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    title           TEXT    NOT NULL,
    company         TEXT    NOT NULL,
    location        TEXT,
    description     TEXT,                         -- full JD text
    apply_url       TEXT    UNIQUE NOT NULL,       -- canonical apply link
    source          TEXT    NOT NULL,             -- e.g. 'linkedin', 'indeed', 'json_feed'
    match_score     REAL,                         -- 0.0–1.0, set by LLM matcher
    status          TEXT    NOT NULL DEFAULT 'saved',
                                -- saved | queued | applying | applied
                                -- | failed | skipped | manual_review
    created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- applications: one row per actual form-fill attempt
-- ============================================================
CREATE TABLE IF NOT EXISTS applications (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
    ats_type        TEXT,                         -- greenhouse | lever | workday | custom
    attempt_number  INTEGER NOT NULL DEFAULT 1,
    status          TEXT    NOT NULL DEFAULT 'in_progress',
                                -- in_progress | submitted | failed | skipped_review
    cover_letter    TEXT,                         -- generated cover letter text
    answers_json    TEXT,                         -- JSON blob: {field_label: generated_answer}
    steps_json      TEXT,                         -- JSON array of discrete step events
    error_message   TEXT,                         -- populated on failure
    screenshot_path TEXT,                         -- path to failure screenshot
    submitted_at    TEXT,                         -- timestamp of final submission
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- llm_calls: audit log for every LLM API call (cost tracking)
-- ============================================================
CREATE TABLE IF NOT EXISTS llm_calls (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    job_id          INTEGER REFERENCES jobs(id),
    purpose         TEXT    NOT NULL,             -- 'match' | 'field_extract' | 'answer_gen' | 'cover_letter'
    model           TEXT    NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    total_tokens    INTEGER,
    cost_usd        REAL,
    created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
);

-- ============================================================
-- Indexes for common query patterns
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_jobs_status        ON jobs(status);
CREATE INDEX IF NOT EXISTS idx_jobs_match_score   ON jobs(match_score DESC);
CREATE INDEX IF NOT EXISTS idx_applications_job   ON applications(job_id);
CREATE INDEX IF NOT EXISTS idx_llm_calls_job      ON llm_calls(job_id);

-- ============================================================
-- Trigger: keep jobs.updated_at current on every UPDATE
-- ============================================================
CREATE TRIGGER IF NOT EXISTS trg_jobs_updated_at
    AFTER UPDATE ON jobs
    FOR EACH ROW
    BEGIN
        UPDATE jobs SET updated_at = datetime('now') WHERE id = OLD.id;
    END;
"""


def init_db(db_path: Path = DB_PATH) -> None:
    """
    Create the database file and apply the schema if it doesn't exist.
    Safe to call on every startup — all DDL uses IF NOT EXISTS.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA_SQL)
        # Safe migration: add steps_json column if an older DB is missing it
        try:
            conn.execute("ALTER TABLE applications ADD COLUMN steps_json TEXT")
            conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists
    logger.info(f"Database initialized at {db_path}")


@contextmanager
def get_db(db_path: Path = DB_PATH) -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a configured SQLite connection.

    Usage:
        with get_db() as conn:
            conn.execute("SELECT ...")

    - row_factory=sqlite3.Row gives dict-like access: row['column_name']
    - WAL mode for better concurrent read performance
    - foreign_keys pragma enforced for referential integrity
    """
    conn = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# Convenience helpers used by multiple modules

def upsert_job(
    title: str,
    company: str,
    apply_url: str,
    source: str,
    description: str = "",
    location: str = "",
) -> int:
    """
    Insert a new job row or ignore if the URL already exists.
    Returns the job's database ID either way.
    """
    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO jobs (title, company, location, description, apply_url, source)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(apply_url) DO NOTHING
            """,
            (title, company, location, description, apply_url, source),
        )
        row = conn.execute(
            "SELECT id FROM jobs WHERE apply_url = ?", (apply_url,)
        ).fetchone()
        return row["id"]


def update_job_status(job_id: int, status: str, match_score: float | None = None) -> None:
    """Transition a job to a new status, optionally recording its match score."""
    with get_db() as conn:
        if match_score is not None:
            conn.execute(
                "UPDATE jobs SET status=?, match_score=? WHERE id=?",
                (status, match_score, job_id),
            )
        else:
            conn.execute("UPDATE jobs SET status=? WHERE id=?", (status, job_id))


def log_llm_call(
    purpose: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    job_id: int | None = None,
) -> None:
    """Record an LLM API call for cost tracking. Pricing constants kept centrally here."""
    # Rough USD cost per 1k tokens (update as pricing changes)
    COST_PER_1K = {"claude-3-5-sonnet-20241022": (0.003, 0.015)}
    in_rate, out_rate = COST_PER_1K.get(model, (0.0, 0.0))
    cost = (prompt_tokens / 1000 * in_rate) + (completion_tokens / 1000 * out_rate)

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO llm_calls
                (job_id, purpose, model, prompt_tokens, completion_tokens, total_tokens, cost_usd)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, purpose, model, prompt_tokens, completion_tokens,
             prompt_tokens + completion_tokens, round(cost, 6)),
        )
