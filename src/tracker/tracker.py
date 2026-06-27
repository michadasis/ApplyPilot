"""
tracker.py — State & Logging Tracker Module

Provides:
  1. Structured logging setup (file + console, JSON-friendly format)
  2. ApplicationTracker class — records every step of a run to SQLite
  3. Run summary reporter — pretty-prints session stats to console
  4. Query helpers — fetch jobs by status for the main pipeline
"""

import json
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from src.tracker.schema import get_db

# Logging configuration

LOGS_DIR = Path(__file__).parent.parent.parent / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logger with:
      - Console handler (colored, human-readable)
      - Rotating file handler (structured, max 5MB × 3 files)

    Call once at application startup (main.py).
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    root = logging.getLogger()
    root.setLevel(log_level)

    # Avoid duplicate handlers on re-import
    if root.handlers:
        return

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(log_level)
    console.setFormatter(_ColorFormatter(
        fmt="%(asctime)s │ %(levelname)-8s │ %(name)-30s │ %(message)s",
        datefmt="%H:%M:%S",
    ))
    root.addHandler(console)

    # Rotating file handler
    today = datetime.now().strftime("%Y-%m-%d")
    log_file = LOGS_DIR / f"auto_apply_{today}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)   # File always gets full debug output
    file_handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root.addHandler(file_handler)

    logging.getLogger("playwright").setLevel(logging.WARNING)   # suppress Playwright noise
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)


class _ColorFormatter(logging.Formatter):
    """ANSI color codes for console log levels."""
    COLORS = {
        "DEBUG":    "\033[36m",    # Cyan
        "INFO":     "\033[32m",    # Green
        "WARNING":  "\033[33m",    # Yellow
        "ERROR":    "\033[31m",    # Red
        "CRITICAL": "\033[35m",    # Magenta
    }
    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)


# ApplicationTracker — per-run state manager

logger = logging.getLogger(__name__)


class ApplicationTracker:
    """
    Records fine-grained events for a single application attempt.

    Usage:
        tracker = ApplicationTracker(job_id=42)
        tracker.start(ats_type="greenhouse")
        tracker.record_step("fields_extracted", {"count": 12})
        tracker.record_step("field_filled",     {"label": "First Name", "value": "John"})
        tracker.finish(success=True, submitted=True)
    """

    def __init__(self, job_id: int):
        self.job_id     = job_id
        self.app_id: int | None = None
        self.steps: list[dict[str, Any]] = []
        self._start_time = datetime.now()

    def start(
        self,
        ats_type: str,
        attempt_number: int = 1,
    ) -> None:
        """Create an applications row and store the returned ID."""
        with get_db() as conn:
            cursor = conn.execute(
                """
                INSERT INTO applications (job_id, ats_type, attempt_number, status)
                VALUES (?, ?, ?, 'in_progress')
                """,
                (self.job_id, ats_type, attempt_number),
            )
            self.app_id = cursor.lastrowid

        logger.info(
            f"[app_id={self.app_id}] Application attempt started "
            f"(job={self.job_id}, ats={ats_type})"
        )

    def record_step(self, step_name: str, metadata: dict[str, Any] | None = None) -> None:
        """
        Log a discrete step in the application process.

        Steps are stored in memory and flushed to the DB on finish().
        This avoids excessive DB writes during the hot path.
        """
        entry = {
            "step":      step_name,
            "timestamp": datetime.now().isoformat(),
            "meta":      metadata or {},
        }
        self.steps.append(entry)
        logger.debug(f"[app_id={self.app_id}] Step: {step_name} | {metadata}")

    def record_field_fill(self, label: str, field_type: str, success: bool) -> None:
        """Convenience wrapper for field-fill events."""
        self.record_step("field_fill", {
            "label":      label,
            "field_type": field_type,
            "success":    success,
        })

    def finish(
        self,
        *,
        success: bool,
        submitted: bool = False,
        cover_letter: str = "",
        answers: str = "",
        error_message: str = "",
        screenshot_path: str = "",
    ) -> None:
        """
        Finalize the application record in SQLite.

        Computes elapsed time, sets final status, and persists step log.

        Args:
            answers: JSON-encoded string of {field_label: generated_answer}.
                     Pass the FillResult.answers_json string directly.
        """
        if self.app_id is None:
            logger.warning("ApplicationTracker.finish() called before start()")
            return

        elapsed = (datetime.now() - self._start_time).total_seconds()

        status = "submitted" if submitted else (
            "failed" if not success else "in_progress"
        )

        steps_json = json.dumps(self.steps)

        with get_db() as conn:
            conn.execute(
                """
                UPDATE applications SET
                    status          = ?,
                    cover_letter    = ?,
                    answers_json    = ?,
                    steps_json      = ?,
                    error_message   = ?,
                    screenshot_path = ?,
                    submitted_at    = ?
                WHERE id = ?
                """,
                (
                    status,
                    cover_letter,
                    answers or "{}",
                    steps_json,
                    error_message,
                    screenshot_path,
                    datetime.now().isoformat() if submitted else None,
                    self.app_id,
                ),
            )

        icon = "✅" if success else "❌"
        logger.info(
            f"{icon} [app_id={self.app_id}] Finished | "
            f"status={status} | elapsed={elapsed:.1f}s | steps={len(self.steps)}"
        )


# Query helpers — used by the main pipeline

def fetch_jobs_for_processing(
    statuses: list[str] | None = None,
    min_match_score: float | None = None,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """
    Retrieve jobs from the DB that are ready to be applied to.

    Default query: jobs with status='saved' that have a match score
    above the configured threshold, ordered by match score descending.

    Returns list of dicts (sqlite3.Row cast to dict).
    """
    if statuses is None:
        statuses = ["queued"]

    placeholders = ", ".join("?" * len(statuses))
    query = f"""
        SELECT id, title, company, location, description, apply_url, match_score, status
        FROM jobs
        WHERE status IN ({placeholders})
    """
    params: list[Any] = list(statuses)

    if min_match_score is not None:
        query += " AND (match_score IS NULL OR match_score >= ?)"
        params.append(min_match_score)

    query += " ORDER BY match_score DESC NULLS LAST LIMIT ?"
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]


def fetch_unscored_jobs(limit: int = 100) -> list[dict[str, Any]]:
    """Return saved jobs that haven't been scored by the LLM matcher yet."""
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, title, company, description
            FROM jobs
            WHERE status = 'saved' AND match_score IS NULL
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]


# Run summary reporter

def print_run_summary() -> None:
    """
    Print a formatted summary table of the current run's application stats.
    Reads directly from SQLite so it reflects the true final state.
    """
    with get_db() as conn:
        stats = conn.execute("""
            SELECT
                COUNT(*)                                  AS total,
                SUM(CASE WHEN status='applied'  THEN 1 ELSE 0 END) AS applied,
                SUM(CASE WHEN status='failed'   THEN 1 ELSE 0 END) AS failed,
                SUM(CASE WHEN status='skipped'  THEN 1 ELSE 0 END) AS skipped,
                SUM(CASE WHEN status='saved'    THEN 1 ELSE 0 END) AS saved,
                SUM(CASE WHEN status='queued'   THEN 1 ELSE 0 END) AS queued
            FROM jobs
        """).fetchone()

        cost = conn.execute(
            "SELECT COALESCE(SUM(cost_usd), 0) FROM llm_calls"
        ).fetchone()[0]

        recent = conn.execute("""
            SELECT j.title, j.company, a.status, a.ats_type, a.submitted_at
            FROM applications a
            JOIN jobs j ON j.id = a.job_id
            ORDER BY a.created_at DESC
            LIMIT 10
        """).fetchall()

    print("\n" + "═" * 64)
    print("  AUTO-APPLY BOT │ RUN SUMMARY")
    print("═" * 64)
    print(f"  Total jobs in DB : {stats['total']}")
    print(f"  Applied          : {stats['applied']}")
    print(f"  Failed           : {stats['failed']}")
    print(f"  Skipped          : {stats['skipped']}")
    print(f"  Saved (pending)  : {stats['saved']}")
    print(f"  Queued           : {stats['queued']}")
    print(f"  LLM cost (total) : ${cost:.4f}")
    print("─" * 64)
    print("  RECENT APPLICATIONS:")
    for row in recent:
        icon = "✅" if row["status"] == "submitted" else (
               "❌" if row["status"] == "failed"    else "⏳")
        ts = (row["submitted_at"] or "—")[:16]
        print(f"  {icon} {row['title'][:28]:<28} @ {row['company'][:18]:<18} [{row['ats_type'] or '?':12}] {ts}")
    print("═" * 64 + "\n")
