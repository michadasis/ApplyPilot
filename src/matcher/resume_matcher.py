"""
resume_matcher.py — Resume Matcher & LLM Orchestrator Pipeline

Bridges the scraper output with the form filler:
  1. Loads unscored jobs from SQLite
  2. Calls LLM to score each job against the resume
  3. Transitions qualifying jobs to status='queued'
  4. Skips jobs below the match threshold
  5. Provides the match data needed by the form filler (skills, rationale)
"""

import logging
from pathlib import Path
from typing import Any

import anthropic

from src.config.config_manager import AppConfig
from src.matcher.llm_orchestrator import (
    build_llm_client,
    parse_resume,
    score_job_match,
)
from src.tracker.schema import update_job_status, get_db
from src.tracker.tracker import fetch_unscored_jobs

logger = logging.getLogger(__name__)


class ResumeMatcher:
    """
    Scores every unscored job in the DB against the user's resume.

    After scoring:
      - score >= min_match_score  → status: 'queued'  (ready to apply)
      - score <  min_match_score  → status: 'skipped' (won't apply)

    The match data (skills, rationale) is stored in the job record
    and passed forward to the form filler for cover letter generation.
    """

    def __init__(self, config: AppConfig):
        self.config     = config
        self.llm_client = build_llm_client(config.anthropic_api_key)
        self.resume     = parse_resume(config.resume_pdf_path)
        logger.info(
            f"ResumeMatcher ready | "
            f"threshold={config.search.min_match_score:.0%} | "
            f"resume={self.resume['word_count']} words"
        )

    def score_pending_jobs(self, batch_size: int = 20) -> dict[str, int]:
        """
        Score up to `batch_size` unscored jobs and update their status.

        Returns a summary dict: {"queued": N, "skipped": M, "errors": K}
        """
        jobs = fetch_unscored_jobs(limit=batch_size)
        logger.info(f"Scoring {len(jobs)} unscored jobs...")

        summary = {"queued": 0, "skipped": 0, "errors": 0}

        for job in jobs:
            job_id      = job["id"]
            title       = job["title"]
            company     = job.get("company", "")
            description = job.get("description", "")

            if not description:
                logger.warning(f"[job_id={job_id}] No description — skipping")
                update_job_status(job_id, "skipped")
                summary["skipped"] += 1
                continue

            try:
                result = score_job_match(
                    client=self.llm_client,
                    resume_text=self.resume["raw_text"],
                    job_title=title,
                    job_description=description,
                    job_id=job_id,
                )

                score  = float(result.get("match_score", 0.0))
                action = result.get("recommended_action", "review")

                # Store match score + transition status
                if score >= self.config.search.min_match_score and action == "apply":
                    update_job_status(job_id, "queued", match_score=score)
                    summary["queued"] += 1
                    logger.info(
                        f"✅ QUEUED [{job_id}] {title} @ {company} "
                        f"({score:.0%}) — {result.get('rationale', '')[:80]}"
                    )
                else:
                    update_job_status(job_id, "skipped", match_score=score)
                    summary["skipped"] += 1
                    logger.info(
                        f"⏭  SKIPPED [{job_id}] {title} @ {company} "
                        f"({score:.0%})"
                    )

                # Cache the full match result on the job row (as JSON in a notes column)
                # We store matched_skills for cover letter generation later
                _store_match_metadata(job_id, result)

            except Exception as e:
                logger.error(f"[job_id={job_id}] Scoring error: {e}", exc_info=True)
                summary["errors"] += 1

        logger.info(
            f"Scoring complete — "
            f"queued={summary['queued']} | "
            f"skipped={summary['skipped']} | "
            f"errors={summary['errors']}"
        )
        return summary

    def get_match_data(self, job_id: int) -> dict[str, Any]:
        """
        Retrieve cached match metadata for a job.
        Used by the form filler to get matched_skills for cover letter generation.
        """
        return _load_match_metadata(job_id)


# Match metadata persistence helpers

def _store_match_metadata(job_id: int, match_result: dict[str, Any]) -> None:
    """
    Store LLM match result JSON in a dedicated column.
    Adds the column if it doesn't exist (safe migration).
    """
    import json
    with get_db() as conn:
        # Add column if missing (SQLite ALTER TABLE is limited but this works)
        try:
            conn.execute("ALTER TABLE jobs ADD COLUMN match_metadata TEXT")
        except Exception:
            pass   # Column already exists

        conn.execute(
            "UPDATE jobs SET match_metadata = ? WHERE id = ?",
            (json.dumps(match_result), job_id),
        )


def _load_match_metadata(job_id: int) -> dict[str, Any]:
    """Load cached match result. Returns empty dict if not found."""
    import json
    with get_db() as conn:
        try:
            row = conn.execute(
                "SELECT match_metadata FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            if row and row["match_metadata"]:
                return json.loads(row["match_metadata"])
        except Exception:
            pass
    return {}
