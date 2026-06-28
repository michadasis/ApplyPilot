"""
Orchestrates the full pipeline:
  Phase 1: SCRAPE  — Discover new jobs from LinkedIn / JSON feed
  Phase 2: SCORE   — LLM match scoring against resume
  Phase 3: APPLY   — Automated form filling for qualifying jobs
  Phase 4: REPORT  — Print run summary

Usage:
  # Full pipeline
  python main.py

  # Scrape only (no applications)
  python main.py --phase scrape

  # Score only (jobs already in DB)
  python main.py --phase score

  # Apply only (jobs already scored and queued)
  python main.py --phase apply

  # Apply to a specific job by ID
  python main.py --phase apply --job-id 42

  # Dry run (fills forms but never submits, regardless of config)
  python main.py --dry-run

  # Use a JSON feed instead of live scraping
  python main.py --feed path/to/jobs.json
"""

import argparse
import asyncio
import logging
import sys
from pathlib import Path

from playwright.async_api import async_playwright

from src.config.config_manager import load_config, write_example_config
from src.filler.form_filler import FormFillerEngine, build_browser_context
from src.matcher.llm_orchestrator import build_llm_client
from src.matcher.resume_matcher import ResumeMatcher
from src.scraper.scraper import RemoteOKScraper, JSONFeedIngestor, ingest_and_store
from src.tracker.schema import init_db, update_job_status
from src.tracker.tracker import (
    ApplicationTracker,
    fetch_jobs_for_processing,
    print_run_summary,
    setup_logging,
)

logger = logging.getLogger(__name__)


# Phase runners

async def run_scrape_phase(config, feed_path: Path | None = None) -> None:
    """Phase 1: Discover and store new job listings via RemoteOK API or JSON feed."""
    logger.info("═" * 50)
    logger.info("PHASE 1: SCRAPING")
    logger.info("═" * 50)

    if feed_path:
        scraper = JSONFeedIngestor(feed_path)
    else:
        scraper = RemoteOKScraper(config)

    await ingest_and_store(
        scraper,
        config,
        max_results=config.behavior.max_applications_per_run * 3,  # buffer
    )


def run_score_phase(config) -> None:
    """Phase 2: LLM match scoring for all unscored jobs."""
    logger.info("═" * 50)
    logger.info("PHASE 2: SCORING")
    logger.info("═" * 50)

    matcher = ResumeMatcher(config)
    summary = matcher.score_pending_jobs(batch_size=50)
    logger.info(f"Scoring done: {summary}")


async def run_apply_phase(
    config,
    context,
    llm_client,
    dry_run: bool = False,
    specific_job_id: int | None = None,
) -> None:
    """Phase 3: Fill and submit application forms."""
    logger.info("═" * 50)
    logger.info("PHASE 3: APPLYING")
    logger.info("═" * 50)

    # If dry_run, force manual_review=True so we never auto-submit
    if dry_run:
        config.behavior.require_manual_review = True
        logger.info("🧪 DRY RUN mode — forms will be filled but submission requires manual confirm")

    # Fetch jobs to process
    if specific_job_id:
        from src.tracker.schema import get_db
        with get_db() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (specific_job_id,)
            ).fetchone()
        jobs = [dict(row)] if row else []
    else:
        jobs = fetch_jobs_for_processing(
            statuses=["queued"],
            min_match_score=config.search.min_match_score,
            limit=config.behavior.max_applications_per_run,
        )

    if not jobs:
        logger.info("No jobs queued for application. Run scrape + score phases first.")
        return

    logger.info(f"Applying to {len(jobs)} job(s)...")

    engine  = FormFillerEngine(config, llm_client)
    matcher = ResumeMatcher(config)

    for job in jobs:
        job_id = job["id"]
        logger.info(f"\n{'─' * 50}")
        logger.info(f"Applying: [{job_id}] {job['title']} @ {job['company']}")
        logger.info(f"URL: {job['apply_url']}")

        # Mark as 'applying' to prevent duplicate runs
        update_job_status(job_id, "applying")

        tracker = ApplicationTracker(job_id=job_id)
        match_data = matcher.get_match_data(job_id)

        try:
            # Detect ATS type for tracker
            from src.filler.form_filler import detect_ats_type
            ats_type = detect_ats_type(job["apply_url"])
            tracker.start(ats_type=ats_type)
            tracker.record_step("apply_start", {"url": job["apply_url"]})

            result = await engine.apply(
                job_id=job_id,
                apply_url=job["apply_url"],
                job_title=job["title"],
                company_name=job["company"],
                job_description=job.get("description", ""),
                match_data=match_data,
                context=context,
            )

            tracker.record_step("apply_complete", {
                "success":   result.success,
                "submitted": result.submitted,
                "ats_type":  result.ats_type,
            })

            tracker.finish(
                success=result.success,
                submitted=result.submitted,
                cover_letter=result.cover_letter_text,
                answers=result.answers_json,
                error_message=result.error_message,
                screenshot_path=result.screenshot_path,
            )

            if result.success:
                logger.info(
                    f"✅ Job {job_id} — "
                    f"{'SUBMITTED' if result.submitted else 'FILLED (review pending)'}"
                )
            else:
                logger.error(f"❌ Job {job_id} failed: {result.error_message}")

            # Retry logic
            if not result.success and config.behavior.retry_failed_attempts > 0:
                logger.info(f"Retrying job {job_id} (attempt 2)...")
                update_job_status(job_id, "queued")   # re-queue for retry

        except Exception as e:
            logger.error(f"Unexpected error on job {job_id}: {e}", exc_info=True)
            update_job_status(job_id, "failed")
            tracker.finish(success=False, error_message=str(e))

        # Polite delay between applications
        await asyncio.sleep(3)


# Main entry point

async def main(args: argparse.Namespace) -> None:
    """Top-level async orchestrator."""

    # Bootstrap
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    write_example_config()   # Write example if missing (first-run UX)

    try:
        config = load_config()
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Configuration error: {e}\n")
        print("   1. Copy config.example.json to config.json and edit it.")
        print("   2. Copy .env.example to .env and add your API keys.")
        sys.exit(1)

    init_db()
    llm_client = build_llm_client(config.groq_api_key)

    phase = args.phase.lower()
    feed_path = Path(args.feed) if args.feed else None

    # Run phases
    async with async_playwright() as playwright:
        _, context = await build_browser_context(playwright, config)

        try:
            if phase in ("all", "scrape"):
                await run_scrape_phase(config, feed_path=feed_path)

            if phase in ("all", "score"):
                run_score_phase(config)   # sync — no browser needed

            if phase in ("all", "apply"):
                await run_apply_phase(
                    config, context, llm_client,
                    dry_run=args.dry_run,
                    specific_job_id=args.job_id,
                )

        finally:
            await context.close()

    # Summary
    print_run_summary()


# CLI argument parser

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-Apply Job Bot — modular LinkedIn application automation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase",
        choices=["all", "scrape", "score", "apply"],
        default="all",
        help="Which pipeline phase to run (default: all)",
    )
    parser.add_argument(
        "--feed",
        metavar="PATH",
        help="Path to a JSON jobs feed file (skips live scraping)",
    )
    parser.add_argument(
        "--job-id",
        type=int,
        metavar="ID",
        help="Apply to a specific job ID (skips queue query)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Fill forms but never auto-submit (forces manual review)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
