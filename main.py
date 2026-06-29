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


from src.config.config_manager import load_config, write_example_config
from src.matcher.llm_orchestrator import build_llm_client
from src.matcher.resume_matcher import ResumeMatcher
from src.scraper.scraper import MultiSourceScraper, JSONFeedIngestor, ingest_and_store
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
        scraper = MultiSourceScraper(config)

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
    llm_client,
    specific_job_id: int | None = None,
) -> None:
    """
    Phase 3: For each scored job —
      1. Generate a tailored cover letter + Q&A pack via Groq
      2. Save the pack to data/packs/job_{id}.txt
      3. Print everything to the terminal for easy copy-pasting
      4. Open the apply URL in the default browser
      5. Wait for Enter (applied) / s (skip) / q (quit)
    """
    import webbrowser
    import json as _json
    import textwrap
    from pathlib import Path as _Path
    from src.matcher.llm_orchestrator import generate_application_pack
    from src.matcher.resume_matcher import ResumeMatcher

    logger.info("═" * 50)
    logger.info("PHASE 3: APPLYING")
    logger.info("═" * 50)

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
        logger.info("No jobs queued. Run scrape + score phases first.")
        return

    # Load resume once
    matcher = ResumeMatcher(config)
    resume_text = matcher.resume_text

    packs_dir = _Path("data/packs")
    packs_dir.mkdir(parents=True, exist_ok=True)

    total   = len(jobs)
    applied = 0
    skipped = 0

    W = 70   # terminal width for separators

    print(f"\n{'═' * W}")
    print(f"  {total} job(s) ready  —  Enter=applied  s=skip  q=quit")
    print(f"{'═' * W}\n")

    for i, job in enumerate(jobs, 1):
        job_id    = job["id"]
        score_pct = int((job.get("match_score") or 0) * 100)
        title     = job["title"]
        company   = job["company"]

        matched_skills: list[str] = []
        if job.get("matched_skills"):
            try:
                matched_skills = _json.loads(job["matched_skills"])
            except Exception:
                pass

        # ── Job header ────────────────────────────────────────────
        print(f"[{i}/{total}] {title} @ {company}")
        print(f"      Score    : {score_pct}%")
        print(f"      Source   : {job.get('source', '?')}")
        print(f"      Location : {job.get('location', '?')}")
        if job.get("match_rationale"):
            print(f"      Why      : {job['match_rationale']}")
        if matched_skills:
            print(f"      Skills   : {', '.join(matched_skills[:6])}")
        print(f"      URL      : {job['apply_url']}")

        # ── Generate application pack ─────────────────────────────
        print(f"\n  Generating cover letter + Q&A... ", end="", flush=True)
        pack_path = packs_dir / f"job_{job_id}_{company.replace(' ', '_')[:30]}.txt"

        try:
            pack = generate_application_pack(
                client=llm_client,
                resume_text=resume_text,
                job_title=title,
                company_name=company,
                job_description=job.get("description", ""),
                matched_skills=matched_skills,
                job_id=job_id,
            )
            print("done.\n")
        except Exception as e:
            print(f"failed ({e}).\n")
            pack = {}

        # ── Display cover letter ──────────────────────────────────
        cover = pack.get("cover_letter", "")
        answers = pack.get("answers", {})
        pitch = pack.get("elevator_pitch", "")
        talking_points = pack.get("key_talking_points", [])

        divider = f"  {'─' * (W - 2)}"

        if cover:
            print(f"  ┌─ COVER LETTER {'─' * (W - 17)}")
            for para in cover.strip().split("\n\n"):
                wrapped = textwrap.fill(para.strip(), width=W - 4,
                                        initial_indent="  │  ",
                                        subsequent_indent="  │  ")
                print(wrapped)
                print("  │")
            print(f"  └{'─' * (W - 1)}\n")

        if answers:
            print(f"  ┌─ COMMON QUESTIONS & ANSWERS {'─' * (W - 31)}")
            for q, a in answers.items():
                print(f"  │  Q: {q}")
                wrapped_a = textwrap.fill(a.strip(), width=W - 7,
                                          initial_indent="  │  A: ",
                                          subsequent_indent="  │     ")
                print(wrapped_a)
                print("  │")
            print(f"  └{'─' * (W - 1)}\n")

        if pitch:
            print(f"  ┌─ ELEVATOR PITCH {'─' * (W - 19)}")
            wrapped_p = textwrap.fill(pitch.strip(), width=W - 4,
                                      initial_indent="  │  ",
                                      subsequent_indent="  │  ")
            print(wrapped_p)
            print(f"  └{'─' * (W - 1)}\n")

        if talking_points:
            print(f"  ┌─ KEY TALKING POINTS {'─' * (W - 23)}")
            for pt in talking_points:
                print(f"  │  • {pt}")
            print(f"  └{'─' * (W - 1)}\n")

        # ── Save pack to file ─────────────────────────────────────
        if pack:
            try:
                with open(pack_path, "w", encoding="utf-8") as f:
                    f.write(f"JOB: {title} @ {company}\n")
                    f.write(f"URL: {job['apply_url']}\n")
                    f.write(f"SCORE: {score_pct}%\n\n")
                    if cover:
                        f.write("=== COVER LETTER ===\n\n")
                        f.write(cover.strip() + "\n\n")
                    if answers:
                        f.write("=== Q&A ===\n\n")
                        for q, a in answers.items():
                            f.write(f"Q: {q}\nA: {a}\n\n")
                    if pitch:
                        f.write("=== ELEVATOR PITCH ===\n\n")
                        f.write(pitch.strip() + "\n\n")
                    if talking_points:
                        f.write("=== KEY TALKING POINTS ===\n\n")
                        for pt in talking_points:
                            f.write(f"• {pt}\n")
                print(f"  💾 Saved to {pack_path}\n")
            except Exception as e:
                logger.warning(f"Could not save pack: {e}")

        # ── Open browser ──────────────────────────────────────────
        try:
            webbrowser.open(job["apply_url"])
        except Exception as e:
            logger.warning(f"Could not open browser: {e}")

        loop = asyncio.get_running_loop()
        answer = await loop.run_in_executor(
            None,
            lambda: input(f"  → Enter=applied  s=skip  q=quit: ").strip().lower(),
        )

        if answer == "q":
            print("\nStopping apply phase.")
            break
        elif answer == "s":
            update_job_status(job_id, "skipped")
            skipped += 1
            print(f"  Skipped.\n")
        else:
            update_job_status(job_id, "applied")
            applied += 1
            print(f"  ✅ Marked as applied.\n")

        print(f"  {'═' * (W - 2)}\n")

    print(f"  Session: {applied} applied | {skipped} skipped | "
          f"{total - applied - skipped} remaining")
    print(f"  Packs saved in: {packs_dir.resolve()}\n")


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

    if phase in ("all", "scrape"):
        await run_scrape_phase(config, feed_path=feed_path)

    if phase in ("all", "score"):
        run_score_phase(config)

    if phase in ("all", "apply"):
        await run_apply_phase(config, llm_client, specific_job_id=args.job_id)

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
        "--verbose", "-v",
        action="store_true",
        help="Enable DEBUG-level logging",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
