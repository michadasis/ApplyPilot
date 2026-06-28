"""
scraper.py — Job Scraper / Ingestor Module

Fetches job listings from the RemoteOK public JSON API and persists
them to SQLite. Falls back to a local JSON feed for offline testing
or curated job lists.

RemoteOK API:
  - Public, no auth required.
  - Base URL: https://remoteok.com/api
  - Filter by tag: https://remoteok.com/api?tag=python
  - First element of the response array is a legal notice — skip it.
  - Rate limit: be polite; we sleep 1 s between tag requests.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import httpx

from src.config.config_manager import AppConfig
from src.tracker.schema import upsert_job

logger = logging.getLogger(__name__)

REMOTEOK_BASE_URL = "https://remoteok.com/api"
REQUEST_TIMEOUT   = 30   # seconds
POLITE_DELAY      = 1.0  # seconds between tag requests

# Maps common job-title keywords → RemoteOK tag slugs
TITLE_TO_TAG: dict[str, str] = {
    "python":             "python",
    "javascript":         "javascript",
    "typescript":         "typescript",
    "react":              "react",
    "node":               "node",
    "backend":            "backend",
    "front end":          "frontend",
    "frontend":           "frontend",
    "full stack":         "fullstack",
    "fullstack":          "fullstack",
    "software engineer":  "software-engineer",
    "software developer": "dev",
    "web developer":      "web",
    "devops":             "devops",
    "go ":                "golang",
    "golang":             "golang",
    "rust":               "rust",
    "java ":              "java",
    "kotlin":             "kotlin",
    "ios":                "ios",
    "android":            "android",
    "machine learning":   "machine-learning",
    "data science":       "data-science",
}


# Data model

@dataclass
class JobListing:
    title:       str
    company:     str
    location:    str
    description: str
    apply_url:   str
    source:      str


# RemoteOK Scraper

class RemoteOKScraper:
    """
    Fetches remote job listings from RemoteOK's public JSON API.

    Derives relevant API tag slugs from the job titles configured in
    config.json, makes one request per tag, de-duplicates by job ID,
    and filters out excluded keywords.

    Usage:
        scraper = RemoteOKScraper(config)
        async for job in scraper.scrape(max_results=50):
            db_id = upsert_job(...)
    """

    HEADERS = {
        "User-Agent": "ApplyPilot/1.0 job-application-bot",
        "Accept":     "application/json",
    }

    def __init__(self, config: AppConfig):
        self.config = config

    def _tag_urls(self) -> list[str]:
        """
        Build a deduplicated list of RemoteOK tag URLs from the
        configured job titles.  Falls back to the generic /api endpoint
        if no title matches a known tag.
        """
        tags: set[str] = set()
        for title in self.config.search.titles:
            title_lower = title.lower()
            for keyword, tag in TITLE_TO_TAG.items():
                if keyword in title_lower:
                    tags.add(tag)

        if not tags:
            logger.info("No title→tag mapping found; fetching all remote dev jobs")
            return [REMOTEOK_BASE_URL + "?tag=dev"]

        return [f"{REMOTEOK_BASE_URL}?tag={tag}" for tag in sorted(tags)]

    async def scrape(
        self,
        max_results: int = 50,
    ) -> AsyncGenerator[JobListing, None]:
        """
        Async generator — yields JobListing objects from RemoteOK.

        Fetches each tag URL in sequence, de-duplicates across tags,
        and stops once max_results jobs have been yielded.
        """
        seen_ids: set[str] = set()
        yielded  = 0
        urls     = self._tag_urls()

        async with httpx.AsyncClient(
            headers=self.HEADERS,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            for url in urls:
                if yielded >= max_results:
                    break

                try:
                    logger.info(f"RemoteOK → {url}")
                    resp = await client.get(url)
                    resp.raise_for_status()
                    raw_list: list = resp.json()
                except Exception as e:
                    logger.warning(f"RemoteOK request failed ({url}): {e}")
                    continue

                for raw in raw_list:
                    if yielded >= max_results:
                        break

                    # Skip the legal-notice object (first element)
                    if not isinstance(raw, dict) or "position" not in raw:
                        continue

                    job_id = str(raw.get("id", ""))
                    if job_id in seen_ids:
                        continue
                    seen_ids.add(job_id)

                    title = raw.get("position", "").strip()
                    if not title or self._is_excluded(title) or not self._is_relevant(title):
                        continue

                    apply_url = (raw.get("apply_url") or raw.get("url") or "").strip()
                    if not apply_url:
                        continue

                    # Prepend tag list to description so the LLM scorer has it
                    tags_str    = ", ".join(raw.get("tags", []))
                    description = raw.get("description", "").strip()
                    if tags_str:
                        description = f"Skills/Tags: {tags_str}\n\n{description}"

                    yield JobListing(
                        title       = title,
                        company     = raw.get("company", "Unknown").strip(),
                        location    = raw.get("location", "Remote").strip() or "Remote",
                        description = description,
                        apply_url   = apply_url,
                        source      = "remoteok",
                    )
                    yielded += 1

                # Polite delay between tag requests
                if urls.index(url) < len(urls) - 1:
                    await asyncio.sleep(POLITE_DELAY)

        logger.info(f"RemoteOK scrape complete — {yielded} jobs fetched")


# JSON Feed Ingestor (generic fallback / custom feeds)

class JSONFeedIngestor:
    """
    Reads jobs from a local JSON file.

    Expected format (array of objects):
    [
      {
        "title":       "Backend Engineer",
        "company":     "Acme Corp",
        "location":    "Remote",
        "description": "We are looking for...",
        "apply_url":   "https://jobs.lever.co/acme/abc123"
      },
      ...
    ]

    Useful for:
      - Curated job lists exported from any job board
      - Internal testing without live API calls
      - Custom company career page exports
    """

    def __init__(self, feed_path: Path | str):
        self.feed_path = Path(feed_path)

    async def ingest(self) -> AsyncGenerator[JobListing, None]:
        """Yield JobListing objects from the JSON feed file."""
        if not self.feed_path.exists():
            raise FileNotFoundError(f"JSON feed not found: {self.feed_path}")

        with open(self.feed_path) as f:
            jobs: list[dict] = json.load(f)

        logger.info(f"JSON feed loaded: {len(jobs)} jobs from {self.feed_path}")

        for raw in jobs:
            yield JobListing(
                title       = raw.get("title",       "Unknown"),
                company     = raw.get("company",     "Unknown"),
                location    = raw.get("location",    ""),
                description = raw.get("description", ""),
                apply_url   = raw.get("apply_url",   ""),
                source      = "json_feed",
            )


# Orchestration helper — save scraped jobs to DB

async def ingest_and_store(
    scraper: RemoteOKScraper | JSONFeedIngestor,
    config:  AppConfig,
    max_results: int = 50,
) -> list[int]:
    """
    Run a scraper and persist all discovered jobs to SQLite.

    Returns list of job IDs that were inserted or already existed.
    """
    new_job_ids: list[int] = []

    if isinstance(scraper, JSONFeedIngestor):
        source = scraper.ingest()
    else:
        source = scraper.scrape(max_results=max_results)

    async for job in source:
        if not job.apply_url:
            continue
        job_id = upsert_job(
            title       = job.title,
            company     = job.company,
            apply_url   = job.apply_url,
            source      = job.source,
            description = job.description,
            location    = job.location,
        )
        new_job_ids.append(job_id)
        logger.info(f"Stored [{job_id}] {job.title} @ {job.company}")

    logger.info(f"Ingestion complete — {len(new_job_ids)} jobs stored/updated")
    return new_job_ids


# Internal helper shared by both scrapers

def _is_excluded(title: str, exclude_keywords: list[str]) -> bool:
    title_lower = title.lower()
    return any(kw.lower() in title_lower for kw in exclude_keywords)


# Patch the method onto both classes so they don't need to duplicate the logic
RemoteOKScraper._is_excluded = lambda self, t: _is_excluded(
    t, self.config.search.exclude_keywords
)

# Keywords derived from each configured title — any one match keeps the job
_RELEVANCE_KEYWORDS = [
    "software", "developer", "engineer", "backend", "back-end",
    "frontend", "front-end", "fullstack", "full-stack", "full stack",
    "python", "javascript", "typescript", "node", "react",
    "web dev", "devops", "sre", "platform", "api", "cloud",
]

def _is_relevant(title: str, search_titles: list[str]) -> bool:
    """
    Return True if the job title plausibly matches what we're looking for.

    Checks against both the user's configured titles and a broad set of
    tech-role keywords so we don't miss slightly differently-worded roles.
    """
    title_lower = title.lower()
    # Match any configured target title word
    for target in search_titles:
        for word in target.lower().split():
            if len(word) > 3 and word in title_lower:
                return True
    # Match broad tech-role vocabulary
    return any(kw in title_lower for kw in _RELEVANCE_KEYWORDS)

RemoteOKScraper._is_relevant = lambda self, t: _is_relevant(
    t, self.config.search.titles
)
