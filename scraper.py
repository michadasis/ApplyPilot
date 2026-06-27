"""
scraper.py — Job Scraper / Ingestor Module

Navigates to job board search URLs, extracts job listings, and persists
them to SQLite. Supports LinkedIn (primary) and a generic JSON feed fallback.

Anti-detection strategy:
  - Persistent browser context (shared with form filler) = logged-in session
  - Randomized scroll behavior and inter-action delays
  - Extracts only publicly visible data (no private API calls)
"""

import asyncio
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

from playwright.async_api import BrowserContext, Page, TimeoutError as PlaywrightTimeout

from src.config.config_manager import AppConfig
from src.tracker.schema import upsert_job, update_job_status, get_db

logger = logging.getLogger(__name__)

NAVIGATION_TIMEOUT = 30_000
SCROLL_PAUSE_MS    = (800, 1800)   # Random range for scroll pauses


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class JobListing:
    title:       str
    company:     str
    location:    str
    description: str
    apply_url:   str
    source:      str


# ---------------------------------------------------------------------------
# LinkedIn Scraper
# ---------------------------------------------------------------------------

class LinkedInScraper:
    """
    Scrapes LinkedIn job search results using a persistent logged-in context.

    LinkedIn heavily rate-limits unauthenticated scraping; a persisted
    browser session (from build_browser_context()) sidesteps most of this.

    Usage:
        async for job in scraper.scrape(search_url):
            db_id = upsert_job(...)
    """

    # CSS selectors (as of 2024 — may need updates if LinkedIn redesigns)
    SELECTORS = {
        "job_cards":    "ul.jobs-search__results-list li",
        "title":        "h3.base-search-card__title",
        "company":      "h4.base-search-card__subtitle",
        "location":     "span.job-search-card__location",
        "apply_link":   "a.base-card__full-link",
        # Detail page selectors
        "description":  "div.show-more-less-html__markup",
        "easy_apply":   "button.jobs-apply-button",
        "external_url": "a[data-tracking-control-name='public_jobs_apply-link-offsite']",
    }

    def __init__(self, config: AppConfig, context: BrowserContext):
        self.config  = config
        self.context = context

    def build_search_url(self, title: str, location: str) -> str:
        """Construct a LinkedIn job search URL for a given title+location."""
        import urllib.parse
        params = urllib.parse.urlencode({
            "keywords": title,
            "location": location,
            "f_TPR":    "r86400",   # Posted in last 24h
            "f_LF":     "f_AL",     # LinkedIn Easy Apply filter (remove if you want all)
            "sortBy":   "DD",       # Most recent first
        })
        return f"https://www.linkedin.com/jobs/search/?{params}"

    async def scrape(
        self,
        search_url: str,
        max_results: int = 25,
    ) -> AsyncGenerator[JobListing, None]:
        """
        Async generator — yields JobListing objects as they're scraped.

        Args:
            search_url  : LinkedIn jobs search URL.
            max_results : Maximum jobs to extract per call.
        """
        page = await self.context.new_page()
        try:
            logger.info(f"Navigating to LinkedIn search: {search_url}")
            await page.goto(search_url, wait_until="networkidle",
                            timeout=NAVIGATION_TIMEOUT)

            # Scroll to load lazy-loaded job cards
            await self._scroll_to_load(page, target_count=max_results)

            # Extract all visible job card URLs first
            cards = await page.query_selector_all(self.SELECTORS["job_cards"])
            logger.info(f"Found {len(cards)} job cards")

            yielded = 0
            for card in cards[:max_results]:
                try:
                    listing = await self._extract_card_preview(card, page)
                    if not listing:
                        continue

                    # Check exclusion keywords before opening detail page
                    if self._is_excluded(listing.title):
                        logger.debug(f"Skipping excluded job: {listing.title}")
                        continue

                    # Fetch full description from detail page
                    listing = await self._fetch_description(listing, page)

                    yield listing
                    yielded += 1

                    # Polite delay between requests
                    await asyncio.sleep(random.uniform(1.5, 3.5))

                except Exception as e:
                    logger.warning(f"Error processing card: {e}")
                    continue

            logger.info(f"Scraped {yielded} jobs from LinkedIn")

        finally:
            await page.close()

    async def _extract_card_preview(
        self, card, page: Page
    ) -> JobListing | None:
        """Extract title, company, location, and URL from a search result card."""
        try:
            title_el   = await card.query_selector(self.SELECTORS["title"])
            company_el = await card.query_selector(self.SELECTORS["company"])
            loc_el     = await card.query_selector(self.SELECTORS["location"])
            link_el    = await card.query_selector(self.SELECTORS["apply_link"])

            if not (title_el and link_el):
                return None

            title    = (await title_el.inner_text()).strip()
            company  = (await company_el.inner_text()).strip() if company_el else "Unknown"
            location = (await loc_el.inner_text()).strip()    if loc_el     else ""
            url      = await link_el.get_attribute("href") or ""

            # Normalize URL — strip tracking params
            url = url.split("?")[0] if "?" in url else url

            return JobListing(
                title=title,
                company=company,
                location=location,
                description="",   # filled in by _fetch_description
                apply_url=url,
                source="linkedin",
            )
        except Exception as e:
            logger.debug(f"Card extraction failed: {e}")
            return None

    async def _fetch_description(
        self, listing: JobListing, page: Page
    ) -> JobListing:
        """
        Open the job detail page and extract the full description + real apply URL.

        LinkedIn Easy Apply jobs have an in-platform apply flow; external jobs
        redirect to the company's ATS. We want the external ATS URL when available.
        """
        try:
            await page.goto(listing.apply_url, wait_until="domcontentloaded",
                            timeout=NAVIGATION_TIMEOUT)

            desc_el = await page.query_selector(self.SELECTORS["description"])
            if desc_el:
                listing.description = (await desc_el.inner_text()).strip()

            # Check for external apply URL (preferred over Easy Apply)
            ext_link = await page.query_selector(self.SELECTORS["external_url"])
            if ext_link:
                ext_url = await ext_link.get_attribute("href")
                if ext_url:
                    listing.apply_url = ext_url
                    logger.debug(f"External apply URL: {ext_url}")

        except PlaywrightTimeout:
            logger.warning(f"Timeout fetching description for: {listing.title}")
        except Exception as e:
            logger.warning(f"Description fetch error: {e}")

        return listing

    async def _scroll_to_load(self, page: Page, target_count: int) -> None:
        """Scroll down the results list to trigger lazy loading of job cards."""
        for _ in range(max(3, target_count // 5)):
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.4)")
            await page.wait_for_timeout(random.randint(*SCROLL_PAUSE_MS))

            # Click "Show more results" button if present
            more_btn = await page.query_selector("button.infinite-scroller__show-more-button")
            if more_btn and await more_btn.is_visible():
                await more_btn.click()
                await page.wait_for_timeout(1500)

    def _is_excluded(self, title: str) -> bool:
        """Return True if the job title contains any exclusion keyword."""
        title_lower = title.lower()
        return any(
            kw.lower() in title_lower
            for kw in self.config.search.exclude_keywords
        )


# ---------------------------------------------------------------------------
# JSON Feed Ingestor (generic fallback / custom feeds)
# ---------------------------------------------------------------------------

class JSONFeedIngestor:
    """
    Reads jobs from a local or remote JSON file.

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
      - Curated job lists exported from job boards
      - Internal testing without live scraping
      - Custom company career page feeds
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
                title       = raw.get("title", "Unknown"),
                company     = raw.get("company", "Unknown"),
                location    = raw.get("location", ""),
                description = raw.get("description", ""),
                apply_url   = raw.get("apply_url", ""),
                source      = "json_feed",
            )


# ---------------------------------------------------------------------------
# Orchestration helper — save scraped jobs to DB
# ---------------------------------------------------------------------------

async def ingest_and_store(
    scraper: LinkedInScraper | JSONFeedIngestor,
    config: AppConfig,
    search_url: str | None = None,
    max_results: int = 50,
) -> list[int]:
    """
    Run a scraper and persist all discovered jobs to SQLite.

    Returns list of newly inserted job IDs (existing jobs are silently skipped).
    """
    new_job_ids: list[int] = []

    if isinstance(scraper, LinkedInScraper):
        # Build search URLs for each title × location combination
        for title in config.search.titles:
            for location in config.search.locations:
                url = search_url or scraper.build_search_url(title, location)
                async for job in scraper.scrape(url, max_results=max_results):
                    job_id = upsert_job(
                        title       = job.title,
                        company     = job.company,
                        apply_url   = job.apply_url,
                        source      = job.source,
                        description = job.description,
                        location    = job.location,
                    )
                    new_job_ids.append(job_id)
                    logger.info(f"Stored job [{job_id}]: {job.title} @ {job.company}")
    else:
        # JSON feed — single pass
        async for job in scraper.ingest():
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

    logger.info(f"Ingestion complete — {len(new_job_ids)} jobs stored/updated")
    return new_job_ids
