"""
scraper.py — Multi-source Job Scraper

All sources are free, require no authentication, and return CS-relevant jobs.

Sources (JSON API):
  1. Remotive       — remotive.com/api/remote-jobs
  2. Arbeitnow      — arbeitnow.com/api/job-board-api  (EU-heavy, great for GR)
  3. Jobicy         — jobicy.com/api/v2/remote-jobs
  4. Himalayas      — himalayas.app/jobs/api
  5. The Muse       — themuse.com/api/public/jobs
  6. HN Who's Hiring — hn.algolia.com (monthly Ask HN thread)

Sources (RSS / XML):
  7. We Work Remotely — weworkremotely.com
  8. Remote.co        — remote.co

Fallback:
  JSONFeedIngestor — reads a local jobs.json you curate manually
"""

import asyncio
import json
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncGenerator

import httpx

from src.config.config_manager import AppConfig
from src.tracker.schema import upsert_job

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT  = 10
POLITE_DELAY     = 0.5   # seconds between requests to the same host

TECH_KEYWORDS = {
    "software", "developer", "engineer", "backend", "back-end", "frontend",
    "front-end", "fullstack", "full-stack", "python", "javascript",
    "typescript", "node", "react", "vue", "angular", "django", "flask",
    "fastapi", "api", "devops", "sre", "platform", "cloud", "data",
    "machine learning", "ml", "ai", "web", "mobile", "ios", "android",
    "kotlin", "swift", "rust", "golang", "java ", "scala", "ruby",
    "php", "c++", "csharp", "c#", "qa", "automation", "test",
    "database", "sql", "nosql", "security", "cyber", "blockchain",
    "embedded", "firmware", "systems", "infrastructure", "architect",
    "tech lead", "engineering manager",
}


@dataclass
class JobListing:
    title:       str
    company:     str
    location:    str
    description: str
    apply_url:   str
    source:      str


def _is_relevant(title: str, exclude_keywords: list[str]) -> bool:
    t = title.lower()
    if any(kw.lower() in t for kw in exclude_keywords):
        return False
    return any(kw in t for kw in TECH_KEYWORDS)


# ── Source 1: Remotive ────────────────────────────────────────────────────────

async def _fetch_remotive(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    categories = ["software-dev", "devops-sysadmin", "data"]
    results: list[JobListing] = []
    seen: set[str] = set()

    for cat in categories:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                "https://remotive.com/api/remote-jobs",
                params={"category": cat, "limit": 100},
            )
            r.raise_for_status()
            for job in r.json().get("jobs", []):
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("url", "") or job.get("apply_url", "")
                if not url:
                    continue
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = job.get("company_name", ""),
                    location    = job.get("candidate_required_location", "Remote"),
                    description = job.get("description", ""),
                    apply_url   = url,
                    source      = "remotive",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"Remotive/{cat} error: {e}")

    logger.info(f"Remotive: {len(results)} jobs fetched")
    return results


# ── Source 2: Arbeitnow ───────────────────────────────────────────────────────

async def _fetch_arbeitnow(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    page = 1
    seen: set[str] = set()

    while len(results) < max_results:
        try:
            r = await client.get(
                "https://www.arbeitnow.com/api/job-board-api",
                params={"page": page},
            )
            r.raise_for_status()
            data = r.json()
            jobs = data.get("data", [])
            if not jobs:
                break
            for job in jobs:
                slug = job.get("slug", "")
                if slug in seen:
                    continue
                seen.add(slug)
                url = job.get("url", "")
                if not url:
                    continue
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = job.get("company_name", ""),
                    location    = job.get("location", "Remote"),
                    description = job.get("description", ""),
                    apply_url   = url,
                    source      = "arbeitnow",
                ))
            page += 1
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"Arbeitnow/p{page} error: {e}")
            break

    logger.info(f"Arbeitnow: {len(results)} jobs fetched")
    return results


# ── Source 3: Jobicy ──────────────────────────────────────────────────────────

async def _fetch_jobicy(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    tags = ["dev", "python", "javascript", "typescript", "fullstack", "backend"]
    results: list[JobListing] = []
    seen: set[str] = set()

    for tag in tags:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                "https://jobicy.com/api/v2/remote-jobs",
                params={"count": 50, "tag": tag},
            )
            r.raise_for_status()
            for job in r.json().get("jobs", []):
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("url", "")
                if not url:
                    continue
                results.append(JobListing(
                    title       = job.get("jobTitle", ""),
                    company     = job.get("companyName", ""),
                    location    = job.get("jobGeo", "Remote"),
                    description = job.get("jobDescription", ""),
                    apply_url   = url,
                    source      = "jobicy",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"Jobicy/{tag} error: {e}")

    logger.info(f"Jobicy: {len(results)} jobs fetched")
    return results


# ── Source 4: Himalayas ───────────────────────────────────────────────────────

async def _fetch_himalayas(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    offset = 0
    seen: set[str] = set()

    while len(results) < max_results:
        try:
            r = await client.get(
                "https://himalayas.app/jobs/api",
                params={"limit": 20, "offset": offset},
            )
            r.raise_for_status()
            jobs = r.json().get("jobs", [])
            if not jobs:
                break
            for job in jobs:
                uid = str(job.get("slug", job.get("id", "")))
                if uid in seen:
                    continue
                seen.add(uid)
                # Apply URL is the Himalayas listing page which links out
                url = job.get("applicationLink") or job.get("url") or \
                      f"https://himalayas.app/jobs/{uid}"
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = job.get("companyName", ""),
                    location    = job.get("location", "Remote"),
                    description = job.get("description", ""),
                    apply_url   = url,
                    source      = "himalayas",
                ))
            offset += 20
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"Himalayas/offset={offset} error: {e}")
            break

    logger.info(f"Himalayas: {len(results)} jobs fetched")
    return results


# ── Source 5: The Muse ────────────────────────────────────────────────────────

async def _fetch_themuse(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    categories = ["Computer and IT", "Data Science", "Software Engineer",
                  "IT Infrastructure", "QA", "Product Management"]
    results: list[JobListing] = []
    seen: set[str] = set()

    for cat in categories:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                "https://www.themuse.com/api/public/jobs",
                params={"category": cat, "page": 1},
            )
            r.raise_for_status()
            for job in r.json().get("results", []):
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = (job.get("refs", {}) or {}).get("landing_page", "")
                if not url:
                    continue
                company = (job.get("company", {}) or {}).get("name", "")
                locations = [
                    loc.get("name", "") for loc in (job.get("locations", []) or [])
                ]
                results.append(JobListing(
                    title       = job.get("name", ""),
                    company     = company,
                    location    = ", ".join(locations) or "Remote",
                    description = job.get("contents", ""),
                    apply_url   = url,
                    source      = "themuse",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"TheMuse/{cat} error: {e}")

    logger.info(f"TheMuse: {len(results)} jobs fetched")
    return results


# ── Source 6: We Work Remotely (RSS) ─────────────────────────────────────────

async def _fetch_weworkremotely(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    feeds = [
        "https://weworkremotely.com/categories/remote-programming-jobs.rss",
        "https://weworkremotely.com/categories/remote-devops-sysadmin-jobs.rss",
        "https://weworkremotely.com/categories/remote-full-stack-programming-jobs.rss",
    ]
    results: list[JobListing] = []
    seen: set[str] = set()

    for feed_url in feeds:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(feed_url)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                link  = (item.findtext("link") or "").strip()
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "").strip()
                # WWR titles look like "Company: Job Title"
                if ": " in title:
                    company, title = title.split(": ", 1)
                else:
                    company = ""
                if link in seen or not link:
                    continue
                seen.add(link)
                results.append(JobListing(
                    title       = title,
                    company     = company,
                    location    = "Remote",
                    description = desc,
                    apply_url   = link,
                    source      = "weworkremotely",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"WeWorkRemotely error: {e}")

    logger.info(f"WeWorkRemotely: {len(results)} jobs fetched")
    return results


# ── Source 7: Remote.co (RSS) ─────────────────────────────────────────────────

async def _fetch_remoteco(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    feeds = [
        "https://remote.co/remote-jobs/developer/feed/",
        "https://remote.co/remote-jobs/engineer/feed/",
    ]
    results: list[JobListing] = []
    seen: set[str] = set()

    for feed_url in feeds:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(feed_url)
            r.raise_for_status()
            root = ET.fromstring(r.content)
            for item in root.findall(".//item"):
                link  = (item.findtext("link") or "").strip()
                title = (item.findtext("title") or "").strip()
                desc  = (item.findtext("description") or "").strip()
                if link in seen or not link:
                    continue
                seen.add(link)
                results.append(JobListing(
                    title       = title,
                    company     = "",
                    location    = "Remote",
                    description = desc,
                    apply_url   = link,
                    source      = "remoteco",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"Remote.co error: {e}")

    logger.info(f"Remote.co: {len(results)} jobs fetched")
    return results


# ── Source 8: HN Who's Hiring (Algolia) ──────────────────────────────────────

async def _fetch_hn_hiring(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    """
    Queries the latest Hacker News 'Ask HN: Who is hiring?' thread via Algolia.
    Each comment is a raw job posting — we parse title/company/url heuristically.
    """
    results: list[JobListing] = []
    try:
        # Find the latest "Who is hiring" story
        r = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "query": "Ask HN: Who is hiring",
                "tags": "story",
                "hitsPerPage": 1,
            },
        )
        r.raise_for_status()
        hits = r.json().get("hits", [])
        if not hits:
            return results
        story_id = hits[0]["objectID"]

        # Fetch top-level comments
        r2 = await client.get(
            "https://hn.algolia.com/api/v1/search",
            params={
                "tags": f"comment,story_{story_id}",
                "hitsPerPage": 100,
            },
        )
        r2.raise_for_status()
        for hit in r2.json().get("hits", []):
            text = hit.get("comment_text", "") or ""
            if not text:
                continue
            # First line usually: "Company | Role | Location | ..."
            first_line = text.split("<p>")[0].strip()
            parts = [p.strip() for p in first_line.split("|")]
            company = parts[0] if parts else "Unknown"
            title   = parts[1] if len(parts) > 1 else "Software Engineer"
            url     = f"https://news.ycombinator.com/item?id={hit['objectID']}"
            results.append(JobListing(
                title       = title,
                company     = company,
                location    = parts[2] if len(parts) > 2 else "Remote",
                description = text,
                apply_url   = url,
                source      = "hn_hiring",
            ))
            if len(results) >= max_results:
                break
    except Exception as e:
        logger.warning(f"HN Hiring error: {e}")

    logger.info(f"HN Who's Hiring: {len(results)} jobs fetched")
    return results


# ── Multi-source aggregator ───────────────────────────────────────────────────

class MultiSourceScraper:
    """
    Aggregates jobs from all 8 free, no-auth sources in parallel.
    Applies relevance + exclusion filtering before yielding.
    """

    HEADERS = {
        "User-Agent": "ApplyPilot/1.0 job-search-bot",
        "Accept":     "application/json, text/xml, */*",
    }

    FETCHERS = [
        ("remotive",       _fetch_remotive),
        ("arbeitnow",      _fetch_arbeitnow),
        ("jobicy",         _fetch_jobicy),
        ("himalayas",      _fetch_himalayas),
        ("themuse",        _fetch_themuse),
        ("weworkremotely", _fetch_weworkremotely),
        ("hn_hiring",      _fetch_hn_hiring),
        ("workingnomads",  _fetch_workingnomads),
        ("authenticjobs",  _fetch_authentic_jobs),
        ("jobspresso",     _fetch_jobspresso),
        ("greenhouse",     _fetch_greenhouse),
        ("lever",          _fetch_lever),
        ("ashby",          _fetch_ashby),
    ]

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    async def scrape(
        self, max_results: int = 100
    ) -> AsyncGenerator[JobListing, None]:
        """
        Fetch from all sources concurrently, then yield de-duplicated,
        relevance-filtered jobs up to max_results.
        """
        per_source = max(20, max_results // len(self.FETCHERS))

        async with httpx.AsyncClient(
            headers=self.HEADERS,
            timeout=REQUEST_TIMEOUT,
            follow_redirects=True,
        ) as client:
            SOURCE_TIMEOUT = 15   # seconds — any single source hanging beyond this is cancelled

            async def _safe_fetch(name: str, fetcher, max_r: int):
                try:
                    return await asyncio.wait_for(
                        fetcher(client, max_r), timeout=SOURCE_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"{name}: timed out after {SOURCE_TIMEOUT}s — skipping")
                    return []
                except Exception as e:
                    logger.warning(f"{name}: error — {e}")
                    return []

            tasks = [
                _safe_fetch(name, fetcher, per_source)
                for name, fetcher in self.FETCHERS
            ]
            all_results = await asyncio.gather(*tasks)

        seen_urls: set[str] = set()
        yielded   = 0

        for batch in all_results:
            for job in batch:
                if yielded >= max_results:
                    return
                if not job.apply_url or job.apply_url in seen_urls:
                    continue
                if not job.title:
                    continue
                if not _is_relevant(job.title, self.config.search.exclude_keywords):
                    continue
                seen_urls.add(job.apply_url)
                yielded += 1
                yield job

        logger.info(f"Multi-source scrape complete — {yielded} jobs total")


# ── JSON Feed fallback ────────────────────────────────────────────────────────

class JSONFeedIngestor:
    """
    Reads jobs from a local JSON file you maintain manually.

    Format:
    [
      {
        "title": "Backend Engineer",
        "company": "Acme",
        "location": "Remote",
        "description": "...",
        "apply_url": "https://..."
      }
    ]
    """

    def __init__(self, feed_path: Path | str) -> None:
        self.feed_path = Path(feed_path)

    async def ingest(self) -> AsyncGenerator[JobListing, None]:
        if not self.feed_path.exists():
            raise FileNotFoundError(f"JSON feed not found: {self.feed_path}")
        with open(self.feed_path) as f:
            jobs: list[dict] = json.load(f)
        logger.info(f"JSON feed: {len(jobs)} jobs from {self.feed_path}")
        for raw in jobs:
            yield JobListing(
                title       = raw.get("title",       "Unknown"),
                company     = raw.get("company",     "Unknown"),
                location    = raw.get("location",    ""),
                description = raw.get("description", ""),
                apply_url   = raw.get("apply_url",   ""),
                source      = "json_feed",
            )


# ── Orchestration helper ──────────────────────────────────────────────────────

async def ingest_and_store(
    scraper: MultiSourceScraper | JSONFeedIngestor,
    config:  AppConfig,
    max_results: int = 200,
) -> list[int]:
    """Run scraper and persist all jobs to SQLite. Returns stored job IDs."""
    new_ids: list[int] = []

    source = (
        scraper.ingest() if isinstance(scraper, JSONFeedIngestor)
        else scraper.scrape(max_results=max_results)
    )

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
        new_ids.append(job_id)
        logger.info(f"Stored [{job_id}] {job.title} @ {job.company} ({job.source})")

    logger.info(f"Ingestion complete — {len(new_ids)} jobs stored/updated")
    return new_ids


# ── Source 8: Working Nomads ──────────────────────────────────────────────────

async def _fetch_workingnomads(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    categories = ["development", "system-admin-devops", "data"]
    seen: set[str] = set()

    for cat in categories:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                "https://www.workingnomads.com/api/exposed_jobs/",
                params={"category": cat, "limit": 50},
            )
            r.raise_for_status()
            for job in r.json():
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("url", "") or job.get("apply_url", "")
                if not url:
                    continue
                company = (job.get("company") or {}).get("name", "") \
                          if isinstance(job.get("company"), dict) \
                          else str(job.get("company", ""))
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = company,
                    location    = job.get("location", "Remote"),
                    description = job.get("description", ""),
                    apply_url   = url,
                    source      = "workingnomads",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.warning(f"WorkingNomads/{cat} error: {e}")

    logger.info(f"WorkingNomads: {len(results)} jobs fetched")
    return results


# ── Source 9: Authentic Jobs (RSS) ────────────────────────────────────────────

async def _fetch_authentic_jobs(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    seen: set[str] = set()
    try:
        r = await client.get(
            "https://authenticjobs.com/feed/",
            params={"type": "1", "location": "remote"},
        )
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            link  = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            creator = (item.findtext("{http://purl.org/dc/elements/1.1/}creator") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            results.append(JobListing(
                title       = title,
                company     = creator,
                location    = "Remote",
                description = desc,
                apply_url   = link,
                source      = "authenticjobs",
            ))
    except Exception as e:
        logger.warning(f"AuthenticJobs error: {e}")

    logger.info(f"AuthenticJobs: {len(results)} jobs fetched")
    return results


# ── Source 10: Jobspresso (RSS) ───────────────────────────────────────────────

async def _fetch_jobspresso(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    seen: set[str] = set()
    try:
        r = await client.get("https://jobspresso.co/feed/")
        r.raise_for_status()
        root = ET.fromstring(r.content)
        for item in root.findall(".//item"):
            link  = (item.findtext("link") or "").strip()
            title = (item.findtext("title") or "").strip()
            desc  = (item.findtext("description") or "").strip()
            creator = (item.findtext("{http://purl.org/dc/elements/1.1/}creator") or "").strip()
            if not link or link in seen:
                continue
            seen.add(link)
            results.append(JobListing(
                title       = title,
                company     = creator,
                location    = "Remote",
                description = desc,
                apply_url   = link,
                source      = "jobspresso",
            ))
    except Exception as e:
        logger.warning(f"Jobspresso error: {e}")

    logger.info(f"Jobspresso: {len(results)} jobs fetched")
    return results


# ── Source 11: Greenhouse (public company boards) ─────────────────────────────
# No auth needed for GET endpoints per Greenhouse docs.
# Token = company's board slug (usually their domain name without TLD).

GREENHOUSE_COMPANIES = [
    # Remote-first / well-known for remote hiring
    "gitlab", "hashicorp", "automattic", "zapier", "invision",
    "hubspot", "datadog", "elastic", "mongodb", "confluent",
    "cloudflare", "twilio", "segment", "postman", "grafana",
    "sourcegraph", "1password", "temporal", "deno", "fly",
    # Startups & scale-ups
    "notion", "linear", "retool", "airtable", "loom",
    "figma", "miro", "coda", "pitch", "rows",
    "vercel", "netlify", "supabase", "appwrite", "novu",
    "resend", "trigger", "inngest", "upstash", "turso",
    # Fintech & SaaS
    "gusto", "rippling", "brex", "mercury", "ramp",
    "plaid", "stripe", "adyen", "checkout", "paddle",
]


async def _fetch_greenhouse(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    seen: set[str] = set()

    for company in GREENHOUSE_COMPANIES:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                f"https://api.greenhouse.io/v1/boards/{company}/jobs",
                params={"content": "true"},
            )
            if r.status_code == 404:
                continue   # company not on Greenhouse or wrong slug
            r.raise_for_status()
            for job in r.json().get("jobs", []):
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("absolute_url", "")
                if not url:
                    continue
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = company.title(),
                    location    = (job.get("location") or {}).get("name", "Remote"),
                    description = job.get("content", ""),
                    apply_url   = url,
                    source      = "greenhouse",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.debug(f"Greenhouse/{company}: {e}")

    logger.info(f"Greenhouse: {len(results)} jobs fetched")
    return results


# ── Source 12: Lever (public company boards) ──────────────────────────────────

LEVER_COMPANIES = [
    # Remote-friendly companies known to use Lever
    "buffer", "doist", "remote", "deel", "hotjar",
    "intercom", "mixpanel", "amplitude", "heap", "fullstory",
    "netlify", "render", "railway", "warp", "zed",
    "scale-ai", "cohere", "perplexity", "together-ai", "replicate",
    "huggingface", "modal", "anyscale", "ray", "prefect",
    "dagster", "airbyte", "fivetran", "dbt-labs", "hightouch",
    "census", "rudderstack", "segment", "mparticle", "snowplow",
]


async def _fetch_lever(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    seen: set[str] = set()

    for company in LEVER_COMPANIES:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                f"https://api.lever.co/v0/postings/{company}",
                params={"mode": "json"},
            )
            if r.status_code in (404, 401):
                continue
            r.raise_for_status()
            for job in r.json():
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("hostedUrl", "") or job.get("applyUrl", "")
                if not url:
                    continue
                cats = job.get("categories", {}) or {}
                results.append(JobListing(
                    title       = job.get("text", ""),
                    company     = company.replace("-", " ").title(),
                    location    = cats.get("location", "Remote"),
                    description = (job.get("descriptionBody") or
                                   job.get("description", "")),
                    apply_url   = url,
                    source      = "lever",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.debug(f"Lever/{company}: {e}")

    logger.info(f"Lever: {len(results)} jobs fetched")
    return results


# ── Source 13: Ashby (public company boards) ──────────────────────────────────

ASHBY_COMPANIES = [
    # Startups that have adopted Ashby as their ATS
    "anthropic", "mistral", "perplexity", "cursor", "replit",
    "fly", "neon", "turso", "convex", "liveblocks",
    "electric-sql", "powersync", "triplit", "evolu", "livestore",
    "val-town", "e2b", "modal", "baseten", "lepton",
    "braintrust", "langchain", "llamaindex", "qdrant", "weaviate",
    "chroma", "pinecone", "milvus", "zilliz", "vespa",
]


async def _fetch_ashby(
    client: httpx.AsyncClient, max_results: int
) -> list[JobListing]:
    results: list[JobListing] = []
    seen: set[str] = set()

    for company in ASHBY_COMPANIES:
        if len(results) >= max_results:
            break
        try:
            r = await client.get(
                f"https://api.ashbyhq.com/posting-api/job-board/{company}",
            )
            if r.status_code in (404, 400):
                continue
            r.raise_for_status()
            for job in r.json().get("jobPostings", []):
                uid = str(job.get("id", ""))
                if uid in seen:
                    continue
                seen.add(uid)
                url = job.get("jobUrl", "") or job.get("applyUrl", "")
                if not url:
                    continue
                results.append(JobListing(
                    title       = job.get("title", ""),
                    company     = company.replace("-", " ").title(),
                    location    = job.get("locationName", "Remote"),
                    description = job.get("descriptionHtml", ""),
                    apply_url   = url,
                    source      = "ashby",
                ))
            await asyncio.sleep(POLITE_DELAY)
        except Exception as e:
            logger.debug(f"Ashby/{company}: {e}")

    logger.info(f"Ashby: {len(results)} jobs fetched")
    return results
