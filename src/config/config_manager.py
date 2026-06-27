"""
config_manager.py — Centralized configuration and secret management.

Loads `config.json` for user preferences and `.env` for secrets.
All other modules import from here — no raw os.getenv() calls elsewhere.
"""

import json
import os
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Resolve project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "config.json"
ENV_PATH     = PROJECT_ROOT / ".env"
RESUME_DIR   = PROJECT_ROOT / "resume"


# Dataclasses — typed config models

@dataclass
class JobSearchConfig:
    """What kinds of jobs to target."""
    titles: list[str]          = field(default_factory=lambda: ["Software Engineer"])
    locations: list[str]       = field(default_factory=lambda: ["Remote"])
    exclude_keywords: list[str]= field(default_factory=list)   # e.g. ["senior", "staff"]
    min_match_score: float     = 0.70                           # 0.0–1.0

@dataclass
class BotBehaviorConfig:
    """Runtime behavior toggles."""
    require_manual_review: bool = True      # Pause before final Submit
    headless: bool              = False     # Show browser window for debugging
    slow_mo_ms: int             = 80        # ms delay between Playwright actions
    max_applications_per_run: int = 10      # Hard cap per invocation
    screenshot_on_failure: bool = True
    retry_failed_attempts: int  = 1         # Times to retry a failed form fill

@dataclass
class ATSConfig:
    """Per-ATS overrides (rarely needed but useful for edge cases)."""
    workday_company_ids: list[str] = field(default_factory=list)
    preferred_ats: list[str]       = field(default_factory=lambda: [
        "greenhouse", "lever", "workday"
    ])

@dataclass
class AppConfig:
    """Top-level config object. Single instance per process."""
    search:   JobSearchConfig   = field(default_factory=JobSearchConfig)
    behavior: BotBehaviorConfig = field(default_factory=BotBehaviorConfig)
    ats:      ATSConfig         = field(default_factory=ATSConfig)

    # Secrets — populated from .env, never from config.json
    anthropic_api_key: str = ""
    resume_pdf_path:   Path = RESUME_DIR / "resume.pdf"
    resume_md_path:    Path = RESUME_DIR / "resume.md"   # optional plaintext version


# Loader

def load_config() -> AppConfig:
    """
    1. Load .env into environment
    2. Parse config.json into AppConfig dataclasses
    3. Inject secrets from environment variables

    Raises FileNotFoundError if config.json is missing.
    Raises ValueError if required secrets are absent.
    """
    # 1. Load .env (silently ignores missing file — CI may set vars directly)
    load_dotenv(dotenv_path=ENV_PATH)

    # 2. Read config.json
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"config.json not found at {CONFIG_PATH}. "
            "Copy config.example.json and fill in your preferences."
        )

    with open(CONFIG_PATH) as f:
        raw: dict[str, Any] = json.load(f)

    # Build typed sub-configs from dict keys (extra keys are silently ignored)
    search_cfg = JobSearchConfig(
        titles           = raw.get("target_titles", ["Software Engineer"]),
        locations        = raw.get("target_locations", ["Remote"]),
        exclude_keywords = raw.get("exclude_keywords", []),
        min_match_score  = float(raw.get("min_match_score", 0.70)),
    )

    behavior_raw = raw.get("bot_behavior", {})
    behavior_cfg = BotBehaviorConfig(
        require_manual_review     = behavior_raw.get("require_manual_review", True),
        headless                  = behavior_raw.get("headless", False),
        slow_mo_ms                = int(behavior_raw.get("slow_mo_ms", 80)),
        max_applications_per_run  = int(behavior_raw.get("max_applications_per_run", 10)),
        screenshot_on_failure     = behavior_raw.get("screenshot_on_failure", True),
        retry_failed_attempts     = int(behavior_raw.get("retry_failed_attempts", 1)),
    )

    ats_raw = raw.get("ats", {})
    ats_cfg = ATSConfig(
        workday_company_ids = ats_raw.get("workday_company_ids", []),
        preferred_ats       = ats_raw.get("preferred_ats", ["greenhouse", "lever", "workday"]),
    )

    # 3. Secrets from environment (never hardcoded)
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise ValueError(
            "ANTHROPIC_API_KEY is not set. "
            "Add it to your .env file or export it as an environment variable."
        )

    cfg = AppConfig(
        search   = search_cfg,
        behavior = behavior_cfg,
        ats      = ats_cfg,
        anthropic_api_key = api_key,
        resume_pdf_path   = Path(os.getenv("RESUME_PDF_PATH", str(RESUME_DIR / "resume.pdf"))),
        resume_md_path    = Path(os.getenv("RESUME_MD_PATH",  str(RESUME_DIR / "resume.md"))),
    )

    logger.info(
        f"Config loaded | targets={cfg.search.titles} | "
        f"manual_review={cfg.behavior.require_manual_review}"
    )
    return cfg


# Example config.json template (written if missing, for first-run UX)

EXAMPLE_CONFIG: dict[str, Any] = {
    "target_titles": ["Software Engineer", "Backend Engineer", "Python Developer"],
    "target_locations": ["Remote", "New York, NY"],
    "exclude_keywords": ["senior", "staff", "principal", "director", "10+ years"],
    "min_match_score": 0.70,
    "bot_behavior": {
        "require_manual_review": True,
        "headless": False,
        "slow_mo_ms": 80,
        "max_applications_per_run": 5,
        "screenshot_on_failure": True,
        "retry_failed_attempts": 1,
    },
    "ats": {
        "workday_company_ids": [],
        "preferred_ats": ["greenhouse", "lever", "workday"],
    },
}


def write_example_config() -> None:
    """Write config.example.json to project root if it doesn't exist."""
    example_path = PROJECT_ROOT / "config.example.json"
    if not example_path.exists():
        with open(example_path, "w") as f:
            json.dump(EXAMPLE_CONFIG, f, indent=2)
        print(f"✅ Example config written to {example_path}")
