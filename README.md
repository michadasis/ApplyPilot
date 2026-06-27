# 🤖 Auto-Apply Job Bot

A modular, local-first job application automation system powered by Playwright and Claude AI. Scrapes job listings, scores them against your resume with an LLM, and fills out ATS application forms automatically — with a human-in-the-loop safeguard before any submission.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                          main.py (Pipeline)                          │
│  Phase 1: SCRAPE → Phase 2: SCORE → Phase 3: APPLY → Phase 4: REPORT│
└──────────┬──────────────┬──────────────┬──────────────┬─────────────┘
           │              │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌───▼────────┐ ┌──▼──────────┐
    │   Scraper   │ │  Matcher   │ │Form Filler │ │  Tracker    │
    │             │ │            │ │  Engine    │ │             │
    │ LinkedIn    │ │ LLM Score  │ │ Playwright │ │ SQLite logs │
    │ JSON Feed   │ │ Resume     │ │ Greenhouse │ │ Screenshots │
    └──────┬──────┘ │ Match      │ │ Lever      │ └─────────────┘
           │        └─────┬──────┘ │ Workday    │
           │              │        │ Generic    │
           └──────────────▼────────▼────────────┘
                          │
                    ┌─────▼──────┐
                    │  SQLite DB │
                    │  jobs.db   │
                    └────────────┘
```

---

## Database Schema

### `jobs` table — every discovered job posting
| Column         | Type    | Description                                      |
|----------------|---------|--------------------------------------------------|
| `id`           | INTEGER | Primary key                                      |
| `title`        | TEXT    | Job title                                        |
| `company`      | TEXT    | Company name                                     |
| `location`     | TEXT    | Job location                                     |
| `description`  | TEXT    | Full job description text                        |
| `apply_url`    | TEXT    | Unique apply link (dedup key)                    |
| `source`       | TEXT    | `linkedin` \| `indeed` \| `json_feed`            |
| `match_score`  | REAL    | LLM match score 0.0–1.0                         |
| `status`       | TEXT    | `saved` → `queued` → `applying` → `applied`      |
| `created_at`   | TEXT    | ISO timestamp                                    |
| `updated_at`   | TEXT    | Auto-updated by trigger                          |

**Status FSM:**
```
saved ──[LLM score pass]──► queued ──[apply starts]──► applying
  │                                                        │
  └──[score too low]──► skipped          ┌─── submitted ──┘
                                         ├─── applied
                                         ├─── failed
                                         └─── skipped (user skip at review)
```

### `applications` table — one row per form-fill attempt
| Column           | Type    | Description                              |
|------------------|---------|------------------------------------------|
| `id`             | INTEGER | Primary key                              |
| `job_id`         | INTEGER | FK → `jobs.id`                           |
| `ats_type`       | TEXT    | `greenhouse` \| `lever` \| `workday` \| `custom` |
| `attempt_number` | INTEGER | Retry counter                            |
| `status`         | TEXT    | `in_progress` \| `submitted` \| `failed` |
| `cover_letter`   | TEXT    | Generated cover letter body              |
| `answers_json`   | TEXT    | JSON: `{field_label: generated_answer}`  |
| `error_message`  | TEXT    | Failure reason                           |
| `screenshot_path`| TEXT    | Path to failure screenshot               |
| `submitted_at`   | TEXT    | Submission timestamp                     |

### `llm_calls` table — API cost tracking
| Column              | Type    | Description               |
|---------------------|---------|---------------------------|
| `purpose`           | TEXT    | `match` \| `field_extract` \| `answer_gen` \| `cover_letter` |
| `model`             | TEXT    | Model ID string           |
| `prompt_tokens`     | INTEGER | Input token count         |
| `completion_tokens` | INTEGER | Output token count        |
| `cost_usd`          | REAL    | Estimated USD cost        |

---

## Project File Tree

```
auto-apply-bot/
│
├── main.py                          # 🚀 Entry point — CLI + pipeline orchestrator
├── requirements.txt                 # Python dependencies
├── config.example.json              # Copy → config.json and customize
├── .env.example                     # Copy → .env and add secrets
│
├── resume/
│   ├── resume.pdf                   # 📄 Your resume (required)
│   └── resume.md                    # Optional plaintext version
│
├── data/
│   ├── jobs.db                      # SQLite database (auto-created)
│   ├── browser_profile/             # Persistent Chrome session (auto-created)
│   └── sample_jobs.json             # Example JSON feed for testing
│
├── logs/
│   └── auto_apply_YYYY-MM-DD.log    # Rotating daily log files
│
├── screenshots/
│   └── failure_job42_20240610.png   # Auto-captured on form errors
│
└── src/
    │
    ├── config/
    │   ├── __init__.py
    │   └── config_manager.py        # ⚙️  Config + secret management
    │
    ├── scraper/
    │   ├── __init__.py
    │   └── scraper.py               # 🕷  LinkedIn scraper + JSON feed ingestor
    │
    ├── matcher/
    │   ├── __init__.py
    │   ├── llm_orchestrator.py      # 🧠 Core LLM module (match, answers, cover letter)
    │   └── resume_matcher.py        # 📊 Batch scoring pipeline
    │
    ├── filler/
    │   ├── __init__.py
    │   └── form_filler.py           # 🖊  Playwright form filler engine
    │
    ├── tracker/
    │   ├── __init__.py
    │   ├── schema.py                # 🗄  SQLite schema + DB helpers
    │   └── tracker.py               # 📋 Logging setup + run reporting
    │
    └── utils/
        └── __init__.py
```

---

## Setup

### 1. Install dependencies

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate

# Install Python packages
pip install -r requirements.txt

# Install Playwright's Chromium browser
playwright install chromium
```

### 2. Configure

```bash
# Copy and edit config
cp config.example.json config.json
nano config.json

# Copy and fill in secrets
cp .env.example .env
nano .env
```

### 3. Add your resume

```bash
cp /path/to/your/resume.pdf resume/resume.pdf
```

### 4. First run — login to LinkedIn

Run with `headless: false` first so you can log in manually:
```bash
python main.py --phase scrape
```
The browser will open. Log in to LinkedIn. The session persists in `data/browser_profile/` — you only need to do this once.

---

## Usage

```bash
# Full pipeline: scrape → score → apply
python main.py

# Scrape new jobs only
python main.py --phase scrape

# Score all unscored jobs (no browser needed)
python main.py --phase score

# Apply to all queued jobs
python main.py --phase apply

# Apply to a specific job by database ID
python main.py --phase apply --job-id 42

# Dry run: fill forms but NEVER auto-submit
python main.py --dry-run

# Use a JSON jobs feed instead of live LinkedIn scraping
python main.py --feed data/sample_jobs.json

# Verbose debug logging
python main.py --verbose
```

---

## ATS Support Matrix

| ATS         | Detection          | Field Extraction      | Multi-step | File Upload | Status  |
|-------------|--------------------|-----------------------|------------|-------------|---------|
| Greenhouse  | URL (`greenhouse.io`) | Static CSS selectors | ❌ Single  | ✅           | ✅ Stable |
| Lever       | URL (`lever.co`)   | React DOM scraping    | ❌ Single  | ✅           | ✅ Stable |
| Workday     | URL (`myworkday.com`) | `data-automation-id` | ✅ Up to 15 steps | ✅  | ✅ Stable |
| Ashby       | URL (`ashbyhq.com`) | Generic fallback     | ❌ Single  | ✅           | ⚠️ Beta  |
| Generic     | Fallback           | LLM semantic parsing  | ❌          | ✅           | ⚠️ Best-effort |

---

## LLM Call Budget (per application)

| Call          | Purpose                        | Avg Tokens | Avg Cost  |
|---------------|--------------------------------|------------|-----------|
| Match score   | Resume vs JD scoring           | ~3,000     | ~$0.015   |
| Field extract | Map fields → values            | ~1,500     | ~$0.008   |
| Cover letter  | Generate 3-paragraph letter    | ~2,000     | ~$0.010   |
| Short answers | Custom question responses (×N) | ~800 each  | ~$0.004   |
| **Total**     | Per application estimate       | **~7,300** | **~$0.037** |

---

## Edge Cases Handled

| Scenario                         | Solution                                                    |
|----------------------------------|-------------------------------------------------------------|
| LinkedIn login required          | Persistent browser context — login once, session reused     |
| Workday multi-step wizard        | Step-by-step pagination with `networkidle` waits            |
| Custom "Why us?" questions       | LLM generates 3-sentence context-aware answers on the fly   |
| Resume file upload               | `page.set_input_files()` with explicit file path            |
| LLM returns invalid JSON         | Auto-retry with correction prompt                           |
| Submit button selector fails     | 5-selector cascade with ATS-specific → generic fallback     |
| Duplicate job URLs               | `ON CONFLICT DO NOTHING` in SQLite upsert                  |
| Anti-bot detection               | `navigator.webdriver` patched, realistic UA + viewport      |
| Form fill failure                | Screenshot captured, job marked `failed`, retry queued      |
| Manual review gate               | CLI pause before Submit — user confirms or skips            |

---

## Security Notes

- API keys are **never** in `config.json` — `.env` only
- `.env` and `data/browser_profile/` should be in `.gitignore`
- The bot never stores passwords beyond the session cookie
- LinkedIn credentials in `.env` are only used for first-time session setup

---

## Extending the Bot

**Add a new ATS (e.g., iCIMS):**
1. Add URL patterns to `ATS_PATTERNS` in `form_filler.py`
2. Write `_extract_fields_icims(page)` following the existing extractor pattern
3. Add submit selectors to `SUBMIT_SELECTORS`
4. Register in `FIELD_EXTRACTORS` dict

**Add a new job source (e.g., Indeed):**
1. Create `IndeedScraper` class in `scraper.py` following `LinkedInScraper`
2. Add a `--source indeed` flag to `main.py`'s argparse
3. Pass the scraper to `ingest_and_store()`
