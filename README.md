# ApplyPilot

A modular, local-first job application automation system Groq. Scrapes job listings, scores them against your resume with an LLM, and presents them to you for filling.

---

## Project File Tree

```
ApplyPilot/
│
├── main.py                          # Entry point CLI
├── requirements.txt                 # Python dependencies
├── config.example.json              # Copy → config.json and customize
├── .env.example                     # Copy → .env and add secrets
│
├── resume/
│   ├── resume.pdf                   # 📄 Your resume (required)
│   └── resume.md                    # Optional plaintext version
│
├── data/
│   └── jobs.db                      # SQLite database (auto-created)
│
├── logs/
│   └── auto_apply_YYYY-MM-DD.log    # Rotating daily log files
│
│
└── src/
    │
    ├── config/
    │   ├── __init__.py
    │   └── config_manager.py        # ⚙️  Config + secret management
    │
    ├── scraper/
    │   ├── __init__.py
    │   └── scraper.py               # 🕷  JSON feed ingestor
    │
    ├── matcher/
    │   ├── __init__.py
    │   ├── llm_orchestrator.py      # 🧠 Core LLM module (match, answers, cover letter)
    │   └── resume_matcher.py        # 📊 Batch scoring pipeline
    │
    ├── filler/
    │   ├── __init__.py
    │   └── form_filler.py
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

# Verbose debug logging
python main.py --verbose
```

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
