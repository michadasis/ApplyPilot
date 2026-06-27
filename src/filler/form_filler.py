"""
form_filler.py — Form Filler Engine

The core Playwright automation module. Orchestrates the full application flow:

  1. Detect ATS type from the apply URL
  2. Navigate to the application form
  3. Extract all form fields from the DOM
  4. Invoke LLM to map fields → values
  5. Fill each field (text, dropdown, file upload, radio/checkbox)
  6. Handle multi-step pagination (critical for Workday)
  7. Generate on-the-fly answers for custom questions
  8. Optionally pause for human review before final submit
  9. Screenshot on any failure

ATS coverage:
  - Greenhouse  (single-page, well-structured DOM)
  - Lever       (single-page, React-based)
  - Workday     (multi-step wizard, heavy JS)
  - Custom/Generic (LLM-driven field detection as fallback)
"""

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from playwright.async_api import (
    Browser,
    BrowserContext,
    Page,
    Playwright,
    async_playwright,
    TimeoutError as PlaywrightTimeout,
)

import anthropic

from src.config.config_manager import AppConfig
from src.matcher.llm_orchestrator import (
    build_llm_client,
    extract_field_mappings,
    generate_cover_letter,
    generate_short_answer,
    parse_resume,
)
from src.tracker.schema import get_db, update_job_status

logger = logging.getLogger(__name__)

# Constants

SCREENSHOTS_DIR = Path(__file__).parent.parent.parent / "screenshots"
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Timeouts (milliseconds)
NAVIGATION_TIMEOUT = 30_000
NETWORK_IDLE_TIMEOUT = 20_000
ELEMENT_TIMEOUT = 10_000

# ATS detection patterns (checked against the apply URL + page content)
ATS_PATTERNS: dict[str, list[str]] = {
    "greenhouse":   ["greenhouse.io", "boards.greenhouse.io", "grnh.se"],
    "lever":        ["lever.co", "jobs.lever.co"],
    "workday":      ["myworkday.com", "wd1.myworkday.com", "wd3.myworkday.com",
                     "wd5.myworkday.com", "workday.com/en-us/applications"],
    "ashby":        ["ashbyhq.com", "jobs.ashbyhq.com"],
}

# CSS selectors for known ATS submit buttons (used as primary; LLM as fallback)
SUBMIT_SELECTORS: dict[str, list[str]] = {
    "greenhouse": [
        "#submit_app",
        "button[data-qa='submit-app-button']",
        "input[type='submit'][value*='Submit']",
    ],
    "lever": [
        ".application-submit button",
        "button[data-qa='btn-submit-application']",
    ],
    "workday": [
        "button[data-automation-id='bottom-navigation-next-button']",
        "button[aria-label='Submit']",
        "[data-automation-id='pageFooterNextButton']",
    ],
    "generic": [
        "button[type='submit']",
        "input[type='submit']",
        "button:has-text('Submit')",
        "button:has-text('Apply Now')",
        "button:has-text('Send Application')",
    ],
}

# Data models

@dataclass
class FormField:
    """Represents a single detected form field."""
    label: str
    field_type: str          # text | textarea | select | checkbox | radio | file | email | tel
    selector: str            # CSS selector or Playwright locator string
    options: list[str]       = field(default_factory=list)  # for <select> and radio groups
    required: bool           = False
    placeholder: str         = ""
    current_value: str       = ""


@dataclass
class FillResult:
    """Outcome of a single form-fill attempt."""
    job_id: int
    ats_type: str
    success: bool
    submitted: bool          = False   # True only if Submit was clicked
    skipped_review: bool     = False   # True if user hit 'skip' at review prompt
    error_message: str       = ""
    screenshot_path: str     = ""
    cover_letter_text: str   = ""
    answers_json: str        = ""      # JSON string of {label: answer}


# ATS Detection

def detect_ats_type(url: str, page_content: str = "") -> str:
    """
    Identify which ATS powers the application page.

    Strategy:
      1. URL substring matching (fast, reliable for major ATS)
      2. Page content scanning for ATS-specific meta tags / scripts
      3. Fallback to 'generic'
    """
    url_lower = url.lower()
    for ats, patterns in ATS_PATTERNS.items():
        if any(p in url_lower for p in patterns):
            logger.info(f"ATS detected via URL: {ats}")
            return ats

    # Content-based detection (e.g., Workday embedded in custom domain)
    content_lower = page_content.lower()
    if "workday" in content_lower and "wd-api" in content_lower:
        return "workday"
    if "greenhouse" in content_lower:
        return "greenhouse"
    if "lever.co" in content_lower:
        return "lever"

    logger.info("ATS not recognized — using generic form filler")
    return "generic"


# Field Extraction — DOM scrapers per ATS type

async def _extract_fields_greenhouse(page: Page) -> list[FormField]:
    """
    Greenhouse has a well-structured form with explicit labels.
    Label elements reliably precede their input counterparts.
    """
    fields: list[FormField] = []

    # Standard text/email/textarea inputs
    input_selectors = [
        "input[type='text']", "input[type='email']", "input[type='tel']",
        "textarea", "input[type='file']", "select",
    ]

    for selector in input_selectors:
        elements = await page.query_selector_all(selector)
        for el in elements:
            # Try to find associated label
            el_id   = await el.get_attribute("id") or ""
            label   = ""
            if el_id:
                label_el = await page.query_selector(f"label[for='{el_id}']")
                if label_el:
                    label = (await label_el.inner_text()).strip()

            if not label:
                # Fallback: nearest ancestor label or aria-label
                label = (await el.get_attribute("aria-label") or
                         await el.get_attribute("placeholder") or
                         el_id or "unknown")

            tag = await el.evaluate("el => el.tagName.toLowerCase()")
            field_type = await el.get_attribute("type") or tag
            if tag == "select":
                field_type = "select"
            if tag == "textarea":
                field_type = "textarea"

            options: list[str] = []
            if field_type == "select":
                opts = await el.query_selector_all("option")
                options = [await o.inner_text() for o in opts if await o.inner_text() != "--"]

            required = (await el.get_attribute("required")) is not None

            fields.append(FormField(
                label=label,
                field_type=field_type,
                selector=f"#{el_id}" if el_id else selector,
                options=options,
                required=required,
                placeholder=await el.get_attribute("placeholder") or "",
            ))

    # EEO/demographic radio groups
    radio_groups = await page.query_selector_all(".field.optional")
    for group in radio_groups:
        legend_el = await group.query_selector("legend")
        if legend_el:
            label = (await legend_el.inner_text()).strip()
            radios = await group.query_selector_all("input[type='radio']")
            options = []
            for r in radios:
                r_id = await r.get_attribute("id") or ""
                r_label_el = await page.query_selector(
                    f"label[for='{r_id}']"
                )
                if r_label_el:
                    options.append(await r_label_el.inner_text())
            fields.append(FormField(
                label=label,
                field_type="radio",
                selector=".field.optional",
                options=options,
            ))

    return fields


async def _extract_fields_lever(page: Page) -> list[FormField]:
    """
    Lever uses a React form. Labels are in <label> tags or data-qa attributes.
    """
    fields: list[FormField] = []
    form_groups = await page.query_selector_all(".application-field, .form-group")

    for group in form_groups:
        label_el = await group.query_selector("label")
        label = (await label_el.inner_text()).strip() if label_el else "unknown"

        input_el = (
            await group.query_selector("input") or
            await group.query_selector("textarea") or
            await group.query_selector("select")
        )
        if not input_el:
            continue

        tag = await input_el.evaluate("el => el.tagName.toLowerCase()")
        field_type = await input_el.get_attribute("type") or tag
        el_id = await input_el.get_attribute("id") or label.lower().replace(" ", "_")

        options: list[str] = []
        if tag == "select":
            field_type = "select"
            opts = await input_el.query_selector_all("option")
            options = [await o.inner_text() for o in opts if await o.inner_text()]

        fields.append(FormField(
            label=label,
            field_type=field_type,
            selector=f"#{el_id}" if await input_el.get_attribute("id") else f"label:has-text('{label}') + input",
            options=options,
            required=(await input_el.get_attribute("required")) is not None,
            placeholder=await input_el.get_attribute("placeholder") or "",
        ))

    return fields


async def _extract_fields_workday(page: Page) -> list[FormField]:
    """
    Workday uses heavily obfuscated class names and data-automation-id attributes.
    We target data-automation-id which is stable across Workday versions.
    """
    fields: list[FormField] = []

    # Workday exposes data-automation-id on inputs
    elements = await page.query_selector_all("[data-automation-id]")
    for el in elements:
        automation_id = await el.get_attribute("data-automation-id") or ""
        tag = await el.evaluate("el => el.tagName.toLowerCase()")

        # Filter to only form input-like elements
        if tag not in ("input", "textarea", "select") and "input" not in automation_id.lower():
            continue

        # Label: look for sibling/ancestor label or aria-labelledby
        label = ""
        labelledby = await el.get_attribute("aria-labelledby")
        if labelledby:
            label_el = await page.query_selector(f"#{labelledby}")
            if label_el:
                label = (await label_el.inner_text()).strip()
        if not label:
            label = await el.get_attribute("aria-label") or automation_id

        field_type = await el.get_attribute("type") or tag
        if tag == "textarea":
            field_type = "textarea"

        options: list[str] = []
        # Workday dropdowns are custom — they render as divs, not <select>
        # Look for associated listbox
        listbox_id = await el.get_attribute("aria-owns") or ""
        if listbox_id:
            field_type = "select"
            opt_els = await page.query_selector_all(f"#{listbox_id} [role='option']")
            options = [await o.inner_text() for o in opt_els]

        fields.append(FormField(
            label=label,
            field_type=field_type,
            selector=f"[data-automation-id='{automation_id}']",
            options=options,
            required=(await el.get_attribute("aria-required")) == "true",
        ))

    return fields


async def _extract_fields_generic(page: Page) -> list[FormField]:
    """
    Universal fallback: harvest all visible form inputs and infer labels
    from surrounding DOM context.
    """
    fields: list[FormField] = []
    all_inputs = await page.query_selector_all(
        "input:not([type='hidden']):not([type='submit']):not([type='button']), "
        "textarea, select"
    )

    for el in all_inputs:
        # Visibility check — skip hidden/off-screen elements
        is_visible = await el.is_visible()
        if not is_visible:
            continue

        el_id   = await el.get_attribute("id") or ""
        el_name = await el.get_attribute("name") or ""
        tag     = await el.evaluate("el => el.tagName.toLowerCase()")
        field_type = await el.get_attribute("type") or tag

        # Label resolution priority:
        # 1. <label for="id"> 2. aria-label 3. placeholder 4. name attr 5. id
        label = ""
        if el_id:
            label_el = await page.query_selector(f"label[for='{el_id}']")
            if label_el:
                label = (await label_el.inner_text()).strip()
        if not label:
            label = (
                await el.get_attribute("aria-label") or
                await el.get_attribute("placeholder") or
                el_name or el_id or "field"
            )

        options: list[str] = []
        if tag == "select":
            field_type = "select"
            opts = await el.query_selector_all("option")
            options = [await o.inner_text() for o in opts]

        # Build the most specific selector possible
        if el_id:
            selector = f"#{el_id}"
        elif el_name:
            selector = f"[name='{el_name}']"
        else:
            selector = tag

        fields.append(FormField(
            label=label,
            field_type=field_type,
            selector=selector,
            options=options,
            required=(await el.get_attribute("required")) is not None,
            placeholder=await el.get_attribute("placeholder") or "",
        ))

    return fields


FIELD_EXTRACTORS = {
    "greenhouse": _extract_fields_greenhouse,
    "lever":      _extract_fields_lever,
    "workday":    _extract_fields_workday,
    "generic":    _extract_fields_generic,
}


# Field Filling — type-aware actions

async def _fill_field(
    page: Page,
    form_field: FormField,
    value: str,
    *,
    resume_path: Path,
    slow_mo: int = 80,
) -> bool:
    """
    Execute the correct Playwright action for a given field type and value.

    Returns True on success, False on failure (caller decides whether to abort).
    """
    selector = form_field.selector

    try:
        el = await page.wait_for_selector(selector, timeout=ELEMENT_TIMEOUT, state="visible")
        if not el:
            logger.warning(f"Selector not found: {selector}")
            return False

        # File upload
        if form_field.field_type == "file" or value == "__UPLOAD_RESUME__":
            if not resume_path.exists():
                logger.error(f"Resume file not found: {resume_path}")
                return False
            await el.set_input_files(str(resume_path))
            logger.debug(f"📎 Uploaded resume to: {selector}")
            return True

        # Select/dropdown
        if form_field.field_type == "select":
            # Workday uses div-based custom dropdowns, not native <select>
            if "workday" in selector:
                await el.click()
                await page.wait_for_timeout(500)
                option_el = page.get_by_role("option", name=re.compile(value, re.IGNORECASE))
                await option_el.click()
                logger.debug(f"🔽 Selected Workday option '{value}' in: {selector}")
                return True

            # Standard <select> — try exact match first, then partial
            try:
                await page.select_option(selector, label=value)
            except Exception:
                # Partial match: find option whose text contains `value`
                matched = next(
                    (opt for opt in form_field.options
                     if value.lower() in opt.lower()),
                    None,
                )
                if matched:
                    await page.select_option(selector, label=matched)
                else:
                    logger.warning(f"No matching option for '{value}' in {selector}")
                    return False
            logger.debug(f"🔽 Selected '{value}' in: {selector}")
            return True

        # Radio button group
        if form_field.field_type == "radio":
            # Click the radio label containing the value text
            radio_label = page.locator(
                f"label:has-text('{value}')"
            ).first
            if await radio_label.is_visible():
                await radio_label.click()
                logger.debug(f"🔘 Selected radio: '{value}'")
                return True
            # Fallback: locate the input by value attribute
            await page.check(f"input[type='radio'][value='{value}']")
            return True

        # Checkbox
        if form_field.field_type == "checkbox":
            should_check = value.lower() in ("true", "yes", "1", "check")
            if should_check:
                await page.check(selector)
            else:
                await page.uncheck(selector)
            logger.debug(f"☑️ Checkbox '{selector}' set to {should_check}")
            return True

        # Text / Email / Tel / Textarea
        # Clear existing content first (important for pre-filled fields)
        await el.triple_click()        # select all
        await el.fill(value)           # Playwright fill is more reliable than type()
        # Trigger change/input events that React/Vue forms often require
        await el.dispatch_event("input")
        await el.dispatch_event("change")
        # Human-like micro-delay
        await page.wait_for_timeout(slow_mo + 20)

        logger.debug(f"✏️  Filled '{selector}' with: {value[:40]}...")
        return True

    except PlaywrightTimeout:
        logger.warning(f"⏱ Timeout finding: {selector}")
        return False
    except Exception as e:
        logger.error(f"❌ Error filling {selector}: {e}")
        return False


# Workday multi-step pagination handler

async def _handle_workday_pagination(page: Page, current_step: int) -> bool:
    """
    Click the Workday 'Next' button and wait for the next step to load.

    Returns True if we advanced, False if no Next button found (probably last page).
    """
    next_selectors = [
        "[data-automation-id='bottom-navigation-next-button']",
        "[data-automation-id='pageFooterNextButton']",
        "button:has-text('Next')",
        "button:has-text('Save and Continue')",
    ]

    for sel in next_selectors:
        btn = await page.query_selector(sel)
        if btn and await btn.is_visible() and await btn.is_enabled():
            await btn.click()
            # Wait for network idle — Workday loads each step via XHR
            try:
                await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)
            except PlaywrightTimeout:
                # Workday sometimes keeps connections open; give it a moment anyway
                await page.wait_for_timeout(2000)
            logger.info(f"➡️  Workday step {current_step} → {current_step + 1}")
            return True

    return False   # No Next button — we're on the last step


# Core Form Filler

class FormFillerEngine:
    """
    Orchestrates the complete form-fill lifecycle for a single job application.

    Usage:
        engine = FormFillerEngine(config, llm_client)
        result = await engine.apply(job_id=42, apply_url="https://...")
    """

    def __init__(self, config: AppConfig, llm_client: anthropic.Anthropic):
        self.config     = config
        self.llm_client = llm_client
        self._resume    = parse_resume(config.resume_pdf_path)

        # Build the candidate profile dict once — reused for every field mapping call
        self._candidate_profile = self._build_candidate_profile()

    def _build_candidate_profile(self) -> dict[str, Any]:
        """
        Assemble a structured profile dict that the LLM uses to fill fields.
        Personal info comes from .env / config; anything not in config is left blank.
        """
        import os
        return {
            "first_name":   os.getenv("CANDIDATE_FIRST_NAME", ""),
            "last_name":    os.getenv("CANDIDATE_LAST_NAME", ""),
            "email":        os.getenv("CANDIDATE_EMAIL", ""),
            "phone":        os.getenv("CANDIDATE_PHONE", ""),
            "linkedin_url": os.getenv("CANDIDATE_LINKEDIN", ""),
            "github_url":   os.getenv("CANDIDATE_GITHUB", ""),
            "portfolio_url":os.getenv("CANDIDATE_PORTFOLIO", ""),
            "city":         os.getenv("CANDIDATE_CITY", ""),
            "state":        os.getenv("CANDIDATE_STATE", ""),
            "work_auth":    os.getenv("CANDIDATE_WORK_AUTH", "Yes"),
            "veteran_status": os.getenv("CANDIDATE_VETERAN_STATUS", "I am not a protected veteran"),
            "disability":   os.getenv("CANDIDATE_DISABILITY", "I don't wish to answer"),
            "ethnicity":    os.getenv("CANDIDATE_ETHNICITY", "I don't wish to answer"),
            "gender":       os.getenv("CANDIDATE_GENDER", "I don't wish to answer"),
            "resume_summary": self._resume["raw_text"][:3000],  # truncated for prompt budget
        }

    async def apply(
        self,
        *,
        job_id: int,
        apply_url: str,
        job_title: str,
        company_name: str,
        job_description: str,
        match_data: dict[str, Any],
        context: BrowserContext,
    ) -> FillResult:
        """
        End-to-end application flow for a single job.

        Steps:
          1. Open page, detect ATS
          2. Generate cover letter
          3. Extract fields
          4. Get LLM field mappings
          5. Fill all fields (multi-step for Workday)
          6. Human-in-the-loop pause if configured
          7. Submit (or hand off to user)

        Args:
            context : Persistent BrowserContext (maintains cookies/session).

        Returns:
            FillResult with success status and metadata.
        """
        page = await context.new_page()
        result = FillResult(job_id=job_id, ats_type="unknown", success=False)

        try:
            # Step 1: Navigate
            logger.info(f"[job_id={job_id}] Navigating to {apply_url}")
            await page.goto(apply_url, wait_until="networkidle",
                            timeout=NAVIGATION_TIMEOUT)
            page_content = await page.content()

            # Step 2: Detect ATS
            ats_type = detect_ats_type(apply_url, page_content)
            result.ats_type = ats_type
            logger.info(f"[job_id={job_id}] ATS type: {ats_type}")

            # Step 3: Generate cover letter (async, before field extraction)
            cover_letter = generate_cover_letter(
                client=self.llm_client,
                resume_text=self._resume["raw_text"],
                job_title=job_title,
                company_name=company_name,
                job_description=job_description,
                matched_skills=match_data.get("matched_skills", []),
                job_id=job_id,
            )
            result.cover_letter_text = cover_letter

            # Step 4 + 5: Fill form (handles multi-step internally)
            answers: dict[str, str] = {}

            if ats_type == "workday":
                answers = await self._fill_workday(
                    page, job_id, job_title, company_name, cover_letter, answers
                )
            else:
                answers = await self._fill_single_page(
                    page, ats_type, job_id, job_title, company_name,
                    cover_letter, answers
                )

            result.answers_json = json.dumps(answers)

            # Step 6: Human-in-the-loop gate
            if self.config.behavior.require_manual_review:
                submitted = await self._human_review_gate(page, job_id)
                result.submitted       = submitted
                result.skipped_review  = not submitted
            else:
                await self._click_submit(page, ats_type)
                result.submitted = True

            result.success = True
            update_job_status(job_id, "applied")
            logger.info(f"✅ [job_id={job_id}] Application {'submitted' if result.submitted else 'filled (skipped by user)'}")

        except Exception as exc:
            result.error_message = str(exc)
            logger.error(f"❌ [job_id={job_id}] Form fill failed: {exc}", exc_info=True)
            update_job_status(job_id, "failed")

            if self.config.behavior.screenshot_on_failure:
                result.screenshot_path = await self._capture_screenshot(page, job_id)

        finally:
            await page.close()

        return result

    # Single-page filler (Greenhouse, Lever, Generic)

    async def _fill_single_page(
        self,
        page: Page,
        ats_type: str,
        job_id: int,
        job_title: str,
        company_name: str,
        cover_letter: str,
        answers: dict[str, str],
    ) -> dict[str, str]:
        """Fill a single-page application form."""

        extractor = FIELD_EXTRACTORS.get(ats_type, _extract_fields_generic)
        fields = await extractor(page)
        logger.info(f"[job_id={job_id}] Extracted {len(fields)} fields")

        # Serialize fields for LLM
        fields_data = [
            {
                "label": f.label,
                "type": f.field_type,
                "selector": f.selector,
                "options": f.options,
                "required": f.required,
                "placeholder": f.placeholder,
            }
            for f in fields
        ]

        # Inject cover letter into profile so LLM can map it correctly
        profile = {**self._candidate_profile, "cover_letter": cover_letter}

        # LLM maps field labels → values
        field_mappings = extract_field_mappings(
            client=self.llm_client,
            form_fields=fields_data,
            candidate_profile=profile,
            job_id=job_id,
        )

        # Fill each field
        for form_field in fields:
            value = field_mappings.get(form_field.selector) or \
                    field_mappings.get(form_field.label)

            if not value:
                logger.debug(f"No mapping for field: {form_field.label} — skipping")
                continue

            # Resolve special value markers
            value, generated = await self._resolve_value_marker(
                value=value,
                form_field=form_field,
                job_title=job_title,
                company_name=company_name,
                job_id=job_id,
                cover_letter=cover_letter,
            )
            if generated:
                answers[form_field.label] = value

            await _fill_field(
                page, form_field, value,
                resume_path=self.config.resume_pdf_path,
                slow_mo=self.config.behavior.slow_mo_ms,
            )
            # Small pause between fields to appear human
            await page.wait_for_timeout(self.config.behavior.slow_mo_ms)

        return answers

    # Workday multi-step filler

    async def _fill_workday(
        self,
        page: Page,
        job_id: int,
        job_title: str,
        company_name: str,
        cover_letter: str,
        answers: dict[str, str],
    ) -> dict[str, str]:
        """
        Handle Workday's multi-step wizard.

        Workday applications typically have 4–8 steps:
          Step 1: Account creation or login
          Step 2: Basic info (name, address, work auth)
          Step 3: Work experience (may be pre-filled from resume upload)
          Step 4: Education
          Step 5: Voluntary self-ID (EEO)
          Step 6: Application questions (custom)
          Step 7: Resume/cover letter upload
          Step Final: Review + Submit
        """
        MAX_STEPS = 15   # Safety ceiling — prevents infinite loops
        step = 1

        while step <= MAX_STEPS:
            logger.info(f"[job_id={job_id}] Workday step {step}")

            # Wait for current step to stabilize
            await page.wait_for_load_state("networkidle", timeout=NETWORK_IDLE_TIMEOUT)

            # Check if we're on the final Review/Submit page
            is_final = await self._is_workday_final_step(page)
            if is_final:
                logger.info(f"[job_id={job_id}] Reached Workday final Review step")
                break

            # Extract and fill current step's fields
            fields = await _extract_fields_workday(page)
            logger.info(f"[job_id={job_id}] Step {step}: {len(fields)} fields")

            if fields:
                fields_data = [
                    {"label": f.label, "type": f.field_type,
                     "selector": f.selector, "options": f.options}
                    for f in fields
                ]
                profile = {**self._candidate_profile, "cover_letter": cover_letter}
                field_mappings = extract_field_mappings(
                    client=self.llm_client,
                    form_fields=fields_data,
                    candidate_profile=profile,
                    job_id=job_id,
                )

                for form_field in fields:
                    value = field_mappings.get(form_field.selector) or \
                            field_mappings.get(form_field.label)
                    if not value:
                        continue

                    value, generated = await self._resolve_value_marker(
                        value=value,
                        form_field=form_field,
                        job_title=job_title,
                        company_name=company_name,
                        job_id=job_id,
                        cover_letter=cover_letter,
                    )
                    if generated:
                        answers[form_field.label] = value

                    await _fill_field(
                        page, form_field, value,
                        resume_path=self.config.resume_pdf_path,
                        slow_mo=self.config.behavior.slow_mo_ms,
                    )
                    await page.wait_for_timeout(self.config.behavior.slow_mo_ms)

            # Advance to next step
            advanced = await _handle_workday_pagination(page, step)
            if not advanced:
                logger.warning(f"[job_id={job_id}] Could not advance past step {step}")
                break

            step += 1

        return answers

    async def _is_workday_final_step(self, page: Page) -> bool:
        """Detect if we're on the Workday Review/Submit final step."""
        # Look for the Submit button (not Next) or a "Review" heading
        submit_btn = await page.query_selector(
            "[data-automation-id='bottom-navigation-finish-button'], "
            "button[aria-label='Submit'], "
            "button:has-text('Submit')"
        )
        if submit_btn and await submit_btn.is_visible():
            return True

        review_heading = await page.query_selector(
            "[data-automation-id='reviewApplication'], "
            "h2:has-text('Review')"
        )
        return review_heading is not None

    # Value marker resolution

    async def _resolve_value_marker(
        self,
        *,
        value: str,
        form_field: FormField,
        job_title: str,
        company_name: str,
        job_id: int,
        cover_letter: str,
    ) -> tuple[str, bool]:
        """
        Resolve special marker values returned by the field mapping LLM.

        Markers:
          __UPLOAD_RESUME__                   → handled in _fill_field
          __GENERATE_COVER_LETTER__           → return pre-generated cover letter
          __GENERATE_ANSWER__:<question>      → call LLM for short answer

        Returns: (resolved_value, was_generated_by_llm)
        """
        if value == "__GENERATE_COVER_LETTER__":
            return cover_letter, False   # Already generated; not a new LLM call

        if value.startswith("__GENERATE_ANSWER__:"):
            question = value[len("__GENERATE_ANSWER__:"):]
            logger.info(f"[job_id={job_id}] Generating answer for: '{question[:60]}'")
            answer = generate_short_answer(
                client=self.llm_client,
                question=question,
                resume_text=self._resume["raw_text"],
                job_title=job_title,
                company_name=company_name,
                job_id=job_id,
            )
            return answer, True

        # Pass-through for normal values and __UPLOAD_RESUME__
        return value, False

    # Submit button

    async def _click_submit(self, page: Page, ats_type: str) -> None:
        """
        Find and click the final Submit button.

        Tries ATS-specific selectors first, then falls back to generic.
        Raises RuntimeError if no submit button found after all attempts.
        """
        candidate_selectors = (
            SUBMIT_SELECTORS.get(ats_type, []) + SUBMIT_SELECTORS["generic"]
        )

        for sel in candidate_selectors:
            try:
                btn = await page.wait_for_selector(sel, timeout=5000, state="visible")
                if btn and await btn.is_enabled():
                    await btn.click()
                    # Wait for confirmation page or navigation
                    await page.wait_for_load_state("networkidle", timeout=NAVIGATION_TIMEOUT)
                    logger.info(f"🚀 Submit clicked via: {sel}")
                    return
            except PlaywrightTimeout:
                continue

        raise RuntimeError(
            "Could not find a Submit button after trying all known selectors. "
            "The form may require manual completion."
        )

    # Human-in-the-loop gate

    async def _human_review_gate(self, page: Page, job_id: int) -> bool:
        """
        Pause execution and prompt the user to review the filled form before submitting.

        This runs in an asyncio context, so we use asyncio.get_running_loop().run_in_executor
        to run the blocking input() call without freezing the event loop.

        Returns True if user confirms submission, False if they skip.
        """
        print("\n" + "═" * 60)
        print(f"  🔍 MANUAL REVIEW REQUIRED  │  Job ID: {job_id}")
        print("═" * 60)
        print("  The form has been filled. Please review the browser window.")
        print("  Press  ENTER  to submit the application.")
        print("  Type  skip  + ENTER  to skip this application.")
        print("═" * 60)

        loop = asyncio.get_running_loop()
        user_input = await loop.run_in_executor(None, input, "  → Your choice: ")

        if user_input.strip().lower() == "skip":
            logger.info(f"[job_id={job_id}] User skipped submission")
            update_job_status(job_id, "skipped")
            return False

        # User pressed Enter — proceed with submit
        await self._click_submit(page, "generic")
        return True

    # Screenshot helper

    async def _capture_screenshot(self, page: Page, job_id: int) -> str:
        """Capture a full-page screenshot and save to the screenshots directory."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = SCREENSHOTS_DIR / f"failure_job{job_id}_{ts}.png"
        try:
            await page.screenshot(path=str(path), full_page=True)
            logger.info(f"📸 Screenshot saved: {path}")
            return str(path)
        except Exception as e:
            logger.warning(f"Screenshot failed: {e}")
            return ""


# Browser Context Factory — persistent context for session reuse

async def build_browser_context(
    playwright: Playwright,
    config: AppConfig,
    user_data_dir: Path | None = None,
) -> tuple[Browser, BrowserContext]:
    """
    Create a Playwright browser with a persistent context.

    Persistent context = cookies/localStorage survive between runs, so
    LinkedIn/Greenhouse/Workday sessions don't require re-login every time.

    Args:
        user_data_dir : Path to the Chrome user-data directory.
                        Defaults to <project_root>/data/browser_profile.
    """
    if user_data_dir is None:
        user_data_dir = Path(__file__).parent.parent.parent / "data" / "browser_profile"
    user_data_dir.mkdir(parents=True, exist_ok=True)

    # Anti-fingerprinting args — makes headless Chrome less detectable
    launch_args = [
        "--disable-blink-features=AutomationControlled",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-web-security",      # needed for some ATS iframes
        "--lang=en-US,en",
    ]

    context = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(user_data_dir),
        headless=config.behavior.headless,
        slow_mo=config.behavior.slow_mo_ms,
        args=launch_args,
        # Realistic viewport + user agent
        viewport={"width": 1440, "height": 900},
        user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/122.0.0.0 Safari/537.36"
        ),
        locale="en-US",
        timezone_id="America/New_York",
        # Inject stealth scripts on every page to mask automation signals
        java_script_enabled=True,
    )

    # Patch navigator.webdriver = undefined on every new page
    await context.add_init_script("""
        Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        window.chrome = { runtime: {} };
        Object.defineProperty(navigator, 'plugins', { get: () => [1, 2, 3] });
        Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
    """)

    # Persistent contexts return a "browser-like" object with no separate browser
    # Return None for browser since persistent context manages its own
    return None, context  # type: ignore[return-value]
