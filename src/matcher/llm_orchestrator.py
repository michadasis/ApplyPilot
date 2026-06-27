"""
llm_orchestrator.py — LLM Orchestrator Module

The AI brain of the bot. Responsible for:
  1. Resume parsing (PDF → structured dict)
  2. Job/resume match scoring
  3. Dynamic form field semantic extraction
  4. On-the-fly short answer generation for custom questions
  5. Cover letter generation

All LLM calls go through a single _call_llm() gateway for unified
error handling, token counting, and cost logging.

Stack: Anthropic Claude (claude-3-5-sonnet) via the official SDK.
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

import anthropic
import pdfplumber  # pip install pdfplumber

from src.tracker.schema import log_llm_call

logger = logging.getLogger(__name__)

# Model config — change here to switch models globally
DEFAULT_MODEL    = "claude-3-5-sonnet-20241022"
MAX_TOKENS       = 2048
RESUME_CACHE: dict[str, Any] = {}   # in-process cache so we only parse once


# Internal gateway — all LLM calls funnel through here

def _call_llm(
    *,
    client: anthropic.Anthropic,
    system_prompt: str,
    user_message: str,
    purpose: str,
    job_id: int | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    expect_json: bool = False,
) -> str:
    """
    Thin wrapper around the Anthropic Messages API.

    Args:
        client        : Authenticated anthropic.Anthropic instance.
        system_prompt : System-level instruction for the model.
        user_message  : The actual user turn content.
        purpose       : Label for the llm_calls audit log row.
        job_id        : Optional FK for cost attribution.
        model         : Model ID string.
        max_tokens    : Max completion tokens.
        expect_json   : If True, validate that the response is parseable JSON
                        and retry once if not.

    Returns:
        The model's text response as a string.

    Raises:
        anthropic.APIError on unrecoverable API failures.
        ValueError if expect_json=True and JSON is malformed after retry.
    """
    messages = [{"role": "user", "content": user_message}]

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=messages,
    )

    text = response.content[0].text.strip()

    # Audit log — non-blocking (we don't fail the app over a logging error)
    try:
        log_llm_call(
            purpose=purpose,
            model=model,
            prompt_tokens=response.usage.input_tokens,
            completion_tokens=response.usage.output_tokens,
            job_id=job_id,
        )
    except Exception as log_err:
        logger.warning(f"LLM call log failed (non-fatal): {log_err}")

    # JSON guard
    if expect_json:
        text = _strip_json_fences(text)
        try:
            json.loads(text)   # validate; caller does the actual parse
        except json.JSONDecodeError:
            # One retry with an explicit correction prompt
            logger.warning(f"[{purpose}] LLM returned invalid JSON — retrying with correction")
            correction_messages = [
                {"role": "user",    "content": user_message},
                {"role": "assistant", "content": text},
                {"role": "user",    "content":
                    "Your response was not valid JSON. "
                    "Return ONLY the JSON object, no markdown fences, no commentary."},
            ]
            retry_resp = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=correction_messages,
            )
            text = _strip_json_fences(retry_resp.content[0].text.strip())
            # If still broken, raise — caller must handle
            json.loads(text)

    logger.debug(f"[{purpose}] LLM response ({len(text)} chars)")
    return text


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` fences that models sometimes emit."""
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$", "", text, flags=re.MULTILINE)
    return text.strip()


# Module 1 — Resume Parser

def parse_resume(resume_path: Path) -> dict[str, Any]:
    """
    Extract text from a PDF resume and return a structured dict.

    Uses pdfplumber for reliable text extraction (handles multi-column layouts).
    Result is cached in RESUME_CACHE so multiple modules share the same parse.

    Returns:
        {
          "raw_text": str,           # Full concatenated text
          "pages": [str, ...],       # Per-page text list
          "word_count": int,
        }
    """
    cache_key = str(resume_path)
    if cache_key in RESUME_CACHE:
        return RESUME_CACHE[cache_key]

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found at {resume_path}")

    pages: list[str] = []
    with pdfplumber.open(resume_path) as pdf:
        for page in pdf.pages:
            text = page.extract_text() or ""
            pages.append(text)

    raw_text = "\n\n".join(pages)
    result = {
        "raw_text": raw_text,
        "pages": pages,
        "word_count": len(raw_text.split()),
    }

    RESUME_CACHE[cache_key] = result
    logger.info(f"Resume parsed: {result['word_count']} words across {len(pages)} page(s)")
    return result


# Module 2 — Job/Resume Match Scorer

MATCH_SYSTEM_PROMPT = """
You are a technical recruiter and resume matcher. Your task is to evaluate how well
a candidate's resume matches a job description.

You MUST return a single valid JSON object. No markdown, no prose, no fences.

Schema:
{
  "match_score": <float 0.0–1.0>,
  "matched_skills": [<string>, ...],
  "missing_skills": [<string>, ...],
  "rationale": "<2-sentence summary>",
  "recommended_action": "apply" | "skip" | "review"
}

Scoring guidelines:
- 0.85+: Strong match — apply confidently
- 0.65–0.84: Decent match — apply with tailored cover letter
- 0.50–0.64: Weak match — skip unless desperate
- <0.50: Poor match — always skip
"""


def score_job_match(
    *,
    client: anthropic.Anthropic,
    resume_text: str,
    job_title: str,
    job_description: str,
    job_id: int | None = None,
) -> dict[str, Any]:
    """
    Ask the LLM to rate how well the resume matches the job description.

    Returns the parsed JSON dict (see MATCH_SYSTEM_PROMPT schema).
    """
    # Truncate inputs to stay within context limits
    resume_snippet = resume_text[:6000]
    jd_snippet = job_description[:4000]

    user_message = f"""
CANDIDATE RESUME:
{resume_snippet}

JOB TITLE: {job_title}

JOB DESCRIPTION:
{jd_snippet}

Evaluate the match and return the JSON object.
""".strip()

    raw = _call_llm(
        client=client,
        system_prompt=MATCH_SYSTEM_PROMPT,
        user_message=user_message,
        purpose="match",
        job_id=job_id,
        expect_json=True,
    )
    result: dict[str, Any] = json.loads(raw)
    logger.info(
        f"[job_id={job_id}] Match score: {result.get('match_score', 0):.2f} "
        f"({result.get('recommended_action')})"
    )
    return result


# Module 3 — Form Field Semantic Extractor

FIELD_EXTRACT_SYSTEM_PROMPT = """
You are an expert web automation engineer specializing in job application forms.
Given a list of raw form field labels and metadata extracted from a webpage,
map each field to the correct value from the candidate's profile.

You MUST return a single valid JSON object. No markdown, no prose, no fences.

The JSON must be a flat object where:
- Key = the exact field label/selector string as provided
- Value = the string value to fill in, OR null if not applicable/unknown

For file upload fields, set the value to "__UPLOAD_RESUME__".
For dropdowns, set the value to the closest matching option text.
For checkboxes asking "Are you authorized to work?", interpret from the profile.
For cover letter fields, set value to "__GENERATE_COVER_LETTER__".
For custom free-text questions, set value to "__GENERATE_ANSWER__:<the question text>".
"""


def extract_field_mappings(
    *,
    client: anthropic.Anthropic,
    form_fields: list[dict[str, Any]],
    candidate_profile: dict[str, Any],
    job_id: int | None = None,
) -> dict[str, str]:
    """
    Given raw form field metadata (label, type, options), return a mapping
    of {field_label: value_to_fill}.

    Args:
        form_fields: List of dicts like:
            {"label": "First Name", "type": "text", "selector": "#firstName", "options": []}
        candidate_profile: Dict with candidate's personal info (from config + resume).
        job_id: For cost attribution.

    Returns:
        {"#firstName": "John", "#lastName": "Doe", ...}
    """
    fields_json = json.dumps(form_fields, indent=2)
    profile_json = json.dumps(candidate_profile, indent=2)

    user_message = f"""
FORM FIELDS (extracted from the page):
{fields_json}

CANDIDATE PROFILE:
{profile_json}

Map each field to the correct value. Return ONLY the JSON object.
""".strip()

    raw = _call_llm(
        client=client,
        system_prompt=FIELD_EXTRACT_SYSTEM_PROMPT,
        user_message=user_message,
        purpose="field_extract",
        job_id=job_id,
        expect_json=True,
    )
    mappings: dict[str, str] = json.loads(raw)
    logger.info(f"[job_id={job_id}] Field mappings extracted: {len(mappings)} fields")
    return mappings


# Module 4 — Short Answer Generator (custom questions)

SHORT_ANSWER_SYSTEM_PROMPT = """
You are writing job application answers on behalf of a candidate.
Your answers must be:
- Concise: exactly 3 sentences unless instructed otherwise
- Specific: reference concrete experience from the resume
- Professional but natural — avoid corporate buzzwords
- First-person, written as the candidate

Return ONLY the answer text. No labels, no JSON, no quotes around the answer.
"""


def generate_short_answer(
    *,
    client: anthropic.Anthropic,
    question: str,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_id: int | None = None,
) -> str:
    """
    Generate a context-aware 3-sentence answer to a custom application question.

    Args:
        question     : The exact question text from the form.
        resume_text  : Full resume text for context grounding.
        job_title    : Target job title for relevance.
        company_name : Target company for personalization.
        job_id       : For cost attribution.

    Returns:
        Plain string — the answer to type into the form field.
    """
    user_message = f"""
QUESTION: {question}

APPLYING FOR: {job_title} at {company_name}

MY RESUME:
{resume_text[:5000]}

Write a 3-sentence answer to this question.
""".strip()

    answer = _call_llm(
        client=client,
        system_prompt=SHORT_ANSWER_SYSTEM_PROMPT,
        user_message=user_message,
        purpose="answer_gen",
        job_id=job_id,
    )
    logger.info(f"[job_id={job_id}] Generated answer for: '{question[:60]}...'")
    return answer


# Module 5 — Cover Letter Generator

COVER_LETTER_SYSTEM_PROMPT = """
You are writing a professional cover letter for a job application.

Rules:
- 3 short paragraphs (no more than 5 sentences each)
- Paragraph 1: Why this role excites the candidate + one specific company detail
- Paragraph 2: 2-3 concrete achievements from the resume most relevant to this role
- Paragraph 3: Forward-looking close with clear call to action
- NO "Dear Hiring Manager" header — return body paragraphs only
- Professional but warm tone; avoid clichés like "I am writing to express my interest"
- First-person, written as the candidate

Return ONLY the letter body. No subject line, no signature block, no JSON.
"""


def generate_cover_letter(
    *,
    client: anthropic.Anthropic,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_description: str,
    matched_skills: list[str],
    job_id: int | None = None,
) -> str:
    """
    Generate a tailored cover letter body for the given job.

    Returns plain text — the 3-paragraph cover letter body.
    """
    user_message = f"""
JOB: {job_title} at {company_name}

TOP MATCHED SKILLS (use these for paragraph 2):
{", ".join(matched_skills[:6])}

JOB DESCRIPTION (first 2000 chars for context):
{job_description[:2000]}

MY RESUME:
{resume_text[:5000]}

Write the 3-paragraph cover letter body.
""".strip()

    letter = _call_llm(
        client=client,
        system_prompt=COVER_LETTER_SYSTEM_PROMPT,
        user_message=user_message,
        purpose="cover_letter",
        job_id=job_id,
        max_tokens=1024,
    )
    logger.info(f"[job_id={job_id}] Cover letter generated ({len(letter)} chars)")
    return letter


# Factory — single client instance for callers

def build_llm_client(api_key: str) -> anthropic.Anthropic:
    """Create and return a configured Anthropic client."""
    return anthropic.Anthropic(api_key=api_key)
