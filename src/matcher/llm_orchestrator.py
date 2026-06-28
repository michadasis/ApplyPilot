"""
llm_orchestrator.py — LLM Orchestrator Module

Stack: Groq (llama-3.3-70b-versatile) via OpenAI-compatible API.
Free tier: https://console.groq.com — no EU restrictions.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Any

from groq import Groq
import pdfplumber

from src.tracker.schema import log_llm_call

logger = logging.getLogger(__name__)

DEFAULT_MODEL   = "llama-3.3-70b-versatile"
MAX_TOKENS      = 2048
# Groq free tier: 30 req/min. One call per 2.5s keeps us safely under.
INTER_CALL_DELAY = 2.5
RESUME_CACHE: dict[str, Any] = {}


# Client wrapper

class GroqClient:
    """Thin wrapper around the Groq SDK client."""
    def __init__(self, api_key: str) -> None:
        self._client = Groq(api_key=api_key)

    @property
    def raw(self) -> Groq:
        return self._client


# Internal gateway

def _call_llm(
    *,
    client: GroqClient,
    system_prompt: str,
    user_message: str,
    purpose: str,
    job_id: int | None = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = MAX_TOKENS,
    expect_json: bool = False,
) -> str:
    """
    Single gateway for all LLM calls.
    Handles rate-limit backoff (429) and optional JSON validation with one retry.
    """
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            time.sleep(INTER_CALL_DELAY)   # polite pacing between every call

            response = client.raw.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_message},
                ],
            )
            break   # success

        except Exception as e:
            status = getattr(e, "status_code", None)
            if status == 429 and attempt < max_attempts - 1:
                # Try to read the suggested wait from the error body
                match = re.search(r"try again in (\d+(?:\.\d+)?)s", str(e), re.IGNORECASE)
                wait = float(match.group(1)) if match else 60.0
                logger.warning(f"[{purpose}] Rate limited — waiting {wait:.0f}s (attempt {attempt + 1})")
                time.sleep(wait)
            else:
                raise

    text = response.choices[0].message.content.strip()

    try:
        log_llm_call(
            purpose=purpose,
            model=model,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            job_id=job_id,
        )
    except Exception as log_err:
        logger.warning(f"LLM call log failed (non-fatal): {log_err}")

    if expect_json:
        text = _strip_json_fences(text)
        try:
            json.loads(text)
        except json.JSONDecodeError:
            logger.warning(f"[{purpose}] Invalid JSON — retrying with correction")
            time.sleep(INTER_CALL_DELAY)
            retry = client.raw.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system",    "content": system_prompt},
                    {"role": "user",      "content": user_message},
                    {"role": "assistant", "content": text},
                    {"role": "user",      "content":
                        "Your response was not valid JSON. "
                        "Return ONLY the JSON object, no markdown fences, no commentary."},
                ],
            )
            text = _strip_json_fences(retry.choices[0].message.content.strip())
            json.loads(text)   # raise if still broken

    logger.debug(f"[{purpose}] LLM response ({len(text)} chars)")
    return text


def _strip_json_fences(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"\s*```$",          "", text, flags=re.MULTILINE)
    return text.strip()


# Module 1 — Resume Parser

def parse_resume(resume_path: Path) -> dict[str, Any]:
    cache_key = str(resume_path)
    if cache_key in RESUME_CACHE:
        return RESUME_CACHE[cache_key]

    if not resume_path.exists():
        raise FileNotFoundError(f"Resume not found at {resume_path}")

    pages: list[str] = []
    with pdfplumber.open(resume_path) as pdf:
        for page in pdf.pages:
            pages.append(page.extract_text() or "")

    raw_text = "\n\n".join(pages)
    result = {"raw_text": raw_text, "pages": pages, "word_count": len(raw_text.split())}
    RESUME_CACHE[cache_key] = result
    logger.info(f"Resume parsed: {result['word_count']} words across {len(pages)} page(s)")
    return result


# Module 2 — Job/Resume Match Scorer

MATCH_SYSTEM_PROMPT = """
You are a technical recruiter and resume matcher. Evaluate how well a candidate's
resume matches a job description.

Return a single valid JSON object — no markdown, no prose, no fences.

Schema:
{
  "match_score": <float 0.0-1.0>,
  "matched_skills": [<string>, ...],
  "missing_skills": [<string>, ...],
  "rationale": "<2-sentence summary>",
  "recommended_action": "apply" | "skip" | "review"
}

Scoring:
- 0.85+  Strong match — apply confidently
- 0.65-0.84  Decent match — apply with tailored cover letter
- 0.50-0.64  Weak match — skip unless desperate
- <0.50  Poor match — always skip
"""


def score_job_match(
    *,
    client: GroqClient,
    resume_text: str,
    job_title: str,
    job_description: str,
    job_id: int | None = None,
) -> dict[str, Any]:
    user_message = f"""
CANDIDATE RESUME:
{resume_text[:6000]}

JOB TITLE: {job_title}

JOB DESCRIPTION:
{job_description[:4000]}

Evaluate the match and return the JSON object.
""".strip()

    raw    = _call_llm(client=client, system_prompt=MATCH_SYSTEM_PROMPT,
                       user_message=user_message, purpose="match",
                       job_id=job_id, expect_json=True)
    result: dict[str, Any] = json.loads(raw)
    logger.info(f"[job_id={job_id}] Match score: {result.get('match_score', 0):.2f} "
                f"({result.get('recommended_action')})")
    return result


# Module 3 — Form Field Extractor

FIELD_EXTRACT_SYSTEM_PROMPT = """
You are an expert web automation engineer. Map form field labels to candidate values.

Return a single valid JSON object — no markdown, no prose, no fences.
Keys = exact field label/selector. Values = string to fill, or null.

Special values:
- File upload -> "__UPLOAD_RESUME__"
- Cover letter -> "__GENERATE_COVER_LETTER__"
- Custom question -> "__GENERATE_ANSWER__:<question text>"
"""


def extract_field_mappings(
    *,
    client: GroqClient,
    form_fields: list[dict[str, Any]],
    candidate_profile: dict[str, Any],
    job_id: int | None = None,
) -> dict[str, str]:
    user_message = f"""
FORM FIELDS:
{json.dumps(form_fields, indent=2)}

CANDIDATE PROFILE:
{json.dumps(candidate_profile, indent=2)}

Map each field. Return ONLY the JSON object.
""".strip()

    raw      = _call_llm(client=client, system_prompt=FIELD_EXTRACT_SYSTEM_PROMPT,
                         user_message=user_message, purpose="field_extract",
                         job_id=job_id, expect_json=True)
    mappings: dict[str, str] = json.loads(raw)
    logger.info(f"[job_id={job_id}] Field mappings: {len(mappings)} fields")
    return mappings


# Module 4 — Short Answer Generator

SHORT_ANSWER_SYSTEM_PROMPT = """
Write job application answers for a candidate.
- Exactly 3 sentences
- Specific: reference concrete experience from the resume
- Professional but natural, first-person
Return ONLY the answer text.
"""


def generate_short_answer(
    *,
    client: GroqClient,
    question: str,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_id: int | None = None,
) -> str:
    user_message = f"""
QUESTION: {question}
APPLYING FOR: {job_title} at {company_name}
MY RESUME:
{resume_text[:5000]}

Write a 3-sentence answer.
""".strip()

    answer = _call_llm(client=client, system_prompt=SHORT_ANSWER_SYSTEM_PROMPT,
                       user_message=user_message, purpose="answer_gen", job_id=job_id)
    logger.info(f"[job_id={job_id}] Answer generated for: '{question[:60]}...'")
    return answer


# Module 5 — Cover Letter Generator

COVER_LETTER_SYSTEM_PROMPT = """
Write a professional cover letter body for a job application.
- 3 short paragraphs (max 5 sentences each)
- Para 1: Why this role excites the candidate + one specific company detail
- Para 2: 2-3 concrete resume achievements relevant to this role
- Para 3: Forward-looking close with call to action
- NO "Dear Hiring Manager" header
- First-person, warm tone, no clichés
Return ONLY the letter body.
"""


def generate_cover_letter(
    *,
    client: GroqClient,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_description: str,
    matched_skills: list[str],
    job_id: int | None = None,
) -> str:
    user_message = f"""
JOB: {job_title} at {company_name}
TOP MATCHED SKILLS: {", ".join(matched_skills[:6])}
JOB DESCRIPTION: {job_description[:2000]}
MY RESUME: {resume_text[:5000]}

Write the 3-paragraph cover letter body.
""".strip()

    letter = _call_llm(client=client, system_prompt=COVER_LETTER_SYSTEM_PROMPT,
                       user_message=user_message, purpose="cover_letter",
                       job_id=job_id, max_tokens=1024)
    logger.info(f"[job_id={job_id}] Cover letter generated ({len(letter)} chars)")
    return letter


# Factory

def build_llm_client(api_key: str) -> GroqClient:
    """Create and return a configured GroqClient."""
    return GroqClient(api_key)
