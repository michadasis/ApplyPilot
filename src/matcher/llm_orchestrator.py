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

DEFAULT_MODEL    = "llama-3.3-70b-versatile"  # generation: cover letters, Q&A
SCORE_MODEL      = "llama-3.1-8b-instant"     # scoring: 500k TPD vs 100k for 70B
MAX_TOKENS       = 2048
INTER_CALL_DELAY = 2.5
RESUME_CACHE: dict[str, Any] = {}

# Token budgets for scoring — kept small to stretch the daily limit
SCORE_RESUME_CHARS = 2000   # ~500 tokens
SCORE_JD_CHARS     = 1500   # ~375 tokens


def _parse_retry_delay(error_str: str) -> float:
    """Extract suggested retry-after seconds from a Groq 429 message."""
    m = re.search(r"retry in (\d+)m([\d.]+)s", error_str)
    if m:
        return int(m.group(1)) * 60 + float(m.group(2))
    m = re.search(r"retry in ([\d.]+)s", error_str)
    if m:
        return float(m.group(1))
    return 65.0


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
            if status == 429:
                delay = _parse_retry_delay(str(e))
                if delay > 600:   # > 10 min = daily quota gone for today
                    raise RuntimeError(
                        f"Daily token quota exhausted (suggested wait: {delay/60:.0f} min). "
                        "Run --phase score again tomorrow, or the quota resets at midnight UTC."
                    ) from e
                if attempt < max_attempts - 1:
                    logger.warning(
                        f"[{purpose}] Rate limited — waiting {delay:.0f}s (attempt {attempt + 1})"
                    )
                    time.sleep(delay)
                else:
                    raise
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
                        "Return ONLY a raw JSON object. "
                        "No markdown, no backticks, no explanation. "
                        "Start with { and end with }."},
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

CRITICAL: recommended_action MUST be exactly one of these three strings: "apply", "skip", "review".
Do NOT write "apply with tailored cover letter" or any other variant. Just "apply".

Scoring:
- 0.85+  Strong match — recommended_action: "apply"
- 0.65-0.84  Decent match — recommended_action: "apply"
- 0.50-0.64  Borderline — recommended_action: "review"
- <0.50  Poor match — recommended_action: "skip"
"""


def score_job_match(
    *,
    client: GroqClient,
    resume_text: str,
    job_title: str,
    job_description: str,
    job_id: int | None = None,
) -> dict[str, Any]:
    # Short excerpts + fast model to stay within the 500k TPD free-tier limit
    user_message = f"""
CANDIDATE RESUME:
{resume_text[:SCORE_RESUME_CHARS]}

JOB TITLE: {job_title}

JOB DESCRIPTION:
{job_description[:SCORE_JD_CHARS]}

Evaluate the match and return the JSON object.
""".strip()

    raw    = _call_llm(client=client, system_prompt=MATCH_SYSTEM_PROMPT,
                       user_message=user_message, purpose="match",
                       job_id=job_id, model=SCORE_MODEL,
                       max_tokens=512, expect_json=True)
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


# Module 6 — Application Pack Generator

COMMON_QUESTIONS = [
    "Tell me about yourself and your background.",
    "Why are you interested in this role and company?",
    "What is your greatest professional achievement?",
    "Describe a challenging technical problem you solved.",
    "Why are you looking for a new position?",
    "What are your salary expectations?",
    "When can you start?",
    "Do you have experience working remotely?",
    "Where do you see yourself in 3 years?",
    "What is your preferred tech stack and why?",
]

APP_PACK_SYSTEM_PROMPT = """
You are a professional job application writer helping a candidate apply for a software role.
Given the candidate's resume, the job posting, and a list of common application questions,
produce a complete application pack.

Return a single valid JSON object with this exact schema:
{
  "cover_letter": "<3-paragraph letter body, no greeting header, no signature>",
  "answers": {
    "<question text>": "<concise, specific answer — 2-4 sentences max>",
    ...
  },
  "elevator_pitch": "<60-second verbal pitch the candidate can use on a call>",
  "key_talking_points": ["<point 1>", "<point 2>", "<point 3>"]
}

Rules:
- Cover letter: para 1 = excitement about role + one specific company detail,
  para 2 = 2-3 concrete resume achievements relevant to this JD,
  para 3 = forward-looking close with call to action.
- Answers: first-person, specific, reference real experience from the resume.
- Salary answer: give a range based on the role/market; if unknown say "open to discussion".
- Keep everything honest — only reference skills actually present in the resume.
- Return ONLY valid JSON. No markdown fences. No commentary.
"""


def generate_application_pack(
    *,
    client: GroqClient,
    resume_text: str,
    job_title: str,
    company_name: str,
    job_description: str,
    matched_skills: list[str],
    job_id: int | None = None,
) -> dict:
    """
    Generate a full application pack for one job:
      - Tailored cover letter
      - Answers to 10 common application questions
      - 60-second elevator pitch
      - Key talking points for interviews

    Returns the parsed JSON dict.
    """
    questions_block = "\n".join(f"- {q}" for q in COMMON_QUESTIONS)

    user_message = f"""
JOB: {job_title} at {company_name}

JOB DESCRIPTION:
{job_description[:3000]}

TOP MATCHED SKILLS FROM RESUME:
{", ".join(matched_skills[:8])}

CANDIDATE RESUME:
{resume_text[:5000]}

COMMON APPLICATION QUESTIONS TO ANSWER:
{questions_block}

Generate the full application pack JSON.
""".strip()

    raw = _call_llm(
        client=client,
        system_prompt=APP_PACK_SYSTEM_PROMPT,
        user_message=user_message,
        purpose="app_pack",
        job_id=job_id,
        max_tokens=2048,
        expect_json=True,
    )

    pack = json.loads(raw)
    logger.info(f"[job_id={job_id}] Application pack generated for {job_title} @ {company_name}")
    return pack
