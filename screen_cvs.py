"""
CV Screening Script (simple version)

How it works:
1. Reads the rules from requirements.txt (years + skills).
2. Reads each CV from the "cvs" folder and gets its text.
3. Groq only EXTRACTS facts from the CV (years, which skills are present).
4. Python applies your rules and decides SHORTLISTED / REJECTED.
   (If Groq fails or the free daily limit is used up, a simple local
   keyword scan is used instead, so you always get a result.)

Run:  python screen_cvs.py
Install:  pip install pypdf python-docx openai
(The "openai" package is only used as a connector - the AI is Groq.)
Put your Groq key in api_key.txt (one line, nothing else).
"""

import csv
import json
import re
import time
from pathlib import Path

from pypdf import PdfReader

from cv_analyzer_table_db import DB_FILE, init_db, save_to_db   # database code

# ---------------- SETTINGS (change these if you want) ----------------
CV_FOLDER = "cvs"
JOB_FILE = "job_description.txt"
REQUIREMENTS_FILE = "requirements.txt"
API_KEY_FILE = "api_key.txt"

RESULTS_CSV = "results.csv"          # final shortlist output
AI_CSV = "groq_output.csv"           # raw Groq replies
ERRORS_CSV = "errors.csv"            # problems (Groq failed, unreadable CV...)

USE_AI = True                        # False = local keyword scan only (instant, no API)
GROQ_URL = "https://api.groq.com/openai/v1"
MODEL = "llama-3.3-70b-versatile"    # first choice; if your key cannot use it, the script
FALLBACK_MODELS = ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "llama-3.1-8b-instant"]  # tries these
MAX_CV_CHARS = 9000                  # only the first part of long CVs is sent (saves free tokens)
WAIT_BETWEEN_CVS = 10                # seconds between calls (free tier: ~8,000 tokens/minute)
MAX_RETRIES = 4                      # tries per CV when Groq says "busy / too fast"
RETRY_WAIT = 15                      # seconds; waits 15, 30, 45... between tries
# ----------------------------------------------------------------------


# ======================= reading the input files =======================

def read_text_file(path):
    return Path(path).read_text(encoding="utf-8").strip()


def read_rules(path):
    """Reads lines like  NAME = value  from requirements.txt."""
    values = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        line = line.split(" #")[0]                 # allow comments after a value
        key, value = line.split("=", 1)
        values[key.strip().upper()] = value.strip()

    def to_list(text):
        return [s.strip() for s in text.split(",") if s.strip()]

    skills = to_list(values.get("SKILLS", ""))
    return {
        "min_years": float(values.get("MIN_YEARS", 0)),
        "years_of": values.get("YEARS_OF", "TOTAL"),
        "skills": skills,
        "optional": to_list(values.get("OPTIONAL_SKILLS", "")),
        "min_skills": int(values.get("MIN_SKILLS_MATCHED", len(skills))),
    }


def extract_cv_text(path):
    """Extract text from PDF, DOCX or TXT."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        from docx import Document
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type: " + suffix)


# ============================ Groq part ==============================

def build_prompt(job_description, rules, cv_text):
    all_skills = rules["skills"] + rules["optional"]
    skill_lines = "\n".join(f'    "{s}": true or false,' for s in all_skills)
    years_line = ""
    if rules["years_of"].upper() != "TOTAL":
        years_line = f'\n  "skill_years": years of professional experience with {rules["years_of"]} (number, or null if not stated),'
    return f"""You are reading a CV. Only EXTRACT facts, do not judge the candidate.

JOB DESCRIPTION (for context):
{job_description}

CV TEXT:
{cv_text}

Return ONLY JSON in exactly this format:
{{
  "name": "candidate full name or Unknown",
  "email": "candidate email from the CV or Not found",
  "total_years": total years of professional experience (number, or null if not stated),{years_line}
  "skills": {{
{skill_lines}
  }},
  "summary": "one short sentence about the candidate"
}}

Rules:
- A skill is true only if the CV clearly mentions it (skills list, summary or work experience).
- Use the years written in the CV. Do not guess.
"""


def is_fatal(error):
    """Errors where retrying will not help (bad key, wrong model, daily limit used up)."""
    text = str(error).lower()
    return any(word in text for word in (
        "invalid_api_key", "invalid api key", "error code: 401",
        "model_not_found", "does not exist", "permission",
        "per day", "(rpd)", "(tpd)", "no credits"))


JSON_MODE_WORKS = True      # switches to False by itself if the model has no JSON mode


def pick_model(client):
    """Asks Groq which models YOUR key can use, and picks the best one."""
    available = [m.id for m in client.models.list().data]
    skip = ("whisper", "guard", "tts", "orpheus", "playai", "embed")
    chat_models = [m for m in available if not any(w in m.lower() for w in skip)]
    for wanted in [MODEL] + FALLBACK_MODELS:
        if wanted in chat_models:
            return wanted
    if chat_models:
        return chat_models[0]
    raise ValueError("No chat models available for this key. Models seen: " + ", ".join(available))


def clean_json(text):
    """Keeps only the {...} part, in case the model adds extra words or code fences."""
    start, end = text.find("{"), text.rfind("}")
    return text[start:end + 1] if start != -1 and end != -1 else text


def ask_groq(client, model, prompt):
    """Call Groq, retry if it is busy or we are going too fast."""
    global JSON_MODE_WORKS
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            options = {}
            if JSON_MODE_WORKS:
                options["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                **options,
            )
            return clean_json(response.choices[0].message.content or "")
        except Exception as e:
            if JSON_MODE_WORKS and "response_format" in str(e).lower():
                JSON_MODE_WORKS = False    # this model has no JSON mode: ask again without it
                print("   ! This model has no JSON mode - trying again without it")
                continue
            last_error = e
            print(f"   ! Groq error (attempt {attempt}/{MAX_RETRIES}): {str(e)[:120]}")
            if is_fatal(e):
                break
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"   ... waiting {wait} seconds, then trying again")
                time.sleep(wait)
    raise last_error


# ======================= local keyword scan (backup) ===================

def skill_in_text(skill, text):
    return re.search(r"\b" + re.escape(skill) + r"\d?\b", text, re.I) is not None


def find_years(label, text):
    """Finds lines like  'WordPress: 3 years'  or  'Total Experience: 3 years'."""
    match = re.search(re.escape(label) + r"\s*:\s*(\d+(?:\.\d+)?)\s*years?", text, re.I)
    return float(match.group(1)) if match else None


def extract_local(cv_text, rules):
    lines = [l.strip() for l in cv_text.splitlines() if l.strip()]
    email = re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", cv_text)
    data = {
        "name": lines[0] if lines else "Unknown",
        "email": email.group(0) if email else "Not found",
        "total_years": find_years("Total Experience", cv_text),
        "skills": {s: skill_in_text(s, cv_text) for s in rules["skills"] + rules["optional"]},
        "summary": "Checked with local keyword scan",
    }
    if rules["years_of"].upper() != "TOTAL":
        data["skill_years"] = find_years(rules["years_of"], cv_text)
    return data


# ============================ the decision =============================

def decide(data, rules):
    """Python applies the rules. Returns (status, reason, years, matched_list)."""
    skills_found = {k.lower(): bool(v) for k, v in (data.get("skills") or {}).items()}
    matched = [s for s in rules["skills"] if skills_found.get(s.lower())]

    if rules["years_of"].upper() == "TOTAL":
        years = data.get("total_years")
    else:
        years = data.get("skill_years")

    problems = []
    if len(matched) < rules["min_skills"]:
        problems.append(f"only {len(matched)} of {rules['min_skills']} required skills")
    if years is not None and years < rules["min_years"]:
        problems.append(f"{years:g} years is less than {rules['min_years']:g}")

    if problems:
        return "REJECTED", "; ".join(problems), years, matched
    if years is None:
        return "REVIEW", "years of experience not found - check manually", years, matched
    return "SHORTLISTED", "meets all rules", years, matched


# ================================ main =================================

def main():
    rules = read_rules(REQUIREMENTS_FILE)
    if not rules["skills"]:
        print("requirements.txt has no SKILLS line. Example:  SKILLS = WordPress, PHP, CSS")
        return

    cv_files = sorted(
        p for p in Path(CV_FOLDER).iterdir()
        if p.suffix.lower() in (".pdf", ".docx", ".txt")
    )
    if not cv_files:
        print(f"No CVs found in the '{CV_FOLDER}' folder.")
        return

    # ----- set up Groq (if it fails to set up, we use the local scan) -----
    ai_on = USE_AI
    client = None
    model = MODEL
    job_description = ""
    error_rows = []
    if ai_on:
        try:
            from openai import OpenAI      # used only as a connector to Groq
            client = OpenAI(api_key=read_text_file(API_KEY_FILE), base_url=GROQ_URL)
            model = pick_model(client)
            print(f"Groq model in use: {model}")
            if Path(JOB_FILE).exists():
                job_description = read_text_file(JOB_FILE)
        except Exception as e:
            print(f"Groq could not start ({e}). Using local keyword scan instead.")
            error_rows.append({"file": "(setup)", "error": "Groq could not start: " + str(e)[:300]})
            ai_on = False

    results = []
    ai_rows = []
    all_skills = rules["skills"] + rules["optional"]

    print(f"\nFound {len(cv_files)} CVs.")
    print(f"Rules: {rules['years_of']} experience >= {rules['min_years']:g} years, "
          f"and at least {rules['min_skills']} of: {', '.join(rules['skills'])}")
    print(f"Mode : {'Groq (local scan as backup)' if ai_on else 'Local keyword scan'}\n")

    for number, cv_path in enumerate(cv_files, 1):
        print("=" * 70)
        print(f"[{number}/{len(cv_files)}] {cv_path.name}")

        try:
            cv_text = extract_cv_text(cv_path)
            if len(cv_text.strip()) < 50:
                raise ValueError("Could not read text (scanned/image CV or empty file)")

            data = None
            source = "Local scan"

            if ai_on:
                try:
                    prompt = build_prompt(job_description, rules, cv_text[:MAX_CV_CHARS])
                    raw = ask_groq(client, model, prompt)
                    ai_rows.append({"file": cv_path.name, "ai_response": raw})
                    data = json.loads(raw)
                    source = "Groq"
                except Exception as e:
                    print("  Groq failed - using local keyword scan for this CV.")
                    error_rows.append({"file": cv_path.name,
                                       "error": "Groq failed, local scan used: " + str(e)[:300]})
                    if is_fatal(e):
                        print("  (Groq turned off for the remaining CVs)")
                        ai_on = False
                if ai_on:
                    time.sleep(WAIT_BETWEEN_CVS)

            if data is None:
                data = extract_local(cv_text, rules)

            status, reason, years, matched = decide(data, rules)
            found = {k.lower(): bool(v) for k, v in (data.get("skills") or {}).items()}

            # ----- print to terminal -----
            print(f"  Name   : {data.get('name', 'Unknown')}")
            print(f"  Email  : {data.get('email', 'Not found')}")
            print(f"  Checked: {source}")
            print("  ---------------------------------------------")
            years_text = "not found" if years is None else f"{years:g} years"
            print(f"  {rules['years_of']} experience: {years_text} (need {rules['min_years']:g})")
            for s in rules["skills"]:
                print(f"  {s:<20} {'MET' if found.get(s.lower()) else 'NOT MET'}")
            for s in rules["optional"]:
                print(f"  {s + ' (optional)':<20} {'found' if found.get(s.lower()) else 'not found'}")
            print("  ---------------------------------------------")
            print(f"  Skills matched: {len(matched)}/{len(rules['skills'])} (need {rules['min_skills']})")
            print(f"  RESULT: {status} - {reason}")

            # ----- save row -----
            row = {
                "file": cv_path.name,
                "name": data.get("name", "Unknown"),
                "email": data.get("email", "Not found"),
                "checked_by": source,
                "years": "" if years is None else years,
            }
            for s in all_skills:
                row[s] = "Yes" if found.get(s.lower()) else "No"
            row["skills_matched"] = len(matched)
            row["final_status"] = status
            row["reason"] = reason
            row["summary"] = data.get("summary", "")
            results.append(row)

        except Exception as e:
            print(f"  ERROR: {e}")
            error_rows.append({"file": cv_path.name, "error": str(e)[:300]})

    # ----- write CSV files -----
    if results:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    if ai_rows:
        with open(AI_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "ai_response"])
            writer.writeheader()
            writer.writerows(ai_rows)

    if error_rows:
        with open(ERRORS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "error"])
            writer.writeheader()
            writer.writerows(error_rows)

    # ----- save to database -----
    db_saved = False
    if results:
        try:
            init_db()
            save_to_db(results)
            db_saved = True
        except Exception as e:
            print(f"\nCould not save to database: {e}")

    # ----- final summary table -----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'EMAIL':<34} {'YEARS':<7} {'SKILLS':<8} {'STATUS'}")
    for r in results:
        years = "?" if r["years"] == "" else f"{r['years']:g}"
        skills = f"{r['skills_matched']}/{len(rules['skills'])}"
        print(f"{r['email']:<34} {years:<7} {skills:<8} {r['final_status']}")
    shortlisted = sum(1 for r in results if r["final_status"] == "SHORTLISTED")
    review = sum(1 for r in results if r["final_status"] == "REVIEW")
    print(f"\nShortlisted: {shortlisted} | Rejected: {len(results) - shortlisted - review} "
          f"| Need review: {review} | Problems logged: {len(error_rows)}")
    print(f"Saved: {RESULTS_CSV}" + (f", {AI_CSV}" if ai_rows else "")
          + (f", {ERRORS_CSV}" if error_rows else "")
          + (f", {DB_FILE}" if db_saved else ""))


if __name__ == "__main__":
    main()