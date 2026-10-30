"""
CV Screening Script (simple version)
Reads CVs from the "cvs" folder, sends each one to Gemini, and shortlists
candidates based on job_description.txt and requirements.txt.

Run:  python screen_cvs.py
"""

import csv
import json
import time
from pathlib import Path

from google import genai
from google.genai import types
from pypdf import PdfReader
from docx import Document

# ---------------- SETTINGS (change these if you want) ----------------
CV_FOLDER = "cvs"
JOB_FILE = "job_description.txt"
REQUIREMENTS_FILE = "requirements.txt"
API_KEY_FILE = "api_key.txt"

RESULTS_CSV = "results.csv"          # final shortlist output
GEMINI_CSV = "gemini_output.csv"     # raw Gemini replies
ERRORS_CSV = "errors.csv"            # CVs that failed

MODEL = "gemini-3.8-flash"           # change if Google renames/retires it
MIN_PERCENT_MET = 100                # 100 = ALL requirements must be met
                                     # 70 = shortlist if 70% or more are met
WAIT_BETWEEN_CVS = 5                 # seconds (free tier has rate limits)
MAX_RETRIES = 6                      # tries per CV when Gemini is busy (503)
RETRY_WAIT = 15                      # seconds; waits 15, 30, 45... between tries
# ----------------------------------------------------------------------


def read_text_file(path):
    return Path(path).read_text(encoding="utf-8").strip()


def read_requirements(path):
    """One requirement per line. Empty lines and lines starting with # are ignored."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [l.strip() for l in lines if l.strip() and not l.strip().startswith("#")]


def extract_cv_text(path):
    """Extract text from PDF, DOCX or TXT."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join((page.extract_text() or "") for page in reader.pages)
    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)
    if suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Unsupported file type: " + suffix)


def build_prompt(job_description, requirements, cv_text):
    numbered = "\n".join(f"{i}. {r}" for i, r in enumerate(requirements, 1))
    return f"""You are an HR assistant screening a CV.

JOB DESCRIPTION:
{job_description}

REQUIREMENTS (check each one):
{numbered}

CV TEXT:
{cv_text}

Return ONLY JSON in exactly this format:
{{
  "name": "candidate full name or Unknown",
  "email": "candidate email from the CV or Not found",
  "requirements": [
    {{"status": "Met" or "Not Met", "reason": "max 12 words"}}
  ],
  "summary": "one short sentence about the candidate"
}}

Rules:
- The "requirements" list must have exactly {len(requirements)} items, in the same order as above.
- Only mark "Met" if the CV clearly shows it. Do not guess.
"""


def ask_gemini(client, prompt):
    """Call Gemini, retry several times if it fails (503 = busy, usually temporary)."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0,
                ),
            )
            return response.text
        except Exception as e:
            last_error = e
            print(f"   ! Gemini error (attempt {attempt}/{MAX_RETRIES}): {str(e)[:150]}")
            if "NOT_FOUND" in str(e) or "API_KEY" in str(e) or "PERMISSION_DENIED" in str(e):
                break  # retrying will not help (wrong model name or bad key)
            if attempt < MAX_RETRIES:
                wait = RETRY_WAIT * attempt
                print(f"   ... waiting {wait} seconds, then trying again")
                time.sleep(wait)
    raise last_error


def main():
    # ----- load inputs -----
    api_key = read_text_file(API_KEY_FILE)
    job_description = read_text_file(JOB_FILE)
    requirements = read_requirements(REQUIREMENTS_FILE)

    if not requirements:
        print("requirements.txt is empty. Add one requirement per line.")
        return

    cv_files = sorted(
        p for p in Path(CV_FOLDER).iterdir()
        if p.suffix.lower() in (".pdf", ".docx", ".txt")
    )
    if not cv_files:
        print(f"No CVs found in the '{CV_FOLDER}' folder.")
        return

    client = genai.Client(api_key=api_key)

    results = []
    gemini_rows = []
    error_rows = []

    print(f"\nFound {len(cv_files)} CVs. Requirements: {len(requirements)}\n")

    # ----- process each CV -----
    for number, cv_path in enumerate(cv_files, 1):
        print("=" * 70)
        print(f"[{number}/{len(cv_files)}] {cv_path.name}")

        try:
            cv_text = extract_cv_text(cv_path)
            if len(cv_text.strip()) < 50:
                raise ValueError("Could not read text (scanned/image CV or empty file)")

            raw = ask_gemini(client, build_prompt(job_description, requirements, cv_text))
            gemini_rows.append({"file": cv_path.name, "gemini_response": raw})

            data = json.loads(raw)
            answers = data["requirements"]
            if len(answers) != len(requirements):
                raise ValueError("Gemini returned wrong number of requirement results")

            # decide shortlist in Python (not by Gemini)
            met_count = sum(1 for a in answers if a["status"].strip().lower() == "met")
            percent = met_count * 100 / len(requirements)
            final = "SHORTLISTED" if percent >= MIN_PERCENT_MET else "REJECTED"

            # ----- print to terminal -----
            print(f"  Name  : {data.get('name', 'Unknown')}")
            print(f"  Email : {data.get('email', 'Not found')}")
            print("  ---------------------------------------------")
            for i, (req, ans) in enumerate(zip(requirements, answers), 1):
                status = "MET    " if ans["status"].strip().lower() == "met" else "NOT MET"
                print(f"  {i}. [{status}] {req[:45]}")
            print("  ---------------------------------------------")
            print(f"  Result: {final}  ({met_count}/{len(requirements)} met)")

            # ----- save row -----
            row = {
                "file": cv_path.name,
                "name": data.get("name", "Unknown"),
                "email": data.get("email", "Not found"),
            }
            for i, ans in enumerate(answers, 1):
                row[f"req{i}_status"] = ans["status"]
                row[f"req{i}_reason"] = ans.get("reason", "")
            row["met_count"] = met_count
            row["percent_met"] = round(percent)
            row["final_status"] = final
            row["summary"] = data.get("summary", "")
            results.append(row)

        except Exception as e:
            print(f"  ERROR: {e}")
            error_rows.append({"file": cv_path.name, "error": str(e)})

        time.sleep(WAIT_BETWEEN_CVS)

    # ----- write CSV files -----
    if results:
        with open(RESULTS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)

    if gemini_rows:
        with open(GEMINI_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "gemini_response"])
            writer.writeheader()
            writer.writerows(gemini_rows)

    if error_rows:
        with open(ERRORS_CSV, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "error"])
            writer.writeheader()
            writer.writerows(error_rows)

    # ----- final summary table -----
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'EMAIL':<35} {'MET':<8} {'STATUS'}")
    for r in results:
        print(f"{r['email']:<35} {str(r['met_count']) + '/' + str(len(requirements)):<8} {r['final_status']}")
    shortlisted = sum(1 for r in results if r["final_status"] == "SHORTLISTED")
    print(f"\nShortlisted: {shortlisted} | Rejected: {len(results) - shortlisted} | Errors: {len(error_rows)}")
    print(f"Saved: {RESULTS_CSV}, {GEMINI_CSV}" + (f", {ERRORS_CSV}" if error_rows else ""))


if __name__ == "__main__":
    main()