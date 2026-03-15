"""
Enhanced scraper for Masters Union UG Data Science & AI programme.
Scrapes the main page plus all sub-pages (curriculum, admissions, career, class profile).
Produces a structured text file with section metadata for RAG ingestion.
"""

import requests
from bs4 import BeautifulSoup
import os
import time

URLS = {
    "main":          "https://mastersunion.org/ug-data-science-and-artificial-intelligence",
    "curriculum":    "https://mastersunion.org/ug-data-science-curriculum",
    "admissions":    "https://mastersunion.org/ug-data-science-admissions-and-fees",
    "career":        "https://mastersunion.org/ug-data-science-career-prospects",
    "class_profile": "https://mastersunion.org/ug-data-science-class-profile",
}

OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "scraped_web_data.txt")

SECTION_KEYWORDS = {
    "OVERVIEW": ["bachelor", "undergraduate programme", "data science & ai", "learn by building",
                 "3+1", "dual degree", "illinois tech", "co-built by pwc", "ola krutrim",
                 "applied, low-theory", "learn-by-shipping", "about the programme",
                 "programme overview", "programme highlights"],
    "CURRICULUM": ["curriculum", "year 1", "year 2", "year 3", "year 4", "semester",
                   "courses", "linear algebra", "machine learning", "deep learning",
                   "nlp", "mlops", "reinforcement", "outclass", "syllabus", "module",
                   "python", "tensorflow", "pytorch", "statistics", "probability",
                   "data structures", "algorithms", "computer vision", "genai",
                   "natural language", "neural network", "capstone"],
    "FACULTY": ["faculty", "professor", "cto", "cxo", "md ", "mentor", "guest", "speaker",
                "google", "microsoft", "amazon", "nasa", "iit", "ph.d", "nitin gaur",
                "ibm", "industry expert", "empaneled"],
    "ADMISSIONS": ["admissions", "apply", "application", "jee", "sat", "musat",
                   "aptitude test", "video essay", "eligibility", "selection",
                   "round 3 deadline", "commencement date", "intake", "how to apply",
                   "application fee", "interview", "shortlist"],
    "FEES": ["fee", "tuition", "inr", "lakh", "scholarship", "25%", "50%", "80%",
             "admission fee", "year 1:", "year 2:", "year 3:", "year 4:",
             "₹", "payment", "emi", "loan", "financial aid", "merit-based"],
    "CAREER": ["placement", "career", "internship", "job", "recruiter", "salary",
               "startup", "entrepreneur", "venture", "hired", "company", "ctc",
               "lpa", "average package", "highest package", "razorpay",
               "placement report", "recruiter", "industry training"],
    "GLOBAL": ["global immersion", "illinois tech", "chicago", "silicon valley",
               "germany", "japan", "singapore", "dubai", "international",
               "nissan", "daikin", "rakuten", "mercedes", "porsche",
               "museum of the future", "careem", "europe"],
    "CAMPUS": ["campus", "hostel", "gurugram", "dlf", "cyberpark", "lab", "gym",
               "cafeteria", "club", "pwc x mu", "maker lab", "residential",
               "5,000 sq", "prototype", "sports", "library"],
    "CLASS_PROFILE": ["class profile", "cohort", "student profile", "batch",
                      "diversity", "background", "average age", "gender ratio",
                      "states represented", "student background"],
    "CONTACT": ["contact", "email", "phone", "address", "ugadmissions", "+91",
                "write to us", "queries"],
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def _classify_line(line: str) -> str:
    lower = line.lower()
    best_section = "GENERAL"
    best_score = 0
    for section, keywords in SECTION_KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in lower)
        if score > best_score:
            best_score = score
            best_section = section
    return best_section


def _extract_tables(soup: BeautifulSoup) -> list[str]:
    """Extract tabular data as structured text lines."""
    results = []
    for table in soup.find_all("table"):
        rows = table.find_all("tr")
        for row in rows:
            cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
            if any(cells):
                results.append(" | ".join(cells))
    return results


def _extract_lists(soup: BeautifulSoup) -> list[str]:
    """Extract list items that might be missed by get_text."""
    results = []
    for ul in soup.find_all(["ul", "ol"]):
        for li in ul.find_all("li"):
            text = li.get_text(strip=True)
            if len(text) > 5:
                results.append(f"- {text}")
    return results


def _scrape_single(url: str) -> list[str]:
    """Fetch one URL and return cleaned, deduplicated lines."""
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "lxml")

    # Remove boilerplate tags
    for tag in soup(["script", "style", "nav", "footer", "noscript", "svg",
                     "img", "iframe", "form", "input", "button", "meta", "link"]):
        tag.decompose()

    # Extract structured data before stripping HTML
    table_lines = _extract_tables(soup)
    list_lines = _extract_lists(soup)

    # Get all text
    raw_text = soup.get_text(separator="\n", strip=True)
    lines = [l.strip() for l in raw_text.splitlines() if len(l.strip()) > 10]

    # Deduplicate while preserving order
    seen: set = set()
    unique: list = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            unique.append(l)

    # Add structured data that may have been flattened
    for l in table_lines + list_lines:
        if l not in seen and len(l.strip()) > 5:
            seen.add(l)
            unique.append(l)

    return unique


def scrape() -> str:
    all_sections: dict = {k: [] for k in SECTION_KEYWORDS}
    all_sections["GENERAL"] = []

    for label, url in URLS.items():
        print(f"[scraper] Fetching {label}: {url} ...")
        try:
            lines = _scrape_single(url)
            print(f"[scraper] {label}: got {len(lines)} lines")
            for line in lines:
                section = _classify_line(line)
                all_sections[section].append(line)
        except Exception as e:
            print(f"[scraper] WARN: Failed to scrape {label}: {e}")
        # Be polite to the server
        time.sleep(1)

    # Build structured output
    output_parts = [
        "=" * 80,
        "MASTERS' UNION — UG DATA SCIENCE & AI",
        "Source: Scraped from official programme webpage and sub-pages",
        f"Pages scraped: {', '.join(URLS.keys())}",
        "=" * 80,
        "",
    ]

    for section_name, section_lines in all_sections.items():
        if not section_lines:
            continue
        output_parts.append("-" * 60)
        output_parts.append(f"SECTION: {section_name}")
        output_parts.append("-" * 60)
        output_parts.extend(section_lines)
        output_parts.append("")

    structured = "\n".join(output_parts)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(structured)

    print(f"[scraper] Saved {len(structured)} chars → {OUTPUT_FILE}")
    return structured


if __name__ == "__main__":
    scrape()
