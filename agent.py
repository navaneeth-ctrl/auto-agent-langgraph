"""
Internship Intelligence Agent (Email + latest_report.md)

What it does:
- Fetches jobs from free sources
- Filters + dedupes
- Uses LLM (Groq/OpenRouter OpenAI-compatible) to score relevance
- Builds a markdown digest
- Writes:
    - out/digest_YYYY-MM-DD.md
    - latest_report.md  (for GitHub auto-commit)
- Sends email via webhook (Google Apps Script)

Required env vars:
- LLM_API_KEY
- LLM_BASE_URL    e.g. https://api.groq.com/openai/v1  OR https://openrouter.ai/api/v1
- LLM_MODEL       e.g. llama-3.1-8b-instant
- MAIL_WEBHOOK_URL
- MAIL_TO

Optional:
- none
"""

import os
import json
import hashlib
from datetime import datetime
from typing import TypedDict, List, Dict, Any

import requests
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END


# ---------------------------
# Config / helpers
# ---------------------------

CONFIG_PATH = "config.json"


def load_config() -> Dict[str, Any]:
    # Safe defaults if config.json missing
    defaults = {
        "keywords_include": [
            "intern", "internship", "trainee",
            "data science", "machine learning", "nlp", "llm", "rag",
            "python", "analytics"
        ],
        "keywords_exclude": [
            "senior", "lead", "principal", "manager", "staff",
            "5+ years", "7+ years", "10+ years"
        ],
        "locations_prefer": ["India", "Remote"],
        "min_score_to_alert": 6,
        "max_alerts": 6,
        "max_jobs_to_score": 30
    }
    if not os.path.exists(CONFIG_PATH):
        return defaults
    try:
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        for k, v in defaults.items():
            cfg.setdefault(k, v)
        return cfg
    except Exception:
        return defaults


def get_llm() -> ChatOpenAI:
    # OpenAI-compatible client, works for Groq/OpenRouter/Together w/ compatible endpoints
    return ChatOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        temperature=0.2,
    )


def stable_job_id(j: Dict[str, Any]) -> str:
    key = f"{j.get('source','')}|{j.get('company','')}|{j.get('title','')}|{j.get('url','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def safe_get(url: str, timeout: int = 30) -> Any:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "internship-agent/1.0"})
    r.raise_for_status()
    # Some endpoints are JSON; this wrapper assumes JSON
    return r.json()


# ---------------------------
# Job sources (free/public)
# ---------------------------

import feedparser

def indeed_india_jobs():
    url = "https://in.indeed.com/rss?q=data+science+intern&l=India"
    feed = feedparser.parse(url)

    jobs = []
    for entry in feed.entries:
        jobs.append({
            "source": "Indeed India",
            "title": entry.title,
            "company": "",
            "location": "India",
            "url": entry.link,
            "tags": [],
            "date": getattr(entry, "published", "")
        })
    return jobs

from bs4 import BeautifulSoup
import requests

def internshala_search_pages() -> list[str]:
    # multiple DS-related internship categories
    return [
        "https://internshala.com/internships/data-science-internship/",
        "https://internshala.com/internships/machine-learning-internship/",
        "https://internshala.com/internships/analytics-internship/",
    ]

def internshala_jobs() -> list[dict]:
    headers = {"User-Agent": "Mozilla/5.0"}
    jobs = []

    for url in internshala_search_pages():
        r = requests.get(url, headers=headers, timeout=30)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "lxml")

        # Internshala cards commonly have this container
        cards = soup.select("div.individual_internship")

        for c in cards:
            title_el = c.select_one("h3.job-internship-name") or c.select_one("h3")
            company_el = c.select_one("p.company-name") or c.select_one("h4")
            loc_el = c.select_one("div#location_names") or c.select_one(".location_link")

            title = title_el.get_text(" ", strip=True) if title_el else ""
            company = company_el.get_text(" ", strip=True) if company_el else ""
            location = loc_el.get_text(" ", strip=True) if loc_el else "India"

            a = c.select_one("a.view_detail_button") or c.select_one("a[href]")
            link = ""
            if a and a.get("href"):
                href = a["href"]
                link = href if href.startswith("http") else ("https://internshala.com" + href)

            if title and link:
                jobs.append({
                    "source": "Internshala",
                    "title": title,
                    "company": company,
                    "location": location,
                    "url": link,
                    "tags": ["internshala"],
                    "date": ""
                })

    return jobs





# ---------------------------
# Email webhook (Apps Script)
# ---------------------------

def send_email_via_webhook(subject: str, body: str) -> None:
    url = os.environ.get("MAIL_WEBHOOK_URL", "")
    to = os.environ.get("MAIL_TO", "")
    if not url or not to:
        # Not configured: just skip
        return

    # Keep body reasonably sized for email
    if len(body) > 15000:
        body = body[:15000] + "\n\n[Truncated]\n"

    try:
        requests.post(
            url,
            json={"to": to, "subject": subject, "body": body},
            timeout=20
        ).raise_for_status()
    except Exception:
        # Avoid crashing the whole run due to mail issues
        pass


# ---------------------------
# LangGraph state + nodes
# ---------------------------

class State(TypedDict):
    raw_jobs: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]
    alert: List[Dict[str, Any]]
    report_md: str


def tool_fetch(state: State) -> State:
    jobs: List[Dict[str, Any]] = []
    for fn in (indeed_india_jobs,):
        try:
            jobs.extend(fn())
        except Exception:
            pass
    return {**state, "raw_jobs": jobs}


def tool_normalize_dedupe_filter(state: State) -> State:
    cfg = load_config()

    exc = [k.lower() for k in cfg.get("keywords_exclude", [])]
    loc_pref = [x.lower() for x in cfg.get("locations_prefer", ["india", "remote"])]

    # STRICT domain filter (Data Science field only)
    domain_terms = [
        "data science", "data scientist", "data analyst",
        "machine learning", "ml", "ai", "artificial intelligence",
        "nlp", "llm", "deep learning",
        "python", "sql", "analytics"
    ]

    intern_terms = ["intern", "internship", "trainee"]

    seen = set()
    out: List[Dict[str, Any]] = []

    for j in state["raw_jobs"]:
        title = (j.get("title") or "").strip()
        if not title:
            continue

        location = (j.get("location") or "").strip()
        tags = " ".join(j.get("tags", []) or [])

        text = " ".join([
            title,
            j.get("company", "") or "",
            location,
            tags
        ]).lower()

        # 1) Exclude senior roles
        if any(x in text for x in exc):
            continue

        # 2) Must be internship
        if not any(t in text for t in intern_terms):
            continue

        # 3) Must be India or Remote
        if location:
            if not any(p in location.lower() for p in loc_pref):
                continue

        # 4) Must be Data Science related
        if not any(d in text for d in domain_terms):
            continue

        jid = stable_job_id(j)
        if jid in seen:
            continue
        seen.add(jid)

        j["id"] = jid
        out.append(j)

    return {**state, "jobs": out}


def agent_rank(state: State) -> State:
    cfg = load_config()
    llm = get_llm()

    items = state["jobs"][: int(cfg.get("max_jobs_to_score", 30))]
    if not items:
        return {**state, "ranked": []}

    prompt = (
    "Return ONLY valid JSON. No markdown. No explanation.\n"
    "You are scoring internships for a Data Science student.\n"
    "Score 1-10 based on relevance to Data Science, ML, NLP, LLM, Python, SQL.\n"
    "Return JSON array format:\n"
    "[{\"id\":\"...\",\"score\":7,\"reason\":\"...\"}]\n\n"
    "Jobs:\n"
    + "\n".join(
        [
            f"- id={j['id']} title={j['title']} company={j.get('company','')} "
            f"location={j.get('location','')} tags={j.get('tags',[])}"
            for j in items
        ]
    )
)
    resp = llm.invoke(prompt).content
    resp = resp.strip()

    ranked: List[Dict[str, Any]] = []
    try:
        parsed = json.loads(resp)
        score_map = {
            x["id"]: x for x in parsed
            if isinstance(x, dict) and "id" in x
        }

        for j in items:
            s = score_map.get(j["id"], {})
            sc = s.get("score", 0)
            try:
                sc_int = int(sc)
            except Exception:
                sc_int = 0
            j["score"] = max(0, min(10, sc_int))
            j["reason"] = (s.get("reason", "") or "").strip()
            ranked.append(j)

    except Exception:
        # Fallback ranking (no JSON parse): use simple heuristic
        for j in items:
            t = (j.get("title") or "").lower()
            score = 6
            if "intern" in t or "internship" in t:
                score += 1
            if "machine learning" in t or "data science" in t:
                score += 1
            j["score"] = min(10, score)
            j["reason"] = "Fallback ranking (LLM JSON parse failed)."
            ranked.append(j)

    ranked.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {**state, "ranked": ranked}


def pick_alert(state: State) -> State:
    cfg = load_config()
    min_score = int(cfg.get("min_score_to_alert", 6))
    max_alerts = int(cfg.get("max_alerts", 6))

    alert = [j for j in state["ranked"] if int(j.get("score", 0)) >= min_score][:max_alerts]
    return {**state, "alert": alert}


def build_report_and_send(state: State) -> State:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    lines = [f"# Internship Intelligence Digest — {today}\n"]

    if not state["alert"]:
        lines.append("No strong matches today based on your keywords/threshold.\n")
    else:
        for j in state["alert"]:
            lines.append(
                f"## {j.get('title','')}\n"
                f"- Source: {j.get('source','')}\n"
                f"- Company: {j.get('company','')}\n"
                f"- Location: {j.get('location','')}\n"
                f"- Score: {j.get('score',0)}/10\n"
                f"- Why: {j.get('reason','')}\n"
                f"- Link: {j.get('url','')}\n"
            )

    report = "\n".join(lines)

    # Save dated digest artifact
    os.makedirs("out", exist_ok=True)
    with open(f"out/digest_{today}.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Save always-latest file (committed by GitHub Actions)
    with open("latest_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # Email (always send daily digest; change to only-send-if-alert if you want)
    send_email_via_webhook(
        subject=f"Internship Intelligence Digest — {today}",
        body=report
    )

    return {**state, "report_md": report}


def build_graph():
    g = StateGraph(State)

    g.add_node("fetch", tool_fetch)
    g.add_node("dedupe_filter", tool_normalize_dedupe_filter)
    g.add_node("rank", agent_rank)
    g.add_node("pick", pick_alert)
    g.add_node("report_send", build_report_and_send)

    g.set_entry_point("fetch")
    g.add_edge("fetch", "dedupe_filter")
    g.add_edge("dedupe_filter", "rank")
    g.add_edge("rank", "pick")
    g.add_edge("pick", "report_send")
    g.add_edge("report_send", END)

    return g.compile()


if __name__ == "__main__":
    graph = build_graph()
    out = graph.invoke({"raw_jobs": [], "jobs": [], "ranked": [], "alert": [], "report_md": ""})
    print(out["report_md"])
