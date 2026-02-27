import os, json, hashlib
from datetime import datetime
from typing import TypedDict, List, Dict, Any

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from sources import remotive_jobs, arbeitnow_jobs, weworkremotely_rss

class State(TypedDict):
    raw_jobs: List[Dict[str, Any]]
    jobs: List[Dict[str, Any]]
    ranked: List[Dict[str, Any]]
    alert: List[Dict[str, Any]]
    report_md: str

def get_llm():
    return ChatOpenAI(
        api_key=os.environ["LLM_API_KEY"],
        base_url=os.environ["LLM_BASE_URL"],
        model=os.environ["LLM_MODEL"],
        temperature=0.2,
    )

def load_config():
    with open("config.json", "r", encoding="utf-8") as f:
        return json.load(f)

def job_id(j):
    key = f"{j.get('source','')}|{j.get('company','')}|{j.get('title','')}|{j.get('url','')}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()

def tool_fetch(state: State) -> State:
    jobs = []
    for fn in (remotive_jobs, arbeitnow_jobs, weworkremotely_rss):
        try:
            jobs.extend(fn())
        except Exception:
            pass
    return {**state, "raw_jobs": jobs}

def tool_normalize_dedupe(state: State) -> State:
    cfg = load_config()
    inc = [k.lower() for k in cfg["keywords_include"]]
    exc = [k.lower() for k in cfg["keywords_exclude"]]

    seen = set()
    out = []
    for j in state["raw_jobs"]:
        title = (j.get("title") or "").strip()
        if not title:
            continue
        text = (title + " " + " ".join(j.get("tags") or []) + " " + (j.get("location") or "")).lower()

        if any(x in text for x in exc):
            continue
        if not any(k in text for k in inc):
            continue

        jid = job_id(j)
        if jid in seen:
            continue
        seen.add(jid)
        j["id"] = jid
        out.append(j)

    return {**state, "jobs": out}

def agent_rank(state: State) -> State:
    cfg = load_config()
    llm = get_llm()

    # Keep prompt short to reduce token usage (free-tier friendly)
    items = state["jobs"][:40]
    prompt = (
        "You are helping a Data Science student find internships/entry-level roles.\n"
        "Score each job from 1-10 based on relevance to: Data Science, ML, NLP, LLM, RAG, Python, analytics.\n"
        "Return JSON array of objects: {id, score, reason}.\n\n"
        "Jobs:\n" + "\n".join([f"- id={j['id']} title={j['title']} location={j.get('location','')} tags={j.get('tags',[])}"
                              for j in items])
    )
    resp = llm.invoke(prompt).content

    # Parse loosely: we’ll do best-effort JSON
    ranked = []
    try:
        parsed = json.loads(resp)
        scores = {x["id"]: x for x in parsed if isinstance(x, dict) and "id" in x}
        for j in items:
            s = scores.get(j["id"], {})
            j["score"] = int(s.get("score", 0)) if str(s.get("score", "0")).isdigit() else 0
            j["reason"] = s.get("reason", "")
            ranked.append(j)
    except Exception:
        # fallback: no LLM parse -> keep but low score
        for j in items:
            j["score"] = 5
            j["reason"] = "Fallback scoring"
            ranked.append(j)

    ranked.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {**state, "ranked": ranked}

def pick_alert(state: State) -> State:
    cfg = load_config()
    min_score = cfg["min_score_to_alert"]
    max_alerts = cfg["max_alerts"]
    alert = [j for j in state["ranked"] if j.get("score", 0) >= min_score][:max_alerts]
    return {**state, "alert": alert}

def build_report_and_send(state: State) -> State:
    import os
    from datetime import datetime

    token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")

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

    # Save dated digest
    os.makedirs("out", exist_ok=True)
    path = f"out/digest_{today}.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report)

    # ✅ Always update latest_report.md
    with open("latest_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    # ✅ Send email (always send daily digest)
    send_email_via_webhook(
        subject=f"Internship Intelligence Digest — {today}",
        body=report
    )

    # Optional Telegram
    if token and chat_id and state["alert"]:
        import requests
        msg = "🔥 Top internship matches:\n\n" + "\n".join(
            [f"• ({j['score']}/10) {j['title']} — {j.get('url','')}" for j in state["alert"]]
        )
        requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": msg},
            timeout=20
        )

    return {**state, "report_md": report}
        

    return {**state, "report_md": report}

def build_graph():
    g = StateGraph(State)
    g.add_node("fetch", tool_fetch)
    g.add_node("dedupe", tool_normalize_dedupe)
    g.add_node("rank", agent_rank)
    g.add_node("pick", pick_alert)
    g.add_node("report_send", build_report_and_send)

    g.set_entry_point("fetch")
    g.add_edge("fetch", "dedupe")
    g.add_edge("dedupe", "rank")
    g.add_edge("rank", "pick")
    g.add_edge("pick", "report_send")
    g.add_edge("report_send", END)
    return g.compile()

def send_email_via_webhook(subject: str, body: str):
    import os
    import requests

    url = os.environ.get("MAIL_WEBHOOK_URL", "")
    to = os.environ.get("MAIL_TO", "")

    if not url or not to:
        return

    requests.post(
        url,
        json={
            "to": to,
            "subject": subject,
            "body": body
        },
        timeout=20
    )
if __name__ == "__main__":
    graph = build_graph()
    out = graph.invoke({"raw_jobs": [], "jobs": [], "ranked": [], "alert": [], "report_md": ""})
    print(out["report_md"])
