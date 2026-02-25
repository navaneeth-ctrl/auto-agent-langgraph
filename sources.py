import requests
import xml.etree.ElementTree as ET

def remotive_jobs():
    # Public API: https://remotive.com/api/remote-jobs
    url = "https://remotive.com/api/remote-jobs"
    data = requests.get(url, timeout=30).json()
    jobs = []
    for j in data.get("jobs", []):
        jobs.append({
            "source": "Remotive",
            "title": j.get("title", ""),
            "company": j.get("company_name", ""),
            "location": j.get("candidate_required_location", ""),
            "url": j.get("url", ""),
            "tags": j.get("tags", []),
            "date": j.get("publication_date", "")
        })
    return jobs

def arbeitnow_jobs():
    # Public API: https://www.arbeitnow.com/api/job-board-api
    url = "https://www.arbeitnow.com/api/job-board-api"
    data = requests.get(url, timeout=30).json()
    jobs = []
    for j in data.get("data", []):
        jobs.append({
            "source": "Arbeitnow",
            "title": j.get("title", ""),
            "company": j.get("company_name", ""),
            "location": j.get("location", ""),
            "url": j.get("url", ""),
            "tags": j.get("tags", []),
            "date": j.get("created_at", "")
        })
    return jobs

def weworkremotely_rss():
    # RSS feed (may change; easy to swap later)
    url = "https://weworkremotely.com/categories/remote-programming-jobs.rss"
    xml = requests.get(url, timeout=30).text
    root = ET.fromstring(xml)
    jobs = []
    for item in root.findall(".//item"):
        jobs.append({
            "source": "WeWorkRemotely",
            "title": (item.findtext("title") or ""),
            "company": "",  # not always present in RSS title cleanly
            "location": "Remote",
            "url": (item.findtext("link") or ""),
            "tags": [],
            "date": (item.findtext("pubDate") or "")
        })
    return jobs
