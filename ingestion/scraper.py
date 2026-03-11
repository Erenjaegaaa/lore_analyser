"""
ingestion/scraper.py — Fetches raw HTML from One Wiki to Rule Them All (lotr.fandom.com).

Public interface:
    scrape_page(url: str) -> dict
        Returns: {url, title, html, scraped_at}
    scrape_pages(urls: list[str], delay: float) -> list[dict]
        Scrapes multiple pages with a polite delay between requests.
"""

import time
from datetime import datetime, timezone

import httpx
from bs4 import BeautifulSoup

from utils.logger import get_logger

log = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

REQUEST_TIMEOUT = 15  # seconds


# ── Public interface ──────────────────────────────────────────────────────────

def scrape_page(url: str) -> dict | None:
    """
    Fetch a single wiki page and return its raw HTML plus metadata.

    Returns None if the request fails, so callers can skip bad URLs
    without crashing the whole pipeline.

    Returns:
        {
            "url": str,
            "title": str,       # extracted from <title> tag
            "html": str,        # full raw HTML of the page
            "scraped_at": str,  # ISO-8601 UTC timestamp
        }
    """
    log.debug("Scraping: %s", url)

    try:
        response = httpx.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        log.warning("HTTP %s for URL: %s", e.response.status_code, url)
        return None
    except httpx.RequestError as e:
        log.warning("Request failed for %s: %s", url, e)
        return None

    html = response.text

    # Extract page title from <title> tag as a lightweight metadata field
    soup = BeautifulSoup(html, "lxml")
    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else url

    # Fandom titles are usually "Page Name | The One Wiki to Rule Them All | Fandom"
    # Keep only the first segment for clarity
    if "|" in title:
        title = title.split("|")[0].strip()

    log.info("Scraped: '%s' (%d chars)", title, len(html))

    return {
        "url": url,
        "title": title,
        "html": html,
        "scraped_at": datetime.now(timezone.utc).isoformat(),
    }


def scrape_pages(urls: list[str], delay: float = 1.5) -> list[dict]:
    """
    Scrape multiple pages with a polite delay between requests.

    Args:
        urls:  List of fully qualified URLs to scrape.
        delay: Seconds to wait between requests. Default 1.5s is
               polite for wiki scraping without being too slow.

    Returns:
        List of page dicts (failed pages are silently skipped).
    """
    results = []

    for i, url in enumerate(urls):
        page = scrape_page(url)
        if page:
            results.append(page)

        # Sleep between requests (skip sleep after the last URL)
        if i < len(urls) - 1:
            time.sleep(delay)

    log.info("Scraped %d / %d pages successfully.", len(results), len(urls))
    return results


# ── Default URL list for Day 1 testing ───────────────────────────────────────

# A small starter set of well-structured lotr.fandom.com pages.
# These are stable, content-rich pages good for testing the pipeline.
DEFAULT_URLS = [
    # ── Characters ────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/Aragorn",
    "https://en.wikipedia.org/wiki/Gandalf",
    "https://en.wikipedia.org/wiki/Frodo_Baggins",
    "https://en.wikipedia.org/wiki/Sauron",
    "https://en.wikipedia.org/wiki/Legolas",
    "https://en.wikipedia.org/wiki/Gimli_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Samwise_Gamgee",
    "https://en.wikipedia.org/wiki/Gollum",
    "https://en.wikipedia.org/wiki/Boromir",
    "https://en.wikipedia.org/wiki/Faramir",
    "https://en.wikipedia.org/wiki/%C3%89owyn",
    "https://en.wikipedia.org/wiki/Th%C3%A9oden",
    "https://en.wikipedia.org/wiki/%C3%89omer",
    "https://en.wikipedia.org/wiki/Elrond",
    "https://en.wikipedia.org/wiki/Galadriel",
    "https://en.wikipedia.org/wiki/Arwen",
    "https://en.wikipedia.org/wiki/Bilbo_Baggins",
    "https://en.wikipedia.org/wiki/Saruman",
    "https://en.wikipedia.org/wiki/Treebeard",
    "https://en.wikipedia.org/wiki/Tom_Bombadil",
    "https://en.wikipedia.org/wiki/Witch-king_of_Angmar",
    "https://en.wikipedia.org/wiki/Balrog",
    "https://en.wikipedia.org/wiki/Shelob",
    "https://en.wikipedia.org/wiki/Morgoth",
    "https://en.wikipedia.org/wiki/F%C3%ABanor",
    "https://en.wikipedia.org/wiki/L%C3%BAthien",
    "https://en.wikipedia.org/wiki/Beren",
    "https://en.wikipedia.org/wiki/T%C3%BArin_Turambar",
    "https://en.wikipedia.org/wiki/Meriadoc_Brandybuck",
    "https://en.wikipedia.org/wiki/Peregrin_Took",
    "https://en.wikipedia.org/wiki/Denethor",
    "https://en.wikipedia.org/wiki/Smaug",
    "https://en.wikipedia.org/wiki/Thorin_Oakenshield",

    # ── Places ────────────────────────────────────────────────
    "https://en.wikipedia.org/wiki/The_Shire",
    "https://en.wikipedia.org/wiki/Minas_Tirith",
    "https://en.wikipedia.org/wiki/Middle-earth",
    "https://en.wikipedia.org/wiki/Mordor",
    "https://en.wikipedia.org/wiki/Rohan_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Gondor",
    "https://en.wikipedia.org/wiki/Rivendell",
    "https://en.wikipedia.org/wiki/Lothl%C3%B3rien",
    "https://en.wikipedia.org/wiki/Isengard",
    "https://en.wikipedia.org/wiki/Helm%27s_Deep",
    "https://en.wikipedia.org/wiki/Moria_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Valinor",
    "https://en.wikipedia.org/wiki/N%C3%BAmenor",
    "https://en.wikipedia.org/wiki/Erebor",
    "https://en.wikipedia.org/wiki/Mirkwood",
    "https://en.wikipedia.org/wiki/Beleriand",
    "https://en.wikipedia.org/wiki/Fangorn_Forest_(Tolkien)",

    # ── Artifacts & Objects ───────────────────────────────────
    "https://en.wikipedia.org/wiki/One_Ring",
    "https://en.wikipedia.org/wiki/Palant%C3%ADr",
    "https://en.wikipedia.org/wiki/Silmarils",
    "https://en.wikipedia.org/wiki/Rings_of_Power_(Tolkien)",

    # ── Races & Peoples ──────────────────────────────────────
    "https://en.wikipedia.org/wiki/Elf_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Dwarf_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Hobbit",
    "https://en.wikipedia.org/wiki/Orc_(Middle-earth)",
    "https://en.wikipedia.org/wiki/Ent",
    "https://en.wikipedia.org/wiki/Nazg%C3%BBl",
    "https://en.wikipedia.org/wiki/Istari",
    "https://en.wikipedia.org/wiki/Vala_(Middle-earth)",
    "https://en.wikipedia.org/wiki/D%C3%BAnedain",
    "https://en.wikipedia.org/wiki/Uruk-hai",

    # ── Events & Wars ────────────────────────────────────────
    "https://en.wikipedia.org/wiki/War_of_the_Ring",
    "https://en.wikipedia.org/wiki/Battle_of_the_Pelennor_Fields",
    "https://en.wikipedia.org/wiki/Quest_of_Erebor",
    "https://en.wikipedia.org/wiki/War_of_the_Last_Alliance",
    "https://en.wikipedia.org/wiki/Fall_of_Gondolin",

    # ── Books & Lore ─────────────────────────────────────────
    "https://en.wikipedia.org/wiki/The_Lord_of_the_Rings",
    "https://en.wikipedia.org/wiki/The_Hobbit",
    "https://en.wikipedia.org/wiki/The_Silmarillion",
    "https://en.wikipedia.org/wiki/Unfinished_Tales",
    "https://en.wikipedia.org/wiki/The_Children_of_H%C3%BArin",
    "https://en.wikipedia.org/wiki/The_Fellowship_of_the_Ring",
    "https://en.wikipedia.org/wiki/The_Two_Towers",
    "https://en.wikipedia.org/wiki/The_Return_of_the_King",
    "https://en.wikipedia.org/wiki/J._R._R._Tolkien",
    "https://en.wikipedia.org/wiki/Middle-earth_in_film",
]