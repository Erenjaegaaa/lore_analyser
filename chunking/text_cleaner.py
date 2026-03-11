# """
# chunking/text_cleaner.py — Strips HTML and normalizes raw Fandom wiki page text.

# 
#Fandom pages contain heavy boilerplate: navigation menus, sidebars, category
# footers, edit buttons, and ad slots. This module removes all of it and returns
# clean prose text suitable for chunking.

# Public interface:
#     clean_html(html: str) -> str
#         Returns plain text with no tags or boilerplate.
#     clean_page(page: dict) -> dict
#         Convenience wrapper: takes a scraper output dict, returns it with a
#         new "clean_text" field added.
# """

# import re
# import unicodedata

# from bs4 import BeautifulSoup, Comment

# from utils.logger import get_logger

# log = get_logger(__name__)

# # ── Fandom-specific CSS classes and IDs to remove entirely ───────────────────
# # These contain navigation, ads, edit controls, and other non-content elements.

# FANDOM_REMOVE_SELECTORS = [
#     # Site navigation and header
#     "header", "nav", "#WikiaBar", ".WikiaBar",
#     # Sidebar and table of contents
#     ".toc", "#toc", ".sidebar", ".infobox",
#     # Edit buttons and page actions
#     ".page-actions", ".editsection", ".mw-editsection",
#     # Footer and categories
#     "footer", ".page-footer", ".categories", ".catlinks",
#     # Fandom-specific UI
#     ".fandom-community-header", ".global-navigation",
#     ".notifications-placeholder", ".wds-community-bar",
#     # Ad slots
#     ".ad-slot", ".advertisement", "#WikiaAdInHouseSkyscraper",
#     # References section (contains [1][2] citation clutter)
#     ".references", "#References",
#     # Navboxes at the bottom of pages
#     ".navbox", ".navigation-not-searchable",
#     # Media captions and thumbnails (keep text, remove decorative structure)
#     ".thumbcaption", ".gallerybox",
#     # Talk page notices
#     ".ambox", ".tmbox",
# ]


# # ── Public interface ──────────────────────────────────────────────────────────

# def clean_html(html: str) -> str:
#     """
#     Parse raw Fandom wiki HTML and return clean plain text.

#     Processing steps:
#       1. Parse with lxml
#       2. Remove HTML comments
#       3. Remove all boilerplate elements (nav, sidebar, footer, ads, etc.)
#       4. Extract text from the main article content div
#       5. Normalize whitespace and Unicode
#       6. Remove citation markers like [1], [2], [note 1]

#     Returns:
#         Clean plain text string. Empty string if no content found.
#     """
#     soup = BeautifulSoup(html, "lxml")

#     # Step 1: Remove HTML comments (<!-- ... -->)
#     for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
#         comment.extract()

#     # Step 2: Remove boilerplate elements
#     for selector in FANDOM_REMOVE_SELECTORS:
#         for element in soup.select(selector):
#             element.decompose()

#     # Step 3: Find the main article content
#     # Fandom wraps article content in <div class="mw-parser-output">
#     content_div = soup.find("div", class_="mw-parser-output")

#     if content_div is None:
#         # Fallback: try the generic article tag, then body
#         content_div = soup.find("article") or soup.find("body")
#         log.warning("mw-parser-output not found, falling back to broader content extraction.")

#     if content_div is None:
#         log.error("Could not find any content container in HTML.")
#         return ""

#     # Step 4: Extract text with newlines between block elements
#     # get_text(separator="\n") inserts newlines between tags
#     raw_text = content_div.get_text(separator="\n")

#     # Step 5: Normalize
#     text = _normalize_text(raw_text)

#     log.debug("Cleaned text: %d chars", len(text))
#     return text


# def clean_page(page: dict) -> dict:
#     """
#     Convenience wrapper for pipeline use.

#     Takes a page dict from scraper.py (must have "html" key) and returns
#     the same dict with a "clean_text" field added.

#     Does not modify the original dict.
#     """
#     clean_text = clean_html(page["html"])
#     return {**page, "clean_text": clean_text}


# # ── Internal helpers ──────────────────────────────────────────────────────────

# def _normalize_text(text: str) -> str:
#     """
#     Clean up raw extracted text:
#       - Normalize Unicode to NFC (handles special characters consistently)
#       - Remove citation markers: [1], [2], [note 1], [a], etc.
#       - Collapse multiple blank lines into a single blank line
#       - Strip leading/trailing whitespace from each line
#       - Remove lines that are pure whitespace or very short (likely artifacts)
#     """
#     # Unicode normalization
#     text = unicodedata.normalize("NFC", text)

#     # Remove citation markers like [1], [22], [note 3], [a]
#     text = re.sub(r"\[\d+\]", "", text)
#     text = re.sub(r"\[note\s*\d+\]", "", text, flags=re.IGNORECASE)
#     text = re.sub(r"\[[a-z]\]", "", text)

#     # Strip each line and remove very short artifact lines (< 3 chars)
#     lines = []
#     for line in text.splitlines():
#         line = line.strip()
#         if len(line) >= 3:
#             lines.append(line)

#     # Collapse 3+ consecutive blank lines into a single blank line
#     text = "\n".join(lines)
#     text = re.sub(r"\n{3,}", "\n\n", text)

#     return text.strip()   

"""
chunking/text_cleaner.py — Strips HTML and normalizes Wikipedia page text.

Public interface:
    clean_html(html: str) -> str
    clean_page(page: dict) -> dict
"""

import re
import unicodedata

from bs4 import BeautifulSoup, Comment

from utils.logger import get_logger

log = get_logger(__name__)

REMOVE_SELECTORS = [
    # Navigation and site chrome
    "#mw-navigation", "#mw-head", "#mw-panel", "#mw-page-base",
    "#mw-head-base", "#catlinks", "#footer", "#siteNotice",
    # Table of contents
    "#toc", ".toc",
    # Edit buttons
    ".mw-editsection", ".mw-editsection-bracket",
    # Infoboxes and sidebars
    ".infobox", ".infobox-full-data", ".sidebar",
    # References section
    ".reflist", ".mw-references-wrap", ".references",
    # Navigation boxes at the bottom
    ".navbox", ".navbox-inner", ".navbox-subgroup",
    # Hatnotes
    ".hatnote",
    # Thumbnail captions
    ".thumbcaption", ".thumb",
    # Wikipedia maintenance tags
    ".ambox", ".tmbox", ".ombox", ".cmbox", ".fmbox",
    # Audio/media elements
    ".audio", ".listen",
    ".mw-authority-control",
    "#coordinates",
    "#p-lang-btn",
]


def clean_html(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    for selector in REMOVE_SELECTORS:
        for el in soup.select(selector):
            el.decompose()

    for tag in soup.find_all(["style", "script", "link", "meta"]):
        tag.decompose()

    content_div = soup.select_one("#mw-content-text .mw-parser-output")

    if content_div is None:
        content_div = (
            soup.find("div", id="bodyContent")
            or soup.find("div", id="content")
            or soup.find("body")
        )
        log.warning("mw-parser-output not found, falling back to broader selector.")

    if content_div is None:
        log.error("No content container found in HTML.")
        return ""

    raw_text = content_div.get_text(separator="\n")
    text = _normalize_text(raw_text)

    log.debug("Cleaned text: %d chars", len(text))
    return text


def clean_page(page: dict) -> dict:
    clean_text = clean_html(page["html"])
    return {**page, "clean_text": clean_text}


def _normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\[note\s*\d+\]", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\[[a-z]\]", "", text)
    text = re.sub(r"\d+°[NS]\s+\d+°[EW]", "", text)

    lines = []
    for line in text.splitlines():
        line = line.strip()
        if len(line) >= 3:
            lines.append(line)

    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()