"""Extract the training text for the GPT from Project Gutenberg's HTML edition of
Freud's "The Interpretation of Dreams" (translated by A. A. Brill) into input.txt.

Keeps the book's paragraphs, one per line. Drops Gutenberg's header and licence,
the footnotes (both the notes and their [1] markers in the text) and everything
from the bibliography or index on.

Usage: download the book's HTML from gutenberg.org into this folder as
interpretation_of_dreams.html, then run: python extract_text.py
"""
from collections import Counter
from pathlib import Path

from bs4 import BeautifulSoup

FOLDER = Path(__file__).parent
BOOK_HTML = FOLDER / "interpretation_of_dreams.html"
OUTPUT = FOLDER / "input.txt"

# The text stops at the first heading that is exactly one of these. An exact match, not
# "contains": chapter I is titled "The Scientific Literature ...", which must not stop it.
STOP_HEADINGS = {"BIBLIOGRAPHY", "LITERATURE", "INDEX"}


def remove_gutenberg_boilerplate(soup):
    """Gutenberg's HTML files wrap their own header and licence in <section class="pg-boilerplate">."""
    sections = soup.select("section.pg-boilerplate")
    if not sections:
        raise ValueError("No <section class=\"pg-boilerplate\"> found: look at how this file marks "
                         "Gutenberg's header and licence, and adapt remove_gutenberg_boilerplate.")
    for section in sections:
        section.decompose()
    return len(sections)


def remove_footnotes(soup):
    """Removes the footnotes and the markers that point to them. Returns (notes, markers) removed."""
    notes = soup.select("div.footnote, div.footnotes")
    markers = soup.select("a.fnanchor") + soup.select('a[href^="#Footnote"]')
    for element in notes + markers:
        element.decompose()  # decomposing an already removed element is harmless
    return len(notes), len(set(markers))


def paragraph_text(p):
    # get_text() without a separator: a separator would put a space before the comma in
    # "<i>Traumdeutung</i>, the". split() then collapses the HTML source's line breaks and runs of spaces.
    return " ".join(p.get_text().split())


def book_paragraphs(soup):
    """The paragraphs in reading order, up to the first stop heading. Returns (paragraphs, stop heading or None)."""
    for br in soup.find_all("br"):
        br.replace_with(" ")
    paragraphs = []
    for element in soup.body.find_all(["h1", "h2", "h3", "h4", "p"]):  # find_all keeps document order
        text = paragraph_text(element)
        if element.name == "p":
            if text:
                paragraphs.append(text)
        elif text.strip(" .").upper() in STOP_HEADINGS:
            return paragraphs, text
    return paragraphs, None


def report(text, boilerplate, notes, markers, stop_heading):
    """What to check before training on the text."""
    print(f"Removed: {boilerplate} Gutenberg sections, {notes} footnotes, {markers} footnote markers")
    print(f"Stopped at heading: {stop_heading!r}" if stop_heading else
          "No bibliography or index heading found: check the end of input.txt")
    print(f"{text.count(chr(10)):,} paragraphs, {len(text):,} characters")  # each paragraph ends with a newline
    print("\nStarts with:\n" + text[:300])
    print("\nEnds with:\n" + text[-300:])
    counts = Counter(text)
    print(f"\n{len(counts)} distinct characters: {''.join(sorted(counts))!r}")
    rare = sorted((c for c in counts if counts[c] < 10), key=lambda c: counts[c])
    print("Rare (fewer than 10 times): " + ", ".join(f"{c!r} {counts[c]}" for c in rare))


def main():
    soup = BeautifulSoup(BOOK_HTML.read_text(encoding="utf-8"), "html.parser")
    boilerplate = remove_gutenberg_boilerplate(soup)
    notes, markers = remove_footnotes(soup)
    paragraphs, stop_heading = book_paragraphs(soup)
    text = "\n".join(paragraphs) + "\n"
    OUTPUT.write_text(text, encoding="utf-8")
    report(text, boilerplate, notes, markers, stop_heading)


if __name__ == "__main__":
    main()
