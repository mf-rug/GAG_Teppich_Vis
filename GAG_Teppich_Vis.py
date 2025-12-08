"""Scrape episode topics, collect linked article paragraphs, and chart year mentions."""

import argparse
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter, Retry

WIKI_BASE = "https://de.wikipedia.org"
EPISODE_URL = f"{WIKI_BASE}/wiki/Geschichten_aus_der_Geschichte_(Podcast)/Episodenliste"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; GAG-Analyzer/1.0)"}
REQUEST_RETRIES = Retry(total=3, backoff_factor=0.5, status_forcelist=(429, 500, 502, 503, 504))


@dataclass
class Fetcher:
    """HTTP client with retry support to be polite to Wikipedia."""

    session: requests.Session

    @classmethod
    def create(cls) -> "Fetcher":
        session = requests.Session()
        adapter = HTTPAdapter(max_retries=REQUEST_RETRIES)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return cls(session=session)

    def fetch_html(self, url: str) -> str:
        response = self.session.get(url, headers=HEADERS, timeout=30)
        response.raise_for_status()
        return response.text


def concat_episode_tables(html: str) -> pd.DataFrame:
    print("Parsing episode tables from HTML ...")
    tables = pd.read_html(html)
    if not tables:
        raise ValueError("No tables found on episode page")
    print(f"Found {len(tables)} tables; concatenating into a single DataFrame")
    return pd.concat(tables, ignore_index=True)


def extract_thema_links(html: str) -> List[str]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[str] = []

    for table in soup.select("#mw-content-text .wikitable"):
        header_cells = [cell.get_text(strip=True) for cell in table.select("thead th")]
        if not header_cells:
            # Some wikitables use the first row as header
            first_row = table.find("tr")
            header_cells = [cell.get_text(strip=True) for cell in first_row.find_all(["th", "td"])] if first_row else []
        try:
            thema_index = header_cells.index("Thema")
        except ValueError:
            continue

        for row in table.select("tbody tr"):
            cells = row.find_all(["td", "th"])
            if len(cells) <= thema_index:
                continue
            thema_cell = cells[thema_index]
            anchor = thema_cell.find("a", href=True)
            if anchor and anchor["href"].startswith("/wiki") and not anchor["href"].startswith("/wiki/Wikipedia:"):
                links.append(WIKI_BASE + anchor["href"])
    print(f"Collected {len(links)} raw topic links from all tables")
    return links


def find_first_paragraph_text(fetcher: Fetcher, url: str) -> Optional[str]:
    print(f"  Fetching article: {url}")
    html = fetcher.fetch_html(url)
    soup = BeautifulSoup(html, "html.parser")
    content = soup.select_one("#mw-content-text .mw-parser-output")
    if not content:
        print("  Skipping: no content section found")
        return None

    for paragraph in content.find_all("p", recursive=False):
        text = paragraph.get_text(strip=True)
        if text:
            return text
    print("  Skipping: no non-empty lead paragraph")
    return None


def year_from_century(century: int, is_bc: bool) -> int:
    year = (century - 1) * 100 + 50
    return -year if is_bc else year


def extract_years(text: str) -> List[int]:
    years: List[int] = []

    # Explicit BCE years (e.g., "300 BC", "300 v. Chr.")
    for match in re.finditer(r"(\d{1,4})\s*(?:BC|BCE|v\.\s?Chr\.)", text, flags=re.IGNORECASE):
        years.append(-int(match.group(1)))

    # Explicit CE years (three or four digits to avoid capturing day numbers)
    for match in re.finditer(r"\b(1[0-9]{3}|2[0-9]{3}|[5-9][0-9]{2})\b", text):
        years.append(int(match.group(1)))

    # Centuries (English/German)
    for match in re.finditer(r"(\d{1,2})(?:st|nd|rd|th)?\s+century", text, flags=re.IGNORECASE):
        years.append(year_from_century(int(match.group(1)), is_bc=False))
    for match in re.finditer(r"(\d{1,2})\.\s?Jahrhundert", text, flags=re.IGNORECASE):
        years.append(year_from_century(int(match.group(1)), is_bc=False))
    for match in re.finditer(r"(\d{1,2})\.\s?Jahrhundert\s*(v\.\s?Chr\.)", text, flags=re.IGNORECASE):
        years.append(year_from_century(int(match.group(1)), is_bc=True))

    return years


def plot_year_histogram(year_counts: Dict[int, int], output: Optional[Path]) -> None:
    if not year_counts:
        print("No years found to plot.")
        return
    years = sorted(year_counts.keys())
    counts = [year_counts[y] for y in years]

    plt.figure(figsize=(12, 6))
    plt.bar(years, counts, width=8, color="skyblue")
    plt.xlabel("Year")
    plt.ylabel("Mentions")
    plt.title("Mentions of Years in Episode Topics")
    plt.tight_layout()
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output)
        print(f"Saved plot to {output}")
    else:
        plt.show()


def unique(iterable: Iterable[str]) -> List[str]:
    seen = set()
    items: List[str] = []
    for value in iterable:
        if value not in seen:
            seen.add(value)
            items.append(value)
    print(f"Reduced to {len(items)} unique links after de-duplication")
    return items


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the histogram image instead of showing it",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    fetcher = Fetcher.create()

    print(f"Fetching episode list page: {EPISODE_URL}")
    html = fetcher.fetch_html(EPISODE_URL)
    episodes_df = concat_episode_tables(html)
    print(f"Loaded {len(episodes_df)} episode rows")

    links = unique(extract_thema_links(html))
    print(f"Found {len(links)} unique linked topics")

    all_years: Counter[int] = Counter()
    for idx, link in enumerate(links, start=1):
        print(f"Processing link {idx}/{len(links)}")
        paragraph = find_first_paragraph_text(fetcher, link)
        if not paragraph:
            print("  No paragraph found; moving on")
            continue
        years = extract_years(paragraph)
        if years:
            print(f"  Extracted years: {years}")
        else:
            print("  No years detected in paragraph")
        all_years.update(years)

    print(f"Extracted {sum(all_years.values())} year mentions across {len(all_years)} distinct years")
    plot_year_histogram(all_years, args.output)


if __name__ == "__main__":
    main()
