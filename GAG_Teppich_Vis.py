"""Scrape episode topics, collect linked article paragraphs, and chart year mentions."""

import argparse
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

import pandas as pd
import requests
from bokeh.io import output_file, show
from bokeh.layouts import row
from bokeh.models import ColumnDataSource, CustomJS, Div
from bokeh.plotting import figure
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


@dataclass
class ThemaLink:
    url: str
    term: str
    episode_index: int


@dataclass
class CombinedLink:
    url: str
    term: str
    episodes: List[int]


def extract_thema_links(html: str, selected_episodes: Optional[Set[int]] = None) -> List[ThemaLink]:
    soup = BeautifulSoup(html, "html.parser")
    links: List[ThemaLink] = []
    row_counter = 0

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
            if len(cells) <= thema_index or not row.find_all("td"):
                continue
            row_counter += 1
            if selected_episodes is not None and row_counter not in selected_episodes:
                continue
            thema_cell = cells[thema_index]
            anchor = thema_cell.find("a", href=True)
            if anchor and anchor["href"].startswith("/wiki") and not anchor["href"].startswith("/wiki/Wikipedia:"):
                links.append(ThemaLink(url=WIKI_BASE + anchor["href"], term=anchor.get_text(strip=True), episode_index=row_counter))
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


def build_interactive_plot(year_details: Dict[int, List[Dict[str, str]]], output: Optional[Path]) -> None:
    if not year_details:
        print("No years found to plot.")
        return

    agg_years = []
    agg_counts = []
    agg_html = []

    for year in sorted(year_details.keys()):
        items = year_details[year]
        agg_years.append(year)
        agg_counts.append(len(items))
        html_rows = []
        for entry in items:
            link_html = f'<a href="{entry["url"]}" target="_blank">{entry["term"]}</a>'
            html_rows.append(f"Episode {entry['episode']}: {link_html}")
        agg_html.append("<br>".join(html_rows))

    source = ColumnDataSource(data=dict(x=agg_years, y=agg_counts, info=agg_html))

    p = figure(
        title="Mentions of Years in Episode Topics",
        x_axis_label="Year",
        y_axis_label="Mentions",
        tools="tap",
        width=900,
        height=500,
    )
    p.vbar(x="x", top="y", width=40, source=source, line_color="white", fill_color="#80b1d3")

    info_panel = Div(text="<b>Click a bar to see topics and episodes</b>", width=350)
    source.selected.js_on_change(
        "indices",
        CustomJS(
            args=dict(source=source, panel=info_panel),
            code="""
        if (cb_obj.indices.length > 0) {
            var idx = cb_obj.indices[0];
            var year = source.data['x'][idx];
            var info = source.data['info'][idx];
            panel.text = `<h3>${year}</h3>${info}`;
        }
        """,
        ),
    )

    layout = row(p, info_panel)
    output_path = output if output else Path("interactive_year_mentions.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_file(output_path)
    show(layout)
    print(f"Interactive plot saved to {output_path}")


def dedupe_links(links: Iterable[ThemaLink]) -> List[CombinedLink]:
    grouped: Dict[str, CombinedLink] = {}
    for link in links:
        if link.url not in grouped:
            grouped[link.url] = CombinedLink(url=link.url, term=link.term, episodes=[link.episode_index])
        else:
            grouped[link.url].episodes.append(link.episode_index)
    print(f"Reduced to {len(grouped)} unique links after de-duplication (while keeping episode references)")
    return list(grouped.values())


def parse_episode_spec(value: str) -> Set[int]:
    if value.isdigit():
        episode = int(value)
        if episode < 1:
            raise argparse.ArgumentTypeError("Episode numbers must be positive")
        return {episode}

    if "-" in value:
        start_str, end_str = value.split("-", maxsplit=1)
        if not start_str.isdigit() or not end_str.isdigit():
            raise argparse.ArgumentTypeError("Ranges must be numeric, e.g., 10-20")
        start, end = int(start_str), int(end_str)
        if start < 1 or end < 1 or start > end:
            raise argparse.ArgumentTypeError("Episode range must be positive and start <= end")
        return set(range(start, end + 1))

    raise argparse.ArgumentTypeError("Provide a single number or a range like 10-20")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the interactive plot HTML (default: interactive_year_mentions.html)",
    )
    parser.add_argument(
        "--episodes",
        type=parse_episode_spec,
        help="Limit processing to a single episode number or an inclusive range like 10-20",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    fetcher = Fetcher.create()

    print(f"Fetching episode list page: {EPISODE_URL}")
    html = fetcher.fetch_html(EPISODE_URL)
    episodes_df = concat_episode_tables(html)
    print(f"Loaded {len(episodes_df)} episode rows")
    if args.episodes:
        valid_episodes = {ep for ep in args.episodes if 1 <= ep <= len(episodes_df)}
        skipped = args.episodes - valid_episodes
        if skipped:
            print(f"Skipping out-of-range episodes: {sorted(skipped)}")
        selected_episodes = valid_episodes
        print(f"Limiting processing to {len(selected_episodes)} episode(s): {sorted(selected_episodes)}")
    else:
        selected_episodes = None

    raw_links = extract_thema_links(html, selected_episodes=selected_episodes)
    links = dedupe_links(raw_links)
    print(f"Found {len(links)} unique linked topics in selection")

    all_years: Counter[int] = Counter()
    year_details: Dict[int, List[Dict[str, str]]] = defaultdict(list)
    for idx, link in enumerate(links, start=1):
        print(f"Processing link {idx}/{len(links)} ({link.term}) from episodes {sorted(link.episodes)}")
        paragraph = find_first_paragraph_text(fetcher, link.url)
        if not paragraph:
            print("  No paragraph found; moving on")
            continue
        years = extract_years(paragraph)
        if years:
            print(f"  Extracted years: {years}")
        else:
            print("  No years detected in paragraph")
        for year in years:
            for episode in link.episodes:
                all_years[year] += 1
                year_details[year].append({"term": link.term, "url": link.url, "episode": episode})

    print(f"Extracted {sum(all_years.values())} year mentions across {len(all_years)} distinct years")
    build_interactive_plot(year_details, args.output)


if __name__ == "__main__":
    main()
