#!/usr/bin/env python3
"""
Crawl a voicelines wiki page, download all WAV files from table rows, and create a CSV
mapping each WAV file path to its dialogue text.
"""

import argparse
import csv
import os
import re
import sys
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

# Optional: Cloudflare bypass (pip install cloudscraper)
try:
    import cloudscraper
except ImportError:
    cloudscraper = None

# Optional: real browser (pip install playwright && playwright install chromium)
try:
    from playwright.sync_api import sync_playwright
except ImportError:
    sync_playwright = None


def _requests_headers() -> dict:
    return {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://voicelines.fandom.com/",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
    }


def fetch_html_requests(url: str, timeout: int = 30) -> str:
    """Fetch HTML with plain requests."""
    resp = requests.get(url, headers=_requests_headers(), timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_html_cloudscraper(url: str, timeout: int = 30) -> str:
    """Fetch HTML using cloudscraper (bypasses Cloudflare JS challenge)."""
    if cloudscraper is None:
        raise RuntimeError("cloudscraper not installed; run: pip install cloudscraper")
    scraper = cloudscraper.create_scraper()
    resp = scraper.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp.text


def fetch_html_browser(url: str, timeout: int = 30000) -> str:
    """Fetch HTML using a real headless browser (bypasses most bot protection)."""
    if sync_playwright is None:
        raise RuntimeError(
            "playwright not installed; run: pip install playwright && playwright install chromium"
        )
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        try:
            page = browser.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=timeout)
            page.wait_for_load_state("networkidle", timeout=timeout)
            return page.content()
        finally:
            browser.close()


def fetch_html(
    url: str,
    timeout: int = 30,
    use_browser: bool = False,
    try_cloudscraper_on_403: bool = True,
) -> str:
    """
    Fetch HTML from URL. On 403, optionally try cloudscraper, or use --use-browser
    for a real browser (most reliable).
    """
    if use_browser:
        return fetch_html_browser(url, timeout=timeout * 1000)

    try:
        return fetch_html_requests(url, timeout=timeout)
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403 and try_cloudscraper_on_403 and cloudscraper:
            try:
                return fetch_html_cloudscraper(url, timeout=timeout)
            except Exception:
                pass
        raise


def parse_rows(html: str, base_url: str) -> list[tuple[str, str]]:
    """
    Parse table rows: extract (wav_url, dialogue_text) from each <tr>.
    Returns list of (wav_url, dialogue_text).
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")
    results = []

    for tr in rows:
        tds = tr.find_all("td")
        if len(tds) < 2:
            continue

        # First td: audio src (direct wav URL)
        audio = tds[0].find("audio", src=True)
        if not audio or not audio.get("src"):
            continue
        wav_url = audio["src"].strip()
        wav_url = urljoin(base_url, wav_url)

        # Second td: dialogue text (strip tags, normalize whitespace)
        dialogue = tds[1].get_text(separator=" ", strip=True)
        dialogue = re.sub(r"\s+", " ", dialogue).strip()
        if not dialogue:
            continue

        results.append((wav_url, dialogue))

    return results


def filename_from_url(url: str) -> str:
    """Get a safe local filename from WAV URL (e.g. last path segment before query)."""
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    # Handle .../Intro02_Hello_ItsMeBEEMO.wav/revision/latest
    if "/revision/" in path:
        path = path.split("/revision/")[0]
    name = os.path.basename(path) or "audio.wav"
    if not name.lower().endswith(".wav"):
        name += ".wav"
    return name


def download_wav(url: str, out_dir: Path, timeout: int = 60) -> Path:
    """Download a WAV file and return the local path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    name = filename_from_url(url)
    path = out_dir / name

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Referer": "https://voicelines.fandom.com/",
    }
    resp = requests.get(url, headers=headers, timeout=timeout, stream=True)
    resp.raise_for_status()

    with open(path, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Crawl voicelines page, download WAVs, and create dialogue CSV."
    )
    parser.add_argument(
        "url",
        help="URL of the wiki page containing the table with audio and dialogue",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        default="wav_output",
        help="Directory for downloaded WAV files and CSV (default: wav_output)",
    )
    parser.add_argument(
        "--csv-name",
        default="dialogue.csv",
        help="CSV filename inside output dir (default: dialogue.csv)",
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="Only build CSV from existing WAV paths; do not download",
    )
    parser.add_argument(
        "--use-browser",
        action="store_true",
        help="Use headless Chromium to load the page (bypasses 403; requires playwright)",
    )
    args = parser.parse_args()

    base_url = args.url
    out_dir = Path(args.output_dir)
    csv_path = out_dir / args.csv_name

    print(f"Fetching: {base_url}")
    try:
        html = fetch_html(
            base_url,
            use_browser=args.use_browser,
        )
    except requests.HTTPError as e:
        if e.response is not None and e.response.status_code == 403:
            print("403 Forbidden. Try: pip install playwright && playwright install chromium", file=sys.stderr)
            print("Then run with: --use-browser", file=sys.stderr)
        print(f"Failed to fetch URL: {e}", file=sys.stderr)
        sys.exit(1)
    except requests.RequestException as e:
        print(f"Failed to fetch URL: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Failed: {e}", file=sys.stderr)
        sys.exit(1)

    pairs = parse_rows(html, base_url)
    if not pairs:
        print("No table rows with audio + dialogue found.", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(pairs)} rows with WAV URL and dialogue.")

    rows_for_csv = []
    for wav_url, dialogue in pairs:
        fname = filename_from_url(wav_url)
        local_path = out_dir / fname

        if not args.no_download:
            try:
                local_path = download_wav(wav_url, out_dir)
                print(f"  Downloaded: {local_path.name}")
            except requests.RequestException as e:
                print(f"  Skip {fname}: {e}", file=sys.stderr)
                continue

        # Store path as relative to current dir or absolute
        path_for_csv = str(local_path)
        rows_for_csv.append((path_for_csv, dialogue))

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["wav_path", "dialogue"])
        writer.writerows(rows_for_csv)

    print(f"CSV written: {csv_path} ({len(rows_for_csv)} entries)")


if __name__ == "__main__":
    main()
