from playwright.sync_api import sync_playwright
import json
import time

BASE = "https://vedabase.io/en/library/bg/"

output_file = "/Users/janusshan/Documents/Krsna LLM/bhagavad_gita.json"
all_data = []

def get_chapter_links(page):
    page.goto(BASE)
    page.wait_for_load_state("networkidle")
    links = page.query_selector_all("a")
    chapter_urls = []
    for a in links:
        href = a.get_attribute("href")
        if href and href.startswith("/en/library/bg/"):
            parts = href.strip("/").split("/")
            if len(parts) == 4 and parts[-1].isdigit():  # chapter links
                chapter_urls.append("https://vedabase.io" + href)
    return sorted(list(set(chapter_urls)), key=lambda x: int(x.rstrip("/").split("/")[-1]))

def get_verse_links(page, chapter_url, chapter_num):
    page.goto(chapter_url)
    page.wait_for_load_state("networkidle")
    links = page.query_selector_all("a")
    verse_urls = []
    for a in links:
        href = a.get_attribute("href")
        if href and href.startswith("/en/library/bg/"):
            parts = href.strip("/").split("/")
            if len(parts) >= 5 and parts[-1].isdigit() and parts[-2].isdigit():
                if int(parts[-2]) == chapter_num:
                    verse_urls.append("https://vedabase.io" + href)
    # sort by verse number
    verse_urls = sorted(list(set(verse_urls)), key=lambda u: int(u.rstrip("/").split("/")[-1]))
    return verse_urls

def scrape_verse(page, url):
    page.goto(url)
    page.wait_for_load_state("networkidle")

    # Extract fields
    try:
        header = page.query_selector("h1").inner_text().strip()
    except:
        header = ""

    try:
        sanskrit = page.query_selector("div.av-verse_text").inner_text().strip()
    except:
        sanskrit = ""

    try:
        translation = page.query_selector("div.av-translation").inner_text().strip()
    except:
        translation = ""

    try:
        purport = page.query_selector("div.av-purport").inner_text().strip()
    except:
        purport = ""

    return {
        "chapter_verse": header,
        "sanskrit": sanskrit,
        "translation": translation,
        "purport": purport
    }

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    chapters = get_chapter_links(page)
    print(f"Found {len(chapters)} chapters")

    for ch_url in chapters:
        chapter_num = int(ch_url.rstrip("/").split("/")[-1])
        print(f"Scraping Chapter {chapter_num} ...")
        verse_urls = get_verse_links(page, ch_url, chapter_num)
        print(f"  Found {len(verse_urls)} verses")

        for v_url in verse_urls:
            print(f"    → Scraping {v_url}")
            data = scrape_verse(page, v_url)
            all_data.append(data)
            time.sleep(0.3)  # polite delay

    browser.close()

# Save JSON
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(all_data, f, ensure_ascii=False, indent=2)

print(f"✅ Done! Saved {len(all_data)} verses to {output_file}")