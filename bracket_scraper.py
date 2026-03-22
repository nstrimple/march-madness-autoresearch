#!/usr/bin/env python3
"""
CBS Sports Bracket Pool Differential Analyzer
Scrapes your CBS bracket pool and shows which remaining games matter most
to you based on where your picks differ from others in your pool.

Usage:
    python bracket_scraper.py

Setup:
    1. Copy .env.example to .env and fill in your credentials
    2. pip install playwright python-dotenv
    3. playwright install chromium
    4. Run this script
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

CBS_EMAIL = os.getenv("CBS_EMAIL")
CBS_PASSWORD = os.getenv("CBS_PASSWORD")
CBS_POOL_URL = os.getenv("CBS_POOL_URL")  # Full URL to your pool standings page
MY_BRACKET_NAME = os.getenv("MY_BRACKET_NAME")  # Your entry name as it appears on CBS

ROUND_NAMES = {
    1: "Round of 64",
    2: "Round of 32",
    3: "Sweet 16",
    4: "Elite 8",
    5: "Final Four",
    6: "Championship",
}

ROUND_POINTS = {1: 1, 2: 2, 3: 4, 4: 8, 5: 16, 6: 32}


def validate_env():
    missing = []
    for var in ["CBS_EMAIL", "CBS_PASSWORD", "CBS_POOL_URL", "MY_BRACKET_NAME"]:
        if not os.getenv(var):
            missing.append(var)
    if missing:
        print(f"ERROR: Missing required env vars: {', '.join(missing)}")
        print("Copy .env.example to .env and fill in your values.")
        sys.exit(1)


def login(page):
    print("Logging into CBS Sports...")
    page.goto("https://www.cbssports.com/login", wait_until="networkidle")
    time.sleep(2)

    # Accept cookies if prompted
    try:
        page.click("button:has-text('Accept')", timeout=3000)
    except Exception:
        pass

    # Fill login form
    page.fill('input[name="email"], input[type="email"], #userid', CBS_EMAIL)
    page.fill('input[name="password"], input[type="password"], #password', CBS_PASSWORD)
    page.click('button[type="submit"], input[type="submit"], .login-btn, button:has-text("Sign In")')
    page.wait_for_load_state("networkidle")
    time.sleep(2)

    # Check if login succeeded
    if "login" in page.url.lower() and "error" in page.content().lower():
        print("ERROR: Login failed. Check CBS_EMAIL and CBS_PASSWORD in .env")
        sys.exit(1)
    print("  Login successful.")


def get_pool_entries(page):
    """Navigate to pool standings and collect all entry bracket URLs."""
    print(f"Loading pool standings: {CBS_POOL_URL}")
    page.goto(CBS_POOL_URL, wait_until="networkidle")
    time.sleep(2)

    # CBS standings pages typically list all entries with links to their brackets
    # Try multiple selector patterns CBS has used over the years
    entry_links = []

    selectors_to_try = [
        "a[href*='/brackets/'][href*='/picks']",
        "a[href*='/brackets/'][href*='/bracket']",
        ".standings-table a[href*='/brackets/']",
        "table a[href*='/brackets/']",
        ".pool-standings a[href*='brackets']",
        "a[href*='brackets']",
    ]

    for selector in selectors_to_try:
        links = page.query_selector_all(selector)
        if links:
            for link in links:
                href = link.get_attribute("href")
                text = link.inner_text().strip()
                if href and "/brackets/" in href:
                    full_url = href if href.startswith("http") else f"https://www.cbssports.com{href}"
                    entry_links.append({"name": text, "url": full_url})
            break

    # Deduplicate by URL
    seen = set()
    unique_entries = []
    for e in entry_links:
        if e["url"] not in seen:
            seen.add(e["url"])
            unique_entries.append(e)

    print(f"  Found {len(unique_entries)} entries in pool.")
    if len(unique_entries) == 0:
        print("  WARNING: No entries found. The page structure may have changed.")
        print("  Current URL:", page.url)
        print("  Saving screenshot to debug_pool.png for inspection.")
        page.screenshot(path="debug_pool.png")
    return unique_entries


def scrape_bracket(page, url, entry_name):
    """Scrape picks from a single bracket page. Returns dict of game_id -> winner."""
    page.goto(url, wait_until="networkidle")
    time.sleep(1.5)

    picks = {}

    # CBS bracket pages render picks as highlighted/selected team names.
    # We look for the standard CBS bracket JSON embedded in the page, or parse DOM.

    # Try 1: Look for embedded JSON data (CBS often embeds bracket state)
    content = page.content()
    for marker in ["bracketData", "bracket_data", "picks_data", "entryData"]:
        idx = content.find(f'"{marker}"')
        if idx == -1:
            idx = content.find(f"var {marker}")
        if idx != -1:
            # Try to extract JSON object after the marker
            try:
                start = content.index("{", idx)
                # Find matching closing brace (naive depth counting)
                depth = 0
                end = start
                for i in range(start, min(start + 50000, len(content))):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                data = json.loads(content[start:end])
                picks = parse_json_picks(data)
                if picks:
                    return picks
            except Exception:
                pass

    # Try 2: DOM scraping — CBS marks winning picks with classes like
    # 'winner', 'correct', 'selected', 'pick', 'chosen'
    pick_selectors = [
        ".bracket-pick.selected .team-name",
        ".matchup-pick .team-name",
        ".pick-winner .team-name",
        "[data-pick='1'] .team-name",
        ".winner .team-name",
        ".bracket-team.picked",
        ".picked-team",
        ".bracket-game .winner",
    ]

    for selector in pick_selectors:
        elements = page.query_selector_all(selector)
        if elements:
            for i, el in enumerate(elements):
                game_id = f"game_{i}"
                # Try to get a more specific game ID from parent element
                parent = el.evaluate_handle("el => el.closest('[data-game-id], [data-matchup-id], [id]')")
                try:
                    gid = parent.get_attribute("data-game-id") or parent.get_attribute("data-matchup-id")
                    if gid:
                        game_id = gid
                except Exception:
                    pass
                picks[game_id] = el.inner_text().strip()
            break

    # Try 3: Look for a JSON API endpoint CBS uses for bracket data
    # Many CBS bracket pages fetch from an API — intercept and replay
    if not picks:
        # Try navigating to the JSON API version directly
        bracket_id = url.rstrip("/").split("/")[-1]
        api_urls = [
            f"https://www.cbssports.com/collegebasketball/brackets/{bracket_id}/picks/?api=json",
            f"https://www.cbssports.com/api/brackets/{bracket_id}/picks",
        ]
        for api_url in api_urls:
            try:
                response = page.request.get(api_url)
                if response.ok:
                    data = response.json()
                    picks = parse_json_picks(data)
                    if picks:
                        break
            except Exception:
                pass

    if not picks:
        print(f"  WARNING: Could not scrape picks for '{entry_name}'. Saving screenshot.")
        page.screenshot(path=f"debug_bracket_{entry_name[:20].replace(' ', '_')}.png")

    return picks


def parse_json_picks(data):
    """Parse picks from various CBS JSON structures."""
    picks = {}
    if not isinstance(data, dict):
        return picks

    # Structure 1: {picks: [{game_id: X, winner: Y}]}
    if "picks" in data:
        for pick in data["picks"]:
            if isinstance(pick, dict):
                gid = str(pick.get("game_id") or pick.get("gameId") or pick.get("id", ""))
                winner = pick.get("winner") or pick.get("team") or pick.get("teamName") or pick.get("pick", "")
                if gid and winner:
                    picks[gid] = str(winner)

    # Structure 2: {rounds: [{games: [{winner_id: X}]}]}
    if "rounds" in data:
        for rnd_idx, rnd in enumerate(data["rounds"]):
            if isinstance(rnd, dict):
                games = rnd.get("games") or rnd.get("matchups") or []
                for game_idx, game in enumerate(games):
                    if isinstance(game, dict):
                        gid = str(game.get("id") or game.get("game_id") or f"r{rnd_idx}g{game_idx}")
                        winner = (
                            game.get("pick")
                            or game.get("winner")
                            or game.get("winner_name")
                            or game.get("pickedTeam", {}).get("name", "")
                        )
                        if gid and winner:
                            picks[gid] = str(winner)

    return picks


def analyze_differentials(my_picks, all_entries, entry_count):
    """
    Compare my picks to the pool. Return games sorted by leverage:
    - Games where I differ from the majority
    - Weighted by points available in that round
    """
    results = []

    all_game_ids = set(my_picks.keys())
    for entry in all_entries:
        all_game_ids.update(entry["picks"].keys())

    for game_id in all_game_ids:
        my_pick = my_picks.get(game_id)
        if not my_pick:
            continue

        others_picks = defaultdict(int)
        others_count = 0
        for entry in all_entries:
            pick = entry["picks"].get(game_id)
            if pick:
                others_picks[pick] += 1
                others_count += 1

        if others_count == 0:
            continue

        total_picks_for_game = sum(others_picks.values())
        my_pick_count = others_picks.get(my_pick, 0)
        others_chose_different = total_picks_for_game - my_pick_count

        if others_chose_different == 0:
            continue  # Everyone agrees — no differential

        pct_with_me = my_pick_count / total_picks_for_game if total_picks_for_game > 0 else 0
        pct_against_me = others_chose_different / total_picks_for_game

        # Determine round from game_id if encoded (e.g. "r3g2") — fallback to 1
        rnd = 1
        if game_id.startswith("r") and "g" in game_id:
            try:
                rnd = int(game_id[1:game_id.index("g")]) + 1
            except Exception:
                pass

        points_at_stake = ROUND_POINTS.get(rnd, 1)

        # Most common non-my pick
        top_other_pick = max(
            (p for p in others_picks if p != my_pick),
            key=lambda p: others_picks[p],
            default=None,
        )

        results.append(
            {
                "game_id": game_id,
                "round": rnd,
                "round_name": ROUND_NAMES.get(rnd, f"Round {rnd}"),
                "my_pick": my_pick,
                "top_other_pick": top_other_pick,
                "pct_with_me": pct_with_me,
                "pct_against_me": pct_against_me,
                "n_against_me": others_chose_different,
                "n_with_me": my_pick_count,
                "points_at_stake": points_at_stake,
                # Leverage = how many extra points I gain vs lose relative to field
                "leverage": pct_against_me * points_at_stake,
            }
        )

    results.sort(key=lambda x: -x["leverage"])
    return results


def print_report(differentials, total_entries):
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print()
    print("=" * 65)
    print(f"  BRACKET DIFFERENTIAL REPORT  —  {now}")
    print(f"  Pool size: {total_entries} entries")
    print("=" * 65)

    if not differentials:
        print("\n  No differentials found — either picks couldn't be scraped")
        print("  or everyone in your pool picked the same bracket.")
        return

    by_round = defaultdict(list)
    for d in differentials:
        by_round[d["round"]].append(d)

    for rnd in sorted(by_round.keys(), reverse=True):
        games = by_round[rnd]
        rnd_name = ROUND_NAMES.get(rnd, f"Round {rnd}")
        pts = ROUND_POINTS.get(rnd, 1)
        print(f"\n{'─'*65}")
        print(f"  {rnd_name.upper()}  ({pts} pts/game)")
        print(f"{'─'*65}")

        for d in games:
            bar_mine = "█" * round(d["pct_with_me"] * 20)
            bar_other = "░" * round(d["pct_against_me"] * 20)
            print(f"\n  Game: {d['game_id']}")
            print(f"    Your pick   : {d['my_pick']}")
            if d["top_other_pick"]:
                print(f"    Pool favors : {d['top_other_pick']}")
            print(
                f"    With you    : {d['n_with_me']} ({d['pct_with_me']*100:.0f}%)  "
                f"[{bar_mine}]"
            )
            print(
                f"    Against you : {d['n_against_me']} ({d['pct_against_me']*100:.0f}%)  "
                f"[{bar_other}]"
            )

            leverage = d["leverage"]
            if leverage >= 8:
                tag = "*** CRITICAL"
            elif leverage >= 4:
                tag = "**  HIGH"
            elif leverage >= 1:
                tag = "*   MEDIUM"
            else:
                tag = "    LOW"
            print(f"    Leverage    : {leverage:.1f}  {tag}")

    print()
    print("=" * 65)
    print("  Leverage = (% pool against you) × (points at stake)")
    print("  Higher leverage = this game swings your pool standing more")
    print("=" * 65)
    print()


def run(headless=True, output_file=None):
    validate_env()

    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=headless)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
        page = context.new_page()

        try:
            login(page)
            entries = get_pool_entries(page)

            if not entries:
                print("ERROR: No entries found. Check CBS_POOL_URL and debug_pool.png")
                browser.close()
                return

            # Find my entry
            my_entry_url = None
            for e in entries:
                if MY_BRACKET_NAME.lower() in e["name"].lower():
                    my_entry_url = e["url"]
                    print(f"  Your bracket: {e['name']} -> {e['url']}")
                    break

            if not my_entry_url:
                print(f"ERROR: Could not find your bracket '{MY_BRACKET_NAME}' in pool entries.")
                print("Entries found:")
                for e in entries:
                    print(f"  {e['name']}")
                browser.close()
                return

            # Scrape all brackets
            all_entries_data = []
            print(f"\nScraping {len(entries)} bracket(s)...")
            my_picks = {}

            for i, entry in enumerate(entries):
                print(f"  [{i+1}/{len(entries)}] {entry['name']}")
                picks = scrape_bracket(page, entry["url"], entry["name"])

                if entry["url"] == my_entry_url:
                    my_picks = picks
                else:
                    all_entries_data.append({"name": entry["name"], "picks": picks})

                time.sleep(0.5)  # polite delay

            if not my_picks:
                print("ERROR: Could not scrape your bracket picks. See debug screenshot.")
                browser.close()
                return

            print(f"\nAnalyzing differentials (your picks: {len(my_picks)} games found)...")
            differentials = analyze_differentials(my_picks, all_entries_data, len(entries))

        finally:
            browser.close()

    print_report(differentials, len(entries))

    if output_file:
        with open(output_file, "w") as f:
            import io
            # Redirect print to file too
            f.write(f"Bracket Differential Report - {datetime.now().isoformat()}\n\n")
            for d in differentials:
                f.write(
                    f"{d['round_name']}\t{d['game_id']}\tYou:{d['my_pick']}\t"
                    f"Pool:{d['top_other_pick']}\t"
                    f"Against:{d['n_against_me']}\tLeverage:{d['leverage']:.2f}\n"
                )
        print(f"Results also saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="CBS bracket pool differential analyzer")
    parser.add_argument(
        "--show-browser",
        action="store_true",
        help="Show the browser window (useful for debugging)",
    )
    parser.add_argument(
        "--output",
        metavar="FILE",
        default=None,
        help="Also save results to a TSV file (e.g. results.tsv)",
    )
    args = parser.parse_args()

    run(headless=not args.show_browser, output_file=args.output)


if __name__ == "__main__":
    main()
