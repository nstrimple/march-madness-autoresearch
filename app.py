#!/usr/bin/env python3
"""
March Madness Bracket Differential — Web UI
Supports both CBS Sports and ESPN Tournament Challenge pools.
"""

import os
import time
import traceback
from collections import defaultdict
from flask import Flask, render_template, request

# Shared analysis logic lives in bracket_scraper
from bracket_scraper import analyze_differentials, ROUND_NAMES, ROUND_POINTS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))

PLAYWRIGHT_ARGS = ["--no-sandbox", "--disable-dev-shm-usage"]
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36"
)


class ScraperError(Exception):
    pass


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    platform = request.form.get("platform", "cbs").lower()
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    pool_url = request.form.get("pool_url", "").strip()
    bracket_name = request.form.get("bracket_name", "").strip()

    if not all([email, password, pool_url, bracket_name]):
        return render_template("index.html", error="Please fill in all four fields.", prefill=request.form)

    if platform == "espn" and "espn.com" not in pool_url:
        return render_template(
            "index.html",
            error="Pool URL must be an ESPN link (fantasy.espn.com/tournament-challenge-bracket/...).",
            prefill=request.form,
        )
    if platform == "cbs" and "cbssports.com" not in pool_url:
        return render_template(
            "index.html",
            error="Pool URL must be a CBS Sports link (cbssports.com/collegebasketball/brackets/...).",
            prefill=request.form,
        )

    try:
        if platform == "espn":
            results = _run_espn(email, password, pool_url, bracket_name)
        else:
            results = _run_cbs(email, password, pool_url, bracket_name)
        return render_template("results.html", **results)
    except ScraperError as e:
        return render_template("index.html", error=str(e), prefill=request.form)
    except Exception as e:
        tb = traceback.format_exc()
        app.logger.error(tb)
        return render_template(
            "index.html",
            error=f"Error: {e}\n\nTraceback:\n{tb}",
            prefill=request.form,
        )


# ---------------------------------------------------------------------------
# CBS scraper runner
# ---------------------------------------------------------------------------

def _run_cbs(email, password, pool_url, bracket_name):
    # Inject credentials so bracket_scraper's module-level functions work
    os.environ["CBS_EMAIL"] = email
    os.environ["CBS_PASSWORD"] = password
    os.environ["CBS_POOL_URL"] = pool_url
    os.environ["MY_BRACKET_NAME"] = bracket_name

    from bracket_scraper import login, get_pool_entries, scrape_bracket
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=PLAYWRIGHT_ARGS)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        try:
            login(page)
            entries = get_pool_entries(page)

            if not entries:
                raise ScraperError(
                    "No bracket entries found in your CBS pool. "
                    "Double-check the pool URL — it should be the Standings page."
                )

            my_entry_url, my_entry_name = _find_my_cbs_entry(entries, bracket_name)

            my_picks, all_entries_data = {}, []
            for entry in entries:
                picks = scrape_bracket(page, entry["url"], entry["name"])
                if entry["url"] == my_entry_url:
                    my_picks = picks
                else:
                    all_entries_data.append({"name": entry["name"], "picks": picks})
                time.sleep(0.4)

            if not my_picks:
                raise ScraperError(
                    "Found your bracket but couldn't read your picks. "
                    "CBS may have updated their page layout."
                )
        finally:
            browser.close()

    return _build_results(my_entry_name, len(entries), my_picks, all_entries_data)


def _find_my_cbs_entry(entries, bracket_name):
    for e in entries:
        if bracket_name.lower() in e["name"].lower():
            return e["url"], e["name"]
    names = ", ".join(e["name"] for e in entries[:10])
    raise ScraperError(
        f"Couldn't find your bracket \"{bracket_name}\" in the CBS pool. "
        f"Entries found: {names}{'...' if len(entries) > 10 else ''}. "
        "Check the spelling matches your CBS entry name exactly."
    )


# ---------------------------------------------------------------------------
# ESPN scraper runner
# ---------------------------------------------------------------------------

def _run_espn(email, password, pool_url, bracket_name):
    from espn_scraper import espn_login, get_espn_pool_entries, scrape_espn_bracket
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=PLAYWRIGHT_ARGS)
        context = browser.new_context(user_agent=USER_AGENT)
        page = context.new_page()
        try:
            espn_login(page, email, password)
            entries = get_espn_pool_entries(page, pool_url)

            if not entries:
                raise ScraperError(
                    "No bracket entries found in your ESPN group. "
                    "Double-check the group URL — it should contain ?groupID=..."
                )

            my_entry = _find_my_espn_entry(entries, bracket_name)
            my_entry_id = my_entry["entry_id"]
            my_entry_name = my_entry["name"]

            my_picks, all_entries_data = {}, []
            for entry in entries:
                picks = scrape_espn_bracket(page, entry["entry_id"], entry["name"])
                if entry["entry_id"] == my_entry_id:
                    my_picks = picks
                else:
                    all_entries_data.append({"name": entry["name"], "picks": picks})
                time.sleep(0.4)

            if not my_picks:
                raise ScraperError(
                    "Found your ESPN bracket but couldn't read your picks. "
                    "ESPN may have updated their API."
                )
        finally:
            browser.close()

    return _build_results(my_entry_name, len(entries), my_picks, all_entries_data)


def _find_my_espn_entry(entries, bracket_name):
    for e in entries:
        if bracket_name.lower() in e["name"].lower():
            return e
    names = ", ".join(e["name"] for e in entries[:10])
    raise ScraperError(
        f"Couldn't find your bracket \"{bracket_name}\" in the ESPN group. "
        f"Entries found: {names}{'...' if len(entries) > 10 else ''}. "
        "Check the spelling matches your ESPN entry name exactly."
    )


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_results(my_entry_name, pool_size, my_picks, all_entries_data):
    differentials = analyze_differentials(my_picks, all_entries_data, pool_size)

    by_round = defaultdict(list)
    for d in differentials:
        by_round[d["round"]].append(d)

    rounds_data = [
        {
            "name": ROUND_NAMES.get(rnd, f"Round {rnd}"),
            "points": ROUND_POINTS.get(rnd, 1),
            "games": by_round[rnd],
        }
        for rnd in sorted(by_round.keys(), reverse=True)
    ]

    return {
        "my_bracket_name": my_entry_name,
        "pool_size": pool_size,
        "total_differentials": len(differentials),
        "rounds": rounds_data,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
