#!/usr/bin/env python3
"""
March Madness Bracket Differential — Web UI
A simple hosted web app so non-technical family members can see
which games matter most to them in their CBS Sports bracket pool.
"""

import os
import json
import traceback
from collections import defaultdict
from flask import Flask, render_template, request, jsonify, session
from bracket_scraper import (
    login,
    get_pool_entries,
    scrape_bracket,
    analyze_differentials,
    ROUND_NAMES,
    ROUND_POINTS,
)

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", os.urandom(24))


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    email = request.form.get("email", "").strip()
    password = request.form.get("password", "").strip()
    pool_url = request.form.get("pool_url", "").strip()
    bracket_name = request.form.get("bracket_name", "").strip()

    if not all([email, password, pool_url, bracket_name]):
        return render_template("index.html", error="Please fill in all four fields.")

    if "cbssports.com" not in pool_url:
        return render_template(
            "index.html",
            error="Pool URL must be a CBS Sports link (cbssports.com).",
            prefill=request.form,
        )

    try:
        results = run_scraper(email, password, pool_url, bracket_name)
        return render_template("results.html", **results)
    except ScraperError as e:
        return render_template("index.html", error=str(e), prefill=request.form)
    except Exception:
        tb = traceback.format_exc()
        app.logger.error(tb)
        return render_template(
            "index.html",
            error="Something went wrong scraping CBS. Try again or check your pool URL.",
            prefill=request.form,
        )


class ScraperError(Exception):
    pass


def run_scraper(email, password, pool_url, bracket_name):
    # Temporarily override env vars so bracket_scraper functions work
    os.environ["CBS_EMAIL"] = email
    os.environ["CBS_PASSWORD"] = password
    os.environ["CBS_POOL_URL"] = pool_url
    os.environ["MY_BRACKET_NAME"] = bracket_name

    from playwright.sync_api import sync_playwright
    import time

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        )
        page = context.new_page()

        try:
            login(page)
            entries = get_pool_entries(page)

            if not entries:
                raise ScraperError(
                    "No bracket entries found in your pool. "
                    "Double-check the pool URL — it should be the Standings page."
                )

            my_entry_url = None
            my_entry_name = None
            for e in entries:
                if bracket_name.lower() in e["name"].lower():
                    my_entry_url = e["url"]
                    my_entry_name = e["name"]
                    break

            if not my_entry_url:
                names = [e["name"] for e in entries[:10]]
                raise ScraperError(
                    f"Couldn't find your bracket "{bracket_name}" in the pool. "
                    f"Entries found: {', '.join(names)}{'...' if len(entries) > 10 else ''}. "
                    "Check the spelling matches your CBS entry name exactly."
                )

            all_entries_data = []
            my_picks = {}
            for i, entry in enumerate(entries):
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

    differentials = analyze_differentials(my_picks, all_entries_data, len(entries))

    # Group by round for display
    by_round = defaultdict(list)
    for d in differentials:
        by_round[d["round"]].append(d)

    rounds_data = []
    for rnd in sorted(by_round.keys(), reverse=True):
        rounds_data.append(
            {
                "name": ROUND_NAMES.get(rnd, f"Round {rnd}"),
                "points": ROUND_POINTS.get(rnd, 1),
                "games": by_round[rnd],
            }
        )

    return {
        "my_bracket_name": my_entry_name,
        "pool_size": len(entries),
        "total_differentials": len(differentials),
        "rounds": rounds_data,
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
