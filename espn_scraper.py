#!/usr/bin/env python3
"""
ESPN Tournament Challenge bracket pool scraper.

Provides the same interface as bracket_scraper.py (CBS) so app.py
can route to either platform transparently.

ESPN's Tournament Challenge API is semi-public — group standings and
individual entry picks are accessible via JSON endpoints using the
session cookies from a normal browser login.
"""

import re
import time
from collections import defaultdict

CURRENT_YEAR = 2026

ESPN_API_BASE = (
    f"https://fantasy.espn.com/apis/v3/games/tcmen/seasons/{CURRENT_YEAR}"
)
ESPN_LOGIN_URL = "https://registerdisney.go.com/jgc/v8/client/ESPN-ONESITE.WEB-PROD/guest/login"


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def espn_login(page, email, password):
    """
    Log into ESPN via their standard web login flow.

    ESPN's login page uses a cross-domain iframe hosted on Disney/BA servers.
    We navigate to the ESPN login page, detect the iframe, and fill it there.
    Falls back to direct form fill if the iframe approach fails.
    """
    page.goto("https://www.espn.com/login", wait_until="networkidle")
    time.sleep(2)

    # Dismiss any cookie/consent banners
    for btn_text in ["Accept All", "Accept", "I Accept"]:
        try:
            page.click(f"button:has-text('{btn_text}')", timeout=2000)
            time.sleep(0.5)
        except Exception:
            pass

    logged_in = False

    # --- Approach 1: iframe-based login (ESPN's standard flow) ---
    try:
        frames = page.frames
        login_frame = None
        for frame in frames:
            url = frame.url or ""
            if "registerdisney" in url or "disneyid" in url or "loginframe" in url.lower():
                login_frame = frame
                break

        # Sometimes the iframe loads lazily — wait for it
        if not login_frame:
            page.wait_for_selector("iframe", timeout=5000)
            for frame in page.frames:
                url = frame.url or ""
                if "registerdisney" in url or "disneyid" in url:
                    login_frame = frame
                    break

        if login_frame:
            login_frame.fill('input[type="email"], input[name="email"], #EmailAddress', email, timeout=5000)
            login_frame.fill('input[type="password"], input[name="password"], #Password', password, timeout=5000)
            login_frame.click('button[type="submit"], .btn-submit, button:has-text("Log In")', timeout=5000)
            page.wait_for_load_state("networkidle")
            time.sleep(3)
            logged_in = True
    except Exception:
        pass

    # --- Approach 2: Direct form fill on main page ---
    if not logged_in:
        try:
            page.fill('input[type="email"], input[name="loginValue"], #EmailAddress', email, timeout=5000)
            page.fill('input[type="password"], input[name="password"], #Password', password, timeout=5000)
            page.click('button[type="submit"], .btn-submit, button:has-text("Log In")', timeout=5000)
            page.wait_for_load_state("networkidle")
            time.sleep(3)
            logged_in = True
        except Exception:
            pass

    # --- Verify login succeeded ---
    # After a successful login ESPN redirects away from /login
    if "login" in page.url.lower():
        # Still on login page — try checking for an error message
        content = page.content()
        if "incorrect" in content.lower() or "invalid" in content.lower():
            raise ValueError(
                "ESPN login failed: incorrect email or password. "
                "Check your credentials and try again."
            )
        # May still be loading — check cookies as fallback
        cookies = page.context.cookies()
        has_session = any(c["name"] in ("espn_s2", "SWID", "s_vi") for c in cookies)
        if not has_session:
            raise ValueError(
                "ESPN login did not complete. ESPN may be showing a CAPTCHA or "
                "bot-check — try again in a few minutes."
            )


# ---------------------------------------------------------------------------
# Group / pool
# ---------------------------------------------------------------------------

def get_espn_group_id(pool_url):
    """Extract the numeric groupID from an ESPN group URL."""
    m = re.search(r"groupID=(\d+)", pool_url, re.IGNORECASE)
    if m:
        return m.group(1)
    # Some URLs embed the ID in the path: /group/12345
    m = re.search(r"/group(?:s)?/(\d+)", pool_url)
    if m:
        return m.group(1)
    raise ValueError(
        "Could not find a groupID in that URL. "
        "It should look like: fantasy.espn.com/tournament-challenge-bracket/2026/en/group?groupID=XXXXXX"
    )


def get_espn_pool_entries(page, pool_url):
    """
    Fetch the list of entries (brackets) in an ESPN group using the API.
    Returns list of dicts with keys: name, entry_id, url
    """
    group_id = get_espn_group_id(pool_url)

    api_url = (
        f"{ESPN_API_BASE}/groups/{group_id}"
        "?view=groupstandingsentry&view=groupmessaging&view=groupsettings"
    )

    response = page.request.get(api_url, headers={"Accept": "application/json"})

    if not response.ok:
        # Try the alternate groups endpoint
        api_url2 = f"{ESPN_API_BASE}/groups/{group_id}?view=groupstandingsentry"
        response = page.request.get(api_url2, headers={"Accept": "application/json"})

    if not response.ok:
        raise ValueError(
            f"ESPN API returned {response.status} for group {group_id}. "
            "The group may be private — ensure you're logged in with an account "
            "that is a member of this group."
        )

    data = response.json()
    return _parse_espn_group_entries(data, group_id)


def _parse_espn_group_entries(data, group_id):
    entries = []

    # Structure A: data["items"] is a list of entry objects
    items = data.get("items", [])

    # Structure B: data["entries"] directly
    if not items:
        items = data.get("entries", [])

    # Structure C: data["groups"][0]["entries"]
    if not items:
        for grp in data.get("groups", []):
            items.extend(grp.get("entries", []))

    for item in items:
        entry_id = (
            item.get("entryId")
            or item.get("id")
            or (item.get("entry") or {}).get("id")
        )
        name = (
            item.get("entryName")
            or item.get("name")
            or (item.get("entry") or {}).get("entryName")
            or f"Entry {entry_id}"
        )
        if entry_id:
            entries.append(
                {
                    "name": str(name),
                    "entry_id": str(entry_id),
                    "url": (
                        f"https://fantasy.espn.com/tournament-challenge-bracket"
                        f"/{CURRENT_YEAR}/en/entry?entryID={entry_id}"
                    ),
                }
            )

    return entries


# ---------------------------------------------------------------------------
# Bracket / picks
# ---------------------------------------------------------------------------

def scrape_espn_bracket(page, entry_id, entry_name=""):
    """
    Fetch picks for a single ESPN Tournament Challenge entry.
    Returns dict of game_id -> team name string (same format as CBS scraper).
    """
    api_url = f"{ESPN_API_BASE}/entries/{entry_id}?view=picks"
    response = page.request.get(api_url, headers={"Accept": "application/json"})

    if not response.ok:
        # Fallback: try fetching the bracket page and parse embedded JSON
        return _scrape_espn_bracket_dom(page, entry_id, entry_name)

    try:
        data = response.json()
        picks = _parse_espn_picks(data)
        if picks:
            return picks
    except Exception:
        pass

    return _scrape_espn_bracket_dom(page, entry_id, entry_name)


def _parse_espn_picks(data):
    """
    Parse picks from ESPN API response.
    ESPN pick objects look like:
      { "slotIndex": 0, "roundId": 1, "winner": { "id": 123, "abbrev": "DUKE", "displayName": "Duke" } }
    or
      { "slotIndex": 0, "roundNum": 1, "selectedEntrant": { "teamId": 123, "name": "Duke" } }
    """
    picks = {}

    def extract_team_name(obj):
        if not obj or not isinstance(obj, dict):
            return None
        return (
            obj.get("displayName")
            or obj.get("teamDisplayName")
            or obj.get("name")
            or obj.get("abbrev")
            or obj.get("teamName")
        )

    # Unwrap top-level wrapper
    items = data.get("items", [data]) if isinstance(data, dict) and "items" in data else [data]

    for item in items:
        raw_picks = item.get("picks", []) if isinstance(item, dict) else []

        for pick in raw_picks:
            if not isinstance(pick, dict):
                continue

            slot = pick.get("slotIndex", pick.get("slot", ""))
            # ESPN rounds are 1-indexed; we normalize to our 1-indexed system
            rnd = pick.get("roundId") or pick.get("roundNum") or pick.get("round", 1)

            game_id = f"r{rnd}g{slot}"

            # Try every possible winner field name ESPN has used
            winner = None
            for key in ("winner", "win", "selectedEntrant", "pickedTeam", "pick", "selectedTeam"):
                candidate = pick.get(key)
                if candidate:
                    winner = extract_team_name(candidate)
                    if winner:
                        break

            # Some structures put teamName directly on the pick
            if not winner:
                winner = pick.get("teamName") or pick.get("teamDisplayName")

            if game_id and winner:
                picks[game_id] = str(winner)

    return picks


def _scrape_espn_bracket_dom(page, entry_id, entry_name):
    """Fallback: load the bracket page and extract picks from the DOM / embedded JSON."""
    url = (
        f"https://fantasy.espn.com/tournament-challenge-bracket"
        f"/{CURRENT_YEAR}/en/entry?entryID={entry_id}"
    )
    page.goto(url, wait_until="networkidle")
    time.sleep(2)

    picks = {}
    content = page.content()

    # Look for embedded JSON blobs ESPN injects into the page
    for marker in ['"picks"', '"bracketData"', '"entryPicks"']:
        idx = content.find(marker)
        while idx != -1:
            try:
                start = content.rindex("{", 0, idx)
                depth = 0
                end = start
                for i in range(start, min(start + 100_000, len(content))):
                    if content[i] == "{":
                        depth += 1
                    elif content[i] == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                import json
                data = json.loads(content[start:end])
                candidate = _parse_espn_picks(data)
                if candidate:
                    picks.update(candidate)
                    break
            except Exception:
                pass
            idx = content.find(marker, idx + 1)
        if picks:
            break

    # DOM fallback — ESPN uses classes like "picked", "winner", "selected"
    if not picks:
        for selector in [
            ".tc-bracket-matchup--winner .tc-team__name",
            ".picked .team-name",
            "[data-selected='true'] .team-name",
            ".bracket-game .winner-name",
        ]:
            els = page.query_selector_all(selector)
            if els:
                for i, el in enumerate(els):
                    picks[f"r1g{i}"] = el.inner_text().strip()
                break

    return picks
