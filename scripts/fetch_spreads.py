"""
Fetch game spreads and save to data/spreads/.

Supports multiple data sources:

1. ESPN (default, no API key needed):
    python scripts/fetch_spreads.py --year 2026 --source espn

2. the-odds-api.com (requires API key):
    python scripts/fetch_spreads.py --year 2026 --source odds-api --api-key YOUR_KEY

3. Merge mode — add to existing spreads without overwriting:
    python scripts/fetch_spreads.py --year 2026 --source espn --merge

Output is written to data/spreads/spreads_YYYY.csv with columns [game_id, spread].
Spread convention: positive = home team favored (matches margin convention).
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import pandas as pd
import requests

# ESPN team abbreviation -> our abbreviation
ESPN_TEAM_MAP = {
    "ATL": "ATL", "BOS": "BOS", "BKN": "BRK", "CHA": "CHO",
    "CHI": "CHI", "CLE": "CLE", "DAL": "DAL", "DEN": "DEN",
    "DET": "DET", "GS": "GSW", "HOU": "HOU", "IND": "IND",
    "LAC": "LAC", "LAL": "LAL", "MEM": "MEM", "MIA": "MIA",
    "MIL": "MIL", "MIN": "MIN", "NO": "NOP", "NY": "NYK",
    "NYK": "NYK", "OKC": "OKC", "ORL": "ORL", "PHI": "PHI",
    "PHX": "PHX", "POR": "POR", "SAC": "SAC", "SA": "SAS",
    "SAS": "SAS", "TOR": "TOR", "UTA": "UTA", "UTAH": "UTA",
    "WSH": "WAS", "WAS": "WAS", "GSW": "GSW", "NOP": "NOP",
}

# the-odds-api team names -> our abbreviation
ODDS_API_TEAM_MAP = {
    "Atlanta Hawks": "ATL",
    "Boston Celtics": "BOS",
    "Brooklyn Nets": "BRK",
    "Charlotte Hornets": "CHO",
    "Chicago Bulls": "CHI",
    "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL",
    "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW",
    "Houston Rockets": "HOU",
    "Indiana Pacers": "IND",
    "Los Angeles Clippers": "LAC",
    "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA",
    "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN",
    "New Orleans Pelicans": "NOP",
    "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC",
    "Orlando Magic": "ORL",
    "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX",
    "Portland Trail Blazers": "POR",
    "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS",
    "Toronto Raptors": "TOR",
    "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}


# ---------------------------------------------------------------------------
# ESPN source
# ---------------------------------------------------------------------------

def espn_map_abbr(abbr: str) -> str:
    return ESPN_TEAM_MAP.get(abbr, abbr)


def fetch_espn_scoreboard(date_str: str) -> list[dict]:
    """Fetch ESPN NBA scoreboard for a date (YYYYMMDD). Returns events list."""
    url = (
        "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"
    )
    resp = requests.get(url, params={"dates": date_str}, timeout=30)
    resp.raise_for_status()
    return resp.json().get("events", [])


def extract_espn_spread(event: dict) -> dict | None:
    """Extract spread from an ESPN event.

    Returns dict with home_abbr, away_abbr, spread (our convention) or None.
    """
    comp = event.get("competitions", [{}])[0]
    odds_list = comp.get("odds", [])
    if not odds_list:
        return None
    odds = odds_list[0]
    spread_val = odds.get("spread")
    if spread_val is None:
        return None

    competitors = comp.get("competitors", [])
    home = next((c for c in competitors if c.get("homeAway") == "home"), None)
    away = next((c for c in competitors if c.get("homeAway") == "away"), None)
    if not home or not away:
        return None

    home_abbr = espn_map_abbr(home["team"]["abbreviation"])
    away_abbr = espn_map_abbr(away["team"]["abbreviation"])

    # ESPN spread: negative = home favored. We negate to our convention.
    return {
        "home_abbr": home_abbr,
        "away_abbr": away_abbr,
        "spread": -float(spread_val),
    }


def build_spreads_espn(games_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch spreads from ESPN for all completed game dates."""
    completed = games_df[games_df["completed"] == True].copy()
    completed["date_str"] = completed["date"].dt.strftime("%Y-%m-%d")
    dates = sorted(completed["date_str"].unique())

    # Build lookup: (date, home, away) -> game_id
    lookup = {}
    for _, row in completed.iterrows():
        key = (row["date_str"], row["team"], row["opponent"])
        lookup[key] = str(row["game_id"])

    spreads = []
    for i, date_str in enumerate(dates):
        espn_date = date_str.replace("-", "")
        print(f"  ESPN: {date_str} ({i+1}/{len(dates)})...", end="", flush=True)
        try:
            events = fetch_espn_scoreboard(espn_date)
        except Exception as e:
            print(f" error: {e}")
            continue

        matched = 0
        for event in events:
            info = extract_espn_spread(event)
            if info is None:
                continue
            key = (date_str, info["home_abbr"], info["away_abbr"])
            game_id = lookup.get(key)
            if game_id:
                spreads.append({"game_id": game_id, "spread": info["spread"]})
                matched += 1

        print(f" {matched}/{len(events)} matched")
        time.sleep(0.5)  # be polite

    return pd.DataFrame(spreads)


# ---------------------------------------------------------------------------
# the-odds-api source
# ---------------------------------------------------------------------------

def fetch_historical_odds(api_key: str, date: str) -> list:
    """Fetch historical NBA spreads for a specific date (YYYY-MM-DDT00:00:00Z)."""
    url = "https://api.the-odds-api.com/v4/historical/sports/basketball_nba/odds/"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american",
        "date": date,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json().get("data", [])


def extract_consensus_spread(event: dict) -> dict | None:
    """Extract consensus spread from an odds-api event."""
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None
    book = bookmakers[0]
    for market in book.get("markets", []):
        if market["key"] == "spreads":
            outcomes = market["outcomes"]
            home_team = event.get("home_team")
            for outcome in outcomes:
                if outcome["name"] == home_team:
                    return {
                        "home_team": home_team,
                        "away_team": event.get("away_team"),
                        "home_spread": float(outcome["point"]),
                    }
    return None


def build_spreads_odds_api(games_df: pd.DataFrame, api_key: str) -> pd.DataFrame:
    """Fetch spreads from the-odds-api for all completed game dates."""
    completed = games_df[games_df["completed"] == True].copy()
    completed["date_str"] = completed["date"].dt.strftime("%Y-%m-%d")
    dates = sorted(completed["date_str"].unique())

    lookup = {}
    for _, row in completed.iterrows():
        key = (row["date_str"], row["team"], row["opponent"])
        lookup[key] = str(row["game_id"])

    spreads = []
    for i, date_str in enumerate(dates):
        iso_date = f"{date_str}T12:00:00Z"
        print(f"  odds-api: {date_str} ({i+1}/{len(dates)})...", end="", flush=True)
        try:
            events = fetch_historical_odds(api_key, iso_date)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                print(" no data")
                continue
            raise
        except Exception as e:
            print(f" error: {e}")
            continue

        matched = 0
        for event in events:
            spread_info = extract_consensus_spread(event)
            if spread_info is None:
                continue
            home_abbr = ODDS_API_TEAM_MAP.get(spread_info["home_team"])
            away_abbr = ODDS_API_TEAM_MAP.get(spread_info["away_team"])
            if not home_abbr or not away_abbr:
                continue
            key = (date_str, home_abbr, away_abbr)
            game_id = lookup.get(key)
            if game_id:
                spreads.append({
                    "game_id": game_id,
                    "spread": -spread_info["home_spread"],
                })
                matched += 1

        print(f" {matched}/{len(events)} matched")
        time.sleep(1.0)

    return pd.DataFrame(spreads)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fetch NBA game spreads")
    parser.add_argument("--year", type=int, required=True, help="Season year")
    parser.add_argument(
        "--source",
        choices=["espn", "odds-api"],
        default="espn",
        help="Data source (default: espn, no API key needed)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="the-odds-api.com API key (or set ODDS_API_KEY env var)",
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help="Merge with existing spreads file (don't overwrite)",
    )
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    games_path = os.path.join(data_dir, "games", f"year_data_{args.year}.csv")
    if not os.path.exists(games_path):
        print(f"Error: game data not found at {games_path}", file=sys.stderr)
        sys.exit(1)

    games_df = pd.read_csv(games_path, dtype={"game_id": str})
    games_df["date"] = pd.to_datetime(games_df["date"], format="mixed")

    print(f"Fetching spreads for {args.year} season from {args.source}...")

    if args.source == "espn":
        new_spreads = build_spreads_espn(games_df)
    elif args.source == "odds-api":
        api_key = args.api_key or os.environ.get("ODDS_API_KEY")
        if not api_key:
            print(
                "Error: --source odds-api requires --api-key or ODDS_API_KEY env var.\n"
                "Get a free key at https://the-odds-api.com/",
                file=sys.stderr,
            )
            sys.exit(1)
        new_spreads = build_spreads_odds_api(games_df, api_key)

    # Merge with existing data if requested
    spreads_dir = os.path.join(data_dir, "spreads")
    os.makedirs(spreads_dir, exist_ok=True)
    output_path = os.path.join(spreads_dir, f"spreads_{args.year}.csv")

    if args.merge and os.path.exists(output_path):
        existing = pd.read_csv(output_path, dtype={"game_id": str})
        existing_ids = set(existing["game_id"])
        # Only add new spreads for games not already covered
        new_only = new_spreads[~new_spreads["game_id"].isin(existing_ids)]
        combined = pd.concat([existing, new_only], ignore_index=True)
        print(f"Merged: {len(existing)} existing + {len(new_only)} new = {len(combined)} total")
        combined.to_csv(output_path, index=False)
    else:
        new_spreads.to_csv(output_path, index=False)
        print(f"Saved {len(new_spreads)} spreads to {output_path}")


if __name__ == "__main__":
    main()
