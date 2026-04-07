"""
Fetch game spreads from the-odds-api.com and save to data/spreads/.

Usage:
    python scripts/fetch_spreads.py --year 2026 --api-key YOUR_KEY

Or set the environment variable:
    export ODDS_API_KEY=YOUR_KEY
    python scripts/fetch_spreads.py --year 2026

The script fetches completed NBA game scores with spreads from the API,
matches them to game IDs in the existing game data, and writes a CSV
to data/spreads/spreads_YYYY.csv.

API docs: https://the-odds-api.com/liveapi/guides/v4/
Free tier: 500 requests/month.
"""

import argparse
import csv
import os
import sys
import time

import pandas as pd
import requests

# Map the-odds-api team names to our abbreviations
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


def fetch_scores(api_key: str, days_from: int = 3) -> list:
    """Fetch recent completed NBA scores."""
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/scores/"
    params = {
        "apiKey": api_key,
        "daysFrom": days_from,
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def extract_consensus_spread(event: dict) -> dict | None:
    """Extract the consensus (first bookmaker) spread for a game event.

    Returns a dict with home_team, away_team, home_spread or None.
    """
    bookmakers = event.get("bookmakers", [])
    if not bookmakers:
        return None

    # Use the first bookmaker (typically the consensus line)
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
                        "commence_time": event.get("commence_time", ""),
                    }
    return None


def match_to_game_id(
    home_abbr: str,
    away_abbr: str,
    game_date: str,
    games_df: pd.DataFrame,
) -> str | None:
    """Match an odds-api game to a game_id in our data."""
    # game_date is like "2024-10-22"
    matches = games_df[
        (games_df["team"] == home_abbr)
        & (games_df["opponent"] == away_abbr)
        & (games_df["date"].astype(str).str.startswith(game_date))
    ]
    if len(matches) == 1:
        return str(matches.iloc[0]["game_id"])
    return None


def build_spreads_from_game_data(
    games_df: pd.DataFrame, api_key: str
) -> pd.DataFrame:
    """Build spread data by fetching historical odds for each game date.

    This iterates through unique dates in the game data and fetches
    historical odds for each date.
    """
    completed = games_df[games_df["completed"] == True].copy()
    dates = sorted(completed["date"].astype(str).str[:10].unique())

    spreads = []
    for i, date_str in enumerate(dates):
        iso_date = f"{date_str}T12:00:00Z"
        print(f"Fetching odds for {date_str} ({i+1}/{len(dates)})...")
        try:
            events = fetch_historical_odds(api_key, iso_date)
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 422:
                print(f"  No data for {date_str}, skipping")
                continue
            raise
        except Exception as e:
            print(f"  Error fetching {date_str}: {e}")
            continue

        for event in events:
            spread_info = extract_consensus_spread(event)
            if spread_info is None:
                continue

            home_name = spread_info["home_team"]
            away_name = spread_info["away_team"]
            home_abbr = ODDS_API_TEAM_MAP.get(home_name)
            away_abbr = ODDS_API_TEAM_MAP.get(away_name)
            if home_abbr is None or away_abbr is None:
                continue

            game_id = match_to_game_id(home_abbr, away_abbr, date_str, completed)
            if game_id is None:
                continue

            # home_spread is negative when home is favored (e.g. -5.5)
            # We want spread in our convention: positive = home favored
            # So we negate: home_spread of -5.5 -> spread of 5.5
            spread = -spread_info["home_spread"]
            spreads.append({"game_id": game_id, "spread": spread})

        # Rate limiting: ~1 request per second
        time.sleep(1.0)

    return pd.DataFrame(spreads)


def main():
    parser = argparse.ArgumentParser(description="Fetch NBA game spreads")
    parser.add_argument("--year", type=int, required=True, help="Season year")
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="the-odds-api.com API key (or set ODDS_API_KEY env var)",
    )
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("ODDS_API_KEY")
    if not api_key:
        print(
            "Error: provide --api-key or set ODDS_API_KEY environment variable.\n"
            "Get a free key at https://the-odds-api.com/",
            file=sys.stderr,
        )
        sys.exit(1)

    # Load game data
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    games_path = os.path.join(data_dir, "games", f"year_data_{args.year}.csv")
    if not os.path.exists(games_path):
        print(f"Error: game data not found at {games_path}", file=sys.stderr)
        sys.exit(1)

    games_df = pd.read_csv(games_path, dtype={"game_id": str})
    games_df["date"] = pd.to_datetime(games_df["date"], format="mixed")

    print(f"Building spreads for {args.year} season...")
    spreads_df = build_spreads_from_game_data(games_df, api_key)

    # Save
    spreads_dir = os.path.join(data_dir, "spreads")
    os.makedirs(spreads_dir, exist_ok=True)
    output_path = os.path.join(spreads_dir, f"spreads_{args.year}.csv")
    spreads_df.to_csv(output_path, index=False)
    print(f"Saved {len(spreads_df)} spreads to {output_path}")


if __name__ == "__main__":
    main()
