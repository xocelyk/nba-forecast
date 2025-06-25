#!/usr/bin/env python3
"""
Script to run 100 simulations once a week throughout the NBA season.
This provides weekly analysis of how team performance and championship odds
evolve throughout the season with faster execution than daily analysis.
"""

import argparse
import datetime
import os
import pickle
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline

# Suppress sklearn warnings about feature names
warnings.filterwarnings("ignore", message="X does not have valid feature names")
warnings.filterwarnings("ignore", category=pd.errors.SettingWithCopyWarning)

import env
import sim_season
import utils
from eval import get_win_margin_model


def get_smoothed_stdev_for_num_games(num_games, spline):
    """Calculate smoothed standard deviation for a given number of games."""
    high = num_games + 50
    low = num_games - 50
    high_weight = 1 - (high - num_games) / 100
    low_weight = 1 - (num_games - low) / 100
    return (spline(high) * high_weight + spline(low) * low_weight) / (
        high_weight + low_weight
    )


class StdevFunction:
    """Wrapper class to make the spline function picklable."""

    def __init__(self, spline):
        self.spline = spline

    def __call__(self, num_games):
        return get_smoothed_stdev_for_num_games(num_games, self.spline)


def load_models_and_data():
    """Load the trained models and game data."""
    print("Loading models and data...")

    # Load the trained model
    model_path = os.path.join(env.DATA_DIR, "win_margin_model_heavy.pkl")
    with open(model_path, "rb") as f:
        win_margin_model = pickle.load(f)

    # Load training data for model parameters
    train_data_path = os.path.join(env.DATA_DIR, "train_data.csv")
    data = pd.read_csv(train_data_path)

    # Get model parameters using eval.py
    _, margin_model_resid_mean, margin_model_resid_std, spline = get_win_margin_model(
        data
    )

    # Create the standard deviation function using the wrapper class
    num_games_to_std_margin_model_resid = StdevFunction(spline)

    # Create a simple win probability model for reporting
    from eval import get_win_probability_model

    win_prob_model = get_win_probability_model(data, win_margin_model)

    # Pace parameters (approximate NBA averages)
    mean_pace = 100.0
    std_pace = 5.0

    print("Models and data loaded successfully.")
    return (
        win_margin_model,
        win_prob_model,
        data,
        margin_model_resid_mean,
        margin_model_resid_std,
        num_games_to_std_margin_model_resid,
        mean_pace,
        std_pace,
    )


def print_weekly_results(
    sim_report_df: pd.DataFrame,
    sim_date: datetime.date,
    completed_games: int,
    future_games: int,
    week_num: int,
    total_weeks: int,
):
    """Pretty print the weekly simulation results."""
    season_progress = completed_games / (completed_games + future_games)

    print(
        f"\n🗓️ WEEKLY NBA SIMULATION - Week {week_num} - {sim_date.strftime('%B %d, %Y')} 🗓️"
    )
    print(
        f"📅 Week {week_num}/{total_weeks} • Season Progress: {season_progress:.1%} • Games Played: {completed_games}"
    )
    print("=" * 85)

    # Championship odds
    print("🏆 CHAMPIONSHIP ODDS")
    print("-" * 35)
    top_12_champ = sim_report_df.nlargest(12, "champion")
    for idx, (team, row) in enumerate(top_12_champ.iterrows(), 1):
        bar_length = int(row["champion"] * 50)  # Scale to 50 chars max
        bar = "█" * bar_length
        print(f"{idx:2d}. {team:3s} {row['champion']:6.1%} {bar}")

    # Playoff odds - East vs West
    print(f"\n🎯 PLAYOFF ODDS")
    print("-" * 35)

    # Define conferences
    east_teams = [
        "ATL",
        "BOS",
        "BRK",
        "CHI",
        "CHO",
        "CLE",
        "DET",
        "IND",
        "MIA",
        "MIL",
        "NYK",
        "ORL",
        "PHI",
        "TOR",
        "WAS",
    ]
    west_teams = [
        "DAL",
        "DEN",
        "GSW",
        "HOU",
        "LAC",
        "LAL",
        "MEM",
        "MIN",
        "NOP",
        "OKC",
        "PHO",
        "POR",
        "SAC",
        "SAS",
        "UTA",
    ]

    # East playoffs
    print("   EASTERN CONFERENCE")
    east_playoff = sim_report_df[sim_report_df.index.isin(east_teams)].nlargest(
        10, "playoffs"
    )
    for idx, (team, row) in enumerate(east_playoff.iterrows(), 1):
        bar_length = int(row["playoffs"] * 25)
        bar = "▓" * bar_length
        print(f"   {idx:2d}. {team:3s} {row['playoffs']:6.1%} {bar}")

    print("   WESTERN CONFERENCE")
    west_playoff = sim_report_df[sim_report_df.index.isin(west_teams)].nlargest(
        10, "playoffs"
    )
    for idx, (team, row) in enumerate(west_playoff.iterrows(), 1):
        bar_length = int(row["playoffs"] * 25)
        bar = "▓" * bar_length
        print(f"   {idx:2d}. {team:3s} {row['playoffs']:6.1%} {bar}")

    # Finals odds
    print(f"\n🔥 NBA FINALS ODDS (Top 10)")
    print("-" * 35)
    top_10_finals = sim_report_df.nlargest(10, "finals")
    for idx, (team, row) in enumerate(top_10_finals.iterrows(), 1):
        bar_length = int(row["finals"] * 30)
        bar = "▒" * bar_length
        print(f"{idx:2d}. {team:3s} {row['finals']:6.1%} {bar}")

    # Expected wins/losses
    print(f"\n📊 PROJECTED FINAL RECORDS (Top 12)")
    print("-" * 50)
    top_12_wins = sim_report_df.nlargest(12, "wins")
    for idx, (team, row) in enumerate(top_12_wins.iterrows(), 1):
        total_games = row["wins"] + row["losses"]
        win_pct = row["wins"] / total_games if total_games > 0 else 0
        print(
            f"{idx:2d}. {team:3s} {row['wins']:5.1f}-{row['losses']:4.1f} ({win_pct:.3f})"
        )

    # Show key changes if this isn't the first simulation
    if hasattr(print_weekly_results, "previous_favorite"):
        current_favorite = top_12_champ.index[0]
        if current_favorite != print_weekly_results.previous_favorite:
            prev_odds = (
                print_weekly_results.previous_odds
                if hasattr(print_weekly_results, "previous_odds")
                else 0
            )
            current_odds = sim_report_df.loc[current_favorite, "champion"]
            print(
                f"\n🚨 NEW CHAMPIONSHIP FAVORITE: {current_favorite} ({current_odds:.1%})"
            )
            print(
                f"   Previous Week Leader: {print_weekly_results.previous_favorite} ({prev_odds:.1%})"
            )

    # Store for next iteration
    print_weekly_results.previous_favorite = top_12_champ.index[0]
    print_weekly_results.previous_odds = sim_report_df.loc[
        top_12_champ.index[0], "champion"
    ]

    # Show interesting weekly stats
    parity_score = 1 / (sim_report_df["champion"] ** 2).sum()
    top_5_champ_share = sim_report_df.nlargest(5, "champion")["champion"].sum()

    print(f"\n📈 WEEKLY INSIGHTS")
    print(f"   🎲 Parity Index: {parity_score:.2f} (higher = more competitive)")
    print(f"   🔝 Top 5 Teams Control: {top_5_champ_share:.1%} of championship odds")

    # Show teams with biggest changes (if we have previous data)
    if hasattr(print_weekly_results, "previous_data"):
        try:
            prev_df = print_weekly_results.previous_data.set_index("team")
            current_df = sim_report_df

            # Calculate championship odds changes
            changes = []
            for team in current_df.index:
                if team in prev_df.index:
                    change = (
                        current_df.loc[team, "champion"] - prev_df.loc[team, "champion"]
                    )
                    changes.append((team, change))

            # Show biggest movers
            changes.sort(key=lambda x: abs(x[1]), reverse=True)
            print(f"\n📈📉 BIGGEST WEEKLY MOVERS (Championship Odds)")
            print("-" * 45)
            for i, (team, change) in enumerate(changes[:6]):
                arrow = "📈" if change > 0 else "📉"
                print(f"   {arrow} {team:3s} {change:+6.1%}")
        except:
            pass  # Skip if comparison fails

    # Store current data for next comparison
    print_weekly_results.previous_data = sim_report_df.reset_index()

    print("=" * 85)
    print()


def get_weekly_dates(year: int = 2025) -> List[datetime.date]:
    """Get weekly simulation dates throughout the NBA season."""
    # NBA season typically runs from mid-October to mid-April
    season_start = datetime.date(year - 1, 10, 15)  # October 15, 2024
    season_end = datetime.date(year, 4, 14)  # April 14, 2025

    # Start with first Monday of the season
    current_date = season_start
    while current_date.weekday() != 0:  # Monday is 0
        current_date += datetime.timedelta(days=1)

    weekly_dates = []
    while current_date <= season_end:
        weekly_dates.append(current_date)
        current_date += datetime.timedelta(days=7)  # Add one week

    return weekly_dates


def run_weekly_simulations(
    data: pd.DataFrame,
    win_margin_model,
    win_prob_model,
    margin_model_resid_mean: float,
    margin_model_resid_std: float,
    num_games_to_std_margin_model_resid,
    mean_pace: float,
    std_pace: float,
    year: int = 2025,
    num_sims: int = 100,
) -> pd.DataFrame:
    """Run simulations once per week throughout the season."""

    print(f"Starting weekly simulation analysis for {year} season...")
    print(f"Running {num_sims} simulations per week")

    # Get weekly simulation dates
    weekly_dates = get_weekly_dates(year)

    # Prepare data
    data["date"] = pd.to_datetime(data["date"]).dt.date
    year_games = data[data["year"] == year].copy()

    # Results storage
    weekly_results = []

    # Track simulation progress
    total_weeks = len(weekly_dates)

    for week_idx, sim_date in enumerate(weekly_dates):
        print(f"\n{'='*60}")
        print(
            f"Week {week_idx + 1}/{total_weeks}: {sim_date} ({sim_date.strftime('%A, %B %d, %Y')})"
        )
        print(f"{'='*60}")

        # Check if there are any games on or after this date
        future_games = year_games[year_games["date"] >= sim_date]
        if future_games.empty:
            print(f"No future games from {sim_date}, skipping...")
            continue

        # Check if we have any completed games before this date
        completed_games = year_games[year_games["date"] < sim_date]
        if completed_games.empty:
            print(f"No completed games before {sim_date}, skipping...")
            continue

        print(f"Completed games: {len(completed_games)}")
        print(f"Future games: {len(future_games)}")

        try:
            # Run simulations from this date
            sim_report_df = sim_season.sim_season(
                data=data,
                win_margin_model=win_margin_model,
                win_prob_model=win_prob_model,
                margin_model_resid_mean=margin_model_resid_mean,
                margin_model_resid_std=margin_model_resid_std,
                num_games_to_std_margin_model_resid=num_games_to_std_margin_model_resid,
                mean_pace=mean_pace,
                std_pace=std_pace,
                year=year,
                num_sims=num_sims,
                parallel=True,
                start_date=sim_date,
            )

            # Add metadata to results
            sim_report_df["simulation_date"] = sim_date
            sim_report_df["week_number"] = week_idx + 1
            sim_report_df["games_completed"] = len(completed_games)
            sim_report_df["games_remaining"] = len(future_games)
            sim_report_df["season_progress"] = len(completed_games) / len(year_games)

            # Store results
            weekly_results.append(sim_report_df.reset_index())

            # Pretty print weekly results
            print_weekly_results(
                sim_report_df,
                sim_date,
                len(completed_games),
                len(future_games),
                week_idx + 1,
                total_weeks,
            )

        except Exception as e:
            print(f"Error running simulations for {sim_date}: {e}")
            continue

    # Combine all results
    if not weekly_results:
        print("No simulation results to combine!")
        return pd.DataFrame()

    print(f"\nCombining results from {len(weekly_results)} simulation weeks...")
    all_results = pd.concat(weekly_results, ignore_index=True)

    return all_results


def save_weekly_results(results_df: pd.DataFrame, year: int = 2025):
    """Save the weekly simulation results to CSV files."""
    if results_df.empty:
        print("No results to save!")
        return

    # Create output directory
    output_dir = os.path.join(env.DATA_DIR, "weekly_simulations")
    os.makedirs(output_dir, exist_ok=True)

    # Save complete results
    output_file = os.path.join(output_dir, f"weekly_sim_results_{year}.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Complete weekly simulation results saved to: {output_file}")

    # Save championship odds progression
    championship_progression = results_df.pivot(
        index="simulation_date", columns="team", values="champion"
    )
    champ_file = os.path.join(output_dir, f"weekly_championship_odds_{year}.csv")
    championship_progression.to_csv(champ_file)
    print(f"Weekly championship odds progression saved to: {champ_file}")

    # Save summary statistics
    summary_stats = []
    for date in results_df["simulation_date"].unique():
        date_data = results_df[results_df["simulation_date"] == date]
        if not date_data.empty:
            top_team = date_data.loc[date_data["champion"].idxmax()]
            summary_stats.append(
                {
                    "date": date,
                    "week_number": top_team["week_number"],
                    "games_completed": top_team["games_completed"],
                    "season_progress": top_team["season_progress"],
                    "favorite_team": top_team["team"],
                    "favorite_odds": top_team["champion"],
                    "parity_index": 1
                    / (date_data["champion"] ** 2).sum(),  # Higher = more parity
                    "top_5_share": date_data.nlargest(5, "champion")["champion"].sum(),
                }
            )

    summary_df = pd.DataFrame(summary_stats)
    summary_file = os.path.join(output_dir, f"weekly_summary_{year}.csv")
    summary_df.to_csv(summary_file, index=False)
    print(f"Weekly summary statistics saved to: {summary_file}")


def analyze_weekly_results(results_df: pd.DataFrame):
    """Analyze and print insights from weekly simulation results."""
    if results_df.empty:
        print("No results to analyze!")
        return

    print(f"\n{'='*70}")
    print("WEEKLY SIMULATION ANALYSIS SUMMARY")
    print(f"{'='*70}")

    # Get unique dates and teams
    dates = sorted(results_df["simulation_date"].unique())
    teams = results_df["team"].unique()

    print(f"Analysis period: {dates[0]} to {dates[-1]}")
    print(f"Total simulation weeks: {len(dates)}")
    print(f"Teams analyzed: {len(teams)}")

    # Championship favorite changes
    print(f"\nCHAMPIONSHIP FAVORITE WEEKLY PROGRESSION:")
    print("-" * 50)

    favorites_by_week = []
    for date in dates:
        date_data = results_df[results_df["simulation_date"] == date]
        if not date_data.empty:
            favorite = date_data.loc[date_data["champion"].idxmax()]
            favorites_by_week.append(
                {
                    "date": date,
                    "week": favorite["week_number"],
                    "team": favorite["team"],
                    "odds": favorite["champion"],
                    "progress": favorite["season_progress"],
                }
            )

    # Show all weekly favorites
    prev_favorite = None
    for fav in favorites_by_week:
        status = " 🆕" if fav["team"] != prev_favorite else ""
        print(
            f"Week {fav['week']:2d} ({fav['date']}): {fav['team']} {fav['odds']:6.1%} - {fav['progress']:.1%} complete{status}"
        )
        prev_favorite = fav["team"]

    # Teams that were #1 at some point
    print(f"\nTEAMS THAT LED CHAMPIONSHIP ODDS:")
    print("-" * 50)
    leaders = set()
    for fav in favorites_by_week:
        leaders.add(fav["team"])

    for team in sorted(leaders):
        team_data = results_df[results_df["team"] == team]
        max_odds = team_data["champion"].max()
        max_week_data = team_data.loc[team_data["champion"].idxmax()]
        max_week = max_week_data["week_number"]
        max_date = max_week_data["simulation_date"]
        print(f"{team}: Peak {max_odds:.1%} in Week {max_week} ({max_date})")

    # Final standings
    print(f"\nFINAL CHAMPIONSHIP ODDS:")
    print("-" * 30)
    final_date = dates[-1]
    final_results = results_df[results_df["simulation_date"] == final_date]
    final_top12 = final_results.nlargest(12, "champion")

    for idx, (_, row) in enumerate(final_top12.iterrows(), 1):
        print(f"{idx:2d}. {row['team']}: {row['champion']:5.1%}")


def main():
    """Main function to run weekly simulation analysis."""
    parser = argparse.ArgumentParser(description="NBA Weekly Simulation Analysis")
    parser.add_argument(
        "--num-sims",
        type=int,
        default=100,
        help="Number of simulations to run per week (default: 100)",
    )
    parser.add_argument(
        "--year",
        type=int,
        default=2025,
        help="NBA season year to analyze (default: 2025)",
    )
    args = parser.parse_args()

    print("NBA Weekly Simulation Analysis")
    print("=" * 60)
    print(f"Season: {args.year}")
    print(f"Simulations per week: {args.num_sims}")
    print("=" * 60)

    try:
        # Load models and data
        (
            win_margin_model,
            win_prob_model,
            data,
            margin_model_resid_mean,
            margin_model_resid_std,
            num_games_to_std_margin_model_resid,
            mean_pace,
            std_pace,
        ) = load_models_and_data()

        # Run weekly simulations
        results_df = run_weekly_simulations(
            data=data,
            win_margin_model=win_margin_model,
            win_prob_model=win_prob_model,
            margin_model_resid_mean=margin_model_resid_mean,
            margin_model_resid_std=margin_model_resid_std,
            num_games_to_std_margin_model_resid=num_games_to_std_margin_model_resid,
            mean_pace=mean_pace,
            std_pace=std_pace,
            year=args.year,
            num_sims=args.num_sims,
        )

        # Save results
        save_weekly_results(results_df, year=args.year)

        # Analyze results
        analyze_weekly_results(results_df)

        print(f"\n{'='*70}")
        print("WEEKLY SIMULATION ANALYSIS COMPLETE!")
        print(f"{'='*70}")

    except Exception as e:
        print(f"Error in weekly simulation analysis: {e}")
        raise


if __name__ == "__main__":
    main()
