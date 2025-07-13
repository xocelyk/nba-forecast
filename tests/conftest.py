"""
Shared test fixtures and configuration for NBA simulation testing.
"""
import os
import datetime
from typing import Dict, List
import pytest
import pandas as pd
import numpy as np


@pytest.fixture(scope="session")
def sample_teams():
    """Sample team data for testing."""
    return {
        "names_to_abbr": {
            "Boston Celtics": "BOS",
            "Los Angeles Lakers": "LAL", 
            "Miami Heat": "MIA",
            "Denver Nuggets": "DEN"
        },
        "abbrs": ["BOS", "LAL", "MIA", "DEN"],
        "abbr_to_name": {
            "BOS": "Boston Celtics",
            "LAL": "Los Angeles Lakers",
            "MIA": "Miami Heat", 
            "DEN": "Denver Nuggets"
        }
    }


@pytest.fixture(scope="session")
def sample_games_data():
    """Sample game data for testing."""
    return pd.DataFrame({
        "boxscore_id": ["202501010BOS", "202501020LAL", "202501030MIA"],
        "date": pd.to_datetime(["2025-01-01", "2025-01-02", "2025-01-03"]),
        "team": ["BOS", "LAL", "MIA"],
        "opponent": ["LAL", "BOS", "DEN"],
        "team_score": [110, 105, 98],
        "opponent_score": [108, 102, 95],
        "margin": [2, 3, 3],
        "location": ["Home", "Home", "Home"],
        "pace": [98.5, 101.2, 96.8],
        "completed": [True, True, True],
        "year": [2025, 2025, 2025],
        "playoff": [0, 0, 0]
    })


@pytest.fixture(scope="session")
def sample_training_data():
    """Sample training data with all required features."""
    np.random.seed(42)  # For reproducible test data
    
    n_games = 100
    teams = ["BOS", "LAL", "MIA", "DEN", "GSW", "NYK"]
    
    data = []
    for i in range(n_games):
        team = np.random.choice(teams)
        opponent = np.random.choice([t for t in teams if t != team])
        team_rating = np.random.normal(0, 5)
        opponent_rating = np.random.normal(0, 5)
        margin = team_rating - opponent_rating + np.random.normal(0, 10)
        
        data.append({
            "team": team,
            "opponent": opponent,
            "team_rating": team_rating,
            "opponent_rating": opponent_rating,
            "rating_diff": team_rating - opponent_rating,
            "last_year_team_rating": np.random.normal(0, 5),
            "last_year_opp_rating": np.random.normal(0, 5),
            "last_year_rating_diff": np.random.normal(0, 5),
            "margin": margin,
            "num_games_into_season": i + 1,
            "team_last_10_rating": np.random.normal(0, 3),
            "opponent_last_10_rating": np.random.normal(0, 3),
            "last_10_rating_diff": np.random.normal(0, 3),
            "team_last_5_rating": np.random.normal(0, 3),
            "opponent_last_5_rating": np.random.normal(0, 3),
            "last_5_rating_diff": np.random.normal(0, 3),
            "team_last_3_rating": np.random.normal(0, 3),
            "opponent_last_3_rating": np.random.normal(0, 3),
            "last_3_rating_diff": np.random.normal(0, 3),
            "team_last_1_rating": np.random.normal(0, 3),
            "opponent_last_1_rating": np.random.normal(0, 3),
            "last_1_rating_diff": np.random.normal(0, 3),
            "team_win_total_future": np.random.uniform(20, 60),
            "opponent_win_total_future": np.random.uniform(20, 60),
            "team_days_since_most_recent_game": np.random.randint(1, 5),
            "opponent_days_since_most_recent_game": np.random.randint(1, 5),
            "hca": 2.5,
            "playoff": np.random.choice([0, 1], p=[0.8, 0.2]),
            "date": datetime.date(2025, 1, 1) + datetime.timedelta(days=i),
            "year": 2025,
            "pace": np.random.normal(100, 5),
            "completed": True
        })
    
    return pd.DataFrame(data)


@pytest.fixture(scope="session") 
def sample_ratings():
    """Sample EM ratings for teams."""
    return {
        "BOS": 5.2,
        "LAL": 3.8,
        "MIA": 1.5,
        "DEN": 4.1,
        "GSW": -1.2,
        "NYK": -2.8
    }


@pytest.fixture(scope="session")
def sample_win_totals():
    """Sample win total futures."""
    return {
        "2025": {
            "BOS": 52.5,
            "LAL": 48.5,
            "MIA": 42.5,
            "DEN": 50.5,
            "GSW": 38.5,
            "NYK": 35.5
        }
    }


@pytest.fixture
def mock_data_dir(tmp_path):
    """Create a temporary directory structure for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create subdirectories
    (data_dir / "games").mkdir()
    (data_dir / "sim_results").mkdir()
    (data_dir / "sim_results" / "archive").mkdir()
    (data_dir / "seed_reports").mkdir()
    (data_dir / "seed_reports" / "archive").mkdir()
    (data_dir / "end_year_ratings").mkdir()
    
    return data_dir


@pytest.fixture
def sample_margin_model_data():
    """Sample data for testing margin models."""
    return {
        "predictions": np.array([2.5, -1.2, 4.8, 0.3, -3.1]),
        "actuals": np.array([3.0, -0.8, 5.2, -0.5, -2.9]),
        "residuals": np.array([0.5, 0.4, 0.4, -0.8, 0.2]),
        "mean_residual": 0.14,
        "std_residual": 0.48
    }


@pytest.fixture
def sample_season_games():
    """Sample season data for simulation testing."""
    np.random.seed(42)
    
    teams = ["BOS", "LAL", "MIA", "DEN"]
    games = []
    
    # Generate round-robin schedule
    for i, team in enumerate(teams):
        for j, opponent in enumerate(teams):
            if i != j:
                # Home and away games
                for home_away in [True, False]:
                    if home_away:
                        home_team, away_team = team, opponent
                    else:
                        home_team, away_team = opponent, team
                        
                    margin = np.random.normal(0, 10)
                    games.append({
                        "date": datetime.date(2025, 1, 1) + datetime.timedelta(days=len(games)),
                        "team": home_team,
                        "opponent": away_team, 
                        "margin": margin,
                        "team_win": 1 if margin > 0 else 0,
                        "team_rating": np.random.normal(0, 5),
                        "opponent_rating": np.random.normal(0, 5),
                        "year": 2025,
                        "playoff": 0,
                        "pace": np.random.normal(100, 5),
                        "completed": True
                    })
    
    return pd.DataFrame(games)


@pytest.fixture(autouse=True)
def reset_random_seed():
    """Reset random seed before each test for reproducibility."""
    np.random.seed(42)


@pytest.fixture
def suppress_warnings():
    """Suppress common warnings during testing."""
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)


# Utility functions for test data generation
def generate_game_result(team: str, opponent: str, date: datetime.date, 
                        team_rating: float = 0, opponent_rating: float = 0,
                        is_home: bool = True, hca: float = 2.5) -> Dict:
    """Generate a single game result for testing."""
    expected_margin = team_rating - opponent_rating + (hca if is_home else -hca)
    actual_margin = expected_margin + np.random.normal(0, 10)
    
    return {
        "date": date,
        "team": team if is_home else opponent,
        "opponent": opponent if is_home else team,
        "margin": actual_margin if is_home else -actual_margin,
        "team_win": 1 if actual_margin > 0 else 0,
        "team_rating": team_rating if is_home else opponent_rating,
        "opponent_rating": opponent_rating if is_home else team_rating,
        "year": date.year,
        "playoff": 0,
        "completed": True
    }