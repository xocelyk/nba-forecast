"""
Unit tests for NBA playoff tiebreaker logic in sim_season.Season.
"""

import random

import numpy as np
import pandas as pd
import pytest

from src.sim_season import Season


def make_game(home, away, home_win, margin=None):
    """Create a single regular-season game row (home-team perspective).

    *margin* is always from the home team's point of view (positive = home
    win).  When *home_win* is True and no margin is given, a default of +5
    is used; when False, -5.
    """
    if margin is None:
        margin = 5.0 if home_win else -5.0
    return {
        "team": home,
        "opponent": away,
        "margin": float(margin),
        "team_win": 1 if margin > 0 else 0,
        "playoff": 0,
    }


def make_season(games):
    """Build a minimal Season-like object with only the attributes the
    tiebreaker engine needs: completed_games and the class constants."""
    obj = object.__new__(Season)
    obj.completed_games = pd.DataFrame(games)
    return obj


# -- helpers to build record_by_team from games --------------------------

def record_from_games(games):
    """Compute {team: [wins, losses]} from a list of game dicts."""
    rec = {}
    for g in games:
        home, away = g["team"], g["opponent"]
        rec.setdefault(home, [0, 0])
        rec.setdefault(away, [0, 0])
        if g["margin"] > 0:
            rec[home][0] += 1
            rec[away][1] += 1
        else:
            rec[home][1] += 1
            rec[away][0] += 1
    return rec


# -----------------------------------------------------------------------
# 1. Two-team tie resolved by head-to-head
# -----------------------------------------------------------------------
class TestTwoTeamHeadToHead:
    def test_h2h_decides(self):
        # BOS beats NYK 3-1 in h2h; both 50-32 overall
        games = []
        # h2h games (BOS wins 3 of 4)
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", False))  # BOS wins on road
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", True))   # NYK wins 1

        # pad both to 50 wins, 32 losses with non-h2h games against other
        # Eastern Conference teams
        filler_opps_bos = ["MIA", "ORL", "CHI", "ATL", "CLE"]
        filler_opps_nyk = ["MIA", "ORL", "CHI", "ATL", "CLE"]
        for _ in range(47):  # BOS needs 47 more wins (3 h2h wins)
            games.append(make_game("BOS", filler_opps_bos[_ % 5], True))
        for _ in range(30):  # BOS needs 30 more losses (2 h2h losses from NYK perspective doesn't apply; BOS lost 1 h2h)
            games.append(make_game(filler_opps_bos[_ % 5], "BOS", True))
        for _ in range(49):  # NYK needs 49 more wins (1 h2h win)
            games.append(make_game("NYK", filler_opps_nyk[_ % 5], True))
        for _ in range(31):  # NYK needs 31 more losses (1 h2h loss)
            games.append(make_game(filler_opps_nyk[_ % 5], "NYK", True))

        record = record_from_games(games)
        assert record["BOS"][0] == 50
        assert record["NYK"][0] == 50

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)
        conference_teams = Season.EASTERN_CONFERENCE

        result = season._break_ties(["BOS", "NYK"], stats, conference_teams, record)
        assert result == ["BOS", "NYK"]


# -----------------------------------------------------------------------
# 2. Two-team tie, h2h split, resolved by division winner (same division)
# -----------------------------------------------------------------------
class TestTwoTeamDivisionWinner:
    def test_division_winner_breaks_tie(self):
        # For two-team ties in the SAME division, both teams with equal
        # wins are co-division-winners, so the div-winner step is always a
        # no-op.  This test verifies the fallthrough to conference record.
        # BOS (Atlantic) vs NYK (Atlantic): h2h split, but BOS has more
        # conference wins.
        games = []
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", True))
        # BOS has more conference wins (plays EC opponents)
        for _ in range(30):
            games.append(make_game("BOS", "MIA", True))
        # NYK has fewer conference wins (plays WC opponents)
        for _ in range(30):
            games.append(make_game("NYK", "LAL", True))

        record = record_from_games(games)
        assert record["BOS"][0] == record["NYK"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        # h2h split, same division (both div winners -- no-op),
        # falls to conference record: BOS=31, NYK=1
        assert stats["BOS"]["conference_wins"] > stats["NYK"]["conference_wins"]

        result = season._break_two_team_tie(
            "BOS", "NYK", stats, Season.EASTERN_CONFERENCE, record
        )
        assert result[0] == "BOS"


# -----------------------------------------------------------------------
# 3. Two-team tie, different divisions, resolved by conference record
# -----------------------------------------------------------------------
class TestTwoTeamConferenceRecord:
    def test_conf_record_breaks_tie(self):
        # BOS (Atlantic) vs CLE (Central) -- different divisions, h2h split
        games = []
        # h2h split
        games.append(make_game("BOS", "CLE", True))
        games.append(make_game("CLE", "BOS", True))

        # BOS: more conference wins (plays against EC teams)
        for _ in range(30):
            games.append(make_game("BOS", "MIA", True))
        # CLE: fewer conference wins (plays against WC teams)
        for _ in range(20):
            games.append(make_game("CLE", "ATL", True))
        for _ in range(10):
            games.append(make_game("CLE", "LAL", True))

        # Pad to equal overall wins (32 each after h2h)
        for _ in range(1):
            games.append(make_game("BOS", "LAL", True))
        for _ in range(1):
            games.append(make_game("CLE", "LAL", True))

        record = record_from_games(games)
        assert record["BOS"][0] == record["CLE"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        # BOS has 31 conference wins (vs MIA + CLE h2h), CLE has 21 (vs ATL + BOS h2h)
        assert stats["BOS"]["conference_wins"] > stats["CLE"]["conference_wins"]

        result = season._break_two_team_tie(
            "BOS", "CLE", stats, Season.EASTERN_CONFERENCE, record
        )
        assert result[0] == "BOS"


# -----------------------------------------------------------------------
# 4. Two-team tie, resolved by net point differential (last real tiebreaker)
# -----------------------------------------------------------------------
class TestTwoTeamPointDifferential:
    def test_point_diff_breaks_tie(self):
        # BOS and NYK: h2h split, same division so div winner matters but
        # both are div winners (tied), same conf record, same eligible records.
        # Only point differential differs.
        games = []
        # h2h split
        games.append(make_game("BOS", "NYK", True, margin=1))
        games.append(make_game("NYK", "BOS", True, margin=1))

        # Both play only vs LAL (non-conference) to avoid conf record difference
        # BOS wins by big margins, NYK wins by small margins
        for _ in range(10):
            games.append(make_game("BOS", "LAL", True, margin=20))
        for _ in range(10):
            games.append(make_game("NYK", "LAL", True, margin=2))

        record = record_from_games(games)
        assert record["BOS"][0] == record["NYK"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)
        assert stats["BOS"]["net_point_differential"] > stats["NYK"]["net_point_differential"]

        result = season._break_two_team_tie(
            "BOS", "NYK", stats, Season.EASTERN_CONFERENCE, record
        )
        assert result[0] == "BOS"


# -----------------------------------------------------------------------
# 5. Three-team tie, division winner separates first, then remaining 2
#    restart from step 1
# -----------------------------------------------------------------------
class TestThreeTeamDivisionWinner:
    def test_div_winner_separates_first(self):
        # BOS (Atlantic), NYK (Atlantic), MIA (Southeast) -- all 50 wins
        # BOS is Atlantic div winner, NYK is not (PHI has 51 wins), MIA is
        # Southeast div winner
        games = []

        # h2h: BOS beats MIA, MIA beats NYK, NYK beats BOS (circular)
        games.append(make_game("BOS", "MIA", True))
        games.append(make_game("MIA", "NYK", True))
        games.append(make_game("NYK", "BOS", True))

        # PHI has 51 wins so NYK (50) is NOT the Atlantic div winner
        for _ in range(51):
            games.append(make_game("PHI", "LAL", True))

        # MIA is Southeast div winner (other SE teams have fewer wins)
        for _ in range(20):
            games.append(make_game("ORL", "LAL", True))

        # Pad BOS, NYK, MIA each to 50 wins
        for _ in range(49):
            games.append(make_game("BOS", "LAL", True))
        for _ in range(49):
            games.append(make_game("NYK", "LAL", True))
        for _ in range(49):
            games.append(make_game("MIA", "LAL", True))

        record = record_from_games(games)
        assert record["BOS"][0] == 50
        assert record["NYK"][0] == 50
        assert record["MIA"][0] == 50

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        # BOS is NOT Atlantic div winner (PHI has 51), but we need BOS to
        # be div winner -- let's make PHI have fewer wins instead.
        # Actually, _is_division_winner checks only teams present in
        # record_by_team.  PHI has 51 wins so BOS (50) is not div winner.
        # Fix: give PHI fewer wins.

        # Restart with correct setup
        games2 = []
        games2.append(make_game("BOS", "MIA", True))
        games2.append(make_game("MIA", "NYK", True))
        games2.append(make_game("NYK", "BOS", True))

        # Atlantic rivals: PHI=40, TOR=30, BRK=25 (all below BOS=50 and NYK=50)
        # But NYK must NOT be div winner. Since BOS and NYK are both at 50,
        # they are co-leaders. _is_division_winner returns True for both
        # because no one has MORE wins. So we need BOS at 50, NYK at 50,
        # but another Atlantic team at 51 to block NYK... but that blocks
        # BOS too.
        #
        # Instead: make BOS=50 and give NYK fewer intra-division standing
        # by making another Atlantic team (PHI) also have 50 wins. Then
        # _is_division_winner returns True for BOS, PHI, and NYK since
        # they're all tied. That won't work either.
        #
        # The _is_division_winner check returns True if no one in the
        # division has MORE wins. With BOS=50 and NYK=50, both return True.
        # To make only BOS the winner, we'd need NYK < BOS in wins, but
        # they're tied -- that's the whole point.
        #
        # For a 3-team tie where division winner separates, we need teams
        # from DIFFERENT divisions. Let's use BOS (Atlantic), CLE (Central),
        # MIA (Southeast). BOS and MIA are division winners, CLE is not
        # (IND has more wins).

        games3 = []
        games3.append(make_game("BOS", "MIA", True))
        games3.append(make_game("MIA", "CLE", True))
        games3.append(make_game("CLE", "BOS", True))

        # IND (Central) has 51 wins, so CLE (50) is NOT Central div winner
        for _ in range(51):
            games3.append(make_game("IND", "LAL", True))

        # Other Atlantic teams below 50
        for _ in range(30):
            games3.append(make_game("PHI", "LAL", True))

        # Other Southeast teams below 50
        for _ in range(30):
            games3.append(make_game("ORL", "LAL", True))

        # Pad BOS, CLE, MIA each to 50 wins
        for _ in range(49):
            games3.append(make_game("BOS", "LAL", True))
        for _ in range(49):
            games3.append(make_game("CLE", "LAL", True))
        for _ in range(49):
            games3.append(make_game("MIA", "LAL", True))

        record3 = record_from_games(games3)
        assert record3["BOS"][0] == 50
        assert record3["CLE"][0] == 50
        assert record3["MIA"][0] == 50

        season3 = make_season(games3)
        stats3 = season3._compute_tiebreaker_stats(record3)

        assert season3._is_division_winner("BOS", record3)
        assert season3._is_division_winner("MIA", record3)
        assert not season3._is_division_winner("CLE", record3)

        result = season3._break_multi_team_tie(
            ["BOS", "CLE", "MIA"], stats3, Season.EASTERN_CONFERENCE, record3
        )
        # CLE should be last (not a division winner)
        assert result[-1] == "CLE"
        # BOS and MIA are div winners, their relative order decided by
        # recursive two-team tiebreaker (h2h: BOS beat MIA)
        assert result[0] == "BOS"
        assert result[1] == "MIA"


# -----------------------------------------------------------------------
# 6. Three-team tie, h2h skipped (unbalanced schedule), falls to
#    conference record
# -----------------------------------------------------------------------
class TestThreeTeamUnbalancedH2H:
    def test_h2h_skipped_unbalanced(self):
        # BOS played NYK 4 times, but BOS played CLE only 2 times
        games = []
        # BOS vs NYK: 4 games
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", True))
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", False))
        # BOS vs CLE: 2 games
        games.append(make_game("BOS", "CLE", True))
        games.append(make_game("CLE", "BOS", False))
        # NYK vs CLE: 3 games (unbalanced with both others)
        games.append(make_game("NYK", "CLE", True))
        games.append(make_game("CLE", "NYK", True))
        games.append(make_game("NYK", "CLE", False))

        # CLE has best conference record via extra EC games
        for _ in range(20):
            games.append(make_game("CLE", "MIA", True))
        # NYK has middle conference record
        for _ in range(12):
            games.append(make_game("NYK", "MIA", True))
        # BOS has worst conference record (pad with WC games)
        for _ in range(5):
            games.append(make_game("BOS", "MIA", True))
        for _ in range(13):
            games.append(make_game("BOS", "LAL", True))

        # Equalize total wins: all should have same wins
        record = record_from_games(games)
        # Adjust so all three have the same wins
        bos_w, nyk_w, cle_w = record["BOS"][0], record["NYK"][0], record["CLE"][0]
        target = max(bos_w, nyk_w, cle_w)
        for _ in range(target - bos_w):
            games.append(make_game("BOS", "LAL", True))
        for _ in range(target - nyk_w):
            games.append(make_game("NYK", "LAL", True))
        for _ in range(target - cle_w):
            games.append(make_game("CLE", "LAL", True))

        record = record_from_games(games)
        assert record["BOS"][0] == record["NYK"][0] == record["CLE"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        # h2h should be unbalanced (BOS-NYK=4, BOS-CLE=2, NYK-CLE=3)
        assert not season._all_played_equal_times(["BOS", "NYK", "CLE"], stats)

        # None are division winners (or all are -- doesn't matter, they're
        # in different divisions), so that step won't separate.
        # Falls to conference record. CLE has most conf wins.
        result = season._break_multi_team_tie(
            ["BOS", "NYK", "CLE"], stats, Season.EASTERN_CONFERENCE, record
        )
        assert result[0] == "CLE"


# -----------------------------------------------------------------------
# 7. Three-team tie, balanced h2h resolves it
# -----------------------------------------------------------------------
class TestThreeTeamBalancedH2H:
    def test_balanced_h2h_resolves(self):
        # All three play each other exactly 2 games
        # BOS beats NYK 2-0, BOS beats CLE 2-0, NYK beats CLE 1-1
        # None are division winners (different divisions)
        games = []
        games.append(make_game("BOS", "NYK", True))
        games.append(make_game("NYK", "BOS", False))  # BOS wins
        games.append(make_game("BOS", "CLE", True))
        games.append(make_game("CLE", "BOS", False))  # BOS wins
        games.append(make_game("NYK", "CLE", True))
        games.append(make_game("CLE", "NYK", True))   # CLE wins 1

        # h2h wins: BOS=4, NYK=1, CLE=1
        # Pad so all three reach 24 total wins
        for _ in range(20):
            games.append(make_game("BOS", "LAL", True))
        for _ in range(23):
            games.append(make_game("NYK", "LAL", True))
        for _ in range(23):
            games.append(make_game("CLE", "LAL", True))

        record = record_from_games(games)
        assert record["BOS"][0] == record["NYK"][0] == record["CLE"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        assert season._all_played_equal_times(["BOS", "NYK", "CLE"], stats)

        # BOS: 4 h2h wins, NYK: 1 h2h win, CLE: 1 h2h win
        result = season._break_multi_team_tie(
            ["BOS", "NYK", "CLE"], stats, Season.EASTERN_CONFERENCE, record
        )
        assert result[0] == "BOS"


# -----------------------------------------------------------------------
# 8. No ties: 15 teams with distinct win totals
# -----------------------------------------------------------------------
class TestNoTies:
    def test_distinct_wins_no_tiebreaker(self):
        games = []
        ec_teams = list(Season.EASTERN_CONFERENCE)
        # Give each team a different number of wins
        for i, team in enumerate(ec_teams):
            wins = 60 - i * 3  # 60, 57, 54, ..., 18
            for _ in range(wins):
                games.append(make_game(team, "LAL", True))
            for _ in range(82 - wins):
                games.append(make_game("LAL", team, True))

        # Need WC teams too for full standings
        wc_teams = list(Season.WESTERN_CONFERENCE)
        for i, team in enumerate(wc_teams):
            wins = 55 - i * 2
            for _ in range(wins):
                games.append(make_game(team, "BOS", True))
            for _ in range(82 - wins):
                games.append(make_game("BOS", team, True))

        record = record_from_games(games)
        season = make_season(games)
        ec_df, wc_df = season.get_playoff_standings(record)

        # Seeds should be in descending order of wins
        ec_wins = ec_df["wins"].tolist()
        assert ec_wins == sorted(ec_wins, reverse=True)
        assert list(ec_df["seed"]) == list(range(1, len(ec_df) + 1))

        wc_wins = wc_df["wins"].tolist()
        assert wc_wins == sorted(wc_wins, reverse=True)
        assert list(wc_df["seed"]) == list(range(1, len(wc_df) + 1))


# -----------------------------------------------------------------------
# 9. Random fallback: two teams identical in every metric
# -----------------------------------------------------------------------
class TestRandomFallback:
    def test_identical_teams_no_crash(self):
        # Two teams with identical records, no h2h games, no conference
        # games, identical point differential (0)
        games = []
        # BOS and NYK each beat LAL 10 times with same margin
        for _ in range(10):
            games.append(make_game("BOS", "LAL", True, margin=5))
        for _ in range(10):
            games.append(make_game("NYK", "LAL", True, margin=5))

        record = record_from_games(games)
        assert record["BOS"][0] == record["NYK"][0]

        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        # Should not crash; result is random but valid
        result = season._break_ties(["BOS", "NYK"], stats, Season.EASTERN_CONFERENCE, record)
        assert set(result) == {"BOS", "NYK"}
        assert len(result) == 2


# -----------------------------------------------------------------------
# 10. Full get_playoff_standings integration test
# -----------------------------------------------------------------------
class TestGetPlayoffStandingsIntegration:
    def test_output_schema_and_seeds(self):
        games = []
        ec_teams = list(Season.EASTERN_CONFERENCE)
        wc_teams = list(Season.WESTERN_CONFERENCE)

        # Give each team different wins to avoid ties (simplifies assertion)
        for i, team in enumerate(ec_teams):
            wins = 65 - i * 3
            losses = 82 - wins
            for w in range(wins):
                opp = wc_teams[w % len(wc_teams)]
                games.append(make_game(team, opp, True))
            for l in range(losses):
                opp = wc_teams[l % len(wc_teams)]
                games.append(make_game(opp, team, True))

        for i, team in enumerate(wc_teams):
            wins = 62 - i * 3
            losses = 82 - wins
            for w in range(wins):
                opp = ec_teams[w % len(ec_teams)]
                games.append(make_game(team, opp, True))
            for l in range(losses):
                opp = ec_teams[l % len(ec_teams)]
                games.append(make_game(opp, team, True))

        record = record_from_games(games)
        season = make_season(games)
        ec_df, wc_df = season.get_playoff_standings(record)

        # Check schema
        assert "team" in ec_df.columns
        assert "wins" in ec_df.columns
        assert "losses" in ec_df.columns
        assert "seed" in ec_df.columns

        # Check all EC teams present
        assert set(ec_df["team"]) == set(ec_teams)
        # Check all WC teams present
        assert set(wc_df["team"]) == set(wc_teams)

        # Seeds are 1..15
        assert list(ec_df["seed"]) == list(range(1, 16))
        assert list(wc_df["seed"]) == list(range(1, 16))

        # Wins are non-increasing (descending order)
        ec_wins = ec_df["wins"].tolist()
        assert ec_wins == sorted(ec_wins, reverse=True)

    def test_tied_teams_get_unique_seeds(self):
        """Two teams with identical records still get distinct seeds."""
        games = []
        ec_teams = list(Season.EASTERN_CONFERENCE)
        wc_teams = list(Season.WESTERN_CONFERENCE)

        # First two EC teams tied at 50 wins
        for i, team in enumerate(ec_teams[:2]):
            for _ in range(50):
                games.append(make_game(team, wc_teams[0], True))
            for _ in range(32):
                games.append(make_game(wc_teams[0], team, True))

        # Rest of EC teams get distinct wins
        for i, team in enumerate(ec_teams[2:], start=2):
            wins = 48 - i
            for _ in range(wins):
                games.append(make_game(team, wc_teams[0], True))
            for _ in range(82 - wins):
                games.append(make_game(wc_teams[0], team, True))

        # WC teams
        for i, team in enumerate(wc_teams):
            wins = 55 - i * 2
            for _ in range(wins):
                games.append(make_game(team, ec_teams[0], True))
            for _ in range(82 - wins):
                games.append(make_game(ec_teams[0], team, True))

        record = record_from_games(games)
        season = make_season(games)
        ec_df, wc_df = season.get_playoff_standings(record)

        # All seeds unique
        assert len(ec_df["seed"].unique()) == len(ec_df)
        assert len(wc_df["seed"].unique()) == len(wc_df)


# -----------------------------------------------------------------------
# Helper method unit tests
# -----------------------------------------------------------------------
class TestHelperMethods:
    def test_compute_tiebreaker_stats(self):
        games = [
            make_game("BOS", "NYK", True, margin=10),
            make_game("NYK", "BOS", True, margin=5),
            make_game("BOS", "LAL", True, margin=15),
        ]
        record = record_from_games(games)
        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)

        assert stats["BOS"]["head_to_head"]["NYK"]["wins"] == 1
        assert stats["BOS"]["head_to_head"]["NYK"]["losses"] == 1
        assert stats["BOS"]["head_to_head"]["NYK"]["games"] == 2
        assert stats["NYK"]["head_to_head"]["BOS"]["wins"] == 1
        assert stats["NYK"]["head_to_head"]["BOS"]["losses"] == 1
        # BOS net PD: +10 (beat NYK at home) - 5 (lost to NYK away) + 15 (beat LAL) = +20
        assert stats["BOS"]["net_point_differential"] == 20.0
        # NYK net PD: -10 + 5 = -5
        assert stats["NYK"]["net_point_differential"] == -5.0
        # Conference: BOS-NYK are both EC, so those games count
        assert stats["BOS"]["conference_wins"] == 1
        assert stats["BOS"]["conference_losses"] == 1
        # BOS-LAL is cross-conference, doesn't count
        assert stats["LAL"]["conference_wins"] == 0

    def test_is_division_winner(self):
        record = {"BOS": [50, 32], "NYK": [45, 37], "PHI": [40, 42],
                  "TOR": [35, 47], "BRK": [30, 52]}
        season = make_season([])
        assert season._is_division_winner("BOS", record)
        assert not season._is_division_winner("NYK", record)

    def test_all_played_equal_times(self):
        games = [
            make_game("BOS", "NYK", True),
            make_game("NYK", "BOS", True),
            make_game("BOS", "CLE", True),
            make_game("CLE", "BOS", True),
            make_game("NYK", "CLE", True),
            make_game("CLE", "NYK", True),
        ]
        record = record_from_games(games)
        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)
        assert season._all_played_equal_times(["BOS", "NYK", "CLE"], stats)

        # Add one more BOS-NYK game to make it unbalanced
        games.append(make_game("BOS", "NYK", True))
        record2 = record_from_games(games)
        season2 = make_season(games)
        stats2 = season2._compute_tiebreaker_stats(record2)
        assert not season2._all_played_equal_times(["BOS", "NYK", "CLE"], stats2)

    def test_get_playoff_eligible_teams(self):
        record = {t: [60 - i * 3, 22 + i * 3] for i, t in enumerate(Season.EASTERN_CONFERENCE)}
        eligible = Season._get_playoff_eligible_teams(Season.EASTERN_CONFERENCE, record)
        assert len(eligible) == 10
        # Should be the top 10 by wins
        assert eligible[0] == Season.EASTERN_CONFERENCE[0]

    def test_get_record_vs_teams(self):
        games = [
            make_game("BOS", "NYK", True),
            make_game("BOS", "CLE", True),
            make_game("BOS", "MIA", False),
        ]
        record = record_from_games(games)
        season = make_season(games)
        stats = season._compute_tiebreaker_stats(record)
        w, l = Season._get_record_vs_teams("BOS", {"NYK", "CLE", "MIA"}, stats)
        assert w == 2
        assert l == 1
