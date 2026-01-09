import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="NBA Simulation Explorer", layout="wide")
st.title("NBA Simulation Explorer")


@st.cache_data
def load_data():
    df = pd.read_csv("data/sim_results/sim_game_results.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df


@st.cache_data
def load_win_totals():
    return pd.read_csv("data/sim_results/sim_raw_results.csv")


df = load_data()
win_totals_df = load_win_totals()

# Color palette for teams (modern, distinguishable)
COLORS = [
    "#4C78A8",  # steel blue
    "#E45756",  # coral red
    "#72B7B2",  # teal
    "#F58518",  # orange
    "#54A24B",  # green
    "#B279A2",  # purple
]


def get_team_ratings(df, team):
    """Get team's rating from all games (home or away)."""
    home = df[df["team"] == team][["simulation_id", "date", "team_rating"]].copy()
    home.columns = ["simulation_id", "date", "rating"]

    away = df[df["opponent"] == team][
        ["simulation_id", "date", "opponent_rating"]
    ].copy()
    away.columns = ["simulation_id", "date", "rating"]

    return pd.concat([home, away]).sort_values("date")


def get_team_games(df, team):
    """Get all games for a team with margin from team's perspective."""
    home = df[df["team"] == team].copy()
    home["is_home"] = True
    home["opp"] = home["opponent"]
    home["margin_for_team"] = home["margin"]
    home["expected_margin_for_team"] = home["expected_margin"]
    home["team_rating_val"] = home["team_rating"]

    away = df[df["opponent"] == team].copy()
    away["is_home"] = False
    away["opp"] = away["team"]
    away["margin_for_team"] = -away["margin"]
    away["expected_margin_for_team"] = -away["expected_margin"]
    away["team_rating_val"] = away["opponent_rating"]

    return pd.concat([home, away]).sort_values("date")


# Sidebar
st.sidebar.header("Filters")
teams = sorted(set(df["team"].unique()) | set(df["opponent"].unique()))
selected_teams = st.sidebar.multiselect(
    "Select Teams (up to 4)", teams, default=["OKC", "BOS"], max_selections=4
)

st.sidebar.subheader("Confidence Band")
pct_low = st.sidebar.slider("Lower percentile", 5, 25, 10)
pct_high = st.sidebar.slider("Upper percentile", 75, 95, 90)

include_playoffs = st.sidebar.checkbox("Include playoffs", value=True)

# Filter data
if not include_playoffs:
    df_filtered = df[df["playoff"] == 0]
else:
    df_filtered = df

# View 1: Team Rating Over Time
st.header("Team Rating Over Time")

if not selected_teams:
    st.warning("Please select at least one team from the sidebar.")
else:
    fig = go.Figure()

    for i, team in enumerate(selected_teams[:4]):
        team_data = get_team_ratings(df_filtered, team)
        if team_data.empty:
            continue

        stats = (
            team_data.groupby("date")["rating"]
            .agg(
                [
                    ("mean", "mean"),
                    ("low", lambda x: np.percentile(x, pct_low)),
                    ("high", lambda x: np.percentile(x, pct_high)),
                    ("n", "count"),
                ]
            )
            .reset_index()
        )

        color = COLORS[i % len(COLORS)]
        # Parse hex to rgb for fill
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)

        # Confidence band (filled area)
        fig.add_trace(
            go.Scatter(
                x=pd.concat([stats["date"], stats["date"][::-1]]),
                y=pd.concat([stats["high"], stats["low"][::-1]]),
                fill="toself",
                fillcolor=f"rgba({r},{g},{b},0.2)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{team} ({pct_low}-{pct_high}%)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Mean line
        stats["rating_rounded"] = stats["mean"].round(1)
        stats["date_str"] = stats["date"].dt.strftime("%b %d")
        fig.add_trace(
            go.Scatter(
                x=stats["date"],
                y=stats["mean"],
                mode="lines",
                name=team,
                line=dict(color=color, width=2.5),
                customdata=np.stack([stats["date_str"], stats["rating_rounded"]], axis=-1),
                hovertemplate=f"<b>{team}</b><br>%{{customdata[0]}}<br>Rating: %{{customdata[1]}}<extra></extra>",
            )
        )

    # Add vertical line for end of regular season
    regular_season_end = df[df["playoff"] == 0]["date"].max()
    fig.add_shape(
        type="line",
        x0=regular_season_end,
        x1=regular_season_end,
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color="rgba(0,0,0,0.4)", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=regular_season_end,
        y=1,
        yref="paper",
        text="Playoffs",
        showarrow=False,
        font=dict(size=10, color="rgba(0,0,0,0.5)"),
        yshift=10,
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Team Rating",
        height=500,
        hovermode="x unified",
        legend=dict(
            yanchor="bottom",
            y=0.01,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="rgba(0,0,0,0.1)",
            borderwidth=1,
        ),
        plot_bgcolor="white",
        yaxis=dict(
            gridcolor="rgba(0,0,0,0.1)",
            zerolinecolor="rgba(0,0,0,0.2)",
            range=[-15, 15],
        ),
        xaxis=dict(
            gridcolor="rgba(0,0,0,0.05)",
            dtick="M1",
            tickformat="%b",
            ticklabelmode="period",
            showgrid=True,
            ticks="outside",
            ticklen=8,
            minor=dict(
                dtick=7 * 24 * 60 * 60 * 1000,
                ticks="outside",
                ticklen=4,
                showgrid=False,
            ),
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

# View 2: Predicted Margin Over Time
st.header("Predicted Margin Over Time")

if not selected_teams:
    st.warning("Please select at least one team from the sidebar.")
else:
    # Let user pick one team for margin timeline
    margin_timeline_team = st.selectbox(
        "Select team for margin timeline",
        selected_teams,
        key="margin_timeline_team",
    )

    # Regular season only for this view
    team_games_timeline = get_team_games(df, margin_timeline_team)
    team_games_timeline = team_games_timeline[team_games_timeline["playoff"] == 0]

    if team_games_timeline.empty:
        st.warning(f"No regular season games found for {margin_timeline_team}")
    else:
        # Calculate stats per game (using expected margin - no noise)
        margin_stats = (
            team_games_timeline.groupby(["date", "opp", "is_home"])
            .agg(
                mean_margin=("expected_margin_for_team", "mean"),
                low_margin=("expected_margin_for_team", lambda x: np.percentile(x, pct_low)),
                high_margin=("expected_margin_for_team", lambda x: np.percentile(x, pct_high)),
                n=("simulation_id", "count"),
            )
            .reset_index()
        )
        margin_stats = margin_stats.sort_values("date").reset_index(drop=True)
        # Round for cleaner display
        margin_stats["mean_margin"] = margin_stats["mean_margin"].round(1)
        margin_stats["low_margin"] = margin_stats["low_margin"].round(1)
        margin_stats["high_margin"] = margin_stats["high_margin"].round(1)
        margin_stats["game_label"] = margin_stats.apply(
            lambda r: f"{'vs' if r['is_home'] else '@'} {r['opp']}", axis=1
        )

        fig_margin_time = go.Figure()

        # Color based on expected outcome: green=favored, red=underdog
        colors = [
            "#2ecc71" if m > 0 else "#e74c3c" for m in margin_stats["mean_margin"]
        ]

        # Error bar colors match point colors
        error_colors = colors.copy()

        # Mean points with error bars showing percentile range
        fig_margin_time.add_trace(
            go.Scatter(
                x=margin_stats["date"],
                y=margin_stats["mean_margin"],
                mode="markers",
                name=margin_timeline_team,
                marker=dict(
                    size=10,
                    color=colors,
                    line=dict(width=1, color="white"),
                ),
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=margin_stats["high_margin"] - margin_stats["mean_margin"],
                    arrayminus=margin_stats["mean_margin"] - margin_stats["low_margin"],
                    thickness=1.5,
                    width=3,
                ),
                customdata=margin_stats["game_label"],
                hovertemplate="<b>%{customdata}</b><br>"
                + "%{x|%b %d}<br>"
                + "Expected Margin: %{y:,.1f}<extra></extra>",
            )
        )

        # Zero line (win/loss threshold)
        fig_margin_time.add_hline(
            y=0, line_dash="solid", line_color="rgba(0,0,0,0.3)", line_width=2
        )

        fig_margin_time.update_layout(
            xaxis_title="",
            yaxis_title="Expected Margin",
            height=500,
            hovermode="closest",
            showlegend=False,
            plot_bgcolor="white",
            yaxis=dict(
                gridcolor="rgba(0,0,0,0.1)",
                zerolinecolor="rgba(0,0,0,0.3)",
                zerolinewidth=2,
            ),
            xaxis=dict(
                gridcolor="rgba(0,0,0,0.05)",
                dtick="M1",  # Tick every month
                tickformat="%b",  # Show month name (Jan, Feb, etc.)
                ticklabelmode="period",
                showgrid=True,
                ticks="outside",
                ticklen=8,
                minor=dict(
                    dtick=7 * 24 * 60 * 60 * 1000,  # Weekly minor ticks (ms)
                    ticks="outside",
                    ticklen=4,
                    showgrid=False,
                ),
            ),
        )

        st.plotly_chart(fig_margin_time, use_container_width=True)

# View 3: Regular Season Win Totals Distribution
st.header("Regular Season Win Totals")

if selected_teams:
    wins_team = st.selectbox(
        "Select team for win total distribution",
        selected_teams,
        key="wins_team",
    )

    if wins_team in win_totals_df.columns:
        wins = win_totals_df[wins_team]

        # Calculate distribution manually for custom hover
        win_counts = wins.value_counts().sort_index()
        all_wins = range(int(wins.min()), int(wins.max()) + 1)

        pct = []
        cum_pct_lt = []  # P(X < x)
        cum_pct_gte = []  # P(X >= x)
        total = len(wins)
        running_sum = 0

        for w in all_wins:
            cum_pct_lt.append(100 * running_sum / total)
            cum_pct_gte.append(100 * (total - running_sum) / total)
            count = win_counts.get(w, 0)
            p = 100 * count / total
            running_sum += count
            pct.append(p)

        fig_wins = go.Figure()

        fig_wins.add_trace(
            go.Bar(
                x=list(all_wins),
                y=pct,
                marker_color="#5DADE2",
                marker_line_color="rgba(255,255,255,0.4)",
                marker_line_width=0.5,
                customdata=np.stack([cum_pct_lt, cum_pct_gte], axis=-1),
                hovertemplate="<b>%{x} wins</b><br>"
                + "P(X < %{x}): %{customdata[0]:.1f}%<br>"
                + "P(X ≥ %{x}): %{customdata[1]:.1f}%<extra></extra>",
            )
        )

        mean_wins = wins.mean()
        std_wins = wins.std()

        fig_wins.add_vline(
            x=mean_wins,
            line_dash="solid",
            line_color=COLORS[1],
            line_width=2,
            annotation_text=f"Mean: {mean_wins:.1f}",
            annotation_position="top",
        )

        fig_wins.update_layout(
            xaxis_title="Wins",
            yaxis_title="Percent of Simulations",
            height=400,
            plot_bgcolor="white",
            xaxis=dict(gridcolor="rgba(0,0,0,0.1)", dtick=5, range=[0, 82]),
            yaxis=dict(gridcolor="rgba(0,0,0,0.1)"),
            bargap=0.1,
        )

        st.plotly_chart(fig_wins, use_container_width=True)

        # Summary stats
        median_wins = np.median(wins)
        p10 = np.percentile(wins, 10)
        p90 = np.percentile(wins, 90)
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Mean Wins", f"{mean_wins:.1f}")
        col2.metric("Median", f"{median_wins:.0f}")
        col3.metric("10th Percentile", f"{p10:.0f}")
        col4.metric("90th Percentile", f"{p90:.0f}")
    else:
        st.warning(f"No win total data found for {wins_team}")

# View 4: Game Margin Distributions
st.header("Game Margin Distributions")

if selected_teams:
    margin_team = st.selectbox("Select team for game details", selected_teams)

    team_games = get_team_games(df_filtered, margin_team)

    if team_games.empty:
        st.warning(f"No games found for {margin_team}")
    else:
        # Build game options
        games_summary = (
            team_games.groupby(["date", "opp", "is_home"])
            .agg(
                {
                    "margin_for_team": ["mean", "std"],
                    "simulation_id": "count",
                }
            )
            .reset_index()
        )
        games_summary.columns = [
            "date",
            "opp",
            "is_home",
            "mean_margin",
            "std_margin",
            "n_sims",
        ]
        games_summary = games_summary.sort_values("date")

        game_options = []
        for _, row in games_summary.iterrows():
            loc = "vs" if row["is_home"] else "@"
            label = f"{row['date'].strftime('%Y-%m-%d')} {loc} {row['opp']}"
            game_options.append(label)

        selected_game_idx = st.selectbox(
            "Select game",
            range(len(game_options)),
            format_func=lambda x: game_options[x],
        )

        if selected_game_idx is not None:
            game_row = games_summary.iloc[selected_game_idx]
            game_date = game_row["date"]
            game_opp = game_row["opp"]
            game_is_home = game_row["is_home"]

            # Filter to this specific game
            game_data = team_games[
                (team_games["date"] == game_date)
                & (team_games["opp"] == game_opp)
                & (team_games["is_home"] == game_is_home)
            ]

            if not game_data.empty:
                col1, col2 = st.columns(2)

                with col1:
                    # Margin histogram
                    fig_margin = go.Figure()
                    fig_margin.add_trace(
                        go.Histogram(
                            x=game_data["margin_for_team"],
                            nbinsx=30,
                            name="Simulated Margin",
                            marker_color="#1f77b4",
                        )
                    )
                    # Vertical line at 0
                    fig_margin.add_vline(
                        x=0, line_dash="dash", line_color="red", line_width=2
                    )

                    mean_margin = game_data["margin_for_team"].mean()
                    fig_margin.add_vline(
                        x=mean_margin,
                        line_dash="solid",
                        line_color="green",
                        line_width=2,
                        annotation_text=f"Mean: {mean_margin:.1f}",
                        annotation_position="top",
                    )

                    fig_margin.update_layout(
                        title=f"{margin_team} Margin Distribution",
                        xaxis_title=f"Margin (positive = {margin_team} wins)",
                        yaxis_title="Count",
                        height=400,
                    )
                    st.plotly_chart(fig_margin, use_container_width=True)

                with col2:
                    # Summary stats
                    win_pct = (game_data["margin_for_team"] > 0).mean() * 100
                    mean_m = game_data["margin_for_team"].mean()
                    std_m = game_data["margin_for_team"].std()
                    mean_exp = game_data["expected_margin_for_team"].mean()
                    n_sims = len(game_data)

                    st.subheader("Summary Statistics")
                    st.metric("Win Probability", f"{win_pct:.1f}%")
                    st.metric("Mean Margin", f"{mean_m:+.1f}")
                    st.metric("Std Dev", f"{std_m:.1f}")
                    st.metric("Expected Margin (pre-noise)", f"{mean_exp:+.1f}")
                    st.metric("Simulations", f"{n_sims}")

                # Rating distribution for this game
                st.subheader(f"{margin_team} Rating at Game Time")
                fig_rating = go.Figure()
                fig_rating.add_trace(
                    go.Histogram(
                        x=game_data["team_rating_val"],
                        nbinsx=30,
                        name="Team Rating",
                        marker_color="#2ca02c",
                    )
                )
                mean_rating = game_data["team_rating_val"].mean()
                fig_rating.add_vline(
                    x=mean_rating,
                    line_dash="solid",
                    line_color="darkgreen",
                    line_width=2,
                    annotation_text=f"Mean: {mean_rating:.1f}",
                    annotation_position="top",
                )
                fig_rating.update_layout(
                    xaxis_title="Team Rating",
                    yaxis_title="Count",
                    height=300,
                )
                st.plotly_chart(fig_rating, use_container_width=True)
