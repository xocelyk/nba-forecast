#!/usr/bin/env python3
"""
Kelly Portfolio Trading System for NBA Championship Futures.
Implements proper long/short portfolio rebalancing with weekly updates.
"""

import argparse
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass

import env


@dataclass
class Position:
    """Represents a long/short position in a championship contract."""
    team: str
    shares: float  # Positive = long, negative = short
    avg_entry_price: float  # Average price of position
    current_price: float  # Current market price
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.current_price
    
    @property
    def notional_value(self) -> float:
        """Absolute notional value of position."""
        return abs(self.shares * self.current_price)
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized P&L of position."""
        return self.shares * (self.current_price - self.avg_entry_price)


def american_odds_to_probability(odds: float) -> float:
    """Convert American odds to implied probability."""
    if pd.isna(odds) or odds == 0:
        return 0.0
    
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def remove_vig_from_odds(market_odds_series: pd.Series) -> pd.Series:
    """Remove vig from market odds using proportional method."""
    implied_probs = market_odds_series.apply(american_odds_to_probability)
    valid_probs = implied_probs[implied_probs > 0]
    
    if len(valid_probs) == 0:
        return implied_probs
    
    total_implied = valid_probs.sum()
    if total_implied > 0:
        fair_probs = valid_probs / total_implied
        result = implied_probs.copy()
        result[valid_probs.index] = fair_probs
        return result
    else:
        return implied_probs


def calculate_kelly_portfolio_weights(model_probs: pd.Series, fair_market_probs: pd.Series) -> pd.Series:
    """
    Calculate Kelly optimal portfolio weights for long/short positions.
    
    Returns:
        Series with Kelly weights (positive = long, negative = short)
    """
    # Calculate edges (model - fair market probability)
    edges = model_probs - fair_market_probs
    
    # Kelly weight proportional to edge
    # Only trade where we have significant edge (>0.5%)
    kelly_weights = pd.Series(0.0, index=model_probs.index)
    
    significant_edges = edges[abs(edges) > 0.005]  # 0.5% minimum edge
    
    if len(significant_edges) > 0:
        # Scale weights by edge magnitude
        total_abs_edge = significant_edges.abs().sum()
        if total_abs_edge > 0:
            kelly_weights[significant_edges.index] = significant_edges / total_abs_edge
    
    return kelly_weights


class KellyPortfolioTrader:
    """Portfolio-based Kelly trading system with long/short positions."""
    
    def __init__(
        self, 
        starting_capital: float = 1000.0,
        transaction_cost: float = 0.01,
        max_gross_exposure: float = 2.0,  # Max 200% gross exposure
        kelly_fraction: float = 1.0
    ):
        self.starting_capital = starting_capital
        self.transaction_cost = transaction_cost
        self.max_gross_exposure = max_gross_exposure
        self.kelly_fraction = kelly_fraction
        
        # Portfolio state
        self.cash = starting_capital
        self.positions: Dict[str, Position] = {}
        self.portfolio_history = []
        self.trades = []
        
        # Performance tracking
        self.total_transaction_costs = 0.0
    
    def get_portfolio_value(self) -> float:
        """Calculate total portfolio value (cash + position values)."""
        position_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + position_value
    
    def get_gross_exposure(self) -> float:
        """Calculate gross exposure (sum of absolute position values)."""
        return sum(pos.notional_value for pos in self.positions.values())
    
    def update_position_prices(self, fair_market_probs: pd.Series, team_mapping: Dict[str, str]):
        """Update current prices for all positions."""
        for team, position in self.positions.items():
            full_name = team_mapping.get(team, team)
            if full_name in fair_market_probs.index and fair_market_probs[full_name] > 0:
                position.current_price = fair_market_probs[full_name]
            else:
                # Team eliminated - contract expires worthless
                position.current_price = 0.0
    
    def rebalance_portfolio(
        self, 
        date: str,
        model_probs: pd.Series, 
        fair_market_probs: pd.Series,
        team_mapping: Dict[str, str]
    ):
        """Rebalance portfolio to Kelly optimal weights."""
        
        print(f"\n🔄 PORTFOLIO REBALANCING - {date}")
        print("=" * 60)
        
        # Update current prices
        self.update_position_prices(fair_market_probs, team_mapping)
        
        # Calculate current portfolio value
        portfolio_value = self.get_portfolio_value()
        
        # Align model and market data
        aligned_model_probs = pd.Series(dtype=float)
        aligned_market_probs = pd.Series(dtype=float)
        
        for team_abbrev, model_prob in model_probs.items():
            full_name = team_mapping.get(team_abbrev, team_abbrev)
            if full_name in fair_market_probs.index:
                aligned_model_probs[team_abbrev] = model_prob
                aligned_market_probs[team_abbrev] = fair_market_probs[full_name]
        
        # Calculate Kelly optimal weights
        kelly_weights = calculate_kelly_portfolio_weights(aligned_model_probs, aligned_market_probs)
        
        # Apply Kelly fraction and exposure limits
        kelly_weights = kelly_weights * self.kelly_fraction
        
        # Scale down if total exposure would exceed limits
        total_abs_weight = kelly_weights.abs().sum()
        if total_abs_weight > self.max_gross_exposure:
            kelly_weights = kelly_weights * (self.max_gross_exposure / total_abs_weight)
        
        print(f"Portfolio Value: ${portfolio_value:.2f}")
        print(f"Target Positions (Kelly {self.kelly_fraction}x):")
        
        # Show significant positions
        significant_weights = kelly_weights[abs(kelly_weights) > 0.01]
        for team, weight in significant_weights.sort_values(key=abs, ascending=False).items():
            model_prob = aligned_model_probs[team]
            market_prob = aligned_market_probs[team]
            edge = model_prob - market_prob
            direction = "LONG" if weight > 0 else "SHORT"
            print(f"  {team}: {direction} {abs(weight):.1%} (Edge: {edge:+.1%})")
        
        # Calculate target position values
        target_values = kelly_weights * portfolio_value
        
        # Execute trades to reach targets
        trades_executed = []
        
        # First, handle eliminated teams (close positions)
        eliminated_teams = []
        for team in list(self.positions.keys()):
            full_name = team_mapping.get(team, team)
            if full_name in fair_market_probs.index and fair_market_probs[full_name] <= 0:
                eliminated_teams.append(team)
        
        for team in eliminated_teams:
            position = self.positions[team]
            if position.shares != 0:
                print(f"  🚫 {team} eliminated - closing position ({position.shares:+.1f} shares)")
                # Position expires worthless
                pnl = -position.shares * position.avg_entry_price
                self.cash += pnl if position.shares < 0 else 0  # Return margin for short positions
                trades_executed.append(f"CLOSE {position.shares:+.1f} {team} (eliminated)")
                del self.positions[team]
        
        # Execute trades for active teams
        for team, target_value in target_values.items():
            current_value = self.positions.get(team, Position(team, 0, 0, 0)).market_value
            trade_value = target_value - current_value
            
            if abs(trade_value) < portfolio_value * 0.001:  # Skip tiny trades
                continue
            
            current_price = aligned_market_probs[team]
            if current_price <= 0:  # Skip eliminated teams
                continue
            
            shares_to_trade = trade_value / current_price
            
            if abs(shares_to_trade) > 0:
                self.execute_trade(team, shares_to_trade, current_price, date)
                direction = "BUY" if shares_to_trade > 0 else "SELL"
                trades_executed.append(f"{direction} {abs(shares_to_trade):.1f} {team}")
        
        # Record portfolio state
        self.portfolio_history.append({
            'date': date,
            'portfolio_value': self.get_portfolio_value(),
            'cash': self.cash,
            'gross_exposure': self.get_gross_exposure(),
            'num_positions': len(self.positions)
        })
        
        print(f"Trades: {len(trades_executed)}")
        for trade in trades_executed[:10]:  # Show first 10 trades
            print(f"  {trade}")
        if len(trades_executed) > 10:
            print(f"  ... and {len(trades_executed) - 10} more")
        
        final_portfolio_value = self.get_portfolio_value()
        print(f"Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"Cash: ${self.cash:.2f} | Gross Exposure: {self.get_gross_exposure()/final_portfolio_value:.1%}")
        
        # Print full portfolio holdings
        print(f"\n📊 FULL PORTFOLIO HOLDINGS:")
        print("-" * 80)
        if not self.positions:
            print("  No positions")
        else:
            # Sort positions by absolute value for better readability
            sorted_positions = sorted(self.positions.items(), 
                                    key=lambda x: abs(x[1].notional_value), 
                                    reverse=True)
            
            print(f"{'Team':4} {'Position':>8} {'Shares':>10} {'Price':>7} {'Value':>10} {'P&L':>10} {'% Port':>7}")
            print("-" * 80)
            
            total_long_value = 0
            total_short_value = 0
            
            for team, pos in sorted_positions:
                position_type = "LONG" if pos.shares > 0 else "SHORT"
                pnl = pos.unrealized_pnl
                pct_of_portfolio = (pos.market_value / final_portfolio_value) * 100
                
                print(f"{team:4} {position_type:>8} {pos.shares:>10.1f} {pos.current_price:>7.3f} "
                      f"${pos.market_value:>9.2f} ${pnl:>9.2f} {pct_of_portfolio:>6.1f}%")
                
                if pos.shares > 0:
                    total_long_value += pos.market_value
                else:
                    total_short_value += abs(pos.market_value)
            
            print("-" * 80)
            print(f"{'Total Long:':>35} ${total_long_value:>9.2f} {(total_long_value/final_portfolio_value)*100:>15.1f}%")
            print(f"{'Total Short:':>35} ${total_short_value:>9.2f} {(total_short_value/final_portfolio_value)*100:>15.1f}%")
            print(f"{'Net Exposure:':>35} ${total_long_value - total_short_value:>9.2f} "
                  f"{((total_long_value - total_short_value)/final_portfolio_value)*100:>15.1f}%")
            print(f"{'Cash:':>35} ${self.cash:>9.2f} {(self.cash/final_portfolio_value)*100:>15.1f}%")
    
    def execute_trade(self, team: str, shares: float, price: float, date: str):
        """Execute a trade and update portfolio."""
        trade_value = abs(shares * price)
        transaction_cost = trade_value * self.transaction_cost
        
        if team not in self.positions:
            self.positions[team] = Position(team, 0, 0, price)
        
        position = self.positions[team]
        
        if shares > 0:  # Buying (going long or covering short)
            if position.shares < 0:  # Covering short
                shares_covered = min(abs(position.shares), shares)
                # When covering a short, we pay current price
                # P&L = (entry_price - exit_price) * shares
                cost_to_cover = shares_covered * price
                self.cash -= cost_to_cover + transaction_cost
                
                position.shares += shares_covered
                shares -= shares_covered
                
                if position.shares == 0:
                    position.avg_entry_price = 0
            
            if shares > 0:  # Additional long position
                self.cash -= shares * price + transaction_cost
                if position.shares >= 0:
                    # Update average entry price for long positions
                    total_cost = position.shares * position.avg_entry_price + shares * price
                    position.shares += shares
                    position.avg_entry_price = total_cost / position.shares if position.shares > 0 else price
                else:
                    position.shares += shares
        
        else:  # Selling (going short or reducing long)
            shares = abs(shares)
            if position.shares > 0:  # Reducing long
                shares_sold = min(position.shares, shares)
                proceeds = shares_sold * price
                pnl = shares_sold * (price - position.avg_entry_price)
                self.cash += proceeds - transaction_cost
                
                position.shares -= shares_sold
                shares -= shares_sold
                
                if position.shares == 0:
                    position.avg_entry_price = 0
            
            if shares > 0:  # Short position
                # For shorts, we receive cash but post margin
                self.cash += shares * price - transaction_cost
                if position.shares <= 0:
                    # Update average entry price for short positions
                    total_margin = abs(position.shares) * position.avg_entry_price + shares * price
                    position.shares -= shares
                    position.avg_entry_price = total_margin / abs(position.shares) if position.shares != 0 else price
                else:
                    position.shares -= shares
        
        position.current_price = price
        
        # Clean up zero positions
        if abs(position.shares) < 0.001:
            del self.positions[team]
        
        # Record trade
        self.trades.append({
            'date': date,
            'team': team,
            'shares': shares if shares > 0 else -abs(shares),
            'price': price,
            'transaction_cost': transaction_cost
        })
        
        self.total_transaction_costs += transaction_cost
    
    def settle_championship(self, winner_team: str):
        """Settle all positions based on championship winner."""
        print(f"\n🏆 CHAMPIONSHIP SETTLEMENT - Winner: {winner_team}")
        print("=" * 60)
        
        total_pnl = 0.0
        
        for team, position in self.positions.items():
            if team == winner_team:
                # Winning contracts settle at $1.00
                if position.shares > 0:  # Long position on winner
                    # We receive $1 per share, but already paid entry price
                    pnl = position.shares * (1.0 - position.avg_entry_price)
                    total_pnl += pnl
                    print(f"  {team}: LONG {position.shares:.1f} shares @ ${position.avg_entry_price:.3f} → $1.00 = ${pnl:+.2f} ✅")
                else:  # Short position on winner (we lose)
                    # We received entry price but must pay $1 per share
                    pnl = position.shares * (position.avg_entry_price - 1.0)  # shares are negative
                    total_pnl += pnl
                    print(f"  {team}: SHORT {abs(position.shares):.1f} shares @ ${position.avg_entry_price:.3f} → $1.00 = ${pnl:+.2f} ❌")
            else:
                # Losing contracts expire at $0.00
                if position.shares > 0:  # Long position loses
                    # We paid entry price, receive nothing
                    pnl = -position.shares * position.avg_entry_price
                    total_pnl += pnl
                    print(f"  {team}: LONG {position.shares:.1f} shares @ ${position.avg_entry_price:.3f} → $0.00 = ${pnl:+.2f} ❌")
                else:  # Short position wins
                    # We received entry price, pay nothing
                    pnl = -position.shares * position.avg_entry_price  # shares are negative, so this is positive
                    total_pnl += pnl
                    print(f"  {team}: SHORT {abs(position.shares):.1f} shares @ ${position.avg_entry_price:.3f} → $0.00 = ${pnl:+.2f} ✅")
        
        final_cash = self.cash + total_pnl
        total_return = (final_cash / self.starting_capital - 1) * 100
        
        print(f"\nFinal Settlement:")
        print(f"  Starting Capital: ${self.starting_capital:.2f}")
        print(f"  Cash: ${self.cash:.2f}")
        print(f"  Championship P&L: ${total_pnl:.2f}")
        print(f"  Final Value: ${final_cash:.2f}")
        print(f"  Total Return: {total_return:+.1f}%")
        print(f"  Transaction Costs: ${self.total_transaction_costs:.2f}")
        
        return final_cash, total_return


def main():
    """Main function to run Kelly portfolio trading analysis."""
    parser = argparse.ArgumentParser(description="Kelly Portfolio Trading Analysis")
    parser.add_argument("--market-odds", type=str, default="nba_2025_odds_wide.csv")
    parser.add_argument("--model-data", type=str, default="aligned_model_predictions_2025.csv")
    parser.add_argument("--starting-capital", type=float, default=1000.0)
    parser.add_argument("--kelly-fractions", nargs="+", type=float, default=[0.1, 0.25, 0.5, 1.0])
    parser.add_argument("--winner", type=str, default="OKC")
    
    args = parser.parse_args()
    
    print("Kelly Portfolio Trading Analysis")
    print("=" * 60)
    
    try:
        # Load data
        market_data = pd.read_csv(args.market_odds)
        model_data = pd.read_csv(args.model_data)
        
        # Team name mapping
        team_mapping = {
            "ATL": "Atlanta Hawks", "BOS": "Boston Celtics", "BRK": "Brooklyn Nets",
            "CHO": "Charlotte Hornets", "CHI": "Chicago Bulls", "CLE": "Cleveland Cavaliers",
            "DAL": "Dallas Mavericks", "DEN": "Denver Nuggets", "DET": "Detroit Pistons",
            "GSW": "Golden State Warriors", "HOU": "Houston Rockets", "IND": "Indiana Pacers",
            "LAC": "Los Angeles Clippers", "LAL": "Los Angeles Lakers", "MEM": "Memphis Grizzlies",
            "MIA": "Miami Heat", "MIL": "Milwaukee Bucks", "MIN": "Minnesota Timberwolves",
            "NOP": "New Orleans Pelicans", "NYK": "New York Knicks", "OKC": "Oklahoma City Thunder",
            "ORL": "Orlando Magic", "PHI": "Philadelphia 76ers", "PHO": "Phoenix Suns",
            "POR": "Portland Trail Blazers", "SAC": "Sacramento Kings", "SAS": "San Antonio Spurs",
            "TOR": "Toronto Raptors", "UTA": "Utah Jazz", "WAS": "Washington Wizards"
        }
        
        # Available dates (excluding playoffs)
        date_order = ["Nov 1", "Dec 1", "Jan 1", "Feb 1", "All-Star Break", "Mar 1", "Apr 1"]
        available_dates = [d for d in date_order if d in model_data['market_date_name'].unique()]
        
        print(f"Regular season dates included: {available_dates}")
        print("Playoff dates excluded from analysis")
        
        # Run analysis for each Kelly fraction
        results = {}
        
        for kelly_frac in args.kelly_fractions:
            print(f"\n{'='*80}")
            print(f"KELLY PORTFOLIO TRADING - {kelly_frac}x Kelly")
            print(f"{'='*80}")
            
            trader = KellyPortfolioTrader(
                starting_capital=args.starting_capital,
                kelly_fraction=kelly_frac
            )
            
            # Portfolio rebalancing simulation
            for date in available_dates:
                # Get model predictions
                model_date_data = model_data[model_data['market_date_name'] == date]
                if model_date_data.empty:
                    continue
                
                model_probs = model_date_data.set_index('team')['champion']
                
                # Get market odds and remove vig
                if date not in market_data.columns:
                    continue
                
                market_odds_all_teams = market_data[date].copy()
                market_odds_all_teams.index = market_data['Team']
                fair_market_probs = remove_vig_from_odds(market_odds_all_teams)
                
                # Rebalance portfolio
                trader.rebalance_portfolio(date, model_probs, fair_market_probs, team_mapping)
            
            # Settle championship
            final_value, total_return = trader.settle_championship(args.winner)
            
            results[kelly_frac] = {
                'final_value': final_value,
                'total_return': total_return,
                'num_trades': len(trader.trades),
                'transaction_costs': trader.total_transaction_costs
            }
        
        # Comparative results
        print(f"\n{'='*80}")
        print("COMPARATIVE RESULTS")
        print(f"{'='*80}")
        
        for kelly_frac, result in results.items():
            print(f"{kelly_frac:4.1f}x Kelly: ${result['final_value']:7.2f} ({result['total_return']:+5.1f}%) | "
                  f"Trades: {result['num_trades']:3d} | Costs: ${result['transaction_costs']:5.2f}")
        
        best_kelly = max(results.keys(), key=lambda k: results[k]['total_return'])
        print(f"\n🏆 Best Strategy: {best_kelly}x Kelly with {results[best_kelly]['total_return']:+.1f}% return")
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()