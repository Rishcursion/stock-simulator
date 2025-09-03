import os
from collections import defaultdict
import warnings

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import linregress


def merge_stocks() -> DataFrame:
    data = {}
    for root, _, files in os.walk("data/merged"):
        for file in files:
            df: DataFrame = pd.read_csv(os.path.join(root, file))

            # Ensure Date column is in datetime format
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

            # Drop any rows where Date is missing or before 2000
            df = df[df["Date"] >= "2000-01-01"]

            data[os.path.basename(file)] = df

    final_data = pd.concat(data, names=["Ticker", "Index"])
    return final_data


class StockTradeEnv(gym.Env):
    def __init__(
        self,
        data,
        initial_cash: float = 50000,
        num_stocks: int = 300,
        liquidity: float = 0.75, 
        brokerage_rate: float = 0.005,   # 0.5% per trade
        slippage_model: str = "volume",  # "fixed" or "volume"
        fixed_slippage: float = 0.0005,  # 0.05% if using fixed
        min_order_size: int = 100,       # Minimum shares per order
        bid_ask_spread: float = 0.001,   # 0.1% bid-ask spread
        price_impact_factor: float = 0.1, # Price impact coefficient
        max_position_pct: float = 0.20,  # Max 20% in single position
        overnight_risk: float = 0.002,   # 0.2% overnight gap risk
        margin_requirement: float = 0.25, # 25% margin requirement
        tax_rate: float = 0.20,          # 20% capital gains tax
    ) -> None:
        super(StockTradeEnv, self).__init__()
        
        # Core parameters
        self.initial_cash = initial_cash
        self.liquidity = liquidity
        self.available_stocks = data
        self.num_stocks = num_stocks
        
        # Trading cost parameters
        self.brokerage_rate = brokerage_rate
        self.slippage_model = slippage_model
        self.fixed_slippage = fixed_slippage
        self.bid_ask_spread = bid_ask_spread
        self.price_impact_factor = price_impact_factor
        
        # Risk management parameters
        self.min_order_size = min_order_size
        self.max_position_pct = max_position_pct
        self.overnight_risk = overnight_risk
        self.margin_requirement = margin_requirement
        self.tax_rate = tax_rate
        
        # State variables
        self.curr_iter = 0
        self.balance = initial_cash
        self.holdings = defaultdict(int)
        self.purchase_prices = defaultdict(list)  # Track purchase prices for tax calculation
        self.portfolio_values = []
        self.action_history = []
        self.excess_returns = []
        self.sharpe_ratios = []
        self.treynor_ratios = []
        self.market_returns = []
        self.portfolio_returns = []
        self.total_trading_costs = 0
        self.total_taxes_paid = 0
        self.drawdown_history = []
        self.peak_value = initial_cash
        
        # Performance metrics
        self.risk_free_rate = 0.02  # More realistic 2% risk-free rate
        
        # Stock selection
        self.selected_tickers = np.random.choice(
            self.available_stocks.index.get_level_values(0).unique(),
            size=self.num_stocks,
            replace=False,
        )

        self.prev_value = self.balance
        self.action_space = spaces.Discrete(self.num_stocks * 5)  # 5 actions per stock
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(10 * self.num_stocks + 5,),  # Extended observations + portfolio metrics
            dtype=np.float32,
        )

        print("Enhanced Realistic Trading Environment Initialized!")
        print(f"Trading costs: Brokerage {self.brokerage_rate:.3f}, Slippage model: {self.slippage_model}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.curr_iter = 0
        self.balance = self.initial_cash
        self.holdings = defaultdict(int)
        self.purchase_prices = defaultdict(list)
        self.portfolio_returns = []
        self.market_returns = []
        self.portfolio_values = []
        self.total_trading_costs = 0
        self.total_taxes_paid = 0
        self.drawdown_history = []
        self.peak_value = self.initial_cash
        self.prev_value = self.balance
        
        # Reselect stocks for each episode
        self.selected_tickers = np.random.choice(
            self.available_stocks.index.get_level_values(0).unique(),
            size=self.num_stocks,
            replace=False,
        )

        return self._get_observations(), {}

    def step(self, action):
        stock_index = action // 5  # Select which stock to act on
        trade_action = action % 5  # 0=Hold, 1=Buy Small, 2=Buy Large, 3=Sell Small, 4=Sell Large

        # Bounds checking
        if stock_index >= len(self.selected_tickers):
            warnings.warn(f"Stock index {stock_index} out of bounds. Using stock 0.")
            stock_index = 0

        ticker = self.selected_tickers[stock_index]

        if ticker not in self.available_stocks.index:
            warnings.warn(f"Ticker {ticker} not in data. Skipping.")
            return self._get_observations(), -0.01, True, self._get_info()

        if self.curr_iter >= len(self.available_stocks.loc[ticker]) - 1:
            return self._get_observations(), 0, True, self._get_info()

        # Apply overnight risk (random gap up/down)
        if self.curr_iter > 0 and np.random.random() < 0.1:  # 10% chance of overnight gap
            gap_magnitude = np.random.normal(0, self.overnight_risk)
            self._apply_overnight_gap(gap_magnitude)

        # Execute trade
        trading_cost = self.execute_trade(trade_action, stock_index)

        # Calculate portfolio metrics
        new_value = self.portfolio_value()
        if new_value is None or new_value <= 0:
            return self._get_observations(), -1.0, True, self._get_info()

        portfolio_return = (new_value - self.prev_value) / (self.prev_value + 1e-6)
        self.portfolio_returns.append(portfolio_return)
        self.prev_value = new_value
        self.portfolio_values.append(new_value)
        
        # Update drawdown
        if new_value > self.peak_value:
            self.peak_value = new_value
        drawdown = (self.peak_value - new_value) / self.peak_value
        self.drawdown_history.append(drawdown)

        # Market return calculation
        try:
            market_return = self._calculate_market_return(ticker)
            self.market_returns.append(market_return)
        except (IndexError, KeyError):
            market_return = 0
            self.market_returns.append(0)

        # Performance metrics
        beta = self.compute_beta(self.portfolio_returns, self.market_returns)
        alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))

        excess_return = portfolio_return - self.risk_free_rate
        self.excess_returns.append(excess_return)

        # Risk-adjusted metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        treynor_ratio = excess_return / (beta + 1e-6) if abs(beta) > 1e-6 else 0
        
        self.sharpe_ratios.append(sharpe_ratio)
        self.treynor_ratios.append(treynor_ratio)

        # Enhanced reward function
        reward = self._calculate_reward(portfolio_return, alpha, sharpe_ratio, treynor_ratio, trading_cost, drawdown)

        self.curr_iter += 1
        done = (self.curr_iter >= len(self.available_stocks.loc[ticker]) - 1 or 
                new_value < 0.1 * self.initial_cash)  # Stop if lost 90%

        if done:
            print(f"Episode finished. Final Portfolio: ${new_value:,.2f}")
            print(f"Total Trading Costs: ${self.total_trading_costs:,.2f}")
            print(f"Total Taxes Paid: ${self.total_taxes_paid:,.2f}")
            print(f"Max Drawdown: {max(self.drawdown_history, default=0):.2%}")

        return self._get_observations(), reward, done, self._get_info()

    def execute_trade(self, action, stock_index):
        """Execute trade with realistic costs and constraints"""
        ticker = self.selected_tickers[stock_index]
        max_idx = len(self.available_stocks.loc[ticker]) - 1
        self.curr_iter = min(self.curr_iter, max_idx)
        row = self.available_stocks.loc[ticker].iloc[self.curr_iter]
        
        base_price = row["Close"]
        vix = row.get("VIX (Volatility Index)", 20)
        volume = row.get("Volume", 1e6)
        
        total_cost = 0
        
        if action == 0:  # Hold
            return 0
            
        # Determine trade size
        trade_sizes = {1: 0.05, 2: 0.15, 3: 0.5, 4: 1.0}  # Small buy, large buy, small sell, large sell
        trade_fraction = trade_sizes.get(action, 0)
        
        total_portfolio_value = self.portfolio_value()
        
        if action in [1, 2]:  # Buy orders
            max_position_value = self.max_position_pct * total_portfolio_value
            current_position_value = self.holdings[ticker] * base_price
            available_position_value = max_position_value - current_position_value
            
            if available_position_value <= 0:
                return 0  # Position limit reached
                
            investable_amount = min(self.balance * trade_fraction, available_position_value)
            max_liquid_shares = volume * self.liquidity
            num_shares = min(investable_amount // base_price, max_liquid_shares)
            
            if num_shares < self.min_order_size:
                return 0  # Order too small
                
            # Calculate actual execution price with slippage and bid-ask spread
            execution_price = self._calculate_execution_price(base_price, num_shares, volume, "buy")
            
            # Calculate total cost including brokerage
            trade_value = num_shares * execution_price
            brokerage_cost = trade_value * self.brokerage_rate
            total_cost = trade_value + brokerage_cost
            
            if total_cost > self.balance:
                return 0  # Insufficient funds
                
            # Execute buy
            self.holdings[ticker] += num_shares
            self.balance -= total_cost
            self.purchase_prices[ticker].extend([execution_price] * int(num_shares))
            self.total_trading_costs += brokerage_cost
            
            print(f"Bought {num_shares} shares of {ticker} at ${execution_price:.2f} (Cost: ${total_cost:.2f})")
            
        elif action in [3, 4]:  # Sell orders
            current_shares = self.holdings[ticker]
            if current_shares <= 0:
                return 0  # No shares to sell
                
            num_shares = min(int(current_shares * trade_fraction), current_shares)
            max_liquid_shares = volume * self.liquidity
            num_shares = min(num_shares, max_liquid_shares)
            
            if num_shares < self.min_order_size:
                return 0  # Order too small
                
            # Calculate execution price
            execution_price = self._calculate_execution_price(base_price, num_shares, volume, "sell")
            
            # Calculate proceeds and costs
            gross_proceeds = num_shares * execution_price
            brokerage_cost = gross_proceeds * self.brokerage_rate
            
            # Calculate capital gains tax
            avg_purchase_price = np.mean(self.purchase_prices[ticker][:num_shares]) if self.purchase_prices[ticker] else execution_price
            capital_gain = max(0, (execution_price - avg_purchase_price) * num_shares)
            tax_cost = capital_gain * self.tax_rate
            
            net_proceeds = gross_proceeds - brokerage_cost - tax_cost
            total_cost = brokerage_cost + tax_cost
            
            # Execute sell
            self.holdings[ticker] -= num_shares
            self.balance += net_proceeds
            self.purchase_prices[ticker] = self.purchase_prices[ticker][num_shares:]  # Remove sold shares
            self.total_trading_costs += brokerage_cost
            self.total_taxes_paid += tax_cost
            
            print(f"Sold {num_shares} shares of {ticker} at ${execution_price:.2f} (Net: ${net_proceeds:.2f})")
            
        return total_cost

    def _calculate_execution_price(self, base_price, num_shares, volume, side):
        """Calculate realistic execution price with slippage and bid-ask spread"""
        # Bid-ask spread
        if side == "buy":
            price_with_spread = base_price * (1 + self.bid_ask_spread / 2)
        else:
            price_with_spread = base_price * (1 - self.bid_ask_spread / 2)
        
        # Market impact / slippage
        if self.slippage_model == "volume":
            # Volume-based slippage
            volume_ratio = num_shares / (volume + 1e-6)
            slippage = self.price_impact_factor * np.sqrt(volume_ratio)
            slippage = min(slippage, 0.05)  # Cap at 5%
        else:
            # Fixed slippage
            slippage = self.fixed_slippage
        
        # Apply slippage
        if side == "buy":
            execution_price = price_with_spread * (1 + slippage)
        else:
            execution_price = price_with_spread * (1 - slippage)
            
        return max(execution_price, 0.01)  # Prevent negative prices

    def _apply_overnight_gap(self, gap_magnitude):
        """Apply overnight price gaps to holdings"""
        for ticker in self.holdings:
            if self.holdings[ticker] > 0:
                try:
                    current_price = self.available_stocks.loc[ticker].iloc[self.curr_iter]["Close"]
                    gap_impact = current_price * gap_magnitude * self.holdings[ticker]
                    # This affects the portfolio value indirectly through price changes
                except (KeyError, IndexError):
                    pass

    def _calculate_market_return(self, ticker):
        """Calculate market return safely"""
        try:
            current_market = self.available_stocks.loc[ticker].iloc[self.curr_iter]["S&P 500 Index"]
            prev_market = self.available_stocks.loc[ticker].iloc[self.curr_iter - 1]["S&P 500 Index"] if self.curr_iter > 0 else current_market
            return (current_market - prev_market) / (prev_market + 1e-6)
        except (KeyError, IndexError):
            return 0

    def _calculate_sharpe_ratio(self, window=30):
        """Calculate rolling Sharpe ratio"""
        if len(self.portfolio_returns) < 2:
            return 0
        
        returns_window = self.portfolio_returns[-window:] if len(self.portfolio_returns) > window else self.portfolio_returns
        excess_returns_window = [r - self.risk_free_rate for r in returns_window]
        
        if len(excess_returns_window) == 0:
            return 0
            
        mean_excess = np.mean(excess_returns_window)
        std_excess = np.std(excess_returns_window)
        
        return mean_excess / (std_excess + 1e-6) if std_excess > 1e-6 else 0

    def _calculate_reward(self, portfolio_return, alpha, sharpe_ratio, treynor_ratio, trading_cost, drawdown):
        """Enhanced reward function with realistic penalties"""
        total_value = self.portfolio_value()
        
        # Base return reward
        return_reward = portfolio_return
        
        # Risk-adjusted rewards
        sharpe_reward = sharpe_ratio * 0.1
        treynor_reward = treynor_ratio * 0.05
        alpha_reward = alpha * 0.1
        
        # Cost penalties
        cost_penalty = -trading_cost / total_value if total_value > 0 else 0
        drawdown_penalty = -drawdown * 2  # Heavily penalize drawdowns
        
        # Diversification bonus
        num_positions = sum(1 for holding in self.holdings.values() if holding > 0)
        diversification_bonus = min(num_positions / 20, 0.05)  # Bonus for up to 20 positions
        
        total_reward = (return_reward + sharpe_reward + treynor_reward + alpha_reward + 
                       cost_penalty + drawdown_penalty + diversification_bonus)
        
        return np.clip(total_reward, -1.0, 1.0)  # Clip to prevent extreme rewards

    def portfolio_value(self):
        """Calculate total portfolio value with error handling"""
        stock_value = 0
        for ticker in self.holdings:
            if self.holdings[ticker] <= 0:
                continue
                
            if ticker not in self.available_stocks.index:
                continue

            try:
                stock_data = self.available_stocks.loc[ticker]
                if self.curr_iter >= len(stock_data):
                    continue
                    
                stock_price = stock_data.iloc[self.curr_iter].get("Close", None)
                if stock_price is not None and stock_price > 0:
                    stock_value += self.holdings[ticker] * stock_price
            except (KeyError, IndexError):
                continue

        return self.balance + stock_value

    def _get_observations(self):
        """Enhanced observations including portfolio metrics"""
        obs = []
        
        # Stock-specific observations
        for ticker in self.selected_tickers:
            if ticker not in self.available_stocks.index or self.curr_iter >= len(self.available_stocks.loc[ticker]):
                obs.extend([0] * 10)  # Fill with zeros for missing data
                continue

            row = self.available_stocks.loc[ticker].iloc[self.curr_iter]
            
            # Price and technical indicators
            stock_obs = [
                row.get("Stock_Return", 0),
                row.get("Market_Return", 0),
                row.get("Beta", 1),
                row.get("Treynor_Ratio", 0),
                row.get("S&P 500 Index", 0) / 4000,  # Normalized
                row.get("VIX (Volatility Index)", 20) / 100,  # Normalized
                row.get("Volume", 1e6) / 1e6,  # Normalized to millions
                row.get("Close", 100) / 100,  # Normalized price
                self.holdings[ticker] / 1000,  # Normalized position size
                (self.holdings[ticker] * row.get("Close", 100)) / self.portfolio_value() if self.portfolio_value() > 0 else 0  # Position weight
            ]
            obs.extend(stock_obs)
        
        # Portfolio-level observations
        portfolio_obs = [
            self.balance / self.initial_cash,  # Cash ratio
            self.portfolio_value() / self.initial_cash,  # Total value ratio
            len([h for h in self.holdings.values() if h > 0]) / self.num_stocks,  # Diversification ratio
            max(self.drawdown_history) if self.drawdown_history else 0,  # Max drawdown
            self.total_trading_costs / self.initial_cash  # Cumulative cost ratio
        ]
        obs.extend(portfolio_obs)

        return np.array(obs, dtype=np.float32)

    def _get_info(self):
        """Return additional information about the environment state"""
        return {
            'portfolio_value': self.portfolio_value(),
            'cash_balance': self.balance,
            'num_positions': sum(1 for h in self.holdings.values() if h > 0),
            'total_trading_costs': self.total_trading_costs,
            'total_taxes_paid': self.total_taxes_paid,
            'max_drawdown': max(self.drawdown_history) if self.drawdown_history else 0,
            'sharpe_ratio': self._calculate_sharpe_ratio(),
            'current_step': self.curr_iter
        }

    @staticmethod
    def compute_beta(portfolio_returns, market_returns, window=30):
        """Compute rolling beta with error handling"""
        if len(portfolio_returns) < 5 or len(market_returns) < 5:
            return 1.0
        
        try:
            returns_window = portfolio_returns[-window:] if len(portfolio_returns) > window else portfolio_returns
            market_window = market_returns[-window:] if len(market_returns) > window else market_returns
            
            if len(returns_window) != len(market_window):
                min_len = min(len(returns_window), len(market_window))
                returns_window = returns_window[-min_len:]
                market_window = market_window[-min_len:]
            
            slope, _, _, _, _ = linregress(market_window, returns_window)
            return slope if np.isfinite(slope) else 1.0
        except:
            return 1.0
