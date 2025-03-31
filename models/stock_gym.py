import os
from collections import defaultdict

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.stats import linregress


def merge_stocks() -> DataFrame:
    data = {}
    for root, _, files in os.walk("../data/merged"):
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
        self, data, initial_cash: float=50000, num_stocks: int = 300
    ) -> None:
        super(StockTradeEnv, self).__init__()
        self.initial_cash = initial_cash
        self.available_stocks = data
        self.curr_iter = 0
        self.balance = initial_cash
        self.holdings = defaultdict(int)
        self.portfolio_values = []
        self.action_history = []
        self.excess_returns = []
        self.sharpe_ratios = []
        self.treynor_ratios = []
        self.market_returns = []
        self.risk_free_rate = 0.10
        self.num_stocks = num_stocks
        self.selected_tickers = np.random.choice(
            self.available_stocks.index.get_level_values(0).unique(),
            size=self.num_stocks,
            replace=False,
        )

        self.prev_value = self.balance
        self.action_space = spaces.Discrete(self.num_stocks * 3)  # 3 actions per stock
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(6 * self.num_stocks,),
            dtype=np.float32,
        )

        print("Gymnasium Environment Initialized!")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.curr_iter = 0
        self.balance = self.initial_cash 
        self.holdings = defaultdict(int)
        self.portfolio_returns = []
        self.market_returns = []
        self.portfolio_values = []
        self.prev_value = self.balance
        self.selected_tickers = np.random.choice(
            self.available_stocks.index.get_level_values(0).unique(),
            size=self.num_stocks,
            replace=False,
        )

        return self._get_observations(), {}

    def step(self, action):
        stock_index = action // 3  # Select which stock to act on
        trade_action = action % 3  # 0 = Hold, 1 = Buy, 2 = Sell

        # Ensure stock_index is within bounds
        if stock_index >= len(self.selected_tickers):
            print(f"Warning: stock_index {stock_index} is out of bounds. Defaulting to 0.")
            stock_index = 0  # Default to first stock

        ticker = self.selected_tickers[stock_index]

        # Ensure ticker exists in available_stocks
        if ticker not in self.available_stocks.index:
            print(f"Warning: {ticker} not found in available_stocks. Skipping.")
            return self._get_observations(), 0, True, {}

        # Ensure curr_iter is within bounds
        if self.curr_iter >= len(self.available_stocks.loc[ticker]) - 1:
            print(f"Warning: curr_iter {self.curr_iter} out of bounds. Ending episode.")
            return self._get_observations(), 0, True, {}

        self.execute_trade(trade_action, stock_index)

        new_value = self.portfolio_value()
        if new_value is None:
            print("Warning: portfolio_value() returned None. Setting reward to 0.")
            return self._get_observations(), 0, True, {}
        portfolio_return = (new_value - self.prev_value) / (self.prev_value + 1e-6)
        self.prev_value = new_value

        self.portfolio_values.append(new_value)
        self.action_history.append(action)

        # Ensure S&P 500 Index data exists and is within bounds
        try:
            market_return = (
                self.available_stocks.loc[ticker].iloc[self.curr_iter + 1]["S&P 500 Index"]
                - self.available_stocks.loc[ticker].iloc[self.curr_iter]["S&P 500 Index"]
            ) / (self.available_stocks.loc[ticker].iloc[self.curr_iter]["S&P 500 Index"] + 1e-6)
        except (IndexError, KeyError):
            print(f"Warning: Missing S&P 500 Index data for {ticker}. Setting market_return to 0.")
            market_return = 0

        self.market_returns.append(market_return)

        # Compute Beta
        if len(self.portfolio_returns) > 1 and len(self.market_returns) > 1:
            beta = self.compute_beta(self.portfolio_returns, self.market_returns)
        else:
            beta = 1  # Default beta

        alpha = portfolio_return - (self.risk_free_rate + beta * (market_return - self.risk_free_rate))

        excess_return = portfolio_return - self.risk_free_rate
        self.excess_returns.append(excess_return)

        # Compute Sharpe Ratio safely
        if len(self.portfolio_returns) > 30:
            std_dev = np.std(self.portfolio_returns[-30:])
        else:
            std_dev = np.std(self.portfolio_returns) if self.portfolio_returns else 1e-6

        sharpe_ratio = excess_return / (std_dev + 1e-6)
        sharpe_ratio = sharpe_ratio if np.isfinite(sharpe_ratio) else 0
        self.sharpe_ratios.append(sharpe_ratio)

        treynor_ratio = excess_return / (beta + 1e-6)
        treynor_ratio = treynor_ratio if np.isfinite(treynor_ratio) else 0
        self.treynor_ratios.append(treynor_ratio)

        reward = (sharpe_ratio * 0.4) + (treynor_ratio * 0.3) + (alpha * 0.3)
        reward = reward if np.isfinite(reward) else 0

        self.curr_iter += 1
        done = self.curr_iter >= len(self.available_stocks.loc[ticker]) - 1
        if done:
            print(f"Best Portfolio: {max(self.portfolio_values)}")
        print("=" * 40)
        print(f"\nStep: {self.curr_iter}")
        print(f"Balance: {self.balance}")
        print(f"Current Portfolio Value: {self.portfolio_value()}")
        print(f"Action Taken: {['Hold', 'Buy', 'Sell'][trade_action]} on {ticker}")
        print("=" * 40)

        return self._get_observations(), reward, done, {}

    def execute_trade(self, action, stock_index):
        ticker = self.selected_tickers[stock_index]
        max_idx = len(self.available_stocks.loc[ticker]) - 1
        self.curr_iter = min(self.curr_iter, max_idx)
        row = self.available_stocks.loc[ticker].iloc[self.curr_iter]
        price = row["Close"]
        vix = row.get("VIX (Volatility Index)", 20)

        # Adaptive investment percentage based on VIX
        base_fraction = 0.10  # Default 10% of balance
        if vix > 25:  # High volatility, invest only 5%
            investment_fraction = 0.05
        elif vix < 15:  # Low volatility, invest 15%
            investment_fraction = 0.15
        else:  # Medium volatility, default to 10%
            investment_fraction = base_fraction

        investable_amount = self.balance * investment_fraction
        total_portfolio_value = self.portfolio_value()

        # Sanity Checks
        position_limit = 0.2 * total_portfolio_value  # No single stock should exceed 20% of portfolio
        max_trade_volume = 0.1 * total_portfolio_value / price  # At most 10% of portfolio in one trade
        min_trade_size = 1  # Minimum of 1 share per trade

        if action == 1 and self.balance >= price:  # Buy
            num_shares = min(investable_amount // price, max_trade_volume)
            if num_shares < min_trade_size:
                print(f"Skipped buying {ticker}, trade size too small.")
                return

            new_position_value = (self.holdings[ticker] + num_shares) * price
            if new_position_value > position_limit:
                print(f"Skipped buying {ticker}, would exceed 20% position limit.")
                return

            self.holdings[ticker] += num_shares
            self.balance -= num_shares * price
            print(f"Bought {num_shares} shares of {ticker} at {price} (VIX: {vix:.2f})")

        elif action == 2 and self.holdings[ticker] > 0:  # Sell
            num_shares = min(self.holdings[ticker], max(1, int(self.holdings[ticker] * 0.5)))
            if num_shares < min_trade_size:
                print(f"Skipped selling {ticker}, trade size too small.")
                return

            self.balance += num_shares * price
            self.holdings[ticker] -= num_shares
            print(f"Sold {num_shares} shares of {ticker} at {price} (VIX: {vix:.2f})")

    def portfolio_value(self):
        stock_value = 0
        for ticker in self.holdings:
            if ticker not in self.available_stocks.index:
                print(f"Warning: {ticker} not found in available_stocks. Skipping.")
                continue  # Skip missing tickers

            stock_data = self.available_stocks.loc[ticker]

            if self.curr_iter >= len(stock_data):  # Prevent out-of-bounds error
                print(f"Warning: Index {self.curr_iter} out of bounds for {ticker}. Skipping.")
                continue

            stock_price = stock_data.iloc[self.curr_iter].get("Close", None)  # Use "Close" (case-sensitive)
            if stock_price is None:
                print(f"Warning: 'Close' price missing for {ticker} at step {self.curr_iter}. Skipping.")
                continue

            stock_value += self.holdings[ticker] * stock_price

        return self.balance + stock_value  # ✅ Return **after** looping through all stocks
    def _get_observations(self):
        obs = []
        for ticker in self.selected_tickers:
            if ticker not in self.available_stocks.index:
                print(f"Warning: {ticker} not found in available_stocks. Skipping.")
                obs.extend([np.nan] * 6)  # Fill with NaNs for later interpolation
                continue

            stock_data = self.available_stocks.loc[ticker]

            if self.curr_iter >= len(stock_data):  # ✅ Check index bounds
                print(f"Warning: Index {self.curr_iter} out of bounds for {ticker}. Skipping.")
                obs.extend([np.nan] * 6)  # Fill with NaNs for later interpolation
                continue

            row = stock_data.iloc[self.curr_iter]

            obs.extend(
                [
                    row.get("Stock_Return", np.nan),
                    row.get("Market_Return", np.nan),
                    row.get("Beta", np.nan),
                    row.get("Treynor_Ratio", np.nan),
                    row.get("S&P 500 Index", np.nan),
                    row.get("VIX (Volatility Index)", np.nan),
                ]
            )

        # Convert to NumPy array and interpolate missing values
        obs_array = np.array(obs, dtype=np.float32)
        
        # Apply interpolation (ignores NaNs)
        obs_array = pd.Series(obs_array).interpolate(method="linear", limit_direction="both").to_numpy()

        return obs_array

    @staticmethod
    def compute_beta(portfolio_returns, market_returns, window=30):
        if len(portfolio_returns) < window:
            return 1
        slope, _, _, _, _ = linregress(
            market_returns[-window:], portfolio_returns[-window:]
        )
        return slope if np.isfinite(slope) else 1
