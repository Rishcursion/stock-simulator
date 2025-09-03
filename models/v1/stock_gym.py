import logging
import os
import time
from collections import defaultdict

import gymnasium as gym
import gymnasium.spaces as spaces
import numpy as np
import pandas as pd
from pandas import DataFrame


def merge_stocks() -> DataFrame:
    """Load and merge stock data with improved error handling and logging"""
    logger = logging.getLogger(__name__)
    start_time = time.time()

    data = {}
    data_path = os.path.join("data", "merged")

    if not os.path.exists(data_path):
        error_msg = f"Data directory not found at: {os.path.abspath(data_path)}"
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    logger.info(f"Loading stock data from {data_path}")

    csv_files = [f for f in os.listdir(data_path) if f.endswith(".csv")]
    logger.info(f"Found {len(csv_files)} CSV files to process")

    for file in csv_files:
        try:
            file_path = os.path.join(data_path, file)
            df: DataFrame = pd.read_csv(file_path)

            # Optimize date parsing
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")

            # Filter data from 2000 onwards and remove invalid dates
            initial_rows = len(df)
            df = df[df["Date"] >= "2000-01-01"].dropna(subset=["Date"])
            filtered_rows = len(df)

            ticker_name = file.replace(".csv", "")
            data[ticker_name] = df

            logger.debug(
                f"Loaded {ticker_name}: {initial_rows} -> {filtered_rows} rows after filtering"
            )

        except Exception as e:
            logger.warning(f"Failed to process file {file}: {e}")
            continue

    if not data:
        error_msg = "No data loaded. Check the 'data/merged' directory."
        logger.error(error_msg)
        raise ValueError(error_msg)

    # Optimize concatenation
    logger.info("Concatenating data from all tickers...")
    final_data = pd.concat(
        data.values(), keys=data.keys(), names=["Ticker", "OldIndex"]
    )
    final_data = final_data.reset_index(level="OldIndex", drop=True).reset_index()

    load_time = time.time() - start_time
    logger.info(f"Data loading completed in {load_time:.2f}s")
    logger.info(f"Total records: {len(final_data)}, Tickers: {len(data)}")

    return final_data


class StockTradeEnv(gym.Env):
    def __init__(
        self,
        data,
        initial_cash: float = 50000,
        num_stocks: int = 50,
        train_test_split_date: str = "2019-01-01",
        mode: str = "train",
        episode_length: int = 252,
        log_level: str = "INFO",
    ) -> None:
        super(StockTradeEnv, self).__init__()

        # Setup logging for this environment instance
        self.logger = self._setup_logging(log_level)
        self.logger.info("=" * 50)
        self.logger.info("Initializing StockTradeEnv")
        self.logger.info("=" * 50)

        # Core parameters
        self.initial_cash = initial_cash
        self.num_stocks = num_stocks
        self.episode_length = episode_length

        # Log initialization parameters
        self.logger.info("Initial parameters:")
        self.logger.info(f"  Initial cash: ${initial_cash:,}")
        self.logger.info(f"  Number of stocks: {num_stocks}")
        self.logger.info(f"  Episode length: {episode_length} days")
        self.logger.info(f"  Train/test split: {train_test_split_date}")

        # Data processing with timing
        start_time = time.time()
        self.all_data = data.sort_values(by="Date").reset_index(drop=True)
        data_process_time = time.time() - start_time
        self.logger.info(f"Data sorting completed in {data_process_time:.3f}s")

        # Data splitting logic with optimization
        self.train_test_split_date = pd.to_datetime(train_test_split_date)

        # Use vectorized operations for better performance
        date_mask = self.all_data["Date"] < self.train_test_split_date
        self.train_data = self.all_data[date_mask]
        self.test_data = self.all_data[~date_mask]

        # Pre-compute unique dates for better performance
        self.train_dates = np.sort(self.train_data["Date"].unique())
        self.test_dates = np.sort(self.test_data["Date"].unique())

        # Log data split statistics
        self.logger.info("Data split statistics:")
        self.logger.info(
            f"  Training data: {len(self.train_data)} records, {len(self.train_dates)} dates"
        )
        self.logger.info(
            f"  Training period: {pd.to_datetime(self.train_dates[0]).to_pydatetime().date()} "
            f"to {pd.to_datetime(self.train_dates[-1]).to_pydatetime().date()}"
        )
        self.logger.info(
            f"  Test data: {len(self.test_data)} records, {len(self.test_dates)} dates"
        )
        self.logger.info(
            f"  Test period: {pd.to_datetime(self.test_dates[0]).to_pydatetime().date()} "
            f"to {pd.to_datetime(self.test_dates[-1]).to_pydatetime().date()}"
        )

        # Environment mode
        self.mode = mode
        self.current_dates = (
            self.train_dates if self.mode == "train" else self.test_dates
        )

        # State variables
        self.balance = initial_cash
        self.holdings = defaultdict(int)
        self.portfolio_values = []
        self.portfolio_returns = []
        self.episode_data = pd.DataFrame()
        self.episode_observations = None
        self.current_step = 0
        self.prev_value = self.balance

        # Performance optimization: pre-compute ticker lists
        self.train_tickers = self.train_data["Ticker"].unique()
        self.test_tickers = self.test_data["Ticker"].unique()

        self.logger.info(
            f"Available tickers - Train: {len(self.train_tickers)}, Test: {len(self.test_tickers)}"
        )

        self.selected_tickers = self._sample_tickers()
        # Action and observation spaces
        self.action_space = spaces.Discrete(self.num_stocks * 3)  # 3 actions per stock
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6 * self.num_stocks,),  # 6 features per stock
            dtype=np.float32,
        )

        # Episode tracking
        self.episode_count = 0
        self.total_episodes = 0
        self.episode_start_time = None

        self.logger.info(f"Environment initialized in '{self.mode}' mode!")
        self.logger.info(f"Action space: {self.action_space.n} actions")
        self.logger.info(f"Observation space: {self.observation_space.shape}")

    def _setup_logging(self, log_level: str):
        """Setup logging for the environment"""
        logger = logging.getLogger(f"{__name__}.StockTradeEnv")

        # Don't add handlers if they already exist
        if not logger.handlers:
            # Create console handler
            console_handler = logging.StreamHandler()
            console_handler.setLevel(getattr(logging, log_level.upper()))

            # Create formatter
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            console_handler.setFormatter(formatter)

            # Add handler to logger
            logger.addHandler(console_handler)
            logger.setLevel(getattr(logging, log_level.upper()))

        return logger

    def set_mode(self, mode: str):
        """Switch between train and test modes with logging"""
        if mode not in ["train", "test"]:
            error_msg = "Mode must be either 'train' or 'test'"
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        old_mode = self.mode
        self.mode = mode
        self.current_dates = (
            self.train_dates if self.mode == "train" else self.test_dates
        )

        self.logger.info(
            f"Environment mode switched from '{old_mode}' to '{self.mode}'"
        )
        self.logger.info(
            f"Available dates in {self.mode} mode: {len(self.current_dates)}"
        )

    def _sample_tickers(self):
        """Sample tickers with performance optimization and logging"""
        start_time = time.time()

        # Use pre-computed ticker lists for better performance
        available_tickers = (
            self.train_tickers if self.mode == "train" else self.test_tickers
        )

        if len(available_tickers) < self.num_stocks:
            self.logger.warning(
                f"Only {len(available_tickers)} tickers available, requested {self.num_stocks}"
            )
            selected_tickers = available_tickers
        else:
            selected_tickers = np.random.choice(
                available_tickers, size=self.num_stocks, replace=False
            )

        sample_time = time.time() - start_time
        self.logger.debug(f"Ticker sampling completed in {sample_time:.4f}s")

        return selected_tickers

    def reset(self, seed=None, options=None):
        """Reset environment with comprehensive logging and optimization"""
        super().reset(seed=seed)

        self.episode_count += 1
        self.total_episodes += 1
        self.episode_start_time = time.time()

        self.logger.info(f"Starting episode {self.episode_count} in {self.mode} mode")

        # Optimized episode data selection
        start_time = time.time()

        max_start_index = len(self.current_dates) - self.episode_length - 1
        if max_start_index <= 0:
            self.logger.warning(
                f"Not enough dates for episode length {self.episode_length}"
            )
            max_start_index = 0

        start_date_index = np.random.randint(0, max(1, max_start_index))
        start_date = self.current_dates[start_date_index]
        end_date_index = min(
            start_date_index + self.episode_length, len(self.current_dates) - 1
        )
        end_date = self.current_dates[end_date_index]

        self.selected_tickers = self._sample_tickers()

        # Optimized data filtering using boolean indexing
        date_mask = (self.all_data["Date"] >= start_date) & (
            self.all_data["Date"] <= end_date
        )
        ticker_mask = self.all_data["Ticker"].isin(self.selected_tickers)
        self.episode_data = self.all_data[date_mask & ticker_mask].copy()

        data_filter_time = time.time() - start_time

        self.logger.info(f"Episode data prepared in {data_filter_time:.3f}s")
        self.logger.info(
            f"Episode period: {pd.to_datetime(start_date).to_pydatetime().date()} to {pd.to_datetime(end_date).to_pydatetime().date()}"
        )
        self.logger.info(f"Episode data shape: {self.episode_data.shape}")
        self.logger.debug(f"Selected tickers: {list(self.selected_tickers)}")

        # Pre-compute observations with error handling
        try:
            self._precompute_observations()
        except Exception as e:
            self.logger.error(f"Failed to precompute observations: {e}")
            raise

        # Reset state variables
        self.current_step = 0
        self.balance = self.initial_cash
        self.holdings = defaultdict(int)
        self.portfolio_values = [self.initial_cash]
        self.portfolio_returns = []
        self.prev_value = self.balance

        self.logger.debug(f"Episode {self.episode_count} reset completed")

        return self._get_observations(), {}

    def _precompute_observations(self):
        """Pre-compute observations with optimization and error handling"""
        start_time = time.time()

        episode_dates = np.sort(self.episode_data["Date"].unique())
        max_len = len(episode_dates)
        if self.episode_length > max_len:
            self.logger.warning(
                f"Requested episode_length {self.episode_length} is greater than available dates {max_len}, using {max_len} instead"
            )
        self.episode_length = min(self.episode_length, max_len)
        episode_dates = episode_dates[
            : self.episode_length
        ]  # Crop to actual episode_length
        self.episode_length = min(self.episode_length, len(episode_dates))

        # Pre-allocate array for better performance
        obs_array = np.zeros(
            (self.episode_length, 6 * self.num_stocks), dtype=np.float32
        )

        # Create a pivot table for faster lookups
        try:
            pivot_data = self.episode_data.pivot_table(
                index="Date",
                columns="Ticker",
                values=[
                    "Stock_Return",
                    "Market_Return",
                    "Beta",
                    "Treynor_Ratio",
                    "S&P 500 Index",
                    "VIX (Volatility Index)",
                ],
                aggfunc="first",
            )
        except Exception as e:
            self.logger.warning(
                f"Pivot table creation failed, using slower method: {e}"
            )
            # Fallback to original method
            self._precompute_observations_fallback()
            return

        # Fill observations efficiently
        for i, current_date in enumerate(episode_dates):
            obs_idx = 0
            for ticker in self.selected_tickers:
                try:
                    # Extract data from pivot table
                    stock_return = (
                        pivot_data.loc[current_date, ("Stock_Return", ticker)]
                        if ("Stock_Return", ticker) in pivot_data.columns
                        else 0
                    )
                    market_return = (
                        pivot_data.loc[current_date, ("Market_Return", ticker)]
                        if ("Market_Return", ticker) in pivot_data.columns
                        else 0
                    )
                    beta = (
                        pivot_data.loc[current_date, ("Beta", ticker)]
                        if ("Beta", ticker) in pivot_data.columns
                        else 1
                    )
                    treynor = (
                        pivot_data.loc[current_date, ("Treynor_Ratio", ticker)]
                        if ("Treynor_Ratio", ticker) in pivot_data.columns
                        else 0
                    )
                    sp500 = (
                        pivot_data.loc[current_date, ("S&P 500 Index", ticker)]
                        if ("S&P 500 Index", ticker) in pivot_data.columns
                        else 0
                    )
                    vix = (
                        pivot_data.loc[current_date, ("VIX (Volatility Index)", ticker)]
                        if ("VIX (Volatility Index)", ticker) in pivot_data.columns
                        else 20
                    )

                    obs_array[i, obs_idx : obs_idx + 6] = [
                        stock_return,
                        market_return,
                        beta,
                        treynor,
                        sp500,
                        vix,
                    ]
                except (KeyError, IndexError):
                    # Use default values if data is missing
                    obs_array[i, obs_idx : obs_idx + 6] = [0, 0, 1, 0, 0, 20]

                obs_idx += 6

        self.episode_observations = obs_array

        precompute_time = time.time() - start_time
        self.logger.debug(f"Observations precomputed in {precompute_time:.3f}s")
        self.logger.debug(f"Observation shape: {self.episode_observations.shape}")

    def _precompute_observations_fallback(self):
        """Fallback method for precomputing observations"""
        obs_list = []
        episode_dates = self.episode_data["Date"].unique()
        self.episode_length = min(self.episode_length, len(episode_dates))

        for i in range(self.episode_length):
            current_date = episode_dates[i]
            obs_step = []

            for ticker in self.selected_tickers:
                row = self.episode_data[
                    (self.episode_data["Date"] == current_date)
                    & (self.episode_data["Ticker"] == ticker)
                ]

                if not row.empty:
                    row = row.iloc[0]
                    obs_step.extend(
                        [
                            row.get("Stock_Return", 0),
                            row.get("Market_Return", 0),
                            row.get("Beta", 1),
                            row.get("Treynor_Ratio", 0),
                            row.get("S&P 500 Index", 0),
                            row.get("VIX (Volatility Index)", 20),
                        ]
                    )
                else:
                    obs_step.extend([0, 0, 1, 0, 0, 20])

            obs_list.append(obs_step)

        self.episode_observations = np.array(obs_list, dtype=np.float32)

    def _get_observations(self):
        """Get current observations with bounds checking"""
        step_index = min(self.current_step, self.episode_observations.shape[0] - 1)
        obs = self.episode_observations[step_index]

        # Handle NaN values
        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)

        return obs

    def step(self, action):
        """Execute one step with comprehensive logging and optimization"""
        step_start_time = time.time()

        # Action decomposition
        stock_index = action // 3
        trade_action = action % 3

        if stock_index >= len(self.selected_tickers):
            self.logger.warning(
                f"Invalid stock index {stock_index}, max is {len(self.selected_tickers)-1}"
            )
            stock_index = stock_index % len(self.selected_tickers)

        ticker = self.selected_tickers[stock_index]

        # Get current date with bounds checking
        episode_dates = np.sort(self.episode_data["Date"].unique())
        if self.current_step >= len(episode_dates):
            self.logger.warning(
                f"Step {self.current_step} exceeds available dates {len(episode_dates)}"
            )
            self.current_step = len(episode_dates) - 1

        current_date = episode_dates[self.current_step]

        # Execute trade with logging
        old_balance = self.balance
        old_holdings = dict(self.holdings)

        self.execute_trade(trade_action, ticker, current_date)

        # Calculate portfolio value and returns
        new_value = self.portfolio_value(current_date)
        portfolio_return = (new_value - self.prev_value) / (self.prev_value + 1e-9)

        # Log trade details if significant change occurred
        if abs(self.balance - old_balance) > 0.01 or self.holdings.get(
            ticker, 0
        ) != old_holdings.get(ticker, 0):
            action_names = ["HOLD", "BUY", "SELL"]
            self.logger.debug(
                f"Step {self.current_step}: {action_names[trade_action]} {ticker} | "
                f"Balance: ${old_balance:.2f} -> ${self.balance:.2f} | "
                f"Holdings[{ticker}]: {old_holdings.get(ticker, 'None')} -> {self.holdings[ticker]} | "
                f"Portfolio: ${self.prev_value:.2f} -> ${new_value:.2f}"
            )

        # Update state
        self.portfolio_returns.append(portfolio_return)
        self.prev_value = new_value
        self.portfolio_values.append(new_value)

        # Calculate reward (scaled for better training)
        reward = portfolio_return * 100
        reward = np.clip(reward, -10, 10)  # Prevent extreme rewards

        # Update step
        self.current_step += 1
        done = self.current_step >= self.episode_length

        # Episode completion logging
        if done:
            episode_time = time.time() - self.episode_start_time
            final_return = (new_value - self.initial_cash) / self.initial_cash

            self.logger.info(
                f"Episode {self.episode_count} completed in {episode_time:.2f}s"
            )
            self.logger.info(f"Final portfolio value: ${new_value:.2f}")
            self.logger.info(f"Total return: {final_return:.2%}")

            if len(self.portfolio_returns) > 1:
                sharpe_ratio = np.mean(self.portfolio_returns) / (
                    np.std(self.portfolio_returns) + 1e-9
                )
                self.logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")

            # Log holdings summary
            total_stock_value = sum(
                self.holdings[ticker] * self._get_current_price(ticker, current_date)
                for ticker in self.holdings
                if self.holdings[ticker] > 0
            )
            self.logger.info(
                f"Final allocation - Cash: ${self.balance:.2f}, Stocks: ${total_stock_value:.2f}"
            )

        step_time = time.time() - step_start_time
        if step_time > 0.001:  # Log only if step took significant time
            self.logger.debug(
                f"Step {self.current_step-1} completed in {step_time:.4f}s"
            )

        return self._get_observations(), reward, done, {}

    def execute_trade(self, action, ticker, current_date):
        """Execute trade with improved error handling and logging"""
        try:
            price = self._get_current_price(ticker, current_date)
            if price is None:
                return

            if action == 1:  # BUY
                if self.balance >= price:
                    # Buy 10% of available balance
                    max_shares = int((self.balance * 0.1) // price)
                    if max_shares > 0:
                        cost = max_shares * price
                        self.holdings[ticker] += max_shares
                        self.balance -= cost

            elif action == 2:  # SELL
                if self.holdings[ticker] > 0:
                    # Sell all holdings of this ticker
                    num_shares = self.holdings[ticker]
                    revenue = num_shares * price
                    self.balance += revenue
                    self.holdings[ticker] = 0

        except Exception as e:
            self.logger.error(
                f"Trade execution failed for {ticker} on {current_date}: {e}"
            )

    def _get_current_price(self, ticker, current_date):
        """Get current price with caching and error handling"""
        try:
            row = self.episode_data[
                (self.episode_data["Date"] == current_date)
                & (self.episode_data["Ticker"] == ticker)
            ]

            if row.empty:
                self.logger.debug(f"No price data for {ticker} on {current_date}")
                return None

            return row.iloc[0]["Close"]

        except (IndexError, KeyError) as e:
            self.logger.debug(
                f"Price lookup failed for {ticker} on {current_date}: {e}"
            )
            return None

    def portfolio_value(self, current_date):
        """Calculate portfolio value with optimization and error handling"""
        stock_value = 0

        for ticker, shares in self.holdings.items():
            if shares > 0:
                price = self._get_current_price(ticker, current_date)
                if price is not None:
                    stock_value += shares * price

        total_value = self.balance + stock_value

        # Sanity check
        if total_value < 0:
            self.logger.warning(
                f"Negative portfolio value detected: ${total_value:.2f}"
            )

        return total_value

    def get_env_stats(self):
        """Get environment statistics for monitoring"""
        return {
            "mode": self.mode,
            "episode_count": self.episode_count,
            "total_episodes": self.total_episodes,
            "current_step": self.current_step,
            "episode_length": self.episode_length,
            "num_selected_tickers": len(self.selected_tickers),
            "episode_data_shape": (
                self.episode_data.shape if not self.episode_data.empty else (0, 0)
            ),
            "current_portfolio_value": (
                self.portfolio_values[-1]
                if self.portfolio_values
                else self.initial_cash
            ),
        }
