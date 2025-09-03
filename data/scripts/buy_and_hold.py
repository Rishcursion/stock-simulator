import os

import numpy as np
import pandas as pd


def load_all_stock_data(data_dir="../data/merged"):
    """
    Loads all stock CSVs from a directory and merges them into a single DataFrame.
    """
    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found.")
        return pd.DataFrame()

    all_files = [
        os.path.join(path, name)
        for path, _, files in os.walk(data_dir)
        for name in files
        if name.endswith(".csv")
    ]

    # Limit to 500 stocks as per the request
    files_to_load = all_files[:500]

    df_list = []
    for file in files_to_load:
        try:
            df = pd.read_csv(file)
            df["Date"] = pd.to_datetime(df["Date"], format="%d-%m-%Y", errors="coerce")
            df = df.dropna(subset=["Date"])
            df["Ticker"] = os.path.basename(file).split(".")[0]
            df_list.append(df)
        except Exception as e:
            print(f"Could not process {file}: {e}")

    if not df_list:
        print("No data was loaded.")
        return pd.DataFrame()

    return pd.concat(df_list, ignore_index=True)


def calculate_multi_stock_bnh(data, initial_capital=50000, risk_free_rate=0.02):
    """
    Calculates the performance of a diversified Buy-and-Hold strategy.
    """
    if data.empty:
        print("Input DataFrame is empty. Cannot perform calculations.")
        return

    print(f"Analyzing {data['Ticker'].nunique()} unique stocks...")

    # Pivot the data to have dates as index and tickers as columns for close prices
    close_prices = data.pivot_table(
        index="Date", columns="Ticker", values="Close"
    ).sort_index()

    # Filter for the relevant date range
    close_prices = close_prices[close_prices.index >= "2000-01-01"]
    close_prices.ffill(inplace=True)  # Forward-fill missing values
    close_prices.bfill(inplace=True)  # Back-fill any remaining NaNs at the start

    # Simulation setup
    start_date = close_prices.index[0]
    end_date = close_prices.index[-1]
    capital_per_stock = initial_capital / len(close_prices.columns)

    # Calculate shares bought at the start
    start_prices = close_prices.loc[start_date]
    shares = capital_per_stock / start_prices

    # Calculate portfolio value over time
    portfolio_values = (close_prices * shares).sum(axis=1)

    # --- Performance Metrics Calculation ---
    daily_returns = portfolio_values.pct_change().dropna()

    total_return = (portfolio_values.iloc[-1] - initial_capital) / initial_capital
    num_years = (end_date - start_date).days / 365.25

    annualized_return = ((1 + total_return) ** (1 / num_years)) - 1
    annualized_volatility = daily_returns.std() * np.sqrt(252)
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility

    # Max Drawdown
    cumulative_returns = (1 + daily_returns).cumprod()
    peak = cumulative_returns.expanding(min_periods=1).max()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    print("\n--- Diversified Buy-and-Hold Baseline Results ---")
    print(f"Initial Portfolio Value: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value:   ${portfolio_values.iloc[-1]:,.2f}")
    print(f"Total Return:            {total_return:.2%}")
    print("-" * 40)
    print(f"Annualized Return:       {annualized_return:.4f}")
    print(f"Annualized Volatility:   {annualized_volatility:.4f}")
    print(f"Sharpe Ratio:            {sharpe_ratio:.4f}")
    print(f"Maximum Drawdown:        {max_drawdown:.4f}")
    print("-" * 40)


if __name__ == "__main__":
    # Ensure the path to your data directory is correct
    stock_data = load_all_stock_data("../merged/")
    calculate_multi_stock_bnh(stock_data)
