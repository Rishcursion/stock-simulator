import os
import pandas as pd

print("Current working directory:", os.getcwd())

# Load macroeconomic data
macro_data = pd.read_csv("../normalized_date_macro.csv")

# Compute Market Return
macro_data["Market_Return"] = macro_data["S&P 500 Index"].pct_change()
macro_data["Market_Return"].fillna(0, inplace=True)

final_data_path = os.path.join(os.pardir, "final_data")
merged_path = os.path.join(os.pardir, "merged")

if not os.path.exists(final_data_path):
    print(f"Error: Directory '{final_data_path}' not found!")
    exit(1)

print("Starting Merge Process")
for root, dirs, files in os.walk(final_data_path):
    for file in files:
        file_path = os.path.join(root, file)
        print(f"\rProcessing {file}", end="", flush=True)

        # Read stock data
        stock_data = pd.read_csv(file_path)

        # Compute Stock Return
        stock_data["Stock_Return"] = stock_data["Close"].pct_change()
        stock_data["Stock_Return"].fillna(0, inplace=True)

        # Merge with macro data on Date
        new_data = pd.merge(left=stock_data, right=macro_data, on="Date", how="left")

        # Forward fill missing macroeconomic data
        new_data.ffill(inplace=True)

        # Compute Beta (Rolling Window)
        rolling_window = 30
        new_data["Beta"] = (
            new_data["Stock_Return"]
            .rolling(rolling_window)
            .cov(new_data["Market_Return"]) / new_data["Market_Return"].rolling(rolling_window).var()
        )
        new_data.loc[:, "Beta"] = new_data["Beta"].fillna(method="ffill").fillna(method="bfill")

        # Compute Treynor Ratio
        risk_free_col = "10Y Treasury Yield"
        if risk_free_col in new_data.columns:
            new_data["Treynor_Ratio"] = (new_data["Stock_Return"] - new_data[risk_free_col]) / new_data["Beta"]
            new_data.loc[:, "Treynor_Ratio"] = new_data["Treynor_Ratio"].replace(
                [float("inf"), -float("inf")], float("nan")
            )
            new_data.loc[:, "Treynor_Ratio"] = new_data["Treynor_Ratio"].fillna(method="ffill")

        # Ensure merged folder exists
        os.makedirs(merged_path, exist_ok=True)

        # Save merged CSV
        new_data.to_csv(os.path.join(merged_path, file), index=False)

print("\nMerge Completed!")
