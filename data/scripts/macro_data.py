import pandas as pd
from fredapi import Fred
from sklearn.preprocessing import MinMaxScaler
import dotenv
import os
dotenv.load_dotenv()
# Replace with your API key
API_KEY = os.getenv("API_KEY") 

# List of macroeconomic indicators from FRED
FRED_SERIES = {
    "GDP": "GDPC1",
    "GDP Growth": "A191RL1Q225SBEA",
    "CPI": "CPIAUCSL",
    "Core CPI": "CPILFESL",
    "PPI": "PPIACO",
    "Unemployment Rate": "UNRATE",
    "Labor Force Participation Rate": "CIVPART",
    "Fed Funds Rate": "FEDFUNDS",
    "10Y Treasury Yield": "DGS10",
    "Money Supply (M2)": "M2SL",
    "Consumer Confidence": "UMCSENT",
    "ISM Manufacturing PMI": "NAPM",
    "Housing Starts": "HOUST",
    "Home Price Index": "CSUSHPISA",
    "S&P 500 Index": "SP500",
    "VIX (Volatility Index)": "VIXCLS",
    "Corporate Bond Yield": "BAA10Y",
    "Oil Prices (WTI)": "DCOILWTICO",
    "Trade Balance": "BOPGSTB",
    "USD Index": "DTWEXM",
}

# Fetch data
def fetch_fred_data(api_key, series_dict, start_date="2000-01-01"):
    fred = Fred(api_key=api_key)
    data = {}

    for label, series_id in series_dict.items():
        try:
            print(f"Fetching: {label} ({series_id})")
            data[label] = fred.get_series(series_id, start_date)
        except Exception as e:
            print(f"Error fetching {label}: {e}")

    return pd.DataFrame(data)

# Normalize the data
def normalize_data(df):
    df = df.interpolate(method="time")  # Interpolate missing values
    df = df.fillna(method="bfill").fillna(method="ffill")  # Fill any remaining NaNs
    return df

# Run the script
df = fetch_fred_data(API_KEY, FRED_SERIES)
df = normalize_data(df)
df.to_csv("us_macro_data_normalized.csv")

print("Normalized data saved as us_macro_data_normalized.csv")
