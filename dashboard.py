import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import torch
from models.dqn import DQN  # Import the DQN model


def load_stock_data() -> pd.DataFrame:
    data = []
    for root, _, files in os.walk("data/merged"):
        for file in files:
            df = pd.read_csv(os.path.join(root, file))
            df["Date"] = pd.to_datetime(df["Date"], format="%Y-%m-%d", errors="coerce")
            df = df[df["Date"] >= "2000-01-01"]
            df["Ticker"] = file.replace(".csv", "")
            data.append(df)
    return pd.concat(data)


data = load_stock_data()
stocks = data["Ticker"].unique()

st.set_page_config(page_title="Stock Trading Dashboard", layout="wide")
st.title("📈 Stock Trading DQN Dashboard")

# Portfolio State
if "portfolio" not in st.session_state:
    st.session_state.portfolio = {
        "cash": 50000,
        "holdings": defaultdict(int),
        "portfolio_value": [],
    }

# Stock Selection
selected_stock = st.selectbox("Select a Stock", stocks)
stock_data = data[data["Ticker"] == selected_stock].iloc[-1]

# Extract Features
features = [col for col in stock_data.index if col not in ["Ticker", "Date"]]


def get_stock_features():
    return {feature: stock_data[feature] for feature in features}


if "user_inputs" not in st.session_state:
    st.session_state.user_inputs = get_stock_features()

st.markdown("### Input Features")
cols = st.columns(3)
for i, feature in enumerate(features):
    with cols[i % 3]:
        st.session_state.user_inputs[feature] = st.number_input(
            f"{feature}", value=st.session_state.user_inputs[feature]
        )

if st.button("🎲 Randomize Inputs"):
    st.session_state.user_inputs = {
        feature: float(np.random.rand()) for feature in features
    }
    st.experimental_rerun()

# Convert user input to tensor
user_inputs = torch.tensor(
    list(st.session_state.user_inputs.values()), dtype=torch.float32
).unsqueeze(0)

# Load DQN model
model = DQN(input_dim=600, output_dim=3)
model.load_state_dict(torch.load("models/Ep_5_Agent.pth"))  # Load trained weights
model.eval()

# Get prediction
with torch.no_grad():
    prediction = torch.argmax(model(user_inputs)).item()

st.markdown("### Model Prediction")
if prediction == 0:
    action_text = "HOLD"
    action_color = "blue"
elif prediction == 1:
    action_text = "BUY"
    action_color = "green"
    st.session_state.portfolio["holdings"][selected_stock] += 1
    st.session_state.portfolio["cash"] -= st.session_state.user_inputs["Close"]
else:
    action_text = "SELL"
    action_color = "red"
    if st.session_state.portfolio["holdings"][selected_stock] > 0:
        st.session_state.portfolio["holdings"][selected_stock] -= 1
        st.session_state.portfolio["cash"] += st.session_state.user_inputs["Close"]

st.markdown(
    f"<h1 style='color:{action_color};'>🚀 Action: {action_text}</h1>",
    unsafe_allow_html=True,
)

# Update Portfolio Value
portfolio_value = st.session_state.portfolio["cash"] + sum(
    st.session_state.portfolio["holdings"][selected_stock]
    * st.session_state.user_inputs["Close"]
)
st.session_state.portfolio["portfolio_value"].append(portfolio_value)

# Portfolio Visualization
st.markdown("### 📊 Portfolio Value Over Time")
fig = px.line(
    y=st.session_state.portfolio["portfolio_value"], title="Portfolio Value Trend"
)
st.plotly_chart(fig)


def update_prices():
    time.sleep(5)
    new_stock_data = data[data["Ticker"] == selected_stock].sample().iloc[0]
    st.session_state.user_inputs = {
        feature: new_stock_data[feature] for feature in features
    }
    st.experimental_rerun()


if st.button("⏳ Start Live Updates"):
    while True:
        update_prices()
        time.sleep(5)
