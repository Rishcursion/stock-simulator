import json

import pandas as pd
import plotly.graph_objects as go

data = json.load(open("Iter_1.json", "r"))


def generate_episode_candlestick_data(all_episodes_portfolio_values):
    """
    Converts step-wise portfolio values into episode-wise candlestick data.

    Args:
        all_episodes_portfolio_values: List of lists containing portfolio values per episode.

    Returns:
        DataFrame with columns ['Episode', 'Open', 'High', 'Low', 'Close']
    """
    episodes = []

    for episode_num, portfolio_values in all_episodes_portfolio_values.items():
        if len(portfolio_values) == 0:  # Skip empty episodes
            continue

        episode_data = {
            "Episode": episode_num,
            "Open": portfolio_values[0],  # First value of the episode
            "High": max(portfolio_values),  # Maximum portfolio value
            "Low": min(portfolio_values),  # Minimum portfolio value
            "Close": portfolio_values[-1],  # Last value of the episode
        }
        episodes.append(episode_data)

    return pd.DataFrame(episodes)


def plot_candlestick(df):
    """
    Plots a candlestick chart for episode-wise portfolio data.
    Assumes df contains columns: ['Episode', 'Open', 'High', 'Low', 'Close']
    """
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=df["Episode"],
                open=df["Open"],
                high=df["High"],
                low=df["Low"],
                close=df["Close"],
            )
        ]
    )

    fig.update_layout(
        title="Portfolio Value Candlestick Chart",
        xaxis_title="Episode",
        yaxis_title="Portfolio Value",
        xaxis_rangeslider_visible=False,
    )

    fig.show()


df_candlestick = generate_episode_candlestick_data(data)
plot_candlestick(df_candlestick)
