import json

import matplotlib.pyplot as plt

data = json.load(open("Iter_1_values.json", "r"))
for episode, values in data.items():
    plt.plot([i for i in range(len(values))], values, label=f"Episode {int(episode)+1}")
plt.xlabel("Steps In Episode")
plt.ylabel("Portfolio Value")
plt.legend()
plt.title("Performance Of Agent Across Episodes")
plt.show()


data2 = json.load(open("Iter_1_holdings.json", "r"))


import plotly.express as px
import plotly.graph_objects as go

def plot_holdings_distribution(holdings, top_n=10):
    """
    Plots the distribution of stock holdings using an interactive bar chart and pie chart.
    
    Args:
        holdings (dict): Dictionary with stock tickers as keys and the number of shares as values.
        top_n (int): Number of top stocks to show individually before grouping the rest into "Others".
    """
    if not holdings:
        print("No holdings to display.")
        return
    
    # Sort holdings in descending order
    sorted_holdings = sorted(holdings.items(), key=lambda x: x[1], reverse=True)
    
    # Split into top N stocks and others
    if len(sorted_holdings) > top_n:
        top_stocks = dict(sorted_holdings[:top_n])
        other_stocks = sum(shares for _, shares in sorted_holdings[top_n:])
        top_stocks["Others"] = other_stocks  # Aggregate smaller holdings
    else:
        top_stocks = dict(sorted_holdings)

    tickers = list(top_stocks.keys())
    shares = list(top_stocks.values())

    # Bar Chart (Log Scale for better visibility)
    bar_fig = px.bar(
        x=tickers, 
        y=shares, 
        text=shares,
        labels={"x": "Stock Ticker", "y": "Number of Shares"},
        title="Stock Holdings Distribution (Top N + Others)",
        color=shares,
        color_continuous_scale="Blues"
    )
    bar_fig.update_traces(textposition='outside')
    bar_fig.update_layout(yaxis_type="log")  # Apply log scale

    # Pie Chart
    pie_fig = go.Figure(
        data=[go.Pie(
            labels=tickers, 
            values=shares, 
            textinfo="percent+label", 
            marker=dict(colors=px.colors.qualitative.Plotly)
        )]
    )
    pie_fig.update_layout(title_text="Stock Holdings Distribution (Top N + Others)")

    # Show figures
    bar_fig.show()
    pie_fig.show()

for key, val in data2.items():
    plot_holdings_distribution(val)
