import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from typing import Dict, List, Tuple

# Set style for visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

def load_portfolio_data(json_file_path: str) -> Dict:
    """Load portfolio values from a JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    return data

def calculate_daily_returns(portfolio_values: List[float]) -> np.ndarray:
    """Calculate daily returns from portfolio values."""
    # Convert to numpy array for easier calculations
    values = np.array(portfolio_values)
    # Calculate daily returns: (current_value - previous_value) / previous_value
    daily_returns = (values[1:] - values[:-1]) / values[:-1]
    return daily_returns

def calculate_metrics(portfolio_values: List[float], risk_free_rate: float = 0.02) -> Dict:
    """
    Calculate portfolio performance metrics.
    
    Args:
        portfolio_values: List of daily portfolio values
        risk_free_rate: Annual risk-free rate (default 2%)
        
    Returns:
        Dictionary containing calculated metrics
    """
    initial_value = portfolio_values[0]
    final_value = portfolio_values[-1]
    
    # Calculate daily returns
    daily_returns = calculate_daily_returns(portfolio_values)
    
    # Number of trading days
    n_days = len(portfolio_values) - 1
    
    # Cumulative return
    cumulative_return = (final_value - initial_value) / initial_value
    
    # Annualized return (geometric average)
    # Formula: (final_value/initial_value)^(252/n_days) - 1
    # Where 252 is the typical number of trading days in a year
    annualized_return = (final_value / initial_value) ** (252 / n_days) - 1
    
    # Annualized volatility
    # Formula: daily_returns_std * sqrt(252)
    annualized_volatility = np.std(daily_returns) * np.sqrt(252)
    
    # Sharpe ratio
    # Formula: (annualized_return - risk_free_rate) / annualized_volatility
    sharpe_ratio = (annualized_return - risk_free_rate) / annualized_volatility
    
    # Maximum drawdown
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_values)
    # Calculate drawdowns
    drawdowns = (running_max - portfolio_values) / running_max
    # Find maximum drawdown
    max_drawdown = np.max(drawdowns)
    
    # Return all metrics in a dictionary
    return {
        "cumulative_return": cumulative_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
        "daily_returns": daily_returns.tolist(),
        "portfolio_values": portfolio_values,
        "drawdowns": drawdowns.tolist()
    }

def analyze_all_episodes(data: Dict, risk_free_rate: float = 0.02) -> Tuple[Dict, Dict]:
    """
    Analyze all episodes in the data.
    
    Args:
        data: Dictionary containing portfolio values for each episode
        risk_free_rate: Annual risk-free rate
        
    Returns:
        Tuple of (episode_metrics, summary_metrics)
    """
    episode_metrics = {}
    
    # Calculate metrics for each episode
    for episode, portfolio_values in data.items():
        episode_metrics[episode] = calculate_metrics(portfolio_values, risk_free_rate)
    
    # Calculate summary statistics across episodes
    summary_metrics = {
        "cumulative_return": {
            "mean": np.mean([m["cumulative_return"] for m in episode_metrics.values()]),
            "std": np.std([m["cumulative_return"] for m in episode_metrics.values()]),
            "min": np.min([m["cumulative_return"] for m in episode_metrics.values()]),
            "max": np.max([m["cumulative_return"] for m in episode_metrics.values()]),
        },
        "annualized_return": {
            "mean": np.mean([m["annualized_return"] for m in episode_metrics.values()]),
            "std": np.std([m["annualized_return"] for m in episode_metrics.values()]),
            "min": np.min([m["annualized_return"] for m in episode_metrics.values()]),
            "max": np.max([m["annualized_return"] for m in episode_metrics.values()]),
        },
        "annualized_volatility": {
            "mean": np.mean([m["annualized_volatility"] for m in episode_metrics.values()]),
            "std": np.std([m["annualized_volatility"] for m in episode_metrics.values()]),
            "min": np.min([m["annualized_volatility"] for m in episode_metrics.values()]),
            "max": np.max([m["annualized_volatility"] for m in episode_metrics.values()]),
        },
        "sharpe_ratio": {
            "mean": np.mean([m["sharpe_ratio"] for m in episode_metrics.values()]),
            "std": np.std([m["sharpe_ratio"] for m in episode_metrics.values()]),
            "min": np.min([m["sharpe_ratio"] for m in episode_metrics.values()]),
            "max": np.max([m["sharpe_ratio"] for m in episode_metrics.values()]),
        },
        "max_drawdown": {
            "mean": np.mean([m["max_drawdown"] for m in episode_metrics.values()]),
            "std": np.std([m["max_drawdown"] for m in episode_metrics.values()]),
            "min": np.min([m["max_drawdown"] for m in episode_metrics.values()]),
            "max": np.max([m["max_drawdown"] for m in episode_metrics.values()]),
        }
    }
    
    return episode_metrics, summary_metrics

def visualize_portfolio_performance(episode_metrics: Dict, summary_metrics: Dict, save_path: str = None):
    """
    Create comprehensive visualizations of portfolio performance.
    
    Args:
        episode_metrics: Dictionary containing metrics for each episode
        summary_metrics: Dictionary containing summary statistics
        save_path: Path to save the visualization (optional)
    """
    # Set up the figure
    fig = plt.figure(figsize=(20, 24))
    gs = GridSpec(5, 2, figure=fig)
    
    # 1. Portfolio Values Over Time (for each episode)
    ax1 = fig.add_subplot(gs[0, :])
    for episode, metrics in episode_metrics.items():
        ax1.plot(metrics["portfolio_values"], alpha=0.3, label=f"Episode {episode}")
    
    # Also plot the mean portfolio value
    # First, ensure all episodes have the same length
    min_length = min(len(metrics["portfolio_values"]) for metrics in episode_metrics.values())
    portfolio_values_array = np.array([metrics["portfolio_values"][:min_length] for metrics in episode_metrics.values()])
    mean_portfolio_values = np.mean(portfolio_values_array, axis=0)
    
    ax1.plot(mean_portfolio_values, linewidth=3, color='red', label="Mean")
    ax1.set_title("Portfolio Values Over Time", fontsize=16)
    ax1.set_xlabel("Trading Days", fontsize=14)
    ax1.set_ylabel("Portfolio Value", fontsize=14)
    ax1.legend(loc='upper left', fontsize=10)
    
    # 2. Cumulative Returns Distribution
    ax2 = fig.add_subplot(gs[1, 0])
    cumulative_returns = [metrics["cumulative_return"] for metrics in episode_metrics.values()]
    sns.histplot(cumulative_returns, kde=True, ax=ax2)
    ax2.axvline(summary_metrics["cumulative_return"]["mean"], color='red', linestyle='--', 
               label=f"Mean: {summary_metrics['cumulative_return']['mean']:.4f}")
    ax2.set_title("Distribution of Cumulative Returns", fontsize=16)
    ax2.set_xlabel("Cumulative Return", fontsize=14)
    ax2.legend()
    
    # 3. Annualized Returns vs. Volatility Scatter Plot (with Sharpe ratio)
    ax3 = fig.add_subplot(gs[1, 1])
    returns = [metrics["annualized_return"] for metrics in episode_metrics.values()]
    volatilities = [metrics["annualized_volatility"] for metrics in episode_metrics.values()]
    sharpe_ratios = [metrics["sharpe_ratio"] for metrics in episode_metrics.values()]
    
    scatter = ax3.scatter(volatilities, returns, c=sharpe_ratios, cmap='coolwarm', s=100, alpha=0.7)
    cbar = plt.colorbar(scatter, ax=ax3)
    cbar.set_label('Sharpe Ratio', fontsize=14)
    
    # Add a diagonal line for constant Sharpe ratios
    risk_free_rate = 0.02  # Assuming the same risk-free rate used in calculations
    for sr in [0.5, 1.0, 1.5, 2.0]:
        x_vals = np.linspace(0, max(volatilities)*1.1, 100)
        y_vals = risk_free_rate + sr * x_vals
        ax3.plot(x_vals, y_vals, 'k--', alpha=0.3, label=f"SR={sr}")
    
    ax3.set_title("Risk-Return Profile", fontsize=16)
    ax3.set_xlabel("Annualized Volatility", fontsize=14)
    ax3.set_ylabel("Annualized Return", fontsize=14)
    ax3.legend(loc='upper left')
    
    # 4. Max Drawdown Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    max_drawdowns = [metrics["max_drawdown"] for metrics in episode_metrics.values()]
    sns.histplot(max_drawdowns, kde=True, ax=ax4)
    ax4.axvline(summary_metrics["max_drawdown"]["mean"], color='red', linestyle='--', 
               label=f"Mean: {summary_metrics['max_drawdown']['mean']:.4f}")
    ax4.set_title("Distribution of Maximum Drawdowns", fontsize=16)
    ax4.set_xlabel("Maximum Drawdown", fontsize=14)
    ax4.legend()
    
    # 5. Sharpe Ratio Distribution
    ax5 = fig.add_subplot(gs[2, 1])
    sharpe_ratios = [metrics["sharpe_ratio"] for metrics in episode_metrics.values()]
    sns.histplot(sharpe_ratios, kde=True, ax=ax5)
    ax5.axvline(summary_metrics["sharpe_ratio"]["mean"], color='red', linestyle='--', 
               label=f"Mean: {summary_metrics['sharpe_ratio']['mean']:.4f}")
    ax5.set_title("Distribution of Sharpe Ratios", fontsize=16)
    ax5.set_xlabel("Sharpe Ratio", fontsize=14)
    ax5.legend()
    
    # 6. Metrics evolution across episodes
    ax6 = fig.add_subplot(gs[3, :])
    episodes = list(episode_metrics.keys())
    metrics_by_episode = {
        "Cumulative Return": [episode_metrics[ep]["cumulative_return"] for ep in episodes],
        "Annualized Return": [episode_metrics[ep]["annualized_return"] for ep in episodes],
        "Sharpe Ratio": [episode_metrics[ep]["sharpe_ratio"] for ep in episodes]
    }
    
    for metric_name, metric_values in metrics_by_episode.items():
        ax6.plot(episodes, metric_values, marker='o', label=metric_name)
    
    ax6.set_title("Evolution of Performance Metrics Across Episodes", fontsize=16)
    ax6.set_xlabel("Episode", fontsize=14)
    ax6.set_ylabel("Metric Value", fontsize=14)
    ax6.legend()
    
    # 7. Drawdown visualization for best episode
    best_episode = max(episode_metrics.items(), key=lambda x: x[1]["sharpe_ratio"])[0]
    ax7 = fig.add_subplot(gs[4, 0])
    
    drawdowns = episode_metrics[best_episode]["drawdowns"]
    ax7.fill_between(range(len(drawdowns)), 0, drawdowns, color='red', alpha=0.3)
    ax7.set_title(f"Drawdowns for Best Episode (Episode {best_episode})", fontsize=16)
    ax7.set_xlabel("Trading Days", fontsize=14)
    ax7.set_ylabel("Drawdown", fontsize=14)
    ax7.set_ylim(0, max(drawdowns) * 1.1)
    
    # 8. Summary metrics table
    ax8 = fig.add_subplot(gs[4, 1])
    ax8.axis('off')
    
    table_data = []
    metrics_names = ["Cumulative Return", "Annualized Return", "Annualized Volatility", 
                     "Sharpe Ratio", "Max Drawdown"]
    
    stats = ["mean", "std", "min", "max"]
    
    for metric in metrics_names:
        metric_key = metric.lower().replace(" ", "_")
        row = [metric]
        for stat in stats:
            row.append(f"{summary_metrics[metric_key][stat]:.4f}")
        table_data.append(row)
    
    table = ax8.table(
        cellText=table_data,
        colLabels=["Metric", "Mean", "Std Dev", "Min", "Max"],
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 2)
    ax8.set_title("Summary Statistics", fontsize=16)
    
    # Add overall title
    plt.suptitle("Portfolio Performance Analysis Across Episodes", fontsize=20, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def generate_performance_report(episode_metrics: Dict, summary_metrics: Dict) -> str:
    """Generate a text report summarizing portfolio performance."""
    report = []
    
    report.append("# Portfolio Performance Analysis Report")
    report.append("\n## Summary Statistics\n")
    
    # Create a summary table
    report.append("| Metric | Mean | Std Dev | Min | Max |")
    report.append("|--------|------|---------|-----|-----|")
    
    metrics_names = {
        "cumulative_return": "Cumulative Return",
        "annualized_return": "Annualized Return",
        "annualized_volatility": "Annualized Volatility",
        "sharpe_ratio": "Sharpe Ratio",
        "max_drawdown": "Max Drawdown"
    }
    
    for metric_key, metric_name in metrics_names.items():
        stats = summary_metrics[metric_key]
        report.append(f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |")
    
    report.append("\n## Best Episodes\n")
    
    # Find best episode by different metrics
    best_return_episode = max(episode_metrics.items(), key=lambda x: x[1]["annualized_return"])[0]
    best_sharpe_episode = max(episode_metrics.items(), key=lambda x: x[1]["sharpe_ratio"])[0]
    min_drawdown_episode = min(episode_metrics.items(), key=lambda x: x[1]["max_drawdown"])[0]
    
    report.append(f"- **Best Annualized Return**: Episode {best_return_episode} with {episode_metrics[best_return_episode]['annualized_return']:.4f}")
    report.append(f"- **Best Sharpe Ratio**: Episode {best_sharpe_episode} with {episode_metrics[best_sharpe_episode]['sharpe_ratio']:.4f}")
    report.append(f"- **Minimum Max Drawdown**: Episode {min_drawdown_episode} with {episode_metrics[min_drawdown_episode]['max_drawdown']:.4f}")
    
    report.append("\n## Performance Trend\n")
    
    # Calculate if performance is improving over episodes
    first_half = list(episode_metrics.keys())[:len(episode_metrics)//2]
    second_half = list(episode_metrics.keys())[len(episode_metrics)//2:]
    
    first_half_sharpe = np.mean([episode_metrics[ep]["sharpe_ratio"] for ep in first_half])
    second_half_sharpe = np.mean([episode_metrics[ep]["sharpe_ratio"] for ep in second_half])
    
    if second_half_sharpe > first_half_sharpe:
        report.append("Performance is improving over episodes. The average Sharpe ratio in the second half "
                     f"of episodes ({second_half_sharpe:.4f}) is higher than in the first half ({first_half_sharpe:.4f}).")
    else:
        report.append("Performance is not improving over episodes. The average Sharpe ratio in the second half "
                     f"of episodes ({second_half_sharpe:.4f}) is lower than in the first half ({first_half_sharpe:.4f}).")
    
    return "\n".join(report)

def main(json_file_path: str, risk_free_rate: float = 0.02, save_plot: bool = True, save_report: bool = True):
    """
    Main function to analyze portfolio performance from a JSON file.
    
    Args:
        json_file_path: Path to the JSON file containing portfolio values
        risk_free_rate: Annual risk-free rate (default 2%)
        save_plot: Whether to save the plot (default True)
        save_report: Whether to save the report (default True)
    """
    # Load data
    print(f"Loading data from {json_file_path}...")
    data = load_portfolio_data(json_file_path)
    
    # Analyze all episodes
    print("Analyzing episodes...")
    episode_metrics, summary_metrics = analyze_all_episodes(data, risk_free_rate)
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_save_path = None
    if save_plot:
        plot_save_path = json_file_path.replace('.json', '_analysis.png')
    visualize_portfolio_performance(episode_metrics, summary_metrics, plot_save_path)
    
    # Generate report
    print("Generating performance report...")
    report = generate_performance_report(episode_metrics, summary_metrics)
    print("\n" + report)
    
    if save_report:
        report_save_path = json_file_path.replace('.json', '_report.md')
        with open(report_save_path, 'w') as f:
            f.write(report)
        print(f"Report saved to {report_save_path}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze portfolio performance from a JSON file.")
    parser.add_argument("json_file", help="Path to the JSON file containing portfolio values")
    parser.add_argument("--risk-free-rate", type=float, default=0.02, help="Annual risk-free rate (default 2%)")
    parser.add_argument("--no-save-plot", action="store_true", help="Don't save the plot")
    parser.add_argument("--no-save-report", action="store_true", help="Don't save the report")
    
    args = parser.parse_args()
    
    main(args.json_file, args.risk_free_rate, not args.no_save_plot, not args.no_save_report)

# Expected JSON structure:
# {
#   "episode_1": [100, 102, 105, ...],  # daily portfolio values
#   "episode_2": [100, 101, 103, ...],
#   ...
#   "episode_25": [100, 99, 102, ...]
# }
