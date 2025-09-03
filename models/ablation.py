import json
import os
import warnings
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict

import gymnasium.spaces as spaces  # Ensure this is imported for AblationEnvironment
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import stats

warnings.filterwarnings("ignore")

from models.v1.stock_gym import StockTradeEnv, merge_stocks
from models.v1.trade_agent import TradingAgent


@dataclass
class ExperimentConfig:
    """Configuration for ablation experiments"""

    initial_cash: float = 50000
    num_stocks: int = 400
    num_episodes: int = 30
    gamma: float = 0.75
    lr: float = 0.001
    random_seed: int = 42


class AblationEnvironment(StockTradeEnv):
    """Modified environment for ablation studies"""

    def __init__(self, data, feature_set: str = "full", **kwargs):
        """
        Initialize environment with specific feature set

        Args:
            data: Stock data
            feature_set: "full" (with macros) or "minimal" (without macros)
        """
        super().__init__(data, **kwargs)
        self.feature_set = feature_set
        self.macro_features = [
            "GDP",
            "GDP Growth",
            "CPI",
            "Core CPI",
            "PPI",
            "Unemployment Rate",
            "Labor Force Participation Rate",
            "Fed Funds Rate",
            "10Y Treasury Yield",
            "Money Supply (M2)",
            "Consumer Confidence",
            "Housing Starts",
            "Home Price Index",
            "Corporate Bond Yield",
            "Oil Prices (WTI)",
            "Trade Balance",
            "USD Index",
        ]
        self.base_features = [
            "Stock_Return",
            "Market_Return",
            "Beta",
            "Treynor_Ratio",
            "S&P 500 Index",
            "VIX (Volatility Index)",
        ]

        # Adjust observation space based on feature set
        if feature_set == "minimal":
            self.observation_space = spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(len(self.base_features) * self.num_stocks,),
                dtype=np.float32,
            )
        elif feature_set == "full":
            total_features = len(self.base_features) + len(self.macro_features)
            self.observation_space = spaces.Box(
                low=-float("inf"),
                high=float("inf"),
                shape=(total_features * self.num_stocks,),
                dtype=np.float32,
            )

    def _get_observations(self):
        """Get observations based on feature set"""
        obs = []

        for ticker in self.selected_tickers:
            if ticker not in self.available_stocks.index:
                # Handle missing ticker based on feature set
                if self.feature_set == "minimal":
                    obs.extend([np.nan] * len(self.base_features))
                else:
                    total_features = len(self.base_features) + len(self.macro_features)
                    obs.extend([np.nan] * total_features)
                continue

            stock_data = self.available_stocks.loc[ticker]

            if self.curr_iter >= len(stock_data):
                if self.feature_set == "minimal":
                    obs.extend([np.nan] * len(self.base_features))
                else:
                    total_features = len(self.base_features) + len(self.macro_features)
                    obs.extend([np.nan] * total_features)
                continue

            row = stock_data.iloc[self.curr_iter]

            # Always include base features
            base_obs = [row.get(feature, np.nan) for feature in self.base_features]
            obs.extend(base_obs)

            # Include macro features only for full model
            if self.feature_set == "full":
                macro_obs = [
                    row.get(feature, np.nan) for feature in self.macro_features
                ]
                obs.extend(macro_obs)

        # Convert to NumPy array and interpolate missing values
        obs_array = np.array(obs, dtype=np.float32)
        obs_array = (
            pd.Series(obs_array)
            .interpolate(method="linear", limit_direction="both")
            .to_numpy()
        )

        # Replace any remaining NaNs with 0
        obs_array = np.nan_to_num(obs_array, nan=0.0)

        return obs_array


class AblationAgent(TradingAgent):
    """Modified agent for ablation studies"""

    def __init__(self, env, experiment_name: str = "default", **kwargs):
        super().__init__(env, **kwargs)
        self.experiment_name = experiment_name
        self.episode_metrics = defaultdict(list)

    def train_step(self, state, action, reward, next_state, done, episode, step):
        """Enhanced training step with metric tracking"""
        # Call parent training step
        super().train_step(state, action, reward, next_state, done, episode, step)

        # Track additional metrics
        self.episode_metrics[episode].append(
            {
                "step": step,
                "reward": reward,
                "epsilon": self.epsilon,
                "portfolio_value": self.env.portfolio_value(),
                "action": action,
            }
        )


class AblationStudy:
    """Main class for running ablation studies"""

    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.results = {}
        self.data = merge_stocks()

    def run_experiment(self, feature_set: str, experiment_name: str) -> Dict:
        """Run single experiment with given feature set"""
        print(f"\n{'='*60}")
        print(f"Running {experiment_name} experiment...")
        print(f"Feature set: {feature_set}")
        print(f"{'='*60}")

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)
        torch.manual_seed(self.config.random_seed)

        # Create environment
        env = AblationEnvironment(
            self.data,
            feature_set=feature_set,
            initial_cash=self.config.initial_cash,
            num_stocks=self.config.num_stocks,
        )

        # Create agent
        agent = AblationAgent(
            env,
            experiment_name=experiment_name,
            gamma=self.config.gamma,
            lr=self.config.lr,
        )

        # Training metrics
        episode_rewards = []
        episode_portfolio_values = []
        episode_final_values = []
        episode_actions = []
        episode_epsilon_values = []

        best_portfolio = float("-inf")
        best_episode = -1

        # Training loop
        for episode in range(self.config.num_episodes):
            if episode % 10 == 0:
                print(f"Episode {episode}/{self.config.num_episodes}")

            state, _ = env.reset()
            done = False
            total_reward = 0
            step = 0
            episode_actions_count = defaultdict(int)

            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)

                # Clip reward
                reward = np.clip(reward, -10, 10)

                # Train agent
                agent.train_step(state, action, reward, next_state, done, episode, step)

                # Update tracking
                state = next_state
                total_reward += reward
                episode_actions_count[action] += 1
                step += 1

            # Record episode metrics
            final_portfolio_value = env.portfolio_value()
            episode_rewards.append(total_reward)
            # Ensure the values appended are plain Python lists if they originated from NumPy arrays
            # The .copy() might return a list of floats, or if env.portfolio_values internally
            # stores numpy arrays, this ensures deep conversion before saving.
            current_portfolio_values_step = env.portfolio_values.copy()
            if isinstance(current_portfolio_values_step, np.ndarray):
                episode_portfolio_values.append(current_portfolio_values_step.tolist())
            else:
                episode_portfolio_values.append(current_portfolio_values_step)

            episode_final_values.append(final_portfolio_value)
            episode_actions.append(
                dict(episode_actions_count)
            )  # Convert defaultdict to dict
            episode_epsilon_values.append(agent.epsilon)

            # Track best model
            if final_portfolio_value > best_portfolio:
                best_portfolio = final_portfolio_value
                best_episode = episode
                torch.save(
                    agent.model.state_dict(), f"best_{experiment_name}_model.pth"
                )

        # Calculate performance metrics
        results = self.calculate_performance_metrics(
            episode_rewards,
            episode_final_values,
            episode_portfolio_values,
            episode_actions,
            episode_epsilon_values,
            experiment_name,
        )

        results["best_episode"] = best_episode
        results["best_portfolio_value"] = best_portfolio

        return results

    def calculate_performance_metrics(
        self,
        episode_rewards,
        episode_final_values,
        episode_portfolio_values,
        episode_actions,
        episode_epsilon_values,
        experiment_name,
    ) -> Dict:
        """Calculate comprehensive performance metrics"""

        # Basic statistics
        final_values = np.array(episode_final_values)
        rewards = np.array(episode_rewards)

        # Calculate returns
        returns = (final_values - self.config.initial_cash) / self.config.initial_cash

        # Ensure all nested lists in episode_portfolio_values are plain Python lists
        # This is a safe guard against any lingering NumPy array objects.
        sanitized_portfolio_values = []
        for pv_list in episode_portfolio_values:
            if isinstance(pv_list, np.ndarray):
                sanitized_portfolio_values.append(pv_list.tolist())
            elif isinstance(pv_list, list):
                # Ensure all elements within the list are primitive types (e.g., floats)
                sanitized_pv_list = [
                    float(x) for x in pv_list
                ]  # Cast to float for consistency
                sanitized_portfolio_values.append(sanitized_pv_list)
            else:
                sanitized_portfolio_values.append(pv_list)

        # Performance metrics
        metrics = {
            "experiment_name": experiment_name,
            "mean_final_value": float(np.mean(final_values)),
            "std_final_value": float(np.std(final_values)),
            "max_final_value": float(np.max(final_values)),
            "min_final_value": float(np.min(final_values)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "sharpe_ratio": float(np.mean(returns) / (np.std(returns) + 1e-8)),
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "success_rate": float(
                np.sum(final_values > self.config.initial_cash) / len(final_values)
            ),
            "episode_rewards": rewards.tolist(),
            "episode_final_values": final_values.tolist(),
            "episode_portfolio_values": sanitized_portfolio_values,  # Use the sanitized version
            "episode_actions": episode_actions,
            "episode_epsilon_values": episode_epsilon_values,
        }

        # Calculate learning curve metrics
        window_size = min(20, len(final_values) // 4)
        if (
            window_size > 0 and len(final_values) >= window_size
        ):  # Ensure enough data for rolling mean
            smoothed_values = pd.Series(final_values).rolling(window=window_size).mean()
            # Ensure the regression can be performed (needs at least 2 points)
            if len(smoothed_values.dropna()) > 1:
                metrics["learning_curve_slope"] = float(
                    stats.linregress(
                        range(len(smoothed_values.dropna())), smoothed_values.dropna()
                    )[0]
                )
            else:
                metrics["learning_curve_slope"] = (
                    0.0  # Or np.nan, or handle as appropriate
                )
        else:
            metrics["learning_curve_slope"] = (
                0.0  # No sufficient data for smoothing/regression
            )

        return metrics

    def run_full_study(self):
        """Run complete ablation study"""
        print("Starting Ablation Study...")
        print(f"Configuration: {self.config}")

        # Run minimal experiment (without macros)
        self.results["minimal"] = self.run_experiment("minimal", "minimal")

        # Run full experiment (with macros)
        self.results["full"] = self.run_experiment("full", "full")

        # Compare results
        self.comparison_results = self.compare_experiments()

        # Save results
        self.save_results()

        # Generate visualizations
        self.generate_visualizations()

        print("\n" + "=" * 60)
        print("ABLATION STUDY COMPLETE")
        print("=" * 60)

        return self.results, self.comparison_results

    def compare_experiments(self) -> Dict:
        """Compare minimal vs full experiments"""
        minimal = self.results["minimal"]
        full = self.results["full"]

        comparison = {
            "improvement_metrics": {
                "mean_final_value_improvement": full["mean_final_value"]
                - minimal["mean_final_value"],
                "mean_return_improvement": full["mean_return"] - minimal["mean_return"],
                "sharpe_ratio_improvement": full["sharpe_ratio"]
                - minimal["sharpe_ratio"],
                "success_rate_improvement": full["success_rate"]
                - minimal["success_rate"],
                "learning_curve_improvement": full.get("learning_curve_slope", 0)
                - minimal.get("learning_curve_slope", 0),
            },
            "relative_improvement": {
                "mean_final_value_pct": (
                    (full["mean_final_value"] - minimal["mean_final_value"])
                    / (
                        minimal["mean_final_value"] + 1e-8
                    )  # Added epsilon for division by zero
                )
                * 100,
                "mean_return_pct": (
                    (full["mean_return"] - minimal["mean_return"])
                    / (abs(minimal["mean_return"]) + 1e-8)
                )
                * 100,
                "sharpe_ratio_pct": (
                    (full["sharpe_ratio"] - minimal["sharpe_ratio"])
                    / (abs(minimal["sharpe_ratio"]) + 1e-8)
                )
                * 100,
                "success_rate_pct": (
                    (full["success_rate"] - minimal["success_rate"])
                    / (minimal["success_rate"] + 1e-8)
                )
                * 100,
            },
            "statistical_significance": {},
        }

        # Statistical significance tests
        try:
            # T-test for final values
            # Ensure there's enough data for t-test
            if (
                len(full["episode_final_values"]) > 1
                and len(minimal["episode_final_values"]) > 1
            ):
                t_stat, p_value = stats.ttest_ind(
                    full["episode_final_values"], minimal["episode_final_values"]
                )
                comparison["statistical_significance"]["final_values_ttest"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            else:
                comparison["statistical_significance"]["final_values_ttest"] = {
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "note": "Not enough data for t-test",
                }

            # T-test for rewards
            if len(full["episode_rewards"]) > 1 and len(minimal["episode_rewards"]) > 1:
                t_stat, p_value = stats.ttest_ind(
                    full["episode_rewards"], minimal["episode_rewards"]
                )
                comparison["statistical_significance"]["rewards_ttest"] = {
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            else:
                comparison["statistical_significance"]["rewards_ttest"] = {
                    "t_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "note": "Not enough data for t-test",
                }

            # Mann-Whitney U test (non-parametric)
            if (
                len(full["episode_final_values"]) > 0
                and len(minimal["episode_final_values"]) > 0
            ):
                u_stat, p_value = stats.mannwhitneyu(
                    full["episode_final_values"],
                    minimal["episode_final_values"],
                    alternative="two-sided",
                )
                comparison["statistical_significance"]["final_values_mannwhitney"] = {
                    "u_statistic": float(u_stat),
                    "p_value": float(p_value),
                    "significant": p_value < 0.05,
                }
            else:
                comparison["statistical_significance"]["final_values_mannwhitney"] = {
                    "u_statistic": np.nan,
                    "p_value": np.nan,
                    "significant": False,
                    "note": "Not enough data for Mann-Whitney U test",
                }

        except Exception as e:
            print(f"Error in statistical tests: {e}")
            comparison["statistical_significance"]["error"] = str(e)

        return comparison

    def save_results(self):
        """Save all results to JSON files"""
        os.makedirs("ablation_results", exist_ok=True)

        try:
            # Save individual results
            with open("ablation_results/minimal_results.json", "w") as f:
                json.dump(self.results["minimal"], f, indent=2)
            print("Saved ablation_results/minimal_results.json")

            with open("ablation_results/full_results.json", "w") as f:
                json.dump(self.results["full"], f, indent=2)
            print("Saved ablation_results/full_results.json")

            # Save comparison results
            with open("ablation_results/comparison_results.json", "w") as f:
                json.dump(self.comparison_results, f, indent=2)
            print("Saved ablation_results/comparison_results.json")

            # Save combined results
            combined_results = {
                "minimal": self.results["minimal"],
                "full": self.results["full"],
                "comparison": self.comparison_results,
                "config": {
                    "initial_cash": self.config.initial_cash,
                    "num_stocks": self.config.num_stocks,
                    "num_episodes": self.config.num_episodes,
                    "gamma": self.config.gamma,
                    "lr": self.config.lr,
                    "random_seed": self.config.random_seed,
                },
            }

            with open("ablation_results/complete_ablation_study.json", "w") as f:
                json.dump(combined_results, f, indent=2)
            print("Saved ablation_results/complete_ablation_study.json")

            print("Results saved to ablation_results/ directory")

        except TypeError as e:
            print(f"ERROR: A TypeError occurred during JSON serialization: {e}")
            print(
                "Please check if all data types within the results dictionaries are JSON serializable (e.g., convert NumPy arrays to lists)."
            )
            # For more detailed debugging, you could print parts of the dictionary here:
            # print(f"Problematic minimal results (first few keys): {dict(list(self.results['minimal'].items())[:5])}")
            # print(f"Problematic full results (first few keys): {dict(list(self.results['full'].items())[:5])}")
        except Exception as e:
            print(f"An unexpected error occurred during JSON saving: {e}")

    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        plt.style.use("seaborn-v0_8")
        fig = plt.figure(figsize=(20, 15))

        # 1. Portfolio Value Comparison
        ax1 = plt.subplot(2, 3, 1)
        episodes = range(self.config.num_episodes)
        plt.plot(
            episodes,
            self.results["minimal"]["episode_final_values"],
            label="Without Macros",
            alpha=0.7,
            linewidth=2,
        )
        plt.plot(
            episodes,
            self.results["full"]["episode_final_values"],
            label="With Macros",
            alpha=0.7,
            linewidth=2,
        )
        plt.title("Portfolio Value Over Episodes", fontsize=14, fontweight="bold")
        plt.xlabel("Episode")
        plt.ylabel("Final Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. Learning Curves (Smoothed)
        ax2 = plt.subplot(2, 3, 2)
        window = min(20, self.config.num_episodes // 4)
        minimal_smooth = (
            pd.Series(self.results["minimal"]["episode_final_values"])
            .rolling(window=window)
            .mean()
        )
        full_smooth = (
            pd.Series(self.results["full"]["episode_final_values"])
            .rolling(window=window)
            .mean()
        )

        plt.plot(
            episodes, minimal_smooth, label="Without Macros (Smoothed)", linewidth=3
        )
        plt.plot(episodes, full_smooth, label="With Macros (Smoothed)", linewidth=3)
        plt.title("Learning Curves (Smoothed)", fontsize=14, fontweight="bold")
        plt.xlabel("Episode")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 3. Performance Distribution
        ax3 = plt.subplot(2, 3, 3)
        plt.hist(
            self.results["minimal"]["episode_final_values"],
            bins=20,
            alpha=0.6,
            label="Without Macros",
            density=True,
        )
        plt.hist(
            self.results["full"]["episode_final_values"],
            bins=20,
            alpha=0.6,
            label="With Macros",
            density=True,
        )
        plt.title("Portfolio Value Distribution", fontsize=14, fontweight="bold")
        plt.xlabel("Final Portfolio Value ($)")
        plt.ylabel("Density")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 4. Reward Comparison
        ax4 = plt.subplot(2, 3, 4)
        plt.plot(
            episodes,
            self.results["minimal"]["episode_rewards"],
            label="Without Macros",
            alpha=0.7,
        )
        plt.plot(
            episodes,
            self.results["full"]["episode_rewards"],
            label="With Macros",
            alpha=0.7,
        )
        plt.title("Episode Rewards", fontsize=14, fontweight="bold")
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Performance Metrics Comparison
        ax5 = plt.subplot(2, 3, 5)
        metrics = ["Mean Return", "Sharpe Ratio", "Success Rate"]
        minimal_vals = [
            self.results["minimal"]["mean_return"],
            self.results["minimal"]["sharpe_ratio"],
            self.results["minimal"]["success_rate"],
        ]
        full_vals = [
            self.results["full"]["mean_return"],
            self.results["full"]["sharpe_ratio"],
            self.results["full"]["success_rate"],
        ]

        x = np.arange(len(metrics))
        width = 0.35

        plt.bar(x - width / 2, minimal_vals, width, label="Without Macros", alpha=0.8)
        plt.bar(x + width / 2, full_vals, width, label="With Macros", alpha=0.8)
        plt.title("Performance Metrics Comparison", fontsize=14, fontweight="bold")
        plt.xlabel("Metrics")
        plt.ylabel("Values")
        plt.xticks(x, metrics)
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 6. Improvement Summary
        ax6 = plt.subplot(2, 3, 6)
        improvements = [
            self.comparison_results["improvement_metrics"][
                "mean_final_value_improvement"
            ],
            self.comparison_results["improvement_metrics"]["mean_return_improvement"],
            self.comparison_results["improvement_metrics"]["sharpe_ratio_improvement"],
            self.comparison_results["improvement_metrics"]["success_rate_improvement"],
        ]
        improvement_labels = [
            "Portfolio Value",
            "Mean Return",
            "Sharpe Ratio",
            "Success Rate",
        ]

        colors = ["green" if x > 0 else "red" for x in improvements]
        bars = plt.bar(improvement_labels, improvements, color=colors, alpha=0.7)
        plt.title("Improvement with Macro Features", fontsize=14, fontweight="bold")
        plt.xlabel("Metrics")
        plt.ylabel("Improvement")
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)

        # Add value labels on bars
        for bar, val in zip(bars, improvements):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{val:.4f}",
                ha="center",
                va="bottom" if height > 0 else "top",
            )

        plt.tight_layout()
        plt.savefig(
            "ablation_results/ablation_study_visualization.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

        # Additional detailed plots
        self.generate_detailed_plots()

    def generate_detailed_plots(self):
        """Generate additional detailed visualizations"""
        # Statistical significance visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # Box plot comparison
        ax1.boxplot(
            [
                self.results["minimal"]["episode_final_values"],
                self.results["full"]["episode_final_values"],
            ],
            labels=["Without Macros", "With Macros"],
        )
        ax1.set_title("Portfolio Value Distribution Comparison")
        ax1.set_ylabel("Final Portfolio Value ($)")
        ax1.grid(True, alpha=0.3)

        # Cumulative improvement
        ax2.plot(
            np.cumsum(self.results["minimal"]["episode_final_values"]),
            label="Without Macros",
            linewidth=2,
        )
        ax2.plot(
            np.cumsum(self.results["full"]["episode_final_values"]),
            label="With Macros",
            linewidth=2,
        )
        ax2.set_title("Cumulative Portfolio Value")
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Cumulative Value ($)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # Volatility comparison
        window = 10
        minimal_vol = (
            pd.Series(self.results["minimal"]["episode_final_values"])
            .rolling(window=window)
            .std()
        )
        full_vol = (
            pd.Series(self.results["full"]["episode_final_values"])
            .rolling(window=window)
            .std()
        )

        ax3.plot(minimal_vol, label="Without Macros", alpha=0.7)
        ax3.plot(full_vol, label="With Macros", alpha=0.7)
        ax3.set_title("Portfolio Volatility (Rolling Standard Deviation)")
        ax3.set_xlabel("Episode")
        ax3.set_ylabel("Volatility")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # Performance improvement over time
        improvement_over_time = np.array(
            self.results["full"]["episode_final_values"]
        ) - np.array(self.results["minimal"]["episode_final_values"])
        ax4.plot(improvement_over_time, linewidth=2, color="purple")
        ax4.axhline(y=0, color="red", linestyle="--", alpha=0.5)
        ax4.set_title("Performance Improvement Over Episodes")
        ax4.set_xlabel("Episode")
        ax4.set_ylabel("Improvement ($)")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "ablation_results/detailed_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print("Visualizations saved to ablation_results/ directory")

    def print_summary(self):
        """Print comprehensive summary of results"""
        print("\n" + "=" * 80)
        print("ABLATION STUDY SUMMARY")
        print("=" * 80)

        print("\nExperiment Configuration:")
        print(f"  - Episodes: {self.config.num_episodes}")
        print(f"  - Stocks: {self.config.num_stocks}")
        print(f"  - Initial Cash: ${self.config.initial_cash:,.2f}")
        print(f"  - Random Seed: {self.config.random_seed}")

        print("\nMINIMAL MODEL (Without Macros):")
        minimal = self.results["minimal"]
        print(
            f"  - Mean Portfolio Value: ${minimal['mean_final_value']:,.2f} ± ${minimal['std_final_value']:,.2f}"
        )
        print(
            f"  - Mean Return: {minimal['mean_return']:.4f} ± {minimal['std_return']:.4f}"
        )
        print(f"  - Sharpe Ratio: {minimal['sharpe_ratio']:.4f}")
        print(f"  - Success Rate: {minimal['success_rate']:.2%}")
        print(f"  - Best Portfolio: ${minimal['max_final_value']:,.2f}")

        print("\nFULL MODEL (With Macros):")
        full = self.results["full"]
        print(
            f"  - Mean Portfolio Value: ${full['mean_final_value']:,.2f} ± ${full['std_final_value']:,.2f}"
        )
        print(f"  - Mean Return: {full['mean_return']:.4f} ± {full['std_return']:.4f}")
        print(f"  - Sharpe Ratio: {full['sharpe_ratio']:.4f}")
        print(f"  - Success Rate: {full['success_rate']:.2%}")
        print(f"  - Best Portfolio: ${full['max_final_value']:,.2f}")

        print("\nIMPROVEMENT WITH MACRO FEATURES:")
        comp = self.comparison_results
        print(
            f"  - Portfolio Value: ${comp['improvement_metrics']['mean_final_value_improvement']:,.2f} ({comp['relative_improvement']['mean_final_value_pct']:.2f}%)"
        )
        print(
            f"  - Mean Return: {comp['improvement_metrics']['mean_return_improvement']:.4f} ({comp['relative_improvement']['mean_return_pct']:.2f}%)"
        )
        print(
            f"  - Sharpe Ratio: {comp['improvement_metrics']['sharpe_ratio_improvement']:.4f} ({comp['relative_improvement']['sharpe_ratio_pct']:.2f}%)"
        )
        print(
            f"  - Success Rate: {comp['improvement_metrics']['success_rate_improvement']:.4f} ({comp['relative_improvement']['success_rate_pct']:.2f}%)"
        )

        print("\nSTATISTICAL SIGNIFICANCE:")
        if "final_values_ttest" in comp["statistical_significance"]:
            ttest = comp["statistical_significance"]["final_values_ttest"]
            print(
                f"  - T-test p-value: {ttest['p_value']:.4f} ({'Significant' if ttest['significant'] else 'Not Significant'})"
            )
            if "note" in ttest:
                print(f"    (Note: {ttest['note']})")

        if "rewards_ttest" in comp["statistical_significance"]:
            ttest_rewards = comp["statistical_significance"]["rewards_ttest"]
            print(
                f"  - Rewards T-test p-value: {ttest_rewards['p_value']:.4f} ({'Significant' if ttest_rewards['significant'] else 'Not Significant'})"
            )
            if "note" in ttest_rewards:
                print(f"    (Note: {ttest_rewards['note']})")

        if "final_values_mannwhitney" in comp["statistical_significance"]:
            utest = comp["statistical_significance"]["final_values_mannwhitney"]
            print(
                f"  - Mann-Whitney U p-value: {utest['p_value']:.4f} ({'Significant' if utest['significant'] else 'Not Significant'})"
            )
            if "note" in utest:
                print(f"    (Note: {utest['note']})")

        if "error" in comp["statistical_significance"]:
            print(
                f"  - Error during statistical tests: {comp['statistical_significance']['error']}"
            )

        print("\n" + "=" * 80)


def main():
    """Main function to run the ablation study"""
    # Configuration
    config = ExperimentConfig(
        initial_cash=50000,
        num_stocks=400,  # Reduced for faster execution
        num_episodes=40,  # Reduced for faster execution
        gamma=0.55,
        lr=0.001,
        random_seed=8008135,
    )

    # Run ablation study
    study = AblationStudy(config)
    results, comparison = study.run_full_study()

    # Print summary
    study.print_summary()

    return study


if __name__ == "__main__":
    # The gymnasium.spaces import was moved to the top with other imports for consistency.
    study = main()
