import json
import logging
import os
from collections import deque
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.v1.dqn import DQN
from models.v1.stock_gym import StockTradeEnv, merge_stocks
from tqdm import tqdm


class ExperienceReplayBuffer:
    """Experience replay buffer for more stable and efficient training"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)


class OptimizedTradingAgent:
    def __init__(
        self,
        env,
        gamma=0.95,
        lr=0.001,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=100,
        min_buffer_size=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_buffer_size = min_buffer_size

        # Device optimization
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Agent using device: {self.device}")

        # Networks with target network for stability
        self.q_network = DQN(env.observation_space.shape[0], env.action_space.n).to(
            self.device
        )
        self.target_network = DQN(
            env.observation_space.shape[0], env.action_space.n
        ).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer with better parameters for faster convergence
        self.optimizer = optim.AdamW(
            self.q_network.parameters(), lr=lr, weight_decay=1e-4, eps=1e-7
        )

        # Learning rate scheduler for adaptive learning
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.8, patience=10
        )

        self.criterion = nn.SmoothL1Loss()  # More stable than Huber for financial data

        # Experience replay buffer
        self.replay_buffer = ExperienceReplayBuffer(buffer_size)

        # Training metrics
        self.train_step_count = 0
        self.losses = []
        self.q_values_history = []

    def select_action(self, state, training=True):
        if training and np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state_tensor = (
            torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            self.q_values_history.append(q_values.mean().item())
        return torch.argmax(q_values).item()

    def store_experience(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train_step(self):
        """Batch training step - more efficient than single-step training"""
        if len(self.replay_buffer) < self.min_buffer_size:
            return None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q values
        current_q_values = (
            self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        )

        # Target Q values using target network (Double DQN)
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(1)
            next_q_values = (
                self.target_network(next_states)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network periodically
        self.train_step_count += 1
        if self.train_step_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Store loss for logging
        self.losses.append(loss.item())

        return loss.item()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_metrics(self):
        """Get current training metrics"""
        return {
            "epsilon": self.epsilon,
            "avg_loss": np.mean(self.losses[-100:]) if self.losses else 0,
            "avg_q_value": (
                np.mean(self.q_values_history[-100:]) if self.q_values_history else 0
            ),
            "buffer_size": len(self.replay_buffer),
            "train_steps": self.train_step_count,
        }


def setup_logging(log_dir="logs"):
    """Setup comprehensive logging"""
    os.makedirs(log_dir, exist_ok=True)

    # Create timestamped log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"trading_agent_{timestamp}.log")

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )

    logger = logging.getLogger(__name__)
    logger.info("=" * 50)
    logger.info("Trading Agent Training Started")
    logger.info("=" * 50)

    return logger


def save_training_results(results, filename="training_results.json"):
    """Save training results to JSON file"""
    os.makedirs("../stats", exist_ok=True)
    with open(f"../stats/{filename}", "w") as f:
        json.dump(results, f, indent=2)


def run_optimized_training():
    """Main training function with optimizations and logging"""

    # Setup logging
    logger = setup_logging()

    # User inputs
    try:
        initial_cash = int(input("Enter Initial Balance (default 50000): ") or 50000)
        stocks_nos = int(input("Enter Number Of Stocks (default 50): ") or 50)
        num_episodes = int(
            input("Enter Number Of Training Episodes (default 100): ") or 100
        )
        split_date = (
            input("Enter Train/Test Split Date (default 2019-01-01): ") or "2019-01-01"
        )

        # Advanced parameters
        batch_size = int(input("Batch size for training (default 64): ") or 64)
        episode_length = int(input("Episode length in days (default 252): ") or 252)

    except ValueError as e:
        logger.error(f"Invalid input: {e}")
        return

    # Log configuration
    logger.info(
        f"Configuration: Cash=${initial_cash}, Stocks={stocks_nos}, Episodes={num_episodes}"
    )
    logger.info(
        f"Split Date={split_date}, Batch Size={batch_size}, Episode Length={episode_length}"
    )

    # Load data and setup environment
    logger.info("Loading stock data...")
    try:
        full_data = merge_stocks()
        logger.info(f"Loaded data with {len(full_data)} records")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return

    # Initialize environment and agent
    env = StockTradeEnv(
        data=full_data,
        initial_cash=initial_cash,
        num_stocks=stocks_nos,
        train_test_split_date=split_date,
        mode="train",
        episode_length=episode_length,
    )

    agent = OptimizedTradingAgent(
        env=env,
        gamma=0.95,
        lr=0.001,
        buffer_size=50000,
        batch_size=batch_size,
        target_update_freq=100,
        min_buffer_size=1000,
    )

    # Training metrics storage
    training_results = {
        "episodes": [],
        "portfolio_values": [],
        "total_rewards": [],
        "losses": [],
        "epsilons": [],
        "sharpe_ratios": [],
        "max_drawdowns": [],
        "trading_costs": [],
    }

    best_portfolio_value = -np.inf
    best_model_path = "../models/optimized_dqn_best_model.pth"
    os.makedirs("../models", exist_ok=True)

    logger.info("Starting training phase...")

    # Training loop with optimizations
    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0
        episode_losses = []

        # Episode progress bar
        pbar = tqdm(
            total=env.episode_length,
            desc=f"Episode {episode + 1}/{num_episodes}",
            leave=False,
        )

        step = 0
        while not done:
            # Select action
            action = agent.select_action(state, training=True)
            next_state, reward, done, info = env.step(action)

            # Clip reward for stability
            reward = np.clip(reward, -5, 5)
            total_reward += reward

            # Store experience
            agent.store_experience(state, action, reward, next_state, done)

            # Train every few steps for efficiency
            if step % 4 == 0:  # Train every 4 steps instead of every step
                loss = agent.train_step()
                if loss is not None:
                    episode_losses.append(loss)

            state = next_state
            step += 1

            # Update progress bar
            if step % 10 == 0:  # Update every 10 steps for performance
                metrics = agent.get_metrics()
                pbar.set_postfix(
                    {
                        "Reward": f"{reward:.3f}",
                        "Portfolio": f"${env.portfolio_values[-1]:,.0f}",
                        "Loss": f'{metrics["avg_loss"]:.4f}',
                        "ε": f'{metrics["epsilon"]:.3f}',
                    }
                )
            pbar.update(1)

        pbar.close()

        # Update epsilon after episode
        agent.update_epsilon()

        # Calculate episode metrics
        final_portfolio_value = env.portfolio_values[-1]
        avg_loss = np.mean(episode_losses) if episode_losses else 0

        # Calculate Sharpe ratio
        returns = np.diff(env.portfolio_values) / (
            np.array(env.portfolio_values[:-1]) + 1e-9
        )
        sharpe_ratio = (
            np.mean(returns) / (np.std(returns) + 1e-9) * np.sqrt(252)
            if len(returns) > 1
            else 0
        )

        # Calculate max drawdown
        peak = np.maximum.accumulate(env.portfolio_values)
        drawdowns = (peak - env.portfolio_values) / peak
        max_drawdown = np.max(drawdowns)

        # Store results
        training_results["episodes"].append(episode + 1)
        training_results["portfolio_values"].append(final_portfolio_value)
        training_results["total_rewards"].append(total_reward)
        training_results["losses"].append(avg_loss)
        training_results["epsilons"].append(agent.epsilon)
        training_results["sharpe_ratios"].append(sharpe_ratio)
        training_results["max_drawdowns"].append(max_drawdown)
        training_results["trading_costs"].append(info.get("total_trading_costs", 0))

        # Update learning rate scheduler
        agent.scheduler.step(final_portfolio_value)

        # Log episode results
        logger.info(
            f"Episode {episode + 1}: Portfolio=${final_portfolio_value:,.2f}, "
            f"Reward={total_reward:.4f}, Sharpe={sharpe_ratio:.4f}, "
            f"Drawdown={max_drawdown:.2%}, Loss={avg_loss:.4f}"
        )

        # Save best model
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value
            torch.save(
                {
                    "model_state_dict": agent.q_network.state_dict(),
                    "optimizer_state_dict": agent.optimizer.state_dict(),
                    "episode": episode + 1,
                    "portfolio_value": final_portfolio_value,
                    "epsilon": agent.epsilon,
                },
                best_model_path,
            )
            logger.info(f"*** New best model saved: ${best_portfolio_value:,.2f} ***")

        # Periodic saves and evaluation
        if (episode + 1) % 20 == 0:
            save_training_results(
                training_results, f"training_checkpoint_ep_{episode + 1}.json"
            )
            logger.info(f"Checkpoint saved at episode {episode + 1}")

    # Final training results
    logger.info("Training completed!")
    logger.info(f"Best portfolio value: ${best_portfolio_value:,.2f}")
    save_training_results(training_results, "final_training_results.json")

    # Evaluation phase
    logger.info("Starting evaluation phase...")

    # Load best model
    checkpoint = torch.load(best_model_path)
    agent.q_network.load_state_dict(checkpoint["model_state_dict"])
    agent.epsilon = 0.0  # No exploration during evaluation

    # Switch to test mode
    env.set_mode("test")

    num_eval_episodes = 20  # Reduced for faster evaluation
    eval_results = []

    for episode in range(num_eval_episodes):
        state, _ = env.reset()
        done = False

        pbar = tqdm(
            total=env.episode_length,
            desc=f"Eval {episode + 1}/{num_eval_episodes}",
            leave=False,
        )

        while not done:
            action = agent.select_action(state, training=False)
            next_state, reward, done, info = env.step(action)
            state = next_state

            pbar.set_postfix({"Portfolio": f"${env.portfolio_values[-1]:,.0f}"})
            pbar.update(1)

        pbar.close()
        eval_results.append(env.portfolio_values[-1])

        logger.info(f"Eval Episode {episode + 1}: ${env.portfolio_values[-1]:,.2f}")

    # Final evaluation metrics
    avg_eval_value = np.mean(eval_results)
    std_eval_value = np.std(eval_results)

    final_results = {
        "training_episodes": num_episodes,
        "best_training_portfolio": best_portfolio_value,
        "evaluation_episodes": num_eval_episodes,
        "avg_eval_portfolio": avg_eval_value,
        "std_eval_portfolio": std_eval_value,
        "eval_results": eval_results,
        "training_results": training_results,
    }

    save_training_results(final_results, "complete_results.json")

    logger.info("=" * 50)
    logger.info("FINAL RESULTS")
    logger.info(f"Best Training Portfolio: ${best_portfolio_value:,.2f}")
    logger.info(
        f"Average Evaluation Portfolio: ${avg_eval_value:,.2f} (±${std_eval_value:,.2f})"
    )
    logger.info(f"Results saved to ../stats/ and model saved to {best_model_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    run_optimized_training()
