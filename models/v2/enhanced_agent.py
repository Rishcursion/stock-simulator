import json
import os
from collections import defaultdict, deque

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from models.enhanced_dqn import create_dqn_model
from models.enhanced_gym import StockTradeEnv, merge_stocks


class SequenceBuffer:
    """Enhanced replay buffer that maintains sequences for LSTM training"""

    def __init__(self, capacity=10000, sequence_length=20):
        self.buffer = deque(maxlen=capacity)
        self.sequence_length = sequence_length

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Sample sequences of experiences for LSTM training"""
        if len(self.buffer) < self.sequence_length:
            return None

        sequences = []
        for _ in range(batch_size):
            # Sample a valid starting point
            start_idx = np.random.randint(0, len(self.buffer) - self.sequence_length)
            sequence = []

            for i in range(self.sequence_length):
                sequence.append(self.buffer[start_idx + i])

            sequences.append(sequence)

        return sequences

    def __len__(self):
        return len(self.buffer)


class EnhancedLSTMTradingAgent:
    """Enhanced trading agent using LSTM-based DQN models"""

    def __init__(
        self,
        env,
        model_type="advanced_lstm",
        gamma=0.95,
        lr=0.0001,
        buffer_size=50000,
        batch_size=32,
        target_update_freq=100,
        min_replay_size=1000,
        sequence_length=20,
    ):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size
        self.sequence_length = sequence_length
        self.model_type = model_type

        # Epsilon scheduling
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01

        # Network dimensions
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.num_stocks = env.num_stocks

        print(f"State space: {self.state_size}, Action space: {self.action_size}")
        print(f"Number of stocks: {self.num_stocks}, Model type: {model_type}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create LSTM-based DQN models
        self.q_network = create_dqn_model(
            model_type=model_type,
            input_dim=self.state_size,
            output_dim=self.action_size,
            num_stocks=self.num_stocks,
            hidden_dim=256,
            num_layers=2,
            sequence_length=sequence_length,
            use_attention=True,
        )
        self.target_network = create_dqn_model(
            model_type=model_type,
            input_dim=self.state_size,
            output_dim=self.action_size,
            num_stocks=self.num_stocks,
            hidden_dim=256,
            num_layers=2,
            sequence_length=sequence_length,
            use_attention=True,
        )

        # Copy weights to target network
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.to(device)
        self.q_network.to(device)
        # Optimizer with gradient clipping support
        self.optimizer = optim.Adam(
            self.q_network.parameters(), lr=lr, weight_decay=1e-5, eps=1e-4
        )
        self.criterion = nn.SmoothL1Loss()

        # Enhanced replay buffer for sequences
        self.replay_buffer = SequenceBuffer(buffer_size, sequence_length)

        # Hidden states for LSTM
        self.hidden_state = None
        self.target_hidden_state = None

        # Training metrics
        self.training_step = 0
        self.episode_losses = []
        self.episode_rewards = []
        self.portfolio_metrics = defaultdict(list)

    def reset_hidden_states(self):
        """Reset LSTM hidden states at episode start"""
        self.hidden_state = None
        self.target_hidden_state = None

    def select_action(self, state, training=True):
        """Select action using LSTM-based Q-network with epsilon-greedy exploration"""
        if training and np.random.rand() < self.epsilon:
            if self.epsilon > 0.5:
                # Early training: prefer safer actions
                safe_actions = []
                for i in range(self.num_stocks):
                    safe_actions.extend(
                        [i * 5, i * 5 + 1, i * 5 + 3]
                    )  # Hold, small buy, small sell
                return np.random.choice(safe_actions)
            else:
                return self.env.action_space.sample()

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            if self.model_type == "ensemble":
                q_values, self.hidden_state = self.q_network(
                    state_tensor, self.hidden_state
                )
            else:
                q_values, self.hidden_state = self.q_network(
                    state_tensor, self.hidden_state
                )

        # Add exploration noise during training
        if training:
            noise = torch.randn_like(q_values) * 0.01
            q_values += noise

        return torch.argmax(q_values).item()

    def train_step(self, state, action, reward, next_state, done):
        """Enhanced training with LSTM sequence processing"""
        # Store experience
        self.replay_buffer.push(state, action, reward, next_state, done)

        # Only train if we have enough experiences
        if len(self.replay_buffer) < self.min_replay_size:
            return None

        # Sample sequences from replay buffer
        sequences = self.replay_buffer.sample(self.batch_size)
        if sequences is None:
            return None

        # Process sequences for LSTM training
        batch_states = []
        batch_actions = []
        batch_rewards = []
        batch_next_states = []
        batch_dones = []

        for sequence in sequences:
            seq_states, seq_actions, seq_rewards, seq_next_states, seq_dones = zip(
                *sequence
            )
            batch_states.append(seq_states[-1])  # Use last state in sequence
            batch_actions.append(seq_actions[-1])  # Use last action
            batch_rewards.append(seq_rewards[-1])  # Use last reward
            batch_next_states.append(seq_next_states[-1])  # Use last next_state
            batch_dones.append(seq_dones[-1])  # Use last done flag

        # Convert to tensors
        states = torch.tensor(batch_states, dtype=torch.float32)
        actions = torch.tensor(batch_actions, dtype=torch.int64)
        rewards = torch.tensor(batch_rewards, dtype=torch.float32)
        next_states = torch.tensor(batch_next_states, dtype=torch.float32)
        dones = torch.tensor(batch_dones, dtype=torch.float32)

        # Reset hidden states for training batch
        if hasattr(self.q_network, "init_hidden"):
            batch_hidden = self.q_network.init_hidden(self.batch_size, states.device)
            target_batch_hidden = self.target_network.init_hidden(
                self.batch_size, states.device
            )
        else:
            batch_hidden = None
            target_batch_hidden = None

        # Current Q-values
        if self.model_type == "ensemble":
            current_q_values_all, _ = self.q_network(states, batch_hidden)
        else:
            current_q_values_all, _ = self.q_network(states, batch_hidden)

        current_q_values = current_q_values_all.gather(1, actions.unsqueeze(1)).squeeze(
            1
        )

        # Target Q-values using Double DQN
        with torch.no_grad():
            if self.model_type == "ensemble":
                next_q_values_main, _ = self.q_network(next_states, batch_hidden)
                next_q_values_target, _ = self.target_network(
                    next_states, target_batch_hidden
                )
            else:
                next_q_values_main, _ = self.q_network(next_states, batch_hidden)
                next_q_values_target, _ = self.target_network(
                    next_states, target_batch_hidden
                )

            next_actions = next_q_values_main.argmax(1)
            next_q_values = next_q_values_target.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))

        # Compute loss
        loss = self.criterion(current_q_values, target_q_values.detach())

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()

        # Gradient clipping for LSTM stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Update target network
        if self.training_step % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        self.training_step += 1
        return loss.item()

    def evaluate_episode(self, num_eval_episodes=5):
        """Evaluate agent performance without exploration"""
        eval_returns = []
        eval_sharpe_ratios = []
        eval_max_drawdowns = []

        original_epsilon = self.epsilon
        self.epsilon = 0  # No exploration during evaluation

        for _ in range(num_eval_episodes):
            state, _ = self.env.reset()
            self.reset_hidden_states()  # Reset LSTM states for evaluation
            done = False
            episode_return = 0

            while not done:
                action = self.select_action(state, training=False)
                next_state, reward, done, info = self.env.step(action)
                episode_return += reward
                state = next_state

            eval_returns.append(episode_return)
            eval_sharpe_ratios.append(info.get("sharpe_ratio", 0))
            eval_max_drawdowns.append(info.get("max_drawdown", 0))

        self.epsilon = original_epsilon

        return {
            "mean_return": np.mean(eval_returns),
            "std_return": np.std(eval_returns),
            "mean_sharpe": np.mean(eval_sharpe_ratios),
            "mean_max_drawdown": np.mean(eval_max_drawdowns),
        }

    def get_uncertainty_estimate(self, state):
        """Get uncertainty estimate for ensemble models"""
        if self.model_type != "ensemble":
            return None

        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        uncertainty = self.q_network.get_uncertainty(state_tensor, self.hidden_state)
        return uncertainty.mean().item()

    def save_model(self, filepath):
        """Save model and training state"""
        save_dict = {
            "q_network_state_dict": self.q_network.state_dict(),
            "target_network_state_dict": self.target_network.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epsilon": self.epsilon,
            "training_step": self.training_step,
            "model_type": self.model_type,
            "sequence_length": self.sequence_length,
            "state_size": self.state_size,
            "action_size": self.action_size,
            "num_stocks": self.num_stocks,
        }
        torch.save(save_dict, filepath)

    def load_model(self, filepath):
        """Load model and training state"""
        checkpoint = torch.load(filepath, map_location="cpu")
        self.q_network.load_state_dict(checkpoint["q_network_state_dict"])
        self.target_network.load_state_dict(checkpoint["target_network_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.training_step = checkpoint["training_step"]


def run_enhanced_lstm_training():
    """Main training loop with LSTM-based DQN models"""

    # Get user inputs
    try:
        initial_cash = float(
            input("Enter Initial Balance (default 100000): ") or 100000
        )
        stocks_nos = int(
            input("Enter Number Of Stocks To Consider (default 50): ") or 50
        )
        stocks_nos = min(stocks_nos, 100)
        num_episodes = int(input("Enter Number Of Episodes (default 1000): ") or 1000)

        # Model selection
        print("\nModel Options:")
        print("1. basic_lstm - Basic LSTM DQN")
        print("2. advanced_lstm - Advanced LSTM with attention")
        print("3. ensemble - Ensemble of LSTM models")
        model_choice = input("Select model (1-3, default 2): ") or "2"
        model_map = {"1": "basic_lstm", "2": "advanced_lstm", "3": "ensemble"}
        model_type = model_map.get(model_choice, "advanced_lstm")

        # Advanced parameters
        print("\nAdvanced Parameters:")
        gamma = float(input("Discount factor gamma (0.95): ") or 0.95)
        lr = float(input("Learning rate (0.0001): ") or 0.0001)
        sequence_length = int(input("LSTM sequence length (20): ") or 20)

    except ValueError:
        print("Invalid input, using defaults")
        initial_cash, stocks_nos, num_episodes = 100000, 50, 1000
        gamma, lr, sequence_length = 0.95, 0.0001, 20
        model_type = "advanced_lstm"

    # Create enhanced environment
    print("Creating enhanced trading environment...")
    env = StockTradeEnv(
        merge_stocks(),
        initial_cash=initial_cash,
        num_stocks=stocks_nos,
        brokerage_rate=0.005,
        slippage_model="volume",
        bid_ask_spread=0.001,
        max_position_pct=0.15,
        overnight_risk=0.002,
    )

    # Create enhanced LSTM agent
    agent = EnhancedLSTMTradingAgent(
        env=env,
        model_type=model_type,
        gamma=gamma,
        lr=lr,
        sequence_length=sequence_length,
    )

    print("\nStarting Enhanced LSTM Training:")
    print(f"Model: {model_type}, Episodes: {num_episodes}")
    print(f"Initial Cash: ${initial_cash:,}, Stocks: {stocks_nos}")
    print(f"Sequence Length: {sequence_length}, Gamma: {gamma}, LR: {lr}")
    print("=" * 60)

    # Training metrics
    best_portfolio_value = float("-inf")
    best_sharpe_ratio = float("-inf")
    episode_metrics = defaultdict(list)

    # Create directories
    os.makedirs("../stats", exist_ok=True)
    os.makedirs("../models", exist_ok=True)

    for episode in range(num_episodes):
        state, _ = env.reset()
        agent.reset_hidden_states()  # Reset LSTM states at episode start
        done = False
        episode_reward = 0
        episode_losses = []
        step = 0

        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Reward clipping
            reward = np.clip(reward, -5, 5)

            # Train agent
            loss = agent.train_step(state, action, reward, next_state, done)
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            episode_reward += reward
            step += 1

            # Progress updates
            if step % 50 == 0:
                portfolio_val = info.get("portfolio_value", 0)
                uncertainty = ""
                if model_type == "ensemble":
                    unc = agent.get_uncertainty_estimate(state)
                    if unc is not None:
                        uncertainty = f", Uncertainty: {unc:.4f}"
                print(
                    f"Step {step}: Portfolio ${portfolio_val:,.2f}, Reward: {reward:.4f}{uncertainty}"
                )

        # Episode summary
        final_info = info
        final_portfolio_value = final_info.get("portfolio_value", 0)
        sharpe_ratio = final_info.get("sharpe_ratio", 0)
        max_drawdown = final_info.get("max_drawdown", 0)
        total_costs = final_info.get("total_trading_costs", 0)

        # Store metrics
        episode_metrics["portfolio_values"].append(final_portfolio_value)
        episode_metrics["rewards"].append(episode_reward)
        episode_metrics["sharpe_ratios"].append(sharpe_ratio)
        episode_metrics["max_drawdowns"].append(max_drawdown)
        episode_metrics["trading_costs"].append(total_costs)
        episode_metrics["num_positions"].append(final_info.get("num_positions", 0))
        episode_metrics["losses"].append(
            np.mean(episode_losses) if episode_losses else 0
        )

        print(f"Episode {episode + 1} Summary:")
        print(f"  Final Portfolio: ${final_portfolio_value:,.2f}")
        print(f"  Total Reward: {episode_reward:.4f}")
        print(f"  Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"  Max Drawdown: {max_drawdown:.2%}")
        print(f"  Trading Costs: ${total_costs:,.2f}")
        print(f"  Epsilon: {agent.epsilon:.4f}")

        # Save best models
        if final_portfolio_value > best_portfolio_value:
            best_portfolio_value = final_portfolio_value
            agent.save_model(
                f"../models/best_portfolio_{model_type}_ep_{episode + 1}.pth"
            )
            print("  New best portfolio value!")

        if sharpe_ratio > best_sharpe_ratio:
            best_sharpe_ratio = sharpe_ratio
            agent.save_model(f"../models/best_sharpe_{model_type}_ep_{episode + 1}.pth")
            print("  New best Sharpe ratio!")

        # Periodic evaluation and saving
        if (episode + 1) % 100 == 0:
            print(f"\n--- Evaluation at Episode {episode + 1} ---")
            eval_results = agent.evaluate_episode()
            print("Evaluation Results:")
            for key, value in eval_results.items():
                print(f"  {key}: {value:.4f}")

            agent.save_model(f"../models/checkpoint_{model_type}_ep_{episode + 1}.pth")

            # Save metrics
            with open(
                f"../stats/lstm_training_metrics_ep_{episode + 1}.json", "w"
            ) as f:
                json_metrics = {}
                for key, values in episode_metrics.items():
                    json_metrics[key] = [
                        float(v) if isinstance(v, np.floating) else v for v in values
                    ]
                json.dump(json_metrics, f, indent=2)

    # Final results
    print("\n" + "=" * 60)
    print("LSTM TRAINING COMPLETED!")
    print(f"Model Type: {model_type}")
    print(f"Best Portfolio Value: ${best_portfolio_value:,.2f}")
    print(f"Best Sharpe Ratio: {best_sharpe_ratio:.4f}")
    print(f"Final Epsilon: {agent.epsilon:.4f}")
    print("=" * 60)

    # Save final results
    final_results = {
        "training_complete": True,
        "model_type": model_type,
        "num_episodes": num_episodes,
        "sequence_length": sequence_length,
        "best_portfolio_value": float(best_portfolio_value),
        "best_sharpe_ratio": float(best_sharpe_ratio),
        "final_epsilon": float(agent.epsilon),
        "episode_metrics": {
            k: [float(v) if isinstance(v, np.floating) else v for v in values]
            for k, values in episode_metrics.items()
        },
    }

    with open(f"../stats/final_lstm_training_results_{model_type}.json", "w") as f:
        json.dump(final_results, f, indent=2)

    agent.save_model(f"../models/final_{model_type}_model.pth")

    return agent, episode_metrics


if __name__ == "__main__":
    try:
        agent, metrics = run_enhanced_lstm_training()
        print("\nLSTM Training completed successfully!")
        print(
            "Check '../stats/' for detailed metrics and '../models/' for saved models."
        )

    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {e}")
        import traceback

        traceback.print_exc()
