import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dqn import DQN
from stock_gym import StockTradeEnv, merge_stocks


class TradingAgent:
    def __init__(self, env, gamma=0.99, lr=0.010):
        self.env = env
        self.gamma = gamma  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_decay = 0.0975
        self.epsilon_min = 0.250
        self.model = DQN(env.observation_space.shape[0], env.action_space.n)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.HuberLoss()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()  # Random action
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state_tensor)
        return torch.argmax(q_values).item()  # Select action with highest Q-value

    def train_step(self, state, action, reward, next_state, done, episode, step):
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)
        action_tensor = torch.tensor([action], dtype=torch.int64)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)

        # Compute Q-value prediction
        q_values = self.model(state_tensor)
        q_value = q_values.gather(1, action_tensor.unsqueeze(1)).squeeze(1)

        # Compute Q-target
        with torch.no_grad():
            next_q_values = self.model(next_state_tensor)
            max_next_q_value = torch.max(next_q_values, dim=1)[0]
            q_target = reward_tensor + (self.gamma * max_next_q_value * (1 - done))

        # Compute loss
        loss = self.criterion(q_value, q_target.detach())

        # Update model
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=1.0
        )  # Gradient clipping
        self.optimizer.step()

        # Decay epsilon (exploration factor)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

        # Debugging information

        if step % 10 == 0:
            print(
                f"""[Episode {episode} | Step {step}] | Action: {action} | Reward: {reward:.4f} 
            | Loss: {loss.item():.6f} | Epsilon: {self.epsilon:.4f}"""
            )


if __name__ == "__main__":
    env = StockTradeEnv(merge_stocks(), initial_cash=25000)
    agent = TradingAgent(env)
    num_episodes = 5
    # num_episodes = int(input("Enter Number Of Episodes: "))
    print("Starting Episodic Training")
    best_portfolio = float("-inf")
    for episode in range(num_episodes):
        print(f"\r Current Episode: {episode} / {num_episodes}")
        state, _ = env.reset()
        done = False
        total_reward = 0
        step = 0

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)

            # Clip reward to prevent extreme values affecting training
            reward = np.clip(reward, -10, 10)

            # Track portfolio value at this step
            portfolio_value = env.portfolio_value()

            # Train the agent
            agent.train_step(state, action, reward, next_state, done, episode, step)

            # Update state and reward tracking
            state = next_state
            total_reward += reward
            step += 1

            # Additional debug info every 10 steps
            if step % 10 == 0:
                print(
                    f"[Episode {episode} | Step {step}] Portfolio Value: {portfolio_value:.2f}"
                )

        final_portfolio_value = env.portfolio_value()
        print(
            f"Episode {episode + 1} completed: Total Reward = {total_reward:.2f} | Final Portfolio Value: {final_portfolio_value:.2f}"
        )

        # Save model if this episode achieved the best portfolio value
        if final_portfolio_value > best_portfolio:
            best_portfolio = final_portfolio_value
            torch.save(agent.model.state_dict(), "best_model.pth")
            print(f"Saved new best model with portfolio value: {best_portfolio:.2f}")
