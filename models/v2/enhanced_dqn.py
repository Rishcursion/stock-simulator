import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head attention for focusing on important stocks/features"""

    def __init__(self, input_dim, num_heads=8):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.input_dim = input_dim
        self.head_dim = input_dim // num_heads

        assert input_dim % num_heads == 0, "input_dim must be divisible by num_heads"

        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.output_proj = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        # Generate Q, K, V
        Q = (
            self.query(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        K = (
            self.key(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        V = (
            self.value(x)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, V)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.input_dim)
        )

        return self.output_proj(context)


class LSTMDQNBasic(nn.Module):
    """Basic LSTM-based DQN for sequential trading decisions"""

    def __init__(
        self, input_dim, output_dim, hidden_dim=256, num_layers=2, sequence_length=20
    ):
        super(LSTMDQNBasic, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length

        # Feature extraction layers
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 128), nn.LayerNorm(128), nn.ReLU(), nn.Dropout(0.2)
        )

        # LSTM for temporal modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=False,
        )

        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, output_dim),
        )

        # Initialize LSTM weights
        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """Initialize LSTM weights for better training"""
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.xavier_uniform_(param)
            elif "bias" in name:
                with torch.no_grad():
                    nn.init.zeros_(param)
                    # Set forget gate bias to 1
                    n = param.size(0)
                    param.data[n // 4 : n // 2].fill_(1.0)

    def init_hidden(self, batch_size, device):
        """Initialize hidden states"""
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
        return (h0, c0)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] if x.dim() > 1 else 1

        # Handle single sample input
        if x.dim() == 1:
            x = x.unsqueeze(0)  # Add batch dimension

        # Extract features
        features = self.feature_extractor(x)

        # Reshape for LSTM: (batch_size, seq_len, features)
        if features.dim() == 2:
            features = features.unsqueeze(1)  # Add sequence dimension

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_out, hidden = self.lstm(features, hidden)

        # Use the last output for Q-values
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim)

        # Generate Q-values
        q_values = self.output_layers(last_output)

        return q_values, hidden


class AdvancedLSTMDQN(nn.Module):
    """Advanced LSTM DQN with attention mechanism and stock-specific processing"""

    def __init__(
        self,
        input_dim,
        output_dim,
        num_stocks,
        hidden_dim=256,
        num_layers=2,
        sequence_length=20,
        use_attention=True,
    ):
        super(AdvancedLSTMDQN, self).__init__()
        self.num_stocks = num_stocks
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.sequence_length = sequence_length
        self.use_attention = use_attention

        # Calculate dimensions
        self.stock_features = 10  # Features per stock from enhanced environment
        self.portfolio_features = 5  # Portfolio-level features
        self.total_stock_features = num_stocks * self.stock_features

        assert (
            input_dim == self.total_stock_features + self.portfolio_features
        ), f"Input dim mismatch: expected {self.total_stock_features + self.portfolio_features}, got {input_dim}"

        # Stock-specific feature processing
        self.stock_processor = nn.Sequential(
            nn.Linear(self.stock_features, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Portfolio feature processing
        self.portfolio_processor = nn.Sequential(
            nn.Linear(self.portfolio_features, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Attention mechanism for stock selection
        if self.use_attention:
            self.stock_attention = AttentionLayer(64, num_heads=8)

        # Combined feature dimension
        combined_dim = 64 * num_stocks + 32  # Stock features + portfolio features

        # Feature fusion layer
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, 256), nn.LayerNorm(256), nn.ReLU(), nn.Dropout(0.2)
        )

        # LSTM layers for temporal modeling
        self.lstm = nn.LSTM(
            input_size=256,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0,
            bidirectional=True,  # Bidirectional for better context
        )

        # Output processing (account for bidirectional LSTM)
        lstm_output_dim = hidden_dim * 2  # Bidirectional doubles the output

        # Action-specific heads for better action value estimation
        self.action_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(lstm_output_dim, 128),
                    nn.LayerNorm(128),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 5),  # 5 actions per stock
                )
                for _ in range(num_stocks)
            ]
        )

        # Value stream (for Dueling DQN architecture)
        self.value_stream = nn.Sequential(
            nn.Linear(lstm_output_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 1),
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize network weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if "weight" in name:
                        nn.init.xavier_uniform_(param)
                    elif "bias" in name:
                        # Use data attribute to avoid in-place operation error
                        with torch.no_grad():
                            nn.init.zeros_(param)
                            # Set forget gate bias to 1
                            n = param.size(0)
                            param.data[n // 4 : n // 2].fill_(1.0)

    def init_hidden(self, batch_size, device):
        """Initialize hidden states for bidirectional LSTM"""
        num_directions = 2  # Bidirectional
        h0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_dim, device=device
        )
        c0 = torch.zeros(
            self.num_layers * num_directions, batch_size, self.hidden_dim, device=device
        )
        return (h0, c0)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] if x.dim() > 1 else 1

        # Handle single sample input
        if x.dim() == 1:
            x = x.unsqueeze(0)

        # Split input into stock and portfolio features
        stock_features = x[
            :, : self.total_stock_features
        ]  # (batch_size, num_stocks * stock_features)
        portfolio_features = x[
            :, self.total_stock_features :
        ]  # (batch_size, portfolio_features)

        # Reshape stock features: (batch_size, num_stocks, stock_features)
        stock_features = stock_features.view(
            batch_size, self.num_stocks, self.stock_features
        )

        # Process each stock's features
        processed_stocks = self.stock_processor(
            stock_features
        )  # (batch_size, num_stocks, 64)

        # Apply attention mechanism
        if self.use_attention:
            processed_stocks = self.stock_attention(processed_stocks)

        # Flatten stock features for concatenation
        flattened_stocks = processed_stocks.view(
            batch_size, -1
        )  # (batch_size, num_stocks * 64)

        # Process portfolio features
        processed_portfolio = self.portfolio_processor(
            portfolio_features
        )  # (batch_size, 32)

        # Combine all features
        combined_features = torch.cat([flattened_stocks, processed_portfolio], dim=1)

        # Feature fusion
        fused_features = self.feature_fusion(combined_features)  # (batch_size, 256)

        # Add sequence dimension if needed
        if fused_features.dim() == 2:
            fused_features = fused_features.unsqueeze(1)  # (batch_size, 1, 256)

        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size, x.device)

        # LSTM forward pass
        lstm_out, new_hidden = self.lstm(fused_features, hidden)

        # Use the last output
        last_output = lstm_out[:, -1, :]  # (batch_size, hidden_dim * 2)

        # Generate action values for each stock using separate heads
        action_values = []
        for i, head in enumerate(self.action_heads):
            stock_q_values = head(last_output)  # (batch_size, 5)
            action_values.append(stock_q_values)

        # Combine all action values
        all_q_values = torch.cat(action_values, dim=1)  # (batch_size, num_stocks * 5)

        # Dueling DQN: Combine value and advantage streams
        state_value = self.value_stream(last_output)  # (batch_size, 1)

        # Calculate advantages and combine with state value
        advantages = all_q_values - all_q_values.mean(dim=1, keepdim=True)
        final_q_values = state_value + advantages

        return final_q_values, new_hidden


class EnsembleDQN(nn.Module):
    """Ensemble of multiple DQN models for better performance and uncertainty estimation"""

    def __init__(self, input_dim, output_dim, num_stocks, num_models=3):
        super(EnsembleDQN, self).__init__()
        self.num_models = num_models

        # Create ensemble of models
        self.models = nn.ModuleList(
            [
                AdvancedLSTMDQN(
                    input_dim,
                    output_dim,
                    num_stocks,
                    hidden_dim=256,
                    num_layers=2,
                    use_attention=True,
                )
                for _ in range(num_models)
            ]
        )

        # Ensemble combination weights
        self.combination_weights = nn.Parameter(torch.ones(num_models) / num_models)

    def forward(self, x, hidden=None):
        batch_size = x.shape[0] if x.dim() > 1 else 1

        # Initialize hidden states for all models if not provided
        if hidden is None:
            hidden = [None] * self.num_models

        # Get predictions from all models
        predictions = []
        new_hiddens = []

        for i, model in enumerate(self.models):
            q_values, new_hidden = model(
                x, hidden[i] if hidden[i] is not None else None
            )
            predictions.append(q_values)
            new_hiddens.append(new_hidden)

        # Combine predictions using learned weights
        weights = F.softmax(self.combination_weights, dim=0)
        ensemble_output = sum(w * pred for w, pred in zip(weights, predictions))

        return ensemble_output, new_hiddens

    def get_uncertainty(self, x, hidden=None):
        """Get uncertainty estimates from ensemble"""
        predictions = []

        for i, model in enumerate(self.models):
            with torch.no_grad():
                q_values, _ = model(x, hidden[i] if hidden is not None else None)
                predictions.append(q_values)

        predictions = torch.stack(predictions)
        uncertainty = torch.std(predictions, dim=0)  # Standard deviation as uncertainty

        return uncertainty


# Convenience function for model selection
def create_dqn_model(model_type, input_dim, output_dim, num_stocks, **kwargs):
    """Factory function to create different DQN models"""

    if model_type == "basic_lstm":
        return LSTMDQNBasic(input_dim, output_dim, **kwargs)

    elif model_type == "advanced_lstm":
        return AdvancedLSTMDQN(input_dim, output_dim, num_stocks, **kwargs)

    elif model_type == "ensemble":
        return EnsembleDQN(input_dim, output_dim, num_stocks, **kwargs)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test parameters
    num_stocks = 50
    stock_features = 10
    portfolio_features = 5
    input_dim = num_stocks * stock_features + portfolio_features
    output_dim = num_stocks * 5  # 5 actions per stock
    batch_size = 32

    print("Testing DQN models:")
    print(f"Input dim: {input_dim}, Output dim: {output_dim}")
    print(f"Num stocks: {num_stocks}, Batch size: {batch_size}")
    print("-" * 50)

    # Create test input
    test_input = torch.randn(batch_size, input_dim)

    # Test basic LSTM model
    print("Testing Basic LSTM DQN...")
    basic_model = create_dqn_model("basic_lstm", input_dim, output_dim, num_stocks)
    basic_output, basic_hidden = basic_model(test_input)
    print(f"Basic LSTM output shape: {basic_output.shape}")

    # Test advanced LSTM model
    print("\nTesting Advanced LSTM DQN...")
    advanced_model = create_dqn_model(
        "advanced_lstm", input_dim, output_dim, num_stocks
    )
    advanced_output, advanced_hidden = advanced_model(test_input)
    print(f"Advanced LSTM output shape: {advanced_output.shape}")

    # Test ensemble model
    print("\nTesting Ensemble DQN...")
    ensemble_model = create_dqn_model(
        "ensemble", input_dim, output_dim, num_stocks, num_models=3
    )
    ensemble_output, ensemble_hidden = ensemble_model(test_input)
    uncertainty = ensemble_model.get_uncertainty(test_input)
    print(f"Ensemble output shape: {ensemble_output.shape}")
    print(f"Uncertainty shape: {uncertainty.shape}")

    print("\nAll models tested successfully!")
