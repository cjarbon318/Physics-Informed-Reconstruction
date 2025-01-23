import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]

class RNNTransformerModel(nn.Module):
    def __init__(self, input_dim=1, rnn_hidden_dim=16, rnn_layers=1, d_model=32, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=64):
        super(RNNTransformerModel, self).__init__()

        self.input_dim = input_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.rnn_layers = rnn_layers
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.dim_feedforward = dim_feedforward

        # RNN for feature extraction
        self.rnn = nn.LSTM(self.input_dim, self.rnn_hidden_dim, num_layers=self.rnn_layers, batch_first=True)

        # Positional encoding for sequential data
        self.positional_encoding = PositionalEncoding(self.d_model)

        # Transformer Encoder-Decoder
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward),
            num_layers=self.num_encoder_layers
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=self.d_model, nhead=self.nhead, dim_feedforward=self.dim_feedforward),
            num_layers=self.num_decoder_layers
        )

        # Fully connected layers
        self.input_fc = nn.Linear(self.rnn_hidden_dim, self.d_model)
        self.output_fc = nn.Linear(self.d_model, self.input_dim)

    def forward(self, src):
        rnn_output, _ = self.rnn(src)
        rnn_encoded = self.input_fc(rnn_output)
        transformer_input = self.positional_encoding(rnn_encoded)
        memory = self.encoder(transformer_input)
        output = self.decoder(transformer_input, memory)

        reconstructed = self.output_fc(output)
        return reconstructed

    @staticmethod
    def physics_loss(t, reconstructed, weight=1.0):
        """
        Computes the physics-informed loss using the known equation: σ = 0.3sin(5t - π).

        Args:
            t (torch.Tensor): Time steps corresponding to the input data.
            reconstructed (torch.Tensor): Reconstructed output from the model.
            weight (float): Weight for the physics-informed loss.

        Returns:
            torch.Tensor: Physics-informed loss.
        """
        expected = 0.3 * torch.sin(5 * t - torch.pi)
        return weight * nn.MSELoss()(reconstructed, expected)
