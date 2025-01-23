import torch
import torch.nn as nn
import numpy as np

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"Sequence length ({seq_len}) exceeds max positional encoding length ({self.pe.size(1)}).")
        return x + self.pe[:, :seq_len, :]


class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, max_len=5000):
        super(TransformerAutoencoder, self).__init__()
        self.input_fc = nn.Linear(input_dim, d_model)
        self.output_fc = nn.Linear(d_model, input_dim)
        self.positional_encoding = PositionalEncoding(d_model, max_len)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_encoder_layers
        )
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward),
            num_layers=num_decoder_layers
        )

    def forward(self, src):
        src = self.input_fc(src)
        src = self.positional_encoding(src)
        memory = self.encoder(src)
        output = self.decoder(src, memory)
        return self.output_fc(output)

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
