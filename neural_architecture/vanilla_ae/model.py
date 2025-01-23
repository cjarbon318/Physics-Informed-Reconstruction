import torch.nn as nn
import torch

class DeeperVanillaNN(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(DeeperVanillaNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, latent_dim),
            nn.Sigmoid()
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

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
