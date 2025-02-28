"""
Defines model architectures
"""

from torch import Tensor, nn


class PolyModelSmall(nn.Module):
    """
    Regression model for the poly dataset
    """

    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass returns a single point estimate

        Args:
            x (Tensor): Input tensor of shape (batch_size, input_dim)

        Returns:
            Tensor: Point estimate prediction of shape (batch_size, output_dim)
        """
        return self.network(x)
