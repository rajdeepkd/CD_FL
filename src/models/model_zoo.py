import torch
import torch.nn as nn


class TabularBinaryMLP(nn.Module):
    """
    A slightly deeper MLP with dropout for binary classification.
    Output: logits for 2 classes.
    """

    def __init__(self, input_dim: int, hidden1: int = 256, hidden2: int = 128, p: float = 0.3):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
        )
        self.classifier = nn.Linear(hidden2, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.feature_extractor(x)
        logits = self.classifier(z)
        return logits

    def get_head(self) -> nn.Module:
        return self.classifier

    def set_head(self, new_head: nn.Module) -> None:
        self.classifier = new_head


def build_model(input_dim: int) -> TabularBinaryMLP:
    return TabularBinaryMLP(input_dim=input_dim)
