import torch.nn as nn


class NN(nn.Module):
    """Neural network for prediction the relation of a question

    Attributes:
        model (nn.Module): PyTorch neural network module
    """
    def __init__(self, emb_dim, out_dim, dropout=0.25, n_hid1=256, n_hid2=256):
        """Initializes all required elements of the neural network

        Args:
            emb_dim: Input size of the embedding
            out_dim: Output size of the linear layer
            dropout: Dropout value
            n_hid1: Size for hidden layer 1
            n_hid2: Size for hidden layer 2
        """
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_dim, n_hid1),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid1),
            nn.Dropout(dropout),
            nn.Linear(n_hid1, n_hid2 // 4),
            nn.ReLU(),
            nn.BatchNorm1d(n_hid2 // 4),
            nn.Dropout(dropout),
            nn.Linear(n_hid2 // 4, out_dim),
        )
        for m in self.model:
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """Application of the neural network on a given input question

        Args:
            x: Tensor containing the embeddings of shape (batch, embedding size)

        Returns:
            Probabilities of the relation classes
        """
        return self.model(x)
