import torch.nn as nn

class ProjectionHead(nn.Module):
    def __init__(
        self,
        embedding_dim,
        projection_dim=256,
        dropout=0.1
    ):
        super().__init__()

        """
        Here you should write simple 2-layer MLP consisting:
        2 Linear layers, GELU activation, Dropout and LayerNorm. 
        Do not forget to send a skip-connection right after projection and before LayerNorm.
        The whole structure should be in the following order:
        [Linear, GELU, Linear, Dropout, Skip, LayerNorm]
        """
        self.linear1 = nn.Linear(embedding_dim, projection_dim)
        self.linear2 = nn.Linear(projection_dim, projection_dim)
        self.gelu = nn.GELU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        """
        Perform forward pass, do not forget about skip-connections.
        """
        start_projection = self.linear1(x)
        mid_projection = self.linear2(self.gelu(start_projection))
        mid_projection = self.dropout(mid_projection)
        final_projection = start_projection + mid_projection
        return nn.functional.normalize(final_projection, dim=-1)
