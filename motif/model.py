"""Inference-only temporal transformer used by both MOTIF branches."""

import torch
from torch import nn
from timm.models.vision_transformer import Block


class TemporalEncoder(nn.Module):
    def __init__(self, input_dim, sequence_length=75, embedding_dim=192, layers=12):
        super().__init__()
        self.embed = nn.Linear(input_dim, embedding_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, sequence_length, embedding_dim))
        self.encoder_blocks = nn.ModuleList(
            [Block(embedding_dim, 3) for _ in range(layers)]
        )
        self.encoder_norm = nn.LayerNorm(embedding_dim)

    def forward(self, sequences):
        tokens = self.embed(sequences) + self.pos_embed[:, : sequences.shape[1]]
        cls = self.cls_token.expand(tokens.shape[0], -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        for block in self.encoder_blocks:
            tokens = block(tokens)
        return self.encoder_norm(tokens)[:, 0]


def load_encoder(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    args = checkpoint["args"]
    model = TemporalEncoder(
        input_dim=args["input_dim"],
        sequence_length=args["window"],
        embedding_dim=args["emb_dim"],
        layers=args["num_layers"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    return model, args
