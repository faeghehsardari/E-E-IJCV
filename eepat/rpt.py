"""Relative Positional Transformer (RPT) components from the EE-PAT paper."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _initialize_module(module):
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)
    elif isinstance(module, nn.Conv1d):
        fan_out = module.kernel_size[0] * module.out_channels // module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class RelativePositionSelfAttention(nn.Module):
    """Multi-head self-attention with translation-invariant relative positions."""

    def __init__(self, embedding_dim, num_heads=8, max_length=256,
                 use_relative_position=True):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.max_length = max_length
        self.use_relative_position = use_relative_position

        self.query = nn.Linear(embedding_dim, embedding_dim)
        self.key_value = nn.Linear(embedding_dim, embedding_dim * 2)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)
        self.relative_embedding = nn.Parameter(torch.randn(max_length, self.head_dim))
        self.apply(_initialize_module)

    @staticmethod
    def _skew(relative_logits):
        padded = F.pad(relative_logits, (1, 0))
        batch_size, num_heads, num_rows, num_columns = padded.shape
        reshaped = padded.reshape(batch_size, num_heads, num_columns, num_rows)
        return reshaped[:, :, 1:, :]

    def forward(self, tokens):
        batch_size, sequence_length, embedding_dim = tokens.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds maximum {self.max_length}"
            )

        query = self.query(tokens).reshape(
            batch_size, sequence_length, self.num_heads, self.head_dim
        ).permute(0, 2, 1, 3)
        key_value = self.key_value(tokens).reshape(
            batch_size, sequence_length, 2, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        key, value = key_value[0], key_value[1]

        attention_logits = query @ key.transpose(-2, -1)
        if self.use_relative_position:
            relative_embedding = self.relative_embedding[
                self.max_length - sequence_length:
            ].transpose(0, 1)
            relative_logits = self._skew(query @ relative_embedding)
            attention_logits = attention_logits + relative_logits

        attention = (attention_logits * self.scale).softmax(dim=-1)
        output = (attention @ value).transpose(1, 2).reshape(
            batch_size, sequence_length, embedding_dim
        )
        return self.output_projection(output)


class LocalRelation(nn.Module):
    """The RPT local relational component: linear, depthwise Conv1d, linear."""

    def __init__(self, embedding_dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.input_projection = nn.Linear(embedding_dim, hidden_dim)
        self.temporal_convolution = nn.Conv1d(
            hidden_dim, hidden_dim, kernel_size=3, padding=1, groups=hidden_dim
        )
        self.activation = nn.GELU()
        self.output_projection = nn.Linear(hidden_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)
        self.apply(_initialize_module)

    def forward(self, tokens):
        tokens = self.input_projection(tokens).transpose(1, 2)
        tokens = self.temporal_convolution(tokens).transpose(1, 2)
        tokens = self.dropout(self.activation(tokens))
        return self.dropout(self.output_projection(tokens))


class RPTBlock(nn.Module):
    """Relative positional self-attention followed by local relation modelling."""

    def __init__(self, embedding_dim, num_heads, max_length, mlp_ratio=4.0,
                 dropout=0.0, use_relative_position=True):
        super().__init__()
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.attention = RelativePositionSelfAttention(
            embedding_dim,
            num_heads=num_heads,
            max_length=max_length,
            use_relative_position=use_relative_position,
        )
        self.local_norm = nn.LayerNorm(embedding_dim)
        self.local_relation = LocalRelation(
            embedding_dim,
            hidden_dim=int(embedding_dim * mlp_ratio),
            dropout=dropout,
        )
        self.apply(_initialize_module)

    def forward(self, tokens):
        tokens = tokens + self.attention(self.attention_norm(tokens))
        return tokens + self.local_relation(self.local_norm(tokens))


class TemporalProjection(nn.Module):
    """Conv1d token projection used before ML-Rel and detection RPT blocks."""

    def __init__(self, input_dim, output_dim, stride=1):
        super().__init__()
        self.projection = nn.Conv1d(
            input_dim, output_dim, kernel_size=3, stride=stride, padding=1
        )
        self.normalization = nn.LayerNorm(output_dim)
        self.apply(_initialize_module)

    def forward(self, features):
        return self.normalization(self.projection(features).transpose(1, 2))
