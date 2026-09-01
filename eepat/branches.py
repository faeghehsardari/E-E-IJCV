"""Assistant and Core branch modules described in the EE-PAT paper."""

import torch.nn as nn
import torch.nn.functional as F

from .rpt import RPTBlock, TemporalProjection, _initialize_module


class MultiLabelRelationshipModule(nn.Module):
    """ML-Rel: encode temporal co-occurrence relations from ground-truth labels."""

    def __init__(self, num_classes, embedding_dim, num_blocks, num_heads,
                 max_length, mlp_ratio):
        super().__init__()
        self.label_projection = TemporalProjection(num_classes, embedding_dim)
        self.rpt_blocks = nn.ModuleList([
            RPTBlock(embedding_dim, num_heads, max_length, mlp_ratio)
            for _ in range(num_blocks)
        ])

    def forward(self, labels):
        relationships = self.label_projection(labels.transpose(1, 2))
        for block in self.rpt_blocks:
            relationships = block(relationships)
        return relationships.transpose(1, 2).contiguous()


class SharedClassificationModule(nn.Module):
    """The shared ML-CLAS/Vid-CLAS 1x1 temporal classification filter."""

    def __init__(self, embedding_dim, num_classes):
        super().__init__()
        self.classifier = nn.Conv1d(embedding_dim, num_classes, kernel_size=1)
        self.apply(_initialize_module)

    def forward(self, features):
        return self.classifier(features).transpose(1, 2).contiguous()


class AssistantBranch(nn.Module):
    """Training-only branch composed of ML-Rel and ML-CLAS."""

    def __init__(self, num_classes, embedding_dim, num_blocks, num_heads,
                 max_length, mlp_ratio):
        super().__init__()
        self.ml_rel = MultiLabelRelationshipModule(
            num_classes, embedding_dim, num_blocks, num_heads, max_length, mlp_ratio
        )
        self.ml_clas = SharedClassificationModule(embedding_dim, num_classes)

    def forward(self, labels):
        return self.ml_clas(self.ml_rel(labels))


class FineDetectionModule(nn.Module):
    """Fine-Det: full-resolution temporal projection and RPT processing."""

    def __init__(self, input_dim, embedding_dim, num_blocks, num_heads,
                 max_length, mlp_ratio):
        super().__init__()
        self.input_projection = TemporalProjection(input_dim, embedding_dim)
        self.rpt_blocks = nn.ModuleList([
            RPTBlock(embedding_dim, num_heads, max_length, mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, video_tokens):
        features = self.input_projection(video_tokens)
        for block in self.rpt_blocks:
            features = block(features)
        return self.output_norm(features).transpose(1, 2).contiguous()


class GranularityBranch(nn.Module):
    """One non-hierarchical Coarse-Det branch operating at a fixed stride."""

    def __init__(self, embedding_dim, stride, num_blocks, num_heads,
                 input_length, mlp_ratio):
        super().__init__()
        self.downsample = TemporalProjection(embedding_dim, embedding_dim, stride=stride)
        branch_length = input_length // stride
        self.rpt_blocks = nn.ModuleList([
            RPTBlock(embedding_dim, num_heads, branch_length, mlp_ratio)
            for _ in range(num_blocks)
        ])
        self.output_norm = nn.LayerNorm(embedding_dim)
        self.mixer_projection = nn.Conv1d(
            embedding_dim, embedding_dim, kernel_size=3, padding=1
        )
        self.mixer_projection.apply(_initialize_module)

    def forward(self, fine_features, output_length):
        features = self.downsample(fine_features)
        for block in self.rpt_blocks:
            features = block(features)
        features = self.output_norm(features).transpose(1, 2).contiguous()
        features = self.mixer_projection(features)
        return F.interpolate(
            features, size=output_length, mode="linear", align_corners=False
        )


class CoarseDetectionModule(nn.Module):
    """Coarse-Det: three parallel, non-hierarchical granularity branches."""

    def __init__(self, embedding_dim, strides, num_blocks, num_heads,
                 input_length, mlp_ratio):
        super().__init__()
        self.granularity_branches = nn.ModuleList([
            GranularityBranch(
                embedding_dim, stride, num_blocks, num_heads, input_length, mlp_ratio
            )
            for stride in strides
        ])

    def forward(self, fine_features):
        output_length = fine_features.shape[-1]
        coarse_features = [
            branch(fine_features, output_length)
            for branch in self.granularity_branches
        ]
        return sum(coarse_features)


class CoreBranch(nn.Module):
    """Inference branch composed of Fine-Det, Coarse-Det, and Vid-CLAS."""

    def __init__(self, input_dim, embedding_dim, num_classes, num_blocks,
                 num_heads, max_length, mlp_ratio, granularity_strides=(2, 4, 8)):
        super().__init__()
        self.fine_det = FineDetectionModule(
            input_dim, embedding_dim, num_blocks, num_heads, max_length, mlp_ratio
        )
        self.coarse_det = CoarseDetectionModule(
            embedding_dim,
            granularity_strides,
            num_blocks,
            num_heads,
            max_length,
            mlp_ratio,
        )
        self.coarse_projection = nn.Conv1d(embedding_dim, embedding_dim, kernel_size=1)
        self.dropout = nn.Dropout()
        self.apply(_initialize_module)

    def forward(self, video_tokens, video_classifier):
        fine_features = self.fine_det(video_tokens)
        coarse_features = self.coarse_det(fine_features)
        fine_logits = video_classifier(fine_features)
        coarse_features = self.dropout(self.coarse_projection(coarse_features))
        coarse_logits = video_classifier(coarse_features)
        return coarse_logits, fine_logits
