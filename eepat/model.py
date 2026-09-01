"""EE-PAT network for dense multi-label action detection.

Paper: "An Effective-Efficient Approach for Dense Multi-Label Action Detection"
(arXiv:2406.06187). The pre-computed I3D features are the Vid-Enc outputs.
"""

import torch
import torch.nn as nn

from .branches import AssistantBranch, CoreBranch, SharedClassificationModule


class EEPAT(nn.Module):
    """Effective-Efficient Position-Aware Transformer (EE-PAT/PAT)."""

    def __init__(self, input_dim=1024, embedding_dim=512, num_classes=157,
                 num_blocks=3, num_heads=8, max_length=256, mlp_ratio=8,
                 granularity_strides=(2, 4, 8)):
        super().__init__()
        self.input_dropout = nn.Dropout()
        self.assistant = AssistantBranch(
            num_classes,
            embedding_dim,
            num_blocks,
            num_heads,
            max_length,
            mlp_ratio,
        )
        self.core = CoreBranch(
            input_dim,
            embedding_dim,
            num_classes,
            num_blocks,
            num_heads,
            max_length,
            mlp_ratio,
            granularity_strides,
        )
        self.video_classifier = SharedClassificationModule(embedding_dim, num_classes)
        self.copy_assistant_classifier()

    def forward_assistant(self, labels):
        """Run training-only ML-Rel and ML-CLAS on ground-truth labels."""
        return self.assistant(labels)

    @torch.no_grad()
    def copy_assistant_classifier(self):
        """Copy ML-CLAS parameters to Vid-CLAS as prescribed by the paper."""
        self.video_classifier.load_state_dict(self.assistant.ml_clas.state_dict())

    def forward_core(self, video_tokens):
        """Run Fine-Det, Coarse-Det, and Vid-CLAS."""
        return self.core(self.input_dropout(video_tokens), self.video_classifier)

    def forward(self, video_tokens):
        """Inference uses only the computationally efficient Core branch."""
        return self.forward_core(video_tokens)
