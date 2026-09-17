from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


def kl_from_topk(student_logits: Tensor, teacher_topk_ids: Tensor, teacher_topk_logits: Tensor, T: float) -> Tensor:
    """KL(teacher || student) computed only over the teacher's top-k tokens.

    student_logits:      [B, T, V]
    teacher_topk_ids:    [B, T, k]  int64
    teacher_topk_logits: [B, T, k]  fp32
    """
    raise NotImplementedError


def cache_teacher_logits(teacher_repo: str, shards: list[str], top_k: int, out_repo: str) -> None:
    """One-off caching pass over the corpus; ~400× storage saving vs full logits."""
    raise NotImplementedError
