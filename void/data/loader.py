from __future__ import annotations

from torch.utils.data import DataLoader, IterableDataset


class ShardDataset(IterableDataset):
    """Streams packed (seq_len+1)-token windows from parquet shards."""

    def __init__(self, corpus_repo: str, seq_len: int, seed: int) -> None:
        super().__init__()
        self.corpus_repo = corpus_repo
        self.seq_len = seq_len
        self.seed = seed

    def __iter__(self):  # yields (input_ids, targets)
        raise NotImplementedError


def build_loader(
    corpus_repo: str,
    seq_len: int,
    batch: int,
    seed: int,
    num_workers: int = 8,
) -> DataLoader:
    ds = ShardDataset(corpus_repo, seq_len, seed)
    return DataLoader(
        ds,
        batch_size=batch,
        num_workers=num_workers,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=num_workers > 0,
    )
