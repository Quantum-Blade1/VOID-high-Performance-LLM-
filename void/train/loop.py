from __future__ import annotations

from void.config import Config


def train(cfg: Config, resume_from: str | None = None) -> None:
    """One-process training entry point.

    Wires: model, optimizers, dataloader, curriculum stage handler, checkpointer,
    logger. Runs to cfg.train.total_tokens or until interrupted; resumes cleanly
    from the latest hub checkpoint if resume_from is None and one exists.
    """
    raise NotImplementedError
