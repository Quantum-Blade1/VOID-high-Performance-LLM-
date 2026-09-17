"""Streaming Dataset over sharded parquet on HF Hub.

Reads pre-tokenized uint32 token IDs from parquet shards, packs into (seq_len + 1)
chunks with EOS separator, returns (input_ids, targets) for causal LM training.
Register tokens are prepended by the collate_fn, not stored in the shards."""
