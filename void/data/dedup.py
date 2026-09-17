"""MinHash near-duplicate detection across sources.

Uses datasketch MinHashLSH; hash 5-shingles of tokenized text. Rejects near-duplicates
across all sources into a global set to avoid over-representing repeated content."""
