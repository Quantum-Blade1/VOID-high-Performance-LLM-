"""Evaluation-corpus contamination scrubbing.

Loads exact-match strings from every planned eval (HumanEval, MBPP, MMLU, GSM8K, ARC,
HellaSwag, void-cli-bench). Rejects any training sample containing an eval-string
substring. Reports rejection statistics per source for the model card."""
