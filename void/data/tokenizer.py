"""BPE tokenizer training and inference wrapper.

Trains a 49,152-vocab BPE tokenizer with byte-level fallback. Special tokens include
FIM sentinels (<|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>), ChatML delimiters
(<|im_start|>, <|im_end|>), and register-token placeholders.

Uses HuggingFace tokenizers lib for training; exports as .json compatible with
transformers.PreTrainedTokenizerFast for inference."""
