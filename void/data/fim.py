"""Fill-in-the-Middle reformatting for code samples.

At data-prep time, 50% of code samples are reformatted into PSM (prefix-suffix-middle)
or SPM (suffix-prefix-middle) formats with sentinels <|fim_prefix|>, <|fim_middle|>,
<|fim_suffix|>. Random split point per sample.

Design ref: Bavarian et al. 2022 (Efficient Training to Fill in the Middle)."""
