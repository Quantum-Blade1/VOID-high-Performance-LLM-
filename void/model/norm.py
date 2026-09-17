"""RMSNorm — root mean square layer normalization.

y = x * (1 / sqrt(mean(x^2) + eps)) * weight.
The reduction is done in float32 regardless of input dtype for numerical stability.
Also used as QK-norm on attention Q/K tensors.

Design ref: Zhang & Sennrich 2019."""
