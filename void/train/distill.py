"""Stage-3 knowledge distillation from a teacher model.

Loads pre-cached top-k=100 teacher logits from HF Hub (compressed, ~400x storage
saving vs full logits). Computes KL(teacher || student) with temperature T,
combined with the standard CE loss as: loss = (1-beta)*CE + beta*T^2*KL.

Design ref: paper/void_v1.tex §5.3."""
