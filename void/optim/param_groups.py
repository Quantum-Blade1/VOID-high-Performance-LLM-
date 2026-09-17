"""Split model parameters into (hidden matrices, embed+head+norms) groups.

Rule: any parameter with ndim >= 2 that is not the token embedding, LM head, or a
norm/scaler goes to Muon. Everything else goes to AdamW."""
