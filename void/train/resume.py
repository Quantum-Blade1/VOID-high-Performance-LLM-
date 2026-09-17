"""Deterministic resume-from-checkpoint.

Restores every piece of persistent state such that a resumed run produces bit-identical
loss / grad_norm to a non-resumed run on the same data. Tested by tests/test_resume.py."""
