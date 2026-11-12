"""Evaluation/comparison tooling for code agents.

The pure-data comparison core (``compare``) imports without torch/CUDA so it can
run in fast host-side unit tests and CI. Render-based helpers import their heavy
dependencies lazily inside the functions that need them.
"""
