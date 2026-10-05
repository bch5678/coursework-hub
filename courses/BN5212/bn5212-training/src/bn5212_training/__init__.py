"""Shared training framework for the BN5212 multimodal mortality project.

The framework owns model construction, the training loop, checkpointing and
prediction export. It does not own cohort construction (see bn5212-data-pipeline)
and it does not own final test metrics (see benchmark-evaluation).
"""
__version__ = "0.1.0"
