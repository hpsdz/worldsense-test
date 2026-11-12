"""Registry for single-sample, pre-LLM multimodal compression strategies."""
from .strategies import create_strategy, register_strategy
from . import omnipack
from . import omniscope

__all__ = ['create_strategy', 'register_strategy']
