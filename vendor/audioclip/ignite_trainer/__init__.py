
from vendor.audioclip.ignite_trainer.version import __version__
from ._utils import load_class
from ._interfaces import AbstractNet, AbstractTransform

__all__ = [
    '__version__',
    'main', 'run',
    'load_class',
    'AbstractNet', 'AbstractTransform'
]



def __getattr__(name):
    # Inference only needs interfaces; training dependencies are optional.
    if name in ('main', 'run'):
        from . import _trainer
        return getattr(_trainer, name)
    raise AttributeError(name)
