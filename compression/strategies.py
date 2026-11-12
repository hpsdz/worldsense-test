from dataclasses import dataclass
import math
from pathlib import Path
import torch

REGISTRY = {}

def register_strategy(name):
    def register(cls):
        if name in REGISTRY:
            raise ValueError(f'Duplicate compression strategy: {name}')
        REGISTRY[name] = cls
        return cls
    return register

@dataclass
class CompressionResult:
    embeddings: torch.Tensor  # full-length, optionally merged
    keep: torch.Tensor        # one boolean per original sequence position

class Strategy:
    needs_audio_importance = False
    defaults = {}
    def __init__(self, params):
        if set(params) - set(self.defaults):
            raise ValueError(f'Unknown compression parameters: {set(params)-set(self.defaults)}')
        self.params = dict(self.defaults, **params)
        for key, value in self.params.items():
            if key in ('rho_audio','rho_video','contextual_ratio','prune_ratio_a','prune_ratio_v'):
                if type(value) not in (int,float) or not math.isfinite(value) or not 0 <= value < 1:
                    raise ValueError(f'{key} must be in [0, 1)')
        if 'g' in self.params and (type(self.params['g']) is not int or self.params['g'] < 0):
            raise ValueError('g must be a nonnegative integer')
    def prepare(self, **kwargs):
        return None

@register_strategy('baseline')
class Baseline(Strategy):
    def compress(self, embeddings, **kwargs):
        return CompressionResult(embeddings, torch.ones(embeddings.shape[1], dtype=torch.bool, device=embeddings.device))

@register_strategy('omnizip')
class OmniZip(Strategy):
    needs_audio_importance = True
    defaults = dict(rho_audio=.4, rho_video=.6, contextual_ratio=.05, g=3)
    def compress(self, embeddings, input_ids, audio_id, video_id, grid, importance, context):
        from .algorithms.omnizip import omnizip
        out, keep = omnizip(embeddings.clone(), importance, input_ids, audio_id, video_id, grid,
            merging_ratio_audio=self.params['rho_audio'], merging_ratio_v=self.params['rho_video'],
            contextual_ratio=self.params['contextual_ratio'], g=self.params['g'])
        return CompressionResult(out, keep)

@register_strategy('random')
class RandomPruning(Strategy):
    """Uniform sampling without replacement over each full modality sequence."""
    defaults = dict(prune_ratio_a=.4, prune_ratio_v=.6, seed=0)

    def __init__(self, params):
        super().__init__(params)
        if type(self.params['seed']) is not int or not 0 <= self.params['seed'] < 2**63:
            raise ValueError('seed must be an integer in [0, 2**63)')

    def compress(self, embeddings, input_ids, audio_id, video_id, **kwargs):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError('Random pruning requires batch=1')
        ids = input_ids[0].detach().cpu()
        keep = torch.ones(ids.numel(), dtype=torch.bool)
        for offset, token_id, key in ((0, video_id, 'prune_ratio_v'), (1, audio_id, 'prune_ratio_a')):
            indices = (ids == token_id).nonzero().flatten()
            # A local CPU generator avoids modifying global RNG or depending on
            # device placement/evaluation order. Separate streams per modality.
            generator = torch.Generator(device='cpu').manual_seed(self.params['seed'] + offset)
            delete_count = math.floor(indices.numel() * self.params[key])
            selected = torch.randperm(indices.numel(), generator=generator)[:delete_count]
            keep[indices[selected]] = False
        return CompressionResult(embeddings, keep.to(embeddings.device))

@register_strategy('omniselect')
class OmniSelect(Strategy):
    needs_audio_importance = True
    defaults = dict(prune_ratio_a=.4, prune_ratio_v=.6, theta=3.7,
        audioclip=None, nltk_data=None,
        audioclip_device='cpu')
    def __init__(self, params):
        super().__init__(params)
        if type(self.params['theta']) not in (int,float) or not math.isfinite(self.params['theta']) or self.params['theta'] < 0:
            raise ValueError('theta must be finite and nonnegative')
    def prepare(self, **kwargs):
        from .relevance import prepare_relevance
        from .resources import resource_path
        params = dict(self.params)
        for key in ('audioclip', 'nltk_data'):
            params[key] = resource_path(key, params[key])
        return prepare_relevance(params, **kwargs)
    def compress(self, embeddings, input_ids, audio_id, video_id, grid, importance, context):
        from .algorithms.omniselect import omniselect
        ids = input_ids[0]
        video = embeddings[0, ids == video_id]
        audio = embeddings[0, ids == audio_id]
        ctx = dict(context)
        for k in ('logits_a','logits_v','pooled_a','pooled_v'):
            ctx[k] = ctx[k].to(embeddings.device)
        keep = omniselect(grid, embeddings, audio, video, importance,
            torch.nn.functional.normalize(video,dim=-1), ids, ctx)
        return CompressionResult(embeddings, keep)

def create_strategy(name='baseline', params=None):
    if name not in REGISTRY:
        raise ValueError(f'Unknown compression method {name}; available: {list(REGISTRY)}')
    return REGISTRY[name](params or {})
