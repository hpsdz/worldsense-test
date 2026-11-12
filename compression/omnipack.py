"""OmniPack adapter: upstream selection/merging with shared-model SDPA hooks."""
import math
import torch
from .strategies import Strategy, CompressionResult, register_strategy


@register_strategy('omnipack')
class OmniPack(Strategy):
    needs_audio_importance = True
    defaults = dict(video_ratio=.2, audio_ratio=.4, encoder_ratio_scale=2.,
                    grid_in_window=2, sec_in_audio_window=2, stage2_layer=18)

    def __init__(self, params):
        super().__init__(params)
        for key in ('video_ratio', 'audio_ratio'):
            value = self.params[key]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= 1:
                raise ValueError(f'{key} must be in (0, 1] (retention ratio)')
        scale = self.params['encoder_ratio_scale']
        if type(scale) not in (int, float) or not math.isfinite(scale) or scale < 1:
            raise ValueError('encoder_ratio_scale must be finite and >= 1')
        for key in ('grid_in_window', 'sec_in_audio_window', 'stage2_layer'):
            if type(self.params[key]) is not int or self.params[key] < 1:
                raise ValueError(f'{key} must be a positive integer')

    def compress(self, embeddings, input_ids, audio_id, video_id, grid, importance, context):
        from .algorithms.omnipack.pre_llm_units import video_spatiotemporal_merge, audio_temporal_merge
        p = self.params
        out, vk = video_spatiotemporal_merge(embeddings, input_ids, context['video_importance'],
            video_id, min(1., p['video_ratio'] * p['encoder_ratio_scale']),
            context['spatial_merge_unit'], grid, p['grid_in_window'])
        out, ak = audio_temporal_merge(out, input_ids, importance, audio_id,
            min(1., p['audio_ratio'] * p['encoder_ratio_scale']), p['sec_in_audio_window'])
        return CompressionResult(out, vk & ak)


@torch.no_grad()
def window_importance(q, k, window):
    """Upstream windowed incoming attention sum, chunked to bound memory."""
    out = torch.zeros(q.shape[1], device=q.device, dtype=q.dtype)
    for start in range(0, q.shape[1], window):
        end = min(start + window, q.shape[1])
        for h in range(q.shape[0]):
            for i in range(start, end, 256):
                scores = (q[h, i:min(i+256,end)] @ k[h,start:end].T) * q.shape[-1]**-.5
                out[start:end] += scores.softmax(-1).sum(0) / q.shape[0]
    return out


def encode_video(thinker, pixels, grid):
    from models.qwen2_5_omni.modeling_qwen2_5_omni import apply_rotary_pos_emb_vision
    scores = []
    def hook(module, args, kwargs):
        hs = args[0] if args else kwargs['hidden_states']
        rotary = kwargs['rotary_pos_emb']
        q = module.q(hs).reshape(len(hs), module.num_heads, -1)
        k = module.k(hs).reshape(len(hs), module.num_heads, -1)
        q = apply_rotary_pos_emb_vision(q.unsqueeze(0), rotary).squeeze(0).transpose(0,1)
        k = apply_rotary_pos_emb_vision(k.unsqueeze(0), rotary).squeeze(0).transpose(0,1)
        window = thinker.compression_strategy.params['grid_in_window'] * int(grid[0,1]*grid[0,2])
        scores.append(window_importance(q,k,window))
    handle = thinker.visual.blocks[-1].attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        result = thinker.get_video_features(pixels, grid)
    finally:
        handle.remove()
    index, _ = thinker.visual.get_window_index(grid)
    scores = scores[0].reshape(-1, thinker.visual.spatial_merge_unit).mean(-1)
    thinker._video_importance = scores[torch.argsort(index).to(scores.device)]
    return result


def stage2(thinker, hidden, causal, positions, rotary, cache_position, past, layer_idx):
    from .algorithms.omnipack.inner_llm_units import stage2_token_selection
    ids = thinker._omnipack_ids.to(hidden.device)
    p = thinker.compression_strategy.params
    cfg = {f'{m}_stage2_keep_ratio': p[f'{m}_ratio'] / min(1., p[f'{m}_ratio'] * p['encoder_ratio_scale'])
           for m in ('audio','video')}
    am, vm = ids == thinker.config.audio_token_id, ids == thinker.config.video_token_id
    hidden, causal, positions, rotary, _, past, keep = stage2_token_selection(
        hidden, causal, positions, rotary, cache_position, past, layer_idx, thinker.model.layers,
        cfg, am, ~(am|vm), vm)
    original_keep = thinker._compression_keep.clone()
    thinker._compression_keep[original_keep] = keep.to(original_keep.device)
    for name, m in [('video_tokens',vm), ('audio_tokens',am), ('query_tokens',~(am|vm)),
                    ('total_tokens',torch.ones_like(am))]:
        stats = thinker.last_compression_stats[name]
        stats['after_stage1'] = stats['after']
        stats['after'] = int((m & keep).sum())
    # All existing layer KV tensors were pruned by upstream; later layers use
    # the same compact physical slots. Semantic rotary positions remain intact.
    return hidden, causal, positions, rotary, torch.arange(hidden.shape[1],device=hidden.device), past
