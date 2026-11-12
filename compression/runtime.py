"""Model integration and compact DynamicCache handling. Baseline is untouched."""
import torch
from transformers.cache_utils import DynamicCache

def audio_importance(attn, hidden_states):
    """Same global Q/K importance reduction as the local upstream implementations."""
    q = attn.q_proj(hidden_states).reshape(-1, attn.num_heads, attn.head_dim).transpose(0,1)
    k = attn.k_proj(hidden_states).reshape(-1, attn.num_heads, attn.head_dim).transpose(0,1)
    score = torch.zeros(q.shape[1], device=q.device)
    for h in range(0, q.shape[0], 4):
        for i in range(0, q.shape[1], 512):
            weights = (q[h:h+4,i:i+512] @ k[h:h+4].transpose(-1,-2)) * attn.head_dim**-.5
            score += weights.softmax(-1).sum((0,1))
    return score / (q.shape[0]*q.shape[1])

def encode_audio(thinker, call):
    strategy = getattr(thinker, 'compression_strategy', None)
    if strategy is None or not strategy.needs_audio_importance:
        return call()
    scores = []
    def hook(module, args, kwargs):
        hs = kwargs.get('hidden_states', args[0] if args else None)
        if getattr(thinker, 'compression_method', None) == 'omnipack':
            from .omnipack import window_importance
            q = module.q_proj(hs).reshape(-1,module.num_heads,module.head_dim).transpose(0,1)
            k = module.k_proj(hs).reshape(-1,module.num_heads,module.head_dim).transpose(0,1)
            scores.append(window_importance(q,k,strategy.params['sec_in_audio_window']*50))
        else:
            scores.append(audio_importance(module, hs))
    handle = thinker.audio_tower.layers[-1].self_attn.register_forward_pre_hook(hook, with_kwargs=True)
    try:
        features = call()
    finally:
        handle.remove()
    score = scores[0]
    score = score[:score.numel()//2*2].reshape(-1,2).mean(-1)
    if score.numel() != features.shape[0]:
        raise ValueError('Audio importance pooling does not match encoded tokens; unsupported audio chunk layout')
    thinker._audio_importance = score
    return features

def compress(thinker, ids, embeddings, mask, positions, grid, cache_position, past, labels):
    strategy = getattr(thinker, 'compression_strategy', None)
    if strategy is None or getattr(thinker, 'compression_method', 'baseline') == 'baseline':
        return embeddings, mask, positions, cache_position
    if thinker.training or labels is not None or ids is None or ids.shape[0] != 1 or not mask.bool().all():
        raise ValueError('Compression supports inference, batch=1, unpadded input only')
    if not isinstance(past, DynamicCache):
        raise ValueError('Compression requires use_cache=True with DynamicCache')
    if grid is None or len(grid) != 1:
        raise ValueError('Compression requires exactly one video')
    aid, vid = thinker.config.audio_token_id, thinker.config.video_token_id
    if not (ids == aid).any() or not (ids == vid).any():
        raise ValueError('This strategy requires both video and audio')
    importance = getattr(thinker, '_audio_importance', None)
    context = thinker.compression_context
    if getattr(thinker, 'compression_method', None) == 'omniscope':
        context = dict(context)
        embed = thinker.get_input_embeddings()
        context['question_embeddings'] = embed(context['question_ids'].to(embeddings.device))[0].to(embeddings.device)
    if getattr(thinker, 'compression_method', None) == 'omnipack':
        if strategy.params['stage2_layer'] >= len(thinker.model.layers):
            raise ValueError('stage2_layer must be smaller than the number of text layers')
        context = dict(video_importance=thinker._video_importance.to(embeddings.device),
                       spatial_merge_unit=thinker.visual.spatial_merge_unit)
    result = strategy.compress(embeddings=embeddings, input_ids=ids, audio_id=aid, video_id=vid,
        grid=grid, importance=importance.to(embeddings.device) if importance is not None else None,
        context=context)
    keep = result.keep.to(embeddings.device)
    if keep.dtype != torch.bool or keep.shape != (ids.shape[1],) or not keep.any():
        raise ValueError('Invalid compression mask')
    text = ((ids[0] != aid) & (ids[0] != vid)).to(keep.device)
    if not keep[text].all() or result.embeddings.shape != embeddings.shape:
        raise ValueError('Compression must preserve text and return full-length embeddings')
    thinker.last_compression_stats = {}
    for name, m in [('video_tokens',ids[0]==vid),('audio_tokens',ids[0]==aid),('query_tokens',text),('total_tokens',torch.ones_like(text))]:
        m=m.to(keep.device)
        thinker.last_compression_stats[name]={'before':int(m.sum()), 'after':int((m&keep).sum())}
    thinker._compression_keep = keep
    if getattr(thinker, 'compression_method', None) == 'omnipack':
        thinker._omnipack_ids = ids[0,keep.to(ids.device)]
    # Semantic RoPE positions stay original; physical cache slots are compact.
    return result.embeddings[:,keep], mask[:,keep], positions[:,:,keep], torch.arange(int(keep.sum()),device=embeddings.device)

def compact_decode(thinker, mask, cache_position):
    keep = getattr(thinker, '_compression_keep', None)
    if keep is None:
        return mask, cache_position
    original = keep.numel()
    mask = torch.cat((mask[:,:original][:,keep.to(mask.device)],mask[:,original:]),dim=1)
    return mask, cache_position - (original-int(keep.sum()))
