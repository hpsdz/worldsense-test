"""OmniScope selection and merging adapted to the shared pre-LLM contract."""
import gc
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
import torch.nn.functional as F
from .strategies import Strategy, CompressionResult, register_strategy
from .algorithms.omniscope import VideoPruner, AudioMerger, compute_audio_tokens_allocation, apply_neighbor_boost


def audio_scores(audio, text):
    n = len(audio)//25
    if n == 0:
        return []
    if len(text) == 0:
        return [1.] * n
    gates = text.norm(dim=-1) / (text.norm(dim=-1).max()+1e-6)
    similarity = F.normalize(audio.float(),dim=-1) @ F.normalize(text.float(),dim=-1).T
    score = ((similarity*gates[None,:]/.05).softmax(-1)*similarity).sum(-1)
    values = score[:n*25].reshape(n,25).topk(5,dim=-1).values.mean(-1).tolist()
    return apply_neighbor_boost([max(0.,v*10)**2 for v in values])


@register_strategy('omniscope')
class OmniScope(Strategy):
    defaults = dict(visual_prune_ratio=.6, audio_prune_ratio=.25,
        clip_model=None, clip_device='cpu', clip_batch_size=4)

    def __init__(self, params):
        super().__init__(params)
        for key in ('visual_prune_ratio','audio_prune_ratio'):
            v = self.params[key]
            if type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v < 1:
                raise ValueError(f'{key} must be in [0,1)')
        if type(self.params['clip_batch_size']) is not int or self.params['clip_batch_size'] < 1:
            raise ValueError('clip_batch_size must be a positive integer')
        if self.params['clip_model'] is not None and (not isinstance(self.params['clip_model'],str) or not self.params['clip_model']):
            raise ValueError('clip_model must be a local model directory')
        try:
            torch.device(self.params['clip_device'])
        except (TypeError, RuntimeError) as exc:
            raise ValueError('clip_device must be a valid torch device') from exc

    @torch.inference_mode()
    def prepare(self, frames, question, inputs, tokenizer, candidates=(), **kwargs):
        from transformers import CLIPModel, CLIPImageProcessor, AutoTokenizer
        from .resources import resource_path
        path, device = resource_path('clip_model', self.params['clip_model']), self.params['clip_device']
        question = question + ('\n' + ' '.join(candidates) if candidates else '')
        if not Path(path).is_dir():
            raise FileNotFoundError(f'OmniScope requires local CLIP ViT-L/14@336 weights: {path}')
        model = CLIPModel.from_pretrained(path, local_files_only=True,
            torch_dtype=torch.float32 if device=='cpu' else torch.float16).to(device).eval()
        try:
            processor = CLIPImageProcessor.from_pretrained(path,local_files_only=True)
            clip_tokenizer = AutoTokenizer.from_pretrained(path,local_files_only=True)
            features=[]
            batch=self.params['clip_batch_size']
            for start in range(0,len(frames),batch):
                pixels=processor(images=[f.cpu().byte() for f in frames[start:start+batch]],return_tensors='pt')
                features.append(F.normalize(model.get_image_features(**pixels.to(device)).float(),dim=-1))
            tokens=clip_tokenizer(question,return_tensors='pt',truncation=True,max_length=77)
            text=F.normalize(model.get_text_features(**tokens.to(device)).float(),dim=-1)
            scores=(torch.cat(features)@text.T).flatten().cpu()
            temporal=int(inputs.video_grid_thw[0,0])
            if len(scores)!=temporal*2:
                raise ValueError('OmniScope requires two sampled frames per temporal grid')
            scores=scores.reshape(temporal,2).mean(-1).clamp_min(0).mul(100).pow(3)
            # Upstream divides by zero for all nonpositive CLIP similarities.
            if not torch.isfinite(scores).all() or scores.sum()<=0:
                scores=torch.ones_like(scores)
            return dict(video_scores=scores,question_ids=tokenizer(question,add_special_tokens=False,return_tensors='pt').input_ids)
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    @torch.inference_mode()
    def compress(self, embeddings, input_ids, audio_id, video_id, grid, context, **kwargs):
        ids=input_ids[0].to(embeddings.device)
        vp=(ids==video_id).nonzero().flatten()
        ap=(ids==audio_id).nonzero().flatten()
        if len(vp)!=int(grid[0].prod())//4 or len(context['video_scores'])!=int(grid[0,0]):
            raise ValueError('OmniScope video scores/grid/token counts do not match')
        keep=torch.ones(len(ids),dtype=torch.bool,device=embeddings.device)
        out=embeddings.clone()
        if self.params['visual_prune_ratio']>0:
            if (grid[0,1:]<4).any():
                raise ValueError('OmniScope requires a visual grid of at least 4x4 patches')
            _, groups=VideoPruner.token_pruning(SimpleNamespace(spatial_merge_size=2),out[0,vp],grid,
                dict(score_list=context['video_scores'].to(embeddings.device),target_prune_ratio=1-self.params['visual_prune_ratio']))
            per_group=int(grid[0,1]*grid[0,2])//4
            selected=torch.cat([g['indices']+i*per_group for i,g in enumerate(groups)])
            keep[vp]=False
            keep[vp[selected]]=True
        if self.params['audio_prune_ratio']>0 and len(ap)>=25:
            scores=audio_scores(out[0,ap],context['question_embeddings'].to(embeddings.device))
            budget=len(scores)*25*(1-self.params['audio_prune_ratio'])
            allocation=compute_audio_tokens_allocation(scores,budget,min_tokens=10,max_tokens=25)
            merger=AudioMerger()
            for second,count in enumerate(allocation):
                source=ap[second*25:(second+1)*25]
                merged=merger._bipartite_soft_matching(out[0,source],25-int(count))
                # Same uniform time anchors as upstream processor; merged
                # features retain upstream ordering. Audio tail remains intact.
                anchors=source[torch.as_tensor(np.linspace(0,24,len(merged),dtype=int),device=source.device)]
                keep[source]=False
                keep[anchors]=True
                out[0,anchors]=merged
        return CompressionResult(out,keep)
