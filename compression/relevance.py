"""AudioCLIP preparation; uses the same frames/audio as Qwen, no second decode."""
from pathlib import Path
from types import SimpleNamespace
import gc
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
import nltk
from nltk import word_tokenize, pos_tag
ROOT = Path(__file__).resolve().parents[1]

def matching(args, frames, audio):
    from vendor.audioclip.model import AudioCLIP
    device = args.audioclip_device
    model = AudioCLIP(pretrained=str(args.audioclip)).to(device).eval()
    tokens = word_tokenize(args.prompt)
    keywords = [w for w, tag in pos_tag(tokens) if tag.startswith(('NN', 'JJ'))]
    # AudioCLIP expects a list of word lists, not individual strings (characters).
    text_features = F.normalize(model.encode_text([[w] for w in (keywords or tokens)]), dim=-1)
    transform = transforms.Compose([
        transforms.ToPILImage(), transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224), transforms.ToTensor(),
        transforms.Normalize((.48145466, .4578275, .40821073), (.26862954, .26130258, .27577711)),
    ])
    n = len(frames)
    track = torch.as_tensor(np.asarray(audio), dtype=torch.float32).flatten()
    chunk = track.numel() // n
    if chunk < 2048:
        raise ValueError('Audio chunks are too short for AudioCLIP; use fewer frames.')
    chunks = track[:chunk*n].reshape(n, 1, chunk)
    audio_scores, video_scores = [], []
    # Small batches and release AudioCLIP before loading Qwen to fit 8 GB GPUs.
    for i in range(n):
        af = F.normalize(model.encode_audio(chunks[i:i+1].to(device)), dim=-1)
        vf = F.normalize(model.encode_image(transform(frames[i].byte()).unsqueeze(0).to(device)), dim=-1)
        audio_scores.append((model.logit_scale_at.exp().clamp(1,100) * af @ text_features.T).mean().cpu())
        video_scores.append((model.logit_scale.exp().clamp(1,100) * vf @ text_features.T).mean().cpu())
    a, v = torch.stack(audio_scores), torch.stack(video_scores)
    mode = 'uniform' if not keywords or abs(int(v.mean()) - int(a.mean())) <= args.theta else bool(v.mean() > a.mean())
    del model, text_features
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return a, v, mode


def groups(indices):
    out, current = [], []
    for i in indices:
        if current and i != current[-1] + 1:
            out.append(current)
            current = []
        current.append(i)
    if current:
        out.append(current)
    return out


@torch.inference_mode()
def prepare_relevance(params, frames, audio, question, inputs, audio_id, video_id):
    if audio is None or not len(audio):
        raise ValueError('OmniSelect requires audio in the video')
    if params['nltk_data'] not in nltk.data.path:
        nltk.data.path.insert(0, params['nltk_data'])
    if not Path(params['audioclip']).is_file():
        raise FileNotFoundError('AudioCLIP weights missing: ' + params['audioclip'] + '; set audioclip in resources.local.json')
    args = SimpleNamespace(**params, prompt=question, prune=True)
    a, v, mode = matching(args, frames, audio)
    ids = inputs.input_ids[0]
    v_lst = (ids == video_id).nonzero().flatten().tolist()
    a_lst = (ids == audio_id).nonzero().flatten().tolist()
    vg, ag = groups(v_lst), groups(a_lst)
    mapping = {idx:g+1 for seq in (vg,ag) for g,group in enumerate(seq) for idx in group}
    grid_tokens = int(inputs.video_grid_thw[0,1]*inputs.video_grid_thw[0,2])//4
    temporal = int(inputs.video_grid_thw[0,0])
    if len(v) != temporal * 2:
        raise ValueError('Expected two sampled frames per Qwen temporal patch.')
    token_scores = v.reshape(temporal, 2).mean(1).repeat_interleave(grid_tokens)
    cursor, vs = 0, []
    for group in vg:
        # Interleaved audio can split a temporal patch into runs of 1 and 143
        # video tokens; map scores per token before pooling these runs.
        vs.append(token_scores[cursor:cursor+len(group)].mean())
        cursor += len(group)
    assert cursor == token_scores.numel()
    # AudioCLIP audio scores represent equal-duration chunks of the full track.
    # Weight overlap with each actual contiguous audio-token time group.
    edges = torch.linspace(0, len(a_lst), len(a)+1)
    cursor, aus = 0, []
    for group in ag:
        overlap = (torch.minimum(edges[1:], torch.tensor(cursor+len(group))) - torch.maximum(edges[:-1], torch.tensor(cursor))).clamp_min(0)
        aus.append((a*overlap).sum()/overlap.sum())
        cursor += len(group)
    prune_need = {'args':args, 'logits_a':a, 'logits_v':v, 'video_first':mode,
                  'pooled_a':torch.stack(aus), 'pooled_v':torch.stack(vs),
                  'token_id2group':mapping, 'v_lst':v_lst, 'a_lst':a_lst}
    return prune_need
