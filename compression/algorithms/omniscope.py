# Extracted from OmniScope d8fb72f; see PROVENANCE.md.

import torch

import torch.nn.functional as F

import numpy as np

def compute_audio_tokens_allocation(audio_semantic_scores, total_budget, min_tokens=3, max_tokens=25):
    if audio_semantic_scores is None or len(audio_semantic_scores) == 0:
        return np.array([])
    scores = np.array(audio_semantic_scores, dtype=np.float64)
    num_seconds = len(scores)
    scores = np.maximum(scores, 0.0)
    if scores.sum() < 1e-12:
        scores = np.ones(num_seconds, dtype=np.float64)
    total_budget = float(np.clip(total_budget, num_seconds * min_tokens, num_seconds * max_tokens))
    allocation = scores / scores.sum() * total_budget
    locked = np.zeros(num_seconds, dtype=bool)
    for _ in range(10):
        allocation = np.clip(allocation, min_tokens, max_tokens)
        clipped_total = allocation.sum()
        diff = clipped_total - total_budget
        if abs(diff) < 1e-06:
            break
        if diff > 0:
            newly_locked = (allocation <= min_tokens) & ~locked
        else:
            newly_locked = (allocation >= max_tokens) & ~locked
        if not newly_locked.any():
            break
        locked[newly_locked] = True
        if locked.all():
            break
        locked_total = allocation[locked].sum()
        remaining_budget = total_budget - locked_total
        unlocked_scores = scores[~locked]
        if unlocked_scores.sum() < 1e-12:
            allocation[~locked] = remaining_budget / (~locked).sum()
        else:
            allocation[~locked] = unlocked_scores / unlocked_scores.sum() * remaining_budget
    int_budget = int(np.round(total_budget))
    floors = np.floor(allocation).astype(np.int32)
    remainders = allocation - floors
    deficit = int_budget - floors.sum()
    if deficit > 0:
        indices = np.argsort(-remainders)
        for i in range(min(deficit, num_seconds)):
            floors[indices[i]] += 1
    elif deficit < 0:
        indices = np.argsort(remainders)
        for i in range(min(-deficit, num_seconds)):
            floors[indices[i]] -= 1
    tokens_per_second = np.clip(floors, min_tokens, max_tokens)
    return tokens_per_second

def apply_neighbor_boost(scores, percentile=80, self_boost=1.2, neighbor_boost=1.5, radius=3, decay=0.5):
    if len(scores) <= 1:
        return scores
    scores_arr = np.array(scores, dtype=float)
    threshold = np.percentile(scores_arr, percentile)
    high_indices = np.where(scores_arr >= threshold)[0]
    multipliers = np.ones_like(scores_arr)
    for idx in high_indices:
        multipliers[idx] = max(multipliers[idx], self_boost)
    for idx in high_indices:
        for d in range(1, radius + 1):
            factor = 1.0 + (neighbor_boost - 1.0) * decay ** d
            for neighbor in [idx - d, idx + d]:
                if 0 <= neighbor < len(scores_arr):
                    multipliers[neighbor] = max(multipliers[neighbor], factor)
    return (scores_arr * multipliers).tolist()

class VideoPruner:

    def token_pruning(self, hidden_states, grid_thw, kwargs):
        frame_keep_indices = None
        importance_scores = kwargs.get('score_list', None)
        target_prune_ratio = kwargs.get('target_prune_ratio', None)
        if importance_scores is not None:
            s = self.spatial_merge_size
            (T_total, H_patch, W_patch) = grid_thw[0].tolist()
            num_groups = T_total
            (llm_h_orig, llm_w_orig) = (H_patch // s, W_patch // s)
            n_tokens_per_group_orig = llm_h_orig * llm_w_orig
            total_target = target_prune_ratio * T_total * llm_h_orig * llm_w_orig
            group_boundaries = [(i * n_tokens_per_group_orig, (i + 1) * n_tokens_per_group_orig) for i in range(num_groups)]
            importance_scores = torch.round(importance_scores * 1000) / 1000
            scores_np = importance_scores.cpu().numpy().astype(np.float64)
            allocation = scores_np / scores_np.sum() * float(total_target)
            target_tokens_per_group = torch.tensor(np.floor(allocation + 0.5).astype(np.int64), device=importance_scores.device)
            resampled_tokens = []
            frame_keep_indices = []
            num_tokens = 0
            token_logs = []
            for i in range(num_groups):
                (start, end) = group_boundaries[i]
                group_tokens = hidden_states[start:end]
                cur_token = min(target_tokens_per_group[i].item(), n_tokens_per_group_orig)
                aspect_ratio = float(llm_w_orig) / float(llm_h_orig)
                target_height = round((cur_token / aspect_ratio) ** 0.5)
                target_width = round(target_height * aspect_ratio)
                target_height = max(2, min(target_height, llm_h_orig))
                target_width = max(2, min(target_width, llm_w_orig))
                actual_frame_tokens = target_height * target_width
                if i % 2 == 1 and prev_group_tokens is not None:
                    num_keep = actual_frame_tokens
                    keep_ratio = actual_frame_tokens / n_tokens_per_group_orig
                    RATIO_THRESHOLD = 0.4
                    if keep_ratio > RATIO_THRESHOLD:
                        with torch.no_grad():
                            normed = F.normalize(group_tokens, dim=1)
                            normed_prev = F.normalize(prev_group_tokens, dim=1)
                            intra_sim = normed @ normed.T
                            intra_sim.fill_diagonal_(-float('inf'))
                            intra_replaceability = intra_sim.max(dim=1).values
                            cross_sim = normed @ normed_prev.T
                            inter_replaceability = cross_sim.max(dim=1).values
                            total_replaceability = intra_replaceability * inter_replaceability
                            score = -total_replaceability
                        if num_keep >= group_tokens.shape[0]:
                            keep_indices = torch.arange(group_tokens.shape[0], device=group_tokens.device)
                        else:
                            keep_indices = score.topk(num_keep).indices
                    else:
                        cos_sim = F.cosine_similarity(group_tokens, prev_group_tokens, dim=1)
                        num_keep = actual_frame_tokens
                        keep_indices = cos_sim.topk(num_keep, largest=False).indices
                    keep_indices = keep_indices.sort().values
                    resampled = group_tokens[keep_indices]
                    frame_keep_indices.append({'indices': keep_indices, 'llm_w_orig': llm_w_orig, 'llm_h_orig': llm_h_orig})
                else:
                    num_keep = actual_frame_tokens
                    if num_keep >= group_tokens.shape[0]:
                        keep_indices = torch.arange(group_tokens.shape[0], device=group_tokens.device)
                    else:
                        with torch.no_grad():
                            normed = F.normalize(group_tokens, dim=1)
                            sim = normed @ normed.T
                            sim.fill_diagonal_(-float('inf'))
                            (knn_vals, _) = torch.topk(sim, min(5, group_tokens.shape[0] - 1), dim=1)
                            density = knn_vals.mean(dim=1)
                            keep_indices = torch.topk(-density, num_keep).indices
                    keep_indices = keep_indices.sort().values
                    resampled = group_tokens[keep_indices]
                    frame_keep_indices.append({'indices': keep_indices, 'llm_w_orig': llm_w_orig, 'llm_h_orig': llm_h_orig})
                prev_group_tokens = group_tokens
                num_tokens += resampled.shape[0]
                token_logs.append(resampled.shape[0])
                resampled_tokens.append(resampled)
            hidden_states = torch.cat(resampled_tokens, dim=0)
            return (hidden_states, frame_keep_indices)
        return (hidden_states, frame_keep_indices)

class AudioMerger:

    def compress_audio_features(self, audio_features, compress_info, tokens_per_second=25):
        if compress_info is None or len(compress_info) == 0:
            return audio_features
        num_seconds = min(len(compress_info), audio_features.shape[0] // tokens_per_second)
        compressed_features = []
        for sec_idx in range(num_seconds):
            sec_start = sec_idx * tokens_per_second
            sec_end = sec_start + tokens_per_second
            sec_features = audio_features[sec_start:sec_end]
            target_tokens = int(compress_info[sec_idx])
            target_tokens = max(1, min(target_tokens, tokens_per_second))
            if target_tokens >= tokens_per_second:
                compressed_features.append(sec_features)
            else:
                r = tokens_per_second - target_tokens
                merged = self._bipartite_soft_matching(sec_features, r)
                compressed_features.append(merged)
        processed_tokens = num_seconds * tokens_per_second
        if processed_tokens < audio_features.shape[0]:
            remaining = audio_features[processed_tokens:]
            compressed_features.append(remaining)
        return torch.cat(compressed_features, dim=0)

    def _bipartite_soft_matching(self, features, r):
        (N, D) = features.shape
        device = features.device
        dtype = features.dtype
        if r <= 0 or N < 2:
            return features
        target_N = N - r
        current_features = features
        while current_features.shape[0] > target_N and current_features.shape[0] >= 2:
            current_N = current_features.shape[0]
            max_r_per_round = current_N // 2 - 1
            r_this_round = min(current_N - target_N, max_r_per_round)
            if r_this_round <= 0:
                break
            current_features = self._bipartite_soft_matching_one_round(current_features, r_this_round)
        return current_features

    def _bipartite_soft_matching_one_round(self, features, r):
        (N, D) = features.shape
        device = features.device
        dtype = features.dtype
        if r <= 0 or N < 2:
            return features
        features_norm = features / (features.norm(dim=-1, keepdim=True) + 1e-08)
        a = features_norm[::2]
        b = features_norm[1::2]
        a_raw = features[::2]
        b_raw = features[1::2]
        (len_a, len_b) = (a.shape[0], b.shape[0])
        scores = a @ b.T
        (node_max, node_idx) = scores.max(dim=-1)
        edge_idx = node_max.argsort(descending=True)
        src_idx = edge_idx[:r]
        unm_idx = edge_idx[r:].sort()[0]
        dst_idx = node_idx[src_idx]
        unmerged_a = a_raw[unm_idx]
        src = a_raw[src_idx]
        merged_b = b_raw.clone()
        b_counts = torch.ones(len_b, 1, device=device, dtype=dtype)
        dst_cpu = dst_idx.cpu().tolist()
        groups = {}
        for (i, d) in enumerate(dst_cpu):
            groups.setdefault(d, []).append(i)
        for (d, src_indices) in groups.items():
            idx_t = torch.tensor(src_indices, device=device)
            merged_b[d] += src[idx_t].sum(dim=0)
            b_counts[d] += len(src_indices)
        merged_b = merged_b / b_counts
        idx_a = torch.arange(0, N, 2, device=device)
        idx_b = torch.arange(1, N, 2, device=device)
        idx_unm = idx_a[unm_idx]
        idx_merged = idx_b
        output_indices = torch.cat([idx_unm, idx_merged], dim=0)
        perm = output_indices.argsort()
        output_features = torch.cat([unmerged_a, merged_b], dim=0)
        return output_features[perm]
