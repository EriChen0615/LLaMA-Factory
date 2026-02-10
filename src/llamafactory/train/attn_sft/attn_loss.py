import torch

def _compute_attn_reranking_scores(attention_weights, attn_source_span, evidence_spans, batch_idx, aggregate_mode="sum", remove_small_attn=False):
    """
    Compute attention-based reranking scores for a single batch item.
    
    Args:
        attention_weights: List of tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
        attn_source_span: Tuple (start_idx, end_idx) for attn source
        evidence_spans: List of tuples, each tuple is (start_idx, end_idx) for evidence
        batch_idx: Index of the current batch item
    
    Returns:
        scores: List of scalar scores for each evidence span
    """
    attn_source_start, attn_source_end = attn_source_span
    scores = []
    
    for evidence_start, evidence_end in evidence_spans:
        # Stack all layers: (num_layers, num_heads, response_len, evidence_len)
        layer_attn = torch.stack([
            attention_weights[l][batch_idx, :, attn_source_start:attn_source_end, evidence_start:evidence_end] 
            for l in range(len(attention_weights))
        ])
        
        if remove_small_attn:
            # only keep attention values that are greater then mean 
            mean = layer_attn.mean() 
            small_mask = (layer_attn > mean)
            layer_attn[small_mask] = 0
        
        # Compute average attention score (equation 1)
        # Sum over all layers and heads, then normalize by evidence length
        evidence_length = evidence_end - evidence_start + 1
        if aggregate_mode == "sum":
            score = layer_attn.sum() / evidence_length
        elif aggregate_mode == "max":
            # layer_attn: (num_layers, num_heads, response_len, evidence_len)
            # Step 1: For each layer, for each response token, for each evidence token, take max over heads
            max_over_heads = layer_attn.max(dim=1).values  # (num_layers, response_len, evidence_len)
            # Step 2: For each layer, for each response token, for each evidence token, take max over evidence tokens
            max_over_evidence = max_over_heads.max(dim=-1).values  # (num_layers, response_len)
            # Step 3: Sum all maximal values
            score = max_over_evidence.sum() / evidence_length
        elif aggregate_mode == "late-interaction":
            score = layer_attn.sum(dim=0).sum(dim=0).max(dim=-1).values.sum()
        else:
            raise ValueError(f"Invalid aggregate mode: {aggregate_mode}")
        scores.append(score)

    scores = torch.stack(scores)
    return scores

def _compute_attn_loss(attention_weights, gt_evidence_labels, evidence_spans, attn_source_spans, aggregate_mode="sum", remove_small_attn=False):
    """
    Compute attention-based reranking loss according to the Attn-SFT formulation.
    
    Args:
        attention_weights: List of tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
        evidence_spans: List of lists of tuples, each tuple is (start_idx, end_idx)
        attn_source_spans: List of tuples, each tuple is (start_idx, end_idx)
        gt_evidence_labels: List of lists, ground truth labels for each evidence
    
    Returns:
        attention_loss: Scalar tensor
    """
    device = attention_weights[0].device
    
    batch_size = attention_weights[0].shape[0]
    loss = torch.tensor(0.0, device=device)
    # gt_probs = []
    evidence_probs = []
    hit_top1 = []
    
    for batch_idx in range(batch_size):
        if batch_idx >= len(evidence_spans) or batch_idx >= len(attn_source_spans):
            continue
            
        batch_evidence_spans = evidence_spans[batch_idx]
        batch_attn_source_span = attn_source_spans[batch_idx]
        batch_gt_labels = torch.as_tensor(gt_evidence_labels[batch_idx], device=device)
        gt_idx = batch_gt_labels.argmax()
        if batch_gt_labels.sum() == 0 or gt_idx < 0 or gt_idx >= len(batch_evidence_spans):
            if gt_idx < 0 or gt_idx >= len(batch_evidence_spans):
                print("[WARNING] gt_idx is out of range")
            continue
        
        # Compute attention-based reranking scores
        rerank_scores = _compute_attn_reranking_scores(
            attention_weights, batch_attn_source_span, batch_evidence_spans, batch_idx, aggregate_mode, remove_small_attn
        )
        
        # Numerically stable softmax using logsumexp trick
        logits = rerank_scores
        loss += torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([gt_idx], device=logits.device))

        max_logit = logits.max()
        shifted_logits = logits - max_logit

        log_probs = shifted_logits - torch.logsumexp(shifted_logits, dim=0)
        probs = torch.exp(log_probs)
        evidence_probs.append(probs)
        # gt_prob = probs[gt_idx]
        # gt_probs.append(gt_prob.item())
        hit_top1.append((logits.argmax() == gt_idx).item())

    return loss, evidence_probs, hit_top1