import torch

def _compute_attn_reranking_scores(attention_weights, response_span, evidence_spans, batch_idx):
    """
    Compute attention-based reranking scores for a single batch item.
    
    Args:
        attention_weights: List of tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
        response_span: Tuple (start_idx, end_idx) for response
        evidence_spans: List of tuples, each tuple is (start_idx, end_idx) for evidence
        batch_idx: Index of the current batch item
    
    Returns:
        scores: List of scalar scores for each evidence span
    """
    response_start, response_end = response_span
    scores = []
    
    for evidence_start, evidence_end in evidence_spans:
        # Stack all layers: (num_layers, num_heads, response_len, evidence_len)
        layer_attn = torch.stack([
            attention_weights[l][batch_idx, :, response_start:response_end+1, evidence_start:evidence_end+1] 
            for l in range(len(attention_weights))
        ])
        
        # Compute average attention score (equation 1)
        # Sum over all layers and heads, then normalize by evidence length
        evidence_length = evidence_end - evidence_start + 1
        score = layer_attn.sum() / evidence_length
        scores.append(score)

    scores = torch.stack(scores)
    return scores

def _compute_attn_loss(attention_weights, gt_evidence_labels, evidence_spans, response_spans):
    """
    Compute attention-based reranking loss according to the Attn-SFT formulation.
    
    Args:
        attention_weights: List of tensors, each of shape (batch_size, num_heads, seq_len, seq_len)
        evidence_spans: List of lists of tuples, each tuple is (start_idx, end_idx)
        response_spans: List of tuples, each tuple is (start_idx, end_idx)
        gt_evidence_labels: List of lists, ground truth labels for each evidence
    
    Returns:
        attention_loss: Scalar tensor
    """
    device = attention_weights[0].device
    
    batch_size = attention_weights[0].shape[0]
    loss = torch.tensor(0.0, device=device)
    gt_probs = []
    
    for batch_idx in range(batch_size):
        if batch_idx >= len(evidence_spans) or batch_idx >= len(response_spans):
            continue
            
        batch_evidence_spans = evidence_spans[batch_idx]
        batch_response_span = response_spans[batch_idx]
        batch_gt_labels = torch.tensor(gt_evidence_labels[batch_idx], device=device)
        if batch_gt_labels.sum() == 0:
            continue
        
        # Compute attention-based reranking scores
        rerank_scores = _compute_attn_reranking_scores(
            attention_weights, batch_response_span, batch_evidence_spans, batch_idx
        )
        
        # Numerically stable softmax using logsumexp trick
        logits = rerank_scores
        gt_idx = batch_gt_labels.argmax()
        loss += torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([gt_idx], device=logits.device))

        max_logit = logits.max()
        shifted_logits = logits - max_logit

        log_probs = shifted_logits - torch.logsumexp(shifted_logits, dim=0)
        probs = torch.exp(log_probs)
        gt_prob = probs[gt_idx]
        gt_probs.append(gt_prob.item())

    return loss, gt_probs