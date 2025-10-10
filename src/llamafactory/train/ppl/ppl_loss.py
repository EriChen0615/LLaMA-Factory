import torch
from typing import Tuple
from ...extras.constants import IGNORE_INDEX

def compute_ppl_loss(pos_logps, neg_logps, tau=1, prior=None, llk_factor=1.0):
    """
    pos_logps: (1, token_num)
    neg_logps: (K-1, token_num)
    tau: temperature
    prior: prior distribution

        L = -log(\frac{exp(r_k)}{\sum_{i=1}^{K} exp(r_i)})
        where r_k = log\pi(y|z_k,x) + log\pi(z_k|x,Z), i.e., log-likelihood plus the log-prior
    
    """
    all_logps = torch.cat([pos_logps, neg_logps], dim=0)
    denom = torch.logsumexp(all_logps, dim=-1)
    posterior_logprob = (all_logps - denom)

    posterior_loss = -posterior_logprob[0]
    llk_loss = llk_factor * -pos_logps # negative log-likelihood loss
    total_loss = posterior_loss + llk_loss

    return total_loss.squeeze(0), posterior_loss.squeeze(0), llk_loss.squeeze(0), posterior_logprob

def compute_joint_loss(pos_logps, logits, labels):
    """
    Assume that the first logits are the positive logits and the other logits are the negative logits.
    """
    cum_logps, _ = _get_token_cumulative_logps(logits, labels) # shape (K, len)
    llk_loss = -pos_logps
    posterior_logprob = cum_logps - torch.logsumexp(cum_logps, dim=0) # shape (K, len)
    posterior_loss = -posterior_logprob[0].sum(-1) # sum over all answer tokens
    total_loss = posterior_loss + llk_loss

    return total_loss.squeeze(0), posterior_loss.squeeze(0), llk_loss.squeeze(0), posterior_logprob

def compute_ensemble_loss(logits, labels):
    """
    Let pk(i) denotes the i-th answer token probability when conditioned on the k-th passage.
    qk(i) denotes the cumulative log-probability of the answer tokens up to the i-th token (not included) when conditioned on the k-th passage.
    loss = - sum_i log(sum_k exp(logpk(i) + logqk(i) + sum_k' logqk'(i)))
    """
    token_logps, cumulative_token_logps, valid_length = _get_token_and_cumulative_logps(logits, labels) # shape (K, N), where N is the number of answer tokens
    K, N = token_logps.shape[0], valid_length

    logp = token_logps # shape (K, N)
    logq = torch.cat([torch.zeros((K, 1), device=logits.device), cumulative_token_logps[:,:N-1]], dim=-1) # shape (K, N) #NOTE log(q0)=0 for all K. i.e., uniform prior. 

    log_passage_posterior = logq - torch.logsumexp(logq, dim=0).unsqueeze(0) # shape (K, N)

    step_passage_marginalized = logp + log_passage_posterior

    step_marginalized = torch.logsumexp(step_passage_marginalized, dim=0) # logsumexp over all K , shape (N, )
    loss = -step_marginalized.sum(dim=0) # sum over all N
    return loss, -log_passage_posterior[0].sum(-1), -logp.sum(dim=0).sum(dim=0), log_passage_posterior

def _get_token_cumulative_logps(
    logits: "torch.Tensor", labels: "torch.Tensor", label_pad_token_id: int = IGNORE_INDEX
) -> Tuple["torch.Tensor", "torch.Tensor"]:
    r"""
    Computes the log probabilities of the given labels under the given logits, cumulative up to the current token
    Assume that labels are pointing to the same target (therefore the same number of valid tokens).
    

    Returns:
        logps: A tensor of shape (batch_size, # of tokens in labels) containing the sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    answer_token_logps = torch.stack(
        [
            per_token_logps[k,loss_mask[k]] for k in range(logits.shape[0])
        ]
    ) #  shape (target len,)

    # perform cumulative sum up to the current 
    cumulative_token_logps = torch.cumsum(answer_token_logps, dim=-1)
    return cumulative_token_logps, loss_mask[0].sum(-1)

def _get_token_and_cumulative_logps(logits, labels, label_pad_token_id=IGNORE_INDEX):
    r"""
    Computes the log probabilities of the given labels under the given logits AND the log probabilities of the labelsc umulative up to the current token
    Assume that labels are pointing to the same target (therefore the same number of valid tokens).
    

    Returns:
        token_logps: A tensor of shape (batch_size, # of tokens in labels) containing the log probabilities of the current token..
        cumulative_token_logps: A tensor of shape (batch_size, # of tokens in labels) containing the cumulative sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    if logits.shape[:-1] != labels.shape:
        raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    logits = logits[:, :-1, :]
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token
    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    answer_token_logps = torch.stack(
        [
            per_token_logps[k,loss_mask[k]] for k in range(logits.shape[0])
        ]
    ) #  shape (target len,)

    # perform cumulative sum up to the current 
    cumulative_token_logps = torch.cumsum(answer_token_logps, dim=-1)
    return answer_token_logps, cumulative_token_logps, loss_mask[0].sum(-1)