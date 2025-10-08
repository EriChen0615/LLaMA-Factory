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