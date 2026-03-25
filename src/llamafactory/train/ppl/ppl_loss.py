import torch
from typing import Tuple
from ...extras.constants import IGNORE_INDEX


def build_dpp_kernel(
    singleton_logits: "torch.Tensor",
    embeddings: "torch.Tensor",
    jitter: float = 1.0e-6,
) -> "torch.Tensor":
    r"""Build a PSD DPP kernel L = D^{1/2} S D^{1/2}.

    - singleton_logits: shape [K, 1] or [K], transformed by sigmoid for D diagonal.
    - embeddings: shape [K, d], normalized then dot-product for S.
    """
    if singleton_logits.dim() > 1:
        singleton_logits = singleton_logits.squeeze(-1)
    d = torch.sigmoid(singleton_logits).clamp_min(1.0e-12)
    sqrt_d = torch.sqrt(d)

    z = torch.nn.functional.normalize(embeddings, p=2, dim=-1, eps=1.0e-12)
    s = z @ z.transpose(0, 1)
    L = (sqrt_d.unsqueeze(1) * s) * sqrt_d.unsqueeze(0)

    if jitter > 0:
        eye = torch.eye(L.size(0), device=L.device, dtype=L.dtype)
        L = L + jitter * eye
    return L


def _stable_logdet_spd(matrix: "torch.Tensor", jitter: float = 1.0e-6) -> "torch.Tensor":
    r"""Numerically stable log-det for positive (semi-)definite matrices."""
    eye = torch.eye(matrix.size(0), device=matrix.device, dtype=matrix.dtype)
    # Retry with exponentially larger jitter if needed.
    cur_jitter = max(float(jitter), 0.0)
    for _ in range(5):
        try:
            chol = torch.linalg.cholesky(matrix + cur_jitter * eye)
            return 2.0 * torch.log(torch.diagonal(chol)).sum()
        except RuntimeError:
            cur_jitter = 10.0 * max(cur_jitter, 1.0e-8)
    # Final fallback via slogdet.
    sign, logabsdet = torch.linalg.slogdet(matrix + cur_jitter * eye)
    if sign <= 0:
        return torch.tensor(float("-inf"), device=matrix.device, dtype=matrix.dtype)
    return logabsdet


def dpp_subset_log_prob(
    L: "torch.Tensor",
    gt_indices: "torch.Tensor",
    jitter: float = 1.0e-6,
) -> "torch.Tensor":
    r"""Compute log P(Y=gt_indices) under L-ensemble DPP."""
    k = L.size(0)
    eye = torch.eye(k, device=L.device, dtype=L.dtype)
    log_norm = _stable_logdet_spd(eye + L, jitter=jitter)
    if gt_indices.numel() == 0:
        return -log_norm

    L_y = L.index_select(0, gt_indices).index_select(1, gt_indices)
    log_num = _stable_logdet_spd(L_y, jitter=jitter)
    return log_num - log_norm


def dpp_subset_nll(
    L: "torch.Tensor",
    gt_indices: "torch.Tensor",
    jitter: float = 1.0e-6,
) -> "torch.Tensor":
    r"""Negative log-likelihood for a GT subset under a DPP kernel."""
    return -dpp_subset_log_prob(L, gt_indices, jitter=jitter)

def compute_ppl_loss(pos_logps, neg_logps, prior_logits=None):
    """
    pos_logps: (1, token_num)
    neg_logps: (K-1, token_num)
    tau: temperature
    prior: prior distribution

        L = -log(\frac{exp(r_k)}{\sum_{i=1}^{K} exp(r_i)})
        where r_k = log\pi(y|z_k,x) + log\pi(z_k|x,Z), i.e., log-likelihood plus the log-prior
    
    """
    if prior_logits is not None:
        log_passage_prior = torch.log_softmax(prior_logits, dim=-1) # shape (K, )
    else:
        log_passage_prior = torch.zeros(logits.shape[0], device=logits.device) # shape (K, )

    logm = log_passage_prior.unsqueeze(1) # shape (K, 1)

    all_logps = torch.cat([pos_logps, neg_logps], dim=0)
    denom = torch.logsumexp(all_logps, dim=-1)
    posterior_logprob = (all_logps - denom)

    posterior_loss = -posterior_logprob[0]
    llk_loss = llk_factor * -pos_logps # negative log-likelihood loss
    total_loss = posterior_loss + llk_loss

    return total_loss.squeeze(0), posterior_loss.squeeze(0), llk_loss.squeeze(0), posterior_logprob, log_passage_prior

def compute_joint_loss(pos_logps, logits, labels, prior_logits=None):
    """
    Assume that the first logits are the positive logits and the other logits are the negative logits.
    """
    if prior_logits is not None:
        log_passage_prior = torch.log_softmax(prior_logits, dim=-1) # shape (K, )
    else:
        log_passage_prior = torch.zeros(logits.shape[0], device=logits.device) # shape (K, )

    logm = log_passage_prior.unsqueeze(1) # shape (K, 1)

    cum_logps, _ = _get_token_cumulative_logps(logits, labels) # shape (K, len)
    llk_loss = -pos_logps
    posterior_logprob = cum_logps - torch.logsumexp(cum_logps, dim=0) # shape (K, len)
    posterior_loss = -posterior_logprob[0].sum(-1) # sum over all answer tokens
    total_loss = posterior_loss + llk_loss

    return total_loss.squeeze(0), posterior_loss.squeeze(0), llk_loss.squeeze(0), posterior_logprob, log_passage_prior


def compute_ensemble_loss(
    logits,
    labels,
    prior_logits=None,
    per_token_logps=None,
    deflection_logits=None,
    deflection_labels=None,
    gt_subset_per_token_logps=None,
    gt_subset_labels=None,
    gt_subset_loss_factor=1.0,
):
    """
    Let pk(i) denotes the i-th answer token probability when conditioned on the k-th passage.
    qk(i) denotes the cumulative log-probability of the answer tokens up to the i-th token (not included) when conditioned on the k-th passage.
    Let mk denote the prior probability of the k-th passage.
    loss = - sum_i log(sum_k exp(log[pk(i)] + log[qk(i)]+log[mk] + sum_k' log[qk'(i)]+log[mk']))

    For deflection (step N+1):
    - p_k,N+1 = s_k^d * (1-s_k)^(1-d) where s_k = sigmoid(deflection_logits[k]) and d is deflection label
    - logp_k,N+1 = d * log(s_k) + (1-d) * log(1-s_k)
    - Uses passage posterior computed from all N answer tokens
    """
    if per_token_logps is None and logits is None:
        raise ValueError("compute_ensemble_loss requires logits or per_token_logps.")

    device = per_token_logps.device if per_token_logps is not None else logits.device
    if prior_logits is not None:
        log_passage_prior = torch.log_softmax(prior_logits, dim=0)
    else:
        batch_size = per_token_logps.shape[0] if per_token_logps is not None else logits.shape[0]
        log_passage_prior = torch.zeros((batch_size, 1), device=device)
    logm = log_passage_prior

    token_logps, cumulative_token_logps, valid_length = _get_token_and_cumulative_logps(
        logits, labels, per_token_logps=per_token_logps
    )
    K, N = token_logps.shape[0], valid_length

    logp = token_logps
    logq = torch.cat([torch.zeros((K, 1), device=device), cumulative_token_logps[:, : N - 1]], dim=-1)

    log_passage_posterior = logq + logm - torch.logsumexp(logq + logm, dim=0).unsqueeze(0)

    step_passage_marginalized = logp + log_passage_posterior
    step_marginalized = torch.logsumexp(step_passage_marginalized, dim=0)
    loss = -step_marginalized.sum(dim=0)

    llk_loss = -logp.sum(dim=0).sum(dim=0)
    if gt_subset_per_token_logps is not None and gt_subset_labels is not None:
        gt_subset_token_logps, _, _ = _get_token_and_cumulative_logps(
            None, gt_subset_labels, per_token_logps=gt_subset_per_token_logps
        )
        llk_loss = -gt_subset_token_logps.sum() * gt_subset_loss_factor
        loss = loss + llk_loss

    if deflection_logits is not None and deflection_labels is not None:
        deflection_logits_flat = deflection_logits.squeeze(-1)
        deflection_labels_flat = deflection_labels.float()

        log_s_k = -torch.nn.functional.softplus(-deflection_logits_flat)
        log_one_minus_s_k = -torch.nn.functional.softplus(deflection_logits_flat)
        logp_deflection = deflection_labels_flat * log_s_k + (1 - deflection_labels_flat) * log_one_minus_s_k

        logq_N_plus_1 = cumulative_token_logps[:, N - 1] if N > 0 else torch.zeros((K,), device=device)
        log_passage_posterior_N_plus_1 = logq_N_plus_1.unsqueeze(-1) + logm - torch.logsumexp(
            logq_N_plus_1.unsqueeze(-1) + logm, dim=0
        )
        log_passage_posterior_N_plus_1 = log_passage_posterior_N_plus_1.squeeze(-1)
        step_marginalized_N_plus_1 = torch.logsumexp(logp_deflection + log_passage_posterior_N_plus_1, dim=0)
        loss = loss - step_marginalized_N_plus_1

    return loss, -log_passage_posterior[0].sum(-1), llk_loss, log_passage_posterior, log_passage_prior.squeeze(1)


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

def _get_token_and_cumulative_logps(logits, labels, label_pad_token_id=IGNORE_INDEX, per_token_logps=None):
    r"""
    Computes the log probabilities of the given labels under the given logits AND the log probabilities of the labelsc umulative up to the current token
    Assume that labels are pointing to the same target (therefore the same number of valid tokens).
    

    Returns:
        token_logps: A tensor of shape (batch_size, # of tokens in labels) containing the log probabilities of the current token..
        cumulative_token_logps: A tensor of shape (batch_size, # of tokens in labels) containing the cumulative sum of log probabilities.
        valid_length: A tensor of shape (batch_size,) containing the number of non-masked tokens.
    """
    # if logits.shape[:-1] != labels.shape:
    #     raise ValueError("Logits (batchsize x seqlen) and labels must have the same shape.")

    labels = labels[:, 1:].clone()
    loss_mask = labels != label_pad_token_id
    labels[labels == label_pad_token_id] = 0  # dummy token

    if per_token_logps is None:
        logits = logits[:, :-1, :]
        per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)

    answer_token_logps = torch.stack(
        [
            per_token_logps[k,loss_mask[k]] for k in range(labels.shape[0])
        ]
    ) #  shape (target len,)

    # perform cumulative sum up to the current 
    cumulative_token_logps = torch.cumsum(answer_token_logps, dim=-1)
    return answer_token_logps, cumulative_token_logps, loss_mask[0].sum(-1)