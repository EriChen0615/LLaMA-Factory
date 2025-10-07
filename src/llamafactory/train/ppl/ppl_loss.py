import torch

def compute_ppl_loss(pos_logps, neg_logps, tau=1, prior=None, llk_factor=1.0):
    """
    pos_logps: (1, token_num)
    neg_logps: (K-1, token_num)
    tau: temperature
    prior: prior distribution

        L = -log(\frac{exp(r_k)}{\sum_{i=1}^{K} exp(r_i)})
        where r_k = log\pi(y|z_k,x) + log\pi(z_k|x,Z), i.e., log-likelihood plus the log-prior
    
    """
    denom = torch.logsumexp(torch.cat([pos_logps, neg_logps], dim=0), dim=-1)
    posterior_loss = -(pos_logps - denom)
    llk_loss = llk_factor * -pos_logps # negative log-likelihood loss
    total_loss = posterior_loss + llk_loss

    return total_loss.squeeze(0), posterior_loss.squeeze(0), llk_loss.squeeze(0)