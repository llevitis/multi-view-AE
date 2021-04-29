import torch

def compute_log_alpha(mu, logvar):
    # clamp because dropout rate p in 0-99%, where p = alpha/(alpha+1)
	return (logvar - 2 * torch.log(torch.abs(mu) + 1e-8)).clamp(min=-8, max=8)

def compute_logvar(mu, log_alpha):
    return log_alpha + 2 * torch.log(torch.abs(mu) + 1e-8)

def compute_kl(mu, logvar):
    return -0.5*torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean(0)

def compute_kl_sparse(mu, logvar):
    log_alpha = compute_log_alpha(mu, logvar)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    return -torch.sum((k1 * torch.sigmoid(k2 + k3 * log_alpha) - 0.5 * torch.log1p(torch.exp(-log_alpha)) - k1), dim=1).mean(0)