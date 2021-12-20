import scipy
import numpy as np
import torch

"""
FUNCTION FOR THE CANONICAL BETA DISTRIBUTION
"""

def compute_constraint(alpha, beta, x):
    """
    Computes the differential of the likelihood w.r.t. alpha (i.e. a). Used for computed saturated parameters.

    Input:
    - alpha: float
        Parameter a (or $\alpha$)
    - beta: float
        Parameter b (or $\beta$)
    - x: float
        Data value
    """
    return scipy.special.digamma(alpha) - scipy.special.digamma(alpha + beta) - np.log(x)


def compute_alpha(beta, x, eps=10**(-8)):
    min_val, max_val = 0, 10**10
    alpha = (min_val + max_val) / 2
    
    while np.abs(compute_constraint(alpha, beta, x)) > eps:
        if compute_constraint(alpha, beta, x) > 0:
            max_val = (min_val + max_val) / 2
        else:
            min_val = (min_val + max_val) / 2

        alpha = (min_val + max_val) / 2

    return alpha

def compute_alpha_gene(beta, x, eps=10**(-8)):
    """
    wrapper to compute the alpha coefficients for a full gene
    """
    print('START ONE', flush=True)
    return [compute_alpha(beta, x[i]) for i in range(x.shape[0])]

"""
FUNCTION FOR THE REPARAMETRIZATION
"""

def log_part_grad_zero_reparam(mu, eta, x):
    return torch.digamma(mu * eta) - torch.digamma(eta - mu * eta) + torch.log(1-x) - torch.log(x)

def compute_mu(eta, x, eps=10**(-6), maxiter=1000):
    min_mu, max_mu = 0, 1
    mu = (min_mu + max_mu) / 2
    
    iter_idx = 0
    current_value = log_part_grad_zero_reparam(mu, eta, x)
    while np.abs(current_value) > eps:
        if current_value > 0:
            max_mu = (min_mu + max_mu) / 2
        else:
            min_mu = (min_mu + max_mu) / 2

        mu = (min_mu + max_mu) / 2
        current_value = log_part_grad_zero_reparam(mu, eta, x)

        if iter_idx > maxiter:
            print('DID NOT CONVERGE WITH LOG PART AT %s'%(current_value))
            break
        iter_idx += 1
    return mu

def compute_mu_gene(eta, X, eps=10**-6, maxiter=1000):
    return [
        compute_mu(eta, x, eps=eps, maxiter=maxiter) for x in X
    ]
