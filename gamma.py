import torch

GAMMA_ZERO_THRESHOLD = 1e-5

def compute_constraint(theta, nu, x):
    """
    """
    return torch.digamma(theta+1) - torch.log(nu) - torch.log(x)


def compute_gamma_saturated_params(nu, x, eps=10**(-6), maxiter=100):
    min_val, max_val = GAMMA_ZERO_THRESHOLD, 10**10
    theta = torch.Tensor([(min_val + max_val) / 2])
    iter_idx = 0 
    
    while torch.abs(compute_constraint(theta, nu, x)) > eps:
        if compute_constraint(theta, nu, x) > 0:
            max_val = (min_val + max_val) / 2
        else:
            min_val = (min_val + max_val) / 2

        theta = torch.Tensor([(min_val + max_val) / 2])

        iter_idx += 1
        if iter_idx > maxiter:
            break

    return theta


def compute_gamma_saturated_params_gene(nu, X, eps=10**-6, maxiter=100):
    return [
        compute_gamma_saturated_params(nu, x, eps=eps, maxiter=maxiter) for x in X
    ]