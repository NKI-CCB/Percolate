import torch
import numpy as np

GAMMA_ZERO_THRESHOLD = 1e-5

def compute_constraint(theta, nu, x):
    """
    """
    return torch.digamma(theta+1) - torch.log(nu) - torch.log(x)


def compute_gamma_saturated_params(nu, x, eps=10**(-6), maxiter=100):
    min_val, max_val = GAMMA_ZERO_THRESHOLD, 10**10
    theta = torch.Tensor([(min_val + max_val) / 2])
    iter_idx = 0 
    
    if compute_constraint(torch.Tensor([min_val]), nu, x) > 0:
        return GAMMA_ZERO_THRESHOLD

    while torch.abs(compute_constraint(theta, nu, x)) > eps:
        if compute_constraint(theta, nu, x) > 0:
            max_val = (min_val + max_val) / 2
        else:
            min_val = (min_val + max_val) / 2

        theta = torch.Tensor([(min_val + max_val) / 2])

        iter_idx += 1
        if iter_idx > maxiter:
            print('CONVERGENCE NOT REACHED FOR GAMMA. LAST VALUE: %1.3f FOR NU=%1.3f AND X=%1.3f YIELDING THETA=%1.3f'%(
                compute_constraint(theta, nu, x), nu, x, theta
            ))
            break

    return theta


def compute_gamma_saturated_params_gene(nu, X, eps=10**-6, maxiter=100):
    return [
        compute_gamma_saturated_params(nu, x, eps=eps, maxiter=maxiter) for x in X.clip(GAMMA_ZERO_THRESHOLD)
    ]