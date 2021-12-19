import scipy
import numpy as np

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