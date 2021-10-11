""" Exponential family using tensor structure

@author: Soufiane Mourragui <soufiane.mourragui@gmail.com>

Codes supporting routines involving the Exponential Family, e.g. Maximum Likelihood
Estimation (MLE) in glm-PCA.

We assume that MLE writes like:
'''
\sum_{i=1}^n G(\theta_{i,j}) - \Theta^T T(x)

for j \in \{1, .., p\} feature index. T(x) is xx^T for most applications we use here.
'''

Example
-------

    
Notes
-------


References
-------


"""

import torch
from copy import deepcopy
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler

saturation_eps = 10**-10

# Compute natural parameters from the parametrization used.
def nu_gaussian(x, params=None):
    return x

def nu_bernoulli(x, params=None):
    return x

def nu_continuous_bernoulli(x, params=None):
    return x

def nu_poisson(x, params=None):
    return x

def nu_multinomial(x, params=None):
    return x

def nu_negative_binomial(x, params=None):
    r = params['r']
    return 1 / (1 + r*torch.exp(-x))

def nu_fun(family):
    if family == 'bernoulli':
        return nu_bernoulli
    elif family == 'continuous_bernoulli':
        return nu_continuous_bernoulli
    elif family == 'poisson':
        return nu_poisson
    elif family == 'gaussian':
        return nu_gaussian
    elif family == 'multinomial':
        return nu_multinomial
    elif family.lower() in ['negative_binomial', 'nb']:
        return nu_negative_binomial


# Function that computes the parametrization of the natural parameters.
def nu_parametrization_gaussian(x, params=None):
    return x

def nu_parametrization_bernoulli(x, params=None):
    return x

def nu_parametrization_continuous_bernoulli(x, params=None):
    return x

def nu_parametrization_poisson(x, params=None):
    return x

def nu_parametrization_multinomial(x, params=None):
    return x

def nu_parametrization_negative_binomial(x, params=None):
    return - torch.log(x/(1-x))

def nu_parametrization_fun(family):
    if family == 'bernoulli':
        return nu_parametrization_bernoulli
    elif family == 'continuous_bernoulli':
        return nu_parametrization_continuous_bernoulli
    elif family == 'poisson':
        return nu_parametrization_poisson
    elif family == 'gaussian':
        return nu_parametrization_gaussian
    elif family == 'multinomial':
        return nu_parametrization_multinomial
    elif family.lower() in ['negative_binomial', 'nb']:
        return nu_parametrization_negative_binomial

# Functions G
def G_gaussian(x, params=None):
    return torch.square(x) / 2

def G_bernoulli(x, params=None):
    return torch.log(1+torch.exp(x))

def G_continuous_bernoulli(x, params=None):
    return torch.log((torch.exp(x)-1)/x)

def G_poisson(x, params=None):
    return torch.exp(x)

def G_multinomial(x, params=None):
    return 0

def G_negative_binomial(x, params=None):
    r = params['r']
    return - r * torch.log(1-torch.exp(x.clip(-np.inf,-1e-7)))


def G_fun(family):
    if family == 'bernoulli':
        return G_bernoulli
    elif family == 'continuous_bernoulli':
        return G_continuous_bernoulli
    elif family == 'poisson':
        return G_poisson
    elif family == 'gaussian':
        return G_gaussian
    elif family == 'multinomial':
        return G_multinomial
    elif family.lower() in ['negative_binomial', 'nb']:
        return G_negative_binomial

# Functions G
def G_grad_gaussian(x, params=None):
    return x

def G_grad_bernoulli(x, params=None):
    # return torch.nn.functional.logsigmoid(x)
    return 1./(1.+torch.exp(-x))

def G_grad_continuous_bernoulli(x, params=None):
    return (torch.exp(-x)+x-1)/(x * (1-torch.exp(-x)))

def G_grad_poisson(x, params=None):
    return torch.exp(x)

def G_grad_multinomial(x, params=None):
    return 0

def G_grad_negative_binomial(x, params=None):
    r = params['r']
    return - r / (1-torch.exp(-x))

def G_grad_fun(family):
    if family == 'bernoulli':
        return G_grad_bernoulli
    elif family == 'continuous_bernoulli':
        return G_grad_continuous_bernoulli
    elif family == 'poisson':
        return G_grad_poisson
    elif family == 'gaussian':
        return G_grad_gaussian
    elif family == 'multinomial':
        raise NotImplementedError('multinomial not implemented')
    elif family.lower() in ['negative_binomial', 'nb']:
        return G_grad_negative_binomial


# g_invert is the inverse of the derivative of A.
def g_invert_bernoulli(x, params=None):
    y = deepcopy(x)
    y = torch.log(y/(1-y))
    # y[y==0] = - np.inf
    # y[y==1] = np.inf
    return y
    # return - np_ag.log((1-x)/(x+saturation_eps))

def g_invert_continuous_bernoulli(x, params=None):
    y = deepcopy(x)
    y[y==0] = - np.inf
    y[y==1] = np.inf
    return y

def g_invert_gaussian(x, params=None):
    return x

def g_invert_poisson(x, params=None):
    y = deepcopy(x)
    y[y==0] = - np.inf
    y[y>0] = torch.log(y[y>0])
    return y

def g_invert_negative_binomial(x, params=None):
    r = params['r']
    return torch.log(x/(x+r))

def g_invertfun(family):
    if family == 'bernoulli':
        return g_invert_bernoulli
    if family == 'continuous_bernoulli':
        return g_invert_continuous_bernoulli
    elif family == 'poisson':
        return g_invert_poisson
    elif family == 'gaussian':
        return g_invert_gaussian
    elif family == 'multinomial':
        raise NotImplementedError('multinomial not implemented')
    elif family.lower() in ['negative_binomial', 'nb']:
        return g_invert_negative_binomial


# Functions h
def h_gaussian(x, params=None):
    return np.power(2*np.pi, -1)

def h_bernoulli(x, params=None):
    # if type(x) == int or type(x) == float:
    #     return torch(1.)
    # else:
    #     return torch.ones(x.shape)
    return 1.

def h_continuous_bernoulli(x, params=None):
    # if type(x) == int or type(x) == float:
    #     return torch(1.)
    # else:
    #     return torch.ones(x.shape)
    return 1.

def h_poisson(x, params=None):
    return torch.Tensor(scipy.special.gamma(x+1))

def h_multinomial(x, params=None):
    pass

def h_negative_binomial(x, params=None):
    r = params['r']
    return torch.Tensor(scipy.special.binom(x+r-1, x))

def h_fun(family):
    if family == 'bernoulli':
        return h_bernoulli
    elif family == 'continuous_bernoulli':
        return h_continuous_bernoulli
    elif family == 'poisson':
        return h_poisson
    elif family == 'gaussian':
        return h_gaussian
    elif family == 'multinomial':
        raise NotImplementedError('multinomial not implemented')
    elif family.lower() in ['negative_binomial', 'nb']:
        return h_negative_binomial

def log_h_negative_binomial(x, params=None):
    r = params['r']

    return torch.lgamma(x+r) - torch.lgamma(x+1) - torch.lgamma(r)

def log_h_fun(family):
    if family == 'bernoulli':
        return NotImplementedError('bernoulli not implemented')
    elif family == 'continuous_bernoulli':
        return NotImplementedError('cont bernoulli not implemented')
    elif family == 'poisson':
        return NotImplementedError('poisson not implemented')
    elif family == 'gaussian':
        return NotImplementedError('multinomial not implemented')
    elif family == 'multinomial':
        raise NotImplementedError('multinomial not implemented')
    elif family.lower() in ['negative_binomial', 'nb']:
        return log_h_negative_binomial


# Compute likelihood
def likelihood(family, data, theta, params=None):
    h = h_fun(family)(data, params)
    nu = nu_fun(family)(data, params)
    exp_term = torch.multiply(nu, data)
    partition =  G_fun(family)(nu, params)
    return h * torch.exp(exp_term - partition)


def log_likelihood(family, data, theta, params=None):
    h = log_h_fun(family)(data, params)
    nu = nu_fun(family)(theta, params)
    exp_term = torch.multiply(nu, data)
    partition =  G_fun(family)(nu, params)
    return - h - exp_term + partition

def natural_parameter_log_likelihood(family, data, theta, params=None):
    nu = nu_fun(family)(theta, params)
    exp_term = torch.multiply(nu, data)
    partition =  G_fun(family)(nu, params)
    return - exp_term + partition

def make_saturated_loading_cost(family, parameters, data, max_value=np.inf, params=None):
    """
    Constructs the likelihood function for a given family.
    """
    loss = G_fun(family)
    nu_mapping = nu_fun(family)
    
    def likelihood(X, intercept=None):
        intercept = intercept if intercept is not None else torch.zeros(parameters.shape[1])
        nu = torch.matmul(parameters - intercept, torch.matmul(X, X.T)) + intercept
        nu = nu_mapping(nu, params)
        c = loss(nu, params)
        c = torch.sum(c)
        d = torch.sum(torch.multiply(data, nu))
        return c - d
    
    return likelihood


def make_saturated_sample_proj_cost(family, parameters, data, max_value=np.inf, params=None):
    loss = G_fun(family)
    nu_mapping = nu_mapping(family)
    
    def likelihood(X, saturated_intercept=None, reconstruction_intercept=None):
        saturated_intercept = saturated_intercept if saturated_intercept is not None else torch.zeros(parameters.shape[1])
        reconstruction_intercept = reconstruction_intercept if reconstruction_intercept is not None else torch.zeros(parameters.shape[1])

        nu = torch.matmul(torch.matmul(X, X.T), parameters - saturated_intercept) + reconstruction_intercept
        nu = nu_mapping(nu, params)
        c = loss(nu, params)
        c = torch.sum(c)
        d = torch.sum(torch.multiply(data, nu))
        return c - d
    
    return likelihood


def make_saturated_subrotation_loading_cost(family, parameters, data, loadings, initial_intercept, max_value=np.inf, params=None):
    """
    Constructs the likelihood function for a given family.
    """
    loss = G_fun(family)
    initial_intercept = initial_intercept if initial_intercept is not None else torch.zeros(parameters.shape[1])

    def likelihood(X, intercept=None):
        intercept = intercept if intercept is not None else torch.zeros(parameters.shape[1])
        # theta = torch.matmul(parameters - intercept, torch.matmul(X, X.T)) + intercept

        theta = parameters - initial_intercept
        theta = torch.matmul(theta, loadings)
        theta = torch.matmul(theta, X)
        theta = torch.matmul(theta, X.T)
        theta = torch.matmul(theta, loadings.T) + intercept

        c = loss(theta, params)
        c = torch.sum(c)
        d = torch.sum(torch.multiply(data, theta))
        return c - d
    
    return likelihood