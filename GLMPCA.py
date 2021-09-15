


import numpy as np
import torch
import torch.optim
from copy import deepcopy
from joblib import Parallel, delayed
from .exponential_family import *
import mctorch.nn as mnn
import mctorch.optim as moptim


def _create_saturated_loading_optim(parameters, data, n_pc, family, learning_rate, intercept=None):
    loadings = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[1], n_pc))
    cost = make_saturated_loading_cost(family, parameters, data, intercept)
    optimizer = moptim.rAdagrad(params = [loadings], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, loadings, lr_scheduler


def _create_saturated_scores_optim(parameters, data, n_pc, family, learning_rate, intercept=None):
    scores = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[0], n_pc))
    cost = make_saturated_sample_proj_cost(family, parameters, data, intercept)
    optimizer = moptim.rAdagrad(params = [scores], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, scores, lr_scheduler


class GLMPCA:

    def __init__(
        self, 
        n_pc, 
        family, 
        maxiter=1000, 
        max_param = 10,
        learning_rate = 0.02
        ):

        self.n_pc = n_pc
        self.family = family
        self.maxiter = maxiter
        self.log_part_theta_matrices_ = None
        self.max_param = np.abs(max_param)
        self.learning_rate_ = learning_rate


    def compute_saturated_loadings(self, X):
        """
        Compute low-rank feature-level projection of saturated parameters.
        """
        self.saturated_param_ = g_invertfun(self.family)(X)
        self.saturated_param_ = torch.clip(self.saturated_param_, -self.max_param, self.max_param)

        self.saturated_loadings_ = self._saturated_loading_iter(self.saturated_param_, X, None)

        return self.saturated_loadings_


    def compute_saturated_orthogonal_scores(self, X):
        """
        Compute low-rank sample-level orthogonal projection of saturated parameters.
        """
        self.saturated_param_ = g_invertfun(self.family)(X)
        self.saturated_param_ = torch.clip(self.saturated_param_, -self.max_param, self.max_param)

        self.saturated_scores_ = self._saturated_score_iter(self.saturated_param_, X, None)

        return self.saturated_scores_


    def compute_equivalent_loadings(self, X, scores, loadings=None):
        """
        Given some orthogonal scores, compute low-rankfeature-level orthogonal projection.
        Additional loadings can be added to constrain the projection to be on these directions (saturated
        parameters are first projected on them).
        """
        saturated_param_ = g_invertfun(self.family)(X)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)
        saturated_param_ = torch.matmul(scores, scores.T).matmul(saturated_param_)
        if loadings is not None:
            saturated_param_ = torch.matmul(saturated_param_, loadings).matmul(loadings.T)

        self.saturated_loadings_ = self._saturated_loading_iter(saturated_param_, X, None)

        return self.saturated_loadings_


    def compute_projected_saturated_params(self, X):
        # Compute saturated params
        saturated_param_ = g_invertfun(self.family)(X)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)

        # Project on loadings
        saturated_param_ = torch.matmul(saturated_param_, self.saturated_loadings_)
        saturated_param_ = torch.matmul(saturated_param_, self.saturated_loadings_.T)

        return saturated_param_


    def project_low_rank(self, X):
        projected_saturated_param_ = self.compute_projected_saturated_params(X)
        return self._saturated_score_iter(projected_saturated_param_, X, intercept=None)


    def project_cell_view(self, X):
        projected_saturated_param_ = self.compute_projected_saturated_params(X)
        return G_grad_fun(self.family)(projected_saturated_param_)


    def clone_empty_GLMPCA(self):
        return GLMPCA( 
            self.n_pc, 
            self.family, 
            maxiter=self.maxiter, 
            max_param=self.max_param,
            learning_rate=self.learning_rate_
        )


    def _saturated_loading_iter(self, saturated_param, data, intercept=None):
        """
        Computes the loadings, i.e. orthogonal low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _loadings, _lr_scheduler = _create_saturated_loading_optim(
            saturated_param.data.clone(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_, 
            deepcopy(intercept)
        )
        
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_loadings)
            self.loadings_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.loadings_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.loadings_learning_scores_[-1]) or np.isnan(self.loadings_learning_scores_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 2
                return self._saturated_loading_iter(saturated_param, data, intercept=intercept)

        return _loadings


    def _saturated_score_iter(self, saturated_param, data, intercept=None):
        """
        Computes the orthogonal scores, i.e. orthogonal sanple low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _scores, _lr_scheduler = _create_saturated_scores_optim(
            saturated_param.data.clone(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_, 
            deepcopy(intercept)
        )
        
        self.scores_learning_scores_ = []
        self.scores_learning_rates_ = []
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_scores)
            self.scores_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.scores_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.scores_learning_scores_[-1]) or np.isnan(self.scores_learning_scores_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 2
                return self._saturated_score_iter(saturated_param, data, intercept=intercept)

        return _scores
