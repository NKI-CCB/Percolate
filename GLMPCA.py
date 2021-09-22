


import numpy as np
import torch
import torch.optim
from copy import deepcopy
from joblib import Parallel, delayed
from .exponential_family import *
import mctorch.nn as mnn
import mctorch.optim as moptim


def _create_saturated_loading_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf):
    loadings = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[1], n_pc))
    intercept = mnn.Parameter(
        data=torch.mean(parameters, axis=0),
        manifold=mnn.Euclidean(parameters.shape[1])
    )
    cost = make_saturated_loading_cost(family, parameters, data, max_value)
    optimizer = moptim.rSGD(params = [loadings, intercept], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, loadings, intercept, lr_scheduler


def _create_saturated_scores_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf):
    scores = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[0], n_pc))
    intercept = mnn.Parameter(
        data=torch.mean(parameters, axis=0),
        manifold=mnn.Euclidean(parameters.shape[1])
    )
    cost = make_saturated_sample_proj_cost(family, parameters, data, max_value)
    optimizer = moptim.rSGD(params = [scores, intercept], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, scores, intercept, lr_scheduler


def _create_saturated_scores_projection_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf):
    scores = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[0], n_pc))
    cost = make_saturated_sample_proj_cost(family, parameters, data, max_value)
    optimizer = moptim.rSGD(params=[scores], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, scores, lr_scheduler


def _create_subrotation_loading_optim(parameters, data, loadings, initial_intercept, n_pc, family, learning_rate, max_value=np.inf):
    subrotation = mnn.Parameter(manifold=mnn.Stiefel(loadings.shape[1], n_pc))
    intercept = mnn.Parameter(
        data=torch.mean(parameters, axis=0),
        manifold=mnn.Euclidean(parameters.shape[1])
    )
    cost = make_saturated_subrotation_loading_cost(family, parameters, data, loadings, initial_intercept, max_value=max_value)
    optimizer = moptim.rSGD(params=[subrotation, intercept], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, subrotation, intercept, lr_scheduler


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
        self.initial_learning_rate_ = learning_rate

        self.saturated_loadings_ = None
        # saturated_intercept_: before projecting
        self.saturated_intercept_ = None
        # reconstruction_intercept: after projecting
        self.reconstruction_intercept_ = None

        # Whether to perform sample or gene projection
        self.sample_projection = False


    def compute_saturated_loadings(self, X):
        """
        Compute low-rank feature-level projection of saturated parameters.
        """
        self.saturated_param_ = g_invertfun(self.family)(X)
        self.saturated_param_ = torch.clip(self.saturated_param_, -self.max_param, self.max_param)

        self.learning_rate_ = self.initial_learning_rate_
        self.saturated_loadings_, self.saturated_intercept_ = self._saturated_loading_iter(self.saturated_param_, X)
        self.saturated_intercept_ = self.saturated_intercept_.clone().detach()
        self.reconstruction_intercept_ = self.saturated_intercept_.clone().detach()
        self.saturated_param_ = self.saturated_param_ - self.saturated_intercept_
        self.sample_projection = False

        return self.saturated_loadings_


    def compute_saturated_orthogonal_scores(self, X, correct_loadings=True):
        """
        Compute low-rank sample-level orthogonal projection of saturated parameters.
        If correct_loadings, align loadings to have perfect match with scores
        """
        if self.saturated_loadings_ is None:
            self.compute_saturated_loadings(X)

        self.saturated_param_ = g_invertfun(self.family)(X)
        self.saturated_param_ = torch.clip(self.saturated_param_, -self.max_param, self.max_param)

        projected_saturated_param_ = self.saturated_param_ - self.saturated_intercept_
        projected_orthogonal_scores_ = projected_saturated_param_.matmul(self.saturated_loadings_)
        projected_orthogonal_scores_ = torch.linalg.svd(projected_orthogonal_scores_, full_matrices=False)
        self.saturated_scores_ = projected_orthogonal_scores_[0]
        if correct_loadings:
            self.saturated_loadings_ = torch.matmul(self.saturated_loadings_, projected_orthogonal_scores_[2].T)
            self.saturated_loadings_ = torch.matmul(self.saturated_loadings_, torch.diag(1./projected_orthogonal_scores_[1]))
            self.sample_projection = True

        return self.saturated_scores_


    def compute_reconstructed_data(self, X, scores):
        """
        Given some orthogonal scores, compute the expected data.
        """

        # Saturated params
        saturated_param_ = g_invertfun(self.family)(X)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)

        # Compute associated cell view
        joint_saturated_param_ = deepcopy(saturated_param_.detach())
        if self.saturated_intercept_ is not None:
            joint_saturated_param_ = joint_saturated_param_ - self.saturated_intercept_
        joint_saturated_param_ = torch.matmul(scores, scores.T).matmul(joint_saturated_param_)
        if self.saturated_intercept_ is not None:
            joint_saturated_param_ = joint_saturated_param_ + self.saturated_intercept_
        self.X_reconstruct_view_ = G_grad_fun(self.family)(joint_saturated_param_)

        return self.X_reconstruct_view_


    def compute_projected_saturated_params(self, X, with_reconstruction_intercept=True):
        # Compute saturated params
        saturated_param_ = g_invertfun(self.family)(X)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)

        # Project on loadings
        saturated_param_ = saturated_param_ - self.saturated_intercept_
        saturated_param_ = torch.matmul(saturated_param_, self.saturated_loadings_)
        saturated_param_ = torch.matmul(saturated_param_, torch.linalg.pinv(self.saturated_loadings_))
        if with_reconstruction_intercept:
            saturated_param_ = saturated_param_ + self.reconstruction_intercept_

        return saturated_param_.clone().detach()


    def compute_saturated_params(self, X):
        # Compute saturated params
        saturated_param_ = g_invertfun(self.family)(X)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)

        # Project on loadings
        saturated_param_ = saturated_param_ - self.saturated_intercept_

        return saturated_param_.clone().detach()


    def project_low_rank(self, X):
        saturated_params = self.compute_saturated_params(X)
        return saturated_params.matmul(self.saturated_loadings_)


    def project_cell_view(self, X):
        # if self.sample_projection:
        #     saturated_params = self.compute_saturated_params(X)
        #     saturated_param_projection_ = saturated_params.matmul(self.saturated_loadings_)
        #     saturated_param_projection_ = saturated_params.matmul(self.saturated_loadings_)
        #     saturated_param_projection_, _, _ = torch.linalg.svd(saturated_param_projection_)
        #     projected_saturated_param_ = saturated_param_projection_.matmul(saturated_param_projection_.T).matmul(saturated_params)
        #     projected_saturated_param_ = projected_saturated_param_ + self.reconstruction_intercept_
        # else:
        projected_saturated_param_ = self.compute_projected_saturated_params(X, with_reconstruction_intercept=True)
        
        return G_grad_fun(self.family)(projected_saturated_param_)


    def clone_empty_GLMPCA(self):
        glmpca_clf = GLMPCA( 
            self.n_pc, 
            self.family, 
            maxiter=self.maxiter, 
            max_param=self.max_param,
            learning_rate=self.learning_rate_
        )
        glmpca_clf.saturated_intercept_ = self.saturated_intercept_.clone().detach()
        glmpca_clf.reconstruction_intercept_ = self.reconstruction_intercept_.clone().detach()

        return glmpca_clf


    def _saturated_loading_iter(self, saturated_param, data):
        """
        Computes the loadings, i.e. orthogonal low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _loadings, _intercept, _lr_scheduler = _create_saturated_loading_optim(
            saturated_param.data.clone(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_,
            self.max_param
        )
        
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []
        previous_loadings = deepcopy(_loadings)
        previous_intercept = deepcopy(_intercept)
        for idx in range(self.maxiter):
            a = deepcopy(_loadings)
            b = deepcopy(_intercept)
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_loadings, _intercept)
            self.loadings_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.loadings_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.loadings_learning_scores_[-1]) or np.isnan(self.loadings_learning_scores_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 1.5
                return self._saturated_loading_iter(saturated_param, data)

            previous_loadings = a
            previous_intercept = b

        return _loadings, _intercept


    def _saturated_score_iter(self, saturated_param, data):
        """
        Computes the orthogonal scores, i.e. orthogonal sanple low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _scores, _intercept, _lr_scheduler = _create_saturated_scores_optim(
            saturated_param.data.clone(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_,
            self.max_param
        )
        
        self.scores_learning_scores_ = []
        self.scores_learning_rates_ = []
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_scores, _intercept, _intercept)
            self.scores_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.scores_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.scores_learning_scores_[-1]) or np.isnan(self.scores_learning_scores_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 1.5
                return self._saturated_score_iter(saturated_param, data)

        return _scores, _intercept


    def _saturated_score_projection_iter(self, saturated_param, data):
        """
        Computes the orthogonal scores, i.e. orthogonal sanple low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _scores, _lr_scheduler = _create_saturated_scores_projection_optim(
            saturated_param.data.clone().detach(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_,
            self.max_param
        )
        
        scores_learning_scores_ = []
        scores_learning_rates_ = []
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_scores, self.saturated_intercept_, self.reconstruction_intercept_)
            scores_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            scores_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(scores_learning_scores_[-1]) or np.isnan(scores_learning_scores_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 1.5
                return self._saturated_score_projection_iter(saturated_param, data)

        return _scores


    def _saturated_subrotation_iter(self, saturated_param, data, loadings, initial_intercept):
        """
        Computes the loadings, i.e. orthogonal low-rank projection, which maximise the likelihood of the data.
        """
        _optimizer, _cost, _subrotation, _intercept, _lr_scheduler = _create_subrotation_loading_optim(
            saturated_param.data.clone(),
            data,
            loadings,
            initial_intercept,
            self.n_pc,
            self.family,
            self.learning_rate_,
            self.max_param
        )
        
        self.subrotation_learning_scores_ = []
        self.subrotation_learning_rates_ = []
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_subrotation, _intercept)
            self.subrotation_learning_scores_.append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.subrotation_learning_rates_.append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.subrotation_learning_scores_[-1]) or np.isnan(self.subrotation_learning_rates_[-1]):
                print('RESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ / 1.5
                return self._saturated_subrotation_iter(saturated_param, data, loadings, initial_intercept)

        _loadings = torch.matmul(loadings, _subrotation)

        # Computation of second intercept
        _reconstruction_intercept = g_invertfun(self.family)(data)
        _reconstruction_intercept = torch.clip(_reconstruction_intercept, -self.max_param, self.max_param)
        _reconstruction_intercept = _reconstruction_intercept - (saturated_param - _intercept).matmul(_loadings).matmul(_loadings.T)
        _reconstruction_intercept = torch.mean(_reconstruction_intercept, axis=0)

        return _loadings, _intercept, _reconstruction_intercept