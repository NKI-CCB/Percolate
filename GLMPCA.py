
import numpy as np
import pandas as pd
import torch
import torch.optim
import matplotlib.pyplot as plt
from copy import deepcopy
from joblib import Parallel, delayed
import mctorch.nn as mnn
import mctorch.optim as moptim
from torch.utils.data import Dataset, TensorDataset, DataLoader
from scipy.stats import beta as beta_dst

from .negative_binomial_routines import compute_dispersion
from .exponential_family import *


def _create_saturated_loading_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf, exp_family_params=None):
    loadings = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[1], n_pc))
    intercept = mnn.Parameter(
        data=torch.mean(parameters, axis=0),
        manifold=mnn.Euclidean(parameters.shape[1])
    )
    params = deepcopy(exp_family_params)

    # Load to GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if params is not None:
        params = {k:params[k].to(device) for k in params}

    if family.lower() in ['negative_binomial', 'nb']:
        params['r'] = params['r'][params['gene_filter']]
    cost = make_saturated_loading_cost(
        family=family,
        max_value=max_value, 
        params=params
    )
    # optimizer = moptim.ConjugateGradient(params = [loadings, intercept], lr=learning_rate)
    optimizer = moptim.rAdagrad(params = [loadings, intercept], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, loadings, intercept, lr_scheduler


def _create_saturated_scores_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf, exp_family_params=None):
    scores = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[0], n_pc))
    intercept = mnn.Parameter(
        data=torch.mean(parameters, axis=0),
        manifold=mnn.Euclidean(parameters.shape[1])
    )
    params = deepcopy(exp_family_params)
    if family.lower() in ['negative_binomial', 'nb']:
        params['r'] = params['r'][params['gene_filter']]
    cost = make_saturated_sample_proj_cost(family, parameters, data, max_value, params)
    # optimizer = moptim.ConjugateGradient(params = [scores, intercept], lr=learning_rate)
    optimizer = moptim.rAdagrad(params = [scores, intercept], lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.9)

    return optimizer, cost, scores, intercept, lr_scheduler


def _create_saturated_scores_projection_optim(parameters, data, n_pc, family, learning_rate, max_value=np.inf):
    scores = mnn.Parameter(manifold=mnn.Stiefel(parameters.shape[0], n_pc))
    cost = make_saturated_sample_proj_cost(family, parameters, data, max_value)
    # optimizer = moptim.ConjugateGradient(params=[scores], lr=learning_rate)
    optimizer = moptim.rAdagrad(params=[scores], lr=learning_rate)
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

        self.exp_family_params = None
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []

        # For NB
        if family.lower() in ['negative_binomial', 'nb']:
            print('SET DEFAULT PARAMETERS FOR NEGATIVE BINOMIAL', flush=True)
            self.nb_params = {
                'r_lr': 1000.,
                'theta_lr': 250.,
                'epochs': 1000
            }


    def compute_saturated_loadings(self, X, exp_family_params=None, batch_size=128, n_init=1):
        """
        Compute low-rank feature-level projection of saturated parameters.
        """
        if exp_family_params is not None:
            self.exp_family_params = exp_family_params
        self.saturated_param_ = self.compute_saturated_params(
            X, 
            with_intercept=False, 
            exp_family_params=self.exp_family_params, 
            save_family_params=True
        )

        self.learning_rate_ = self.initial_learning_rate_
        self.loadings_learning_scores_ = []
        self.loadings_learning_rates_ = []
        if n_init == 1:
            self.saturated_loadings_, self.saturated_intercept_ = self._saturated_loading_iter(
                self.saturated_param_, 
                X[:,self.exp_family_params['gene_filter']] if self.family.lower() in ['negative_binomial', 'nb'] else X,
                batch_size=batch_size
            )
        else:
            # Perform several initializations and select the top ones.
            init_results = [
                self._saturated_loading_iter(
                    self.saturated_param_, 
                    X[:,self.exp_family_params['gene_filter']] if self.family.lower() in ['negative_binomial', 'nb'] else X,
                    batch_size=batch_size,
                    return_train_likelihood=True
                ) for _ in range(n_init)
            ]
            self.iter_likelihood_results_ = [e[-1].detach().numpy() for e in init_results]
            self.optimal_iter_arg_ = np.argmin(self.iter_likelihood_results_)
            self.saturated_loadings_, self.saturated_intercept_ = init_results[self.optimal_iter_arg_][:2]


        self.saturated_intercept_ = self.saturated_intercept_.clone().detach()
        self.reconstruction_intercept_ = self.saturated_intercept_.clone().detach()
        self.saturated_param_ = self.saturated_param_ - self.saturated_intercept_
        self.sample_projection = False

        return self.saturated_loadings_


    def compute_saturated_orthogonal_scores(self, X, correct_loadings=False):
        """
        Compute low-rank sample-level orthogonal projection of saturated parameters.
        If correct_loadings, align loadings to have perfect match with scores
        """
        if self.saturated_loadings_ is None:
            self.compute_saturated_loadings(X)

        self.saturated_param_ = self.compute_saturated_params(
            X, 
            with_intercept=True, 
            exp_family_params=self.exp_family_params,
            save_family_params=False
        )

        projected_orthogonal_scores_ = self.saturated_param_.matmul(self.saturated_loadings_).matmul(self.saturated_loadings_.T)
        # projected_orthogonal_scores_ /= torch.linalg.norm(projected_orthogonal_scores_, axis=0)
        # self.saturated_scores_ = projected_orthogonal_scores_.detach().numpy()
        # self.sample_projection = True

        self.projected_orthogonal_scores_svd_ = torch.linalg.svd(projected_orthogonal_scores_, full_matrices=False)
        # Restrict to top components and return SVD in form U@S@V^T
        svd_results_ = []
        svd_results_.append(self.projected_orthogonal_scores_svd_[0][:,:self.n_pc])
        svd_results_.append(self.projected_orthogonal_scores_svd_[1][:self.n_pc])
        svd_results_.append(self.projected_orthogonal_scores_svd_[2].T[:,:self.n_pc])
        self.projected_orthogonal_scores_svd_ = svd_results_

        # Compute saturated scores by taking the left singular values
        self.saturated_scores_ = self.projected_orthogonal_scores_svd_[0]

        if correct_loadings:
            self.saturated_loadings_ = torch.matmul(self.saturated_loadings_, self.projected_orthogonal_scores_svd_[2].T)
            # self.saturated_loadings_weights_ = 1./self.projected_orthogonal_scores_svd_[1]
            # It is Sigma_A (or Sigma_B) in the derivation
            self.saturated_loadings_weights_ = self.projected_orthogonal_scores_svd_[1]
            self.saturated_loadings_ = torch.matmul(self.saturated_loadings_, torch.diag(1./self.projected_orthogonal_scores_svd_[1]))
            self.sample_projection = True

        return self.saturated_scores_


    def compute_reconstructed_data(self, X, scores):
        """
        Given some orthogonal scores, compute the expected data.
        """

        saturated_param_ = self.compute_saturated_params(
            X, with_intercept=False, exp_family_params=self.exp_family_params
        )

        # Compute associated cell view
        joint_saturated_param_ = deepcopy(saturated_param_.detach())
        if self.saturated_intercept_ is not None:
            joint_saturated_param_ = joint_saturated_param_ - self.saturated_intercept_
        joint_saturated_param_ = torch.matmul(scores, scores.T).matmul(joint_saturated_param_)
        if self.saturated_intercept_ is not None:
            joint_saturated_param_ = joint_saturated_param_ + self.reconstruction_intercept_

        params = deepcopy(self.exp_family_params)
        if self.family.lower() in ['negative_binomial', 'nb']:
            params['r'] = params['r'][params['gene_filter']]
        self.X_reconstruct_view_ = G_grad_fun(self.family)(joint_saturated_param_, params)

        return self.X_reconstruct_view_


    def compute_projected_saturated_params(self, X, with_reconstruction_intercept=True, exp_family_params=None):
        # Compute saturated params
        saturated_param_ = self.compute_saturated_params(
            X, 
            with_intercept=True, 
            exp_family_params=exp_family_params, 
            save_family_params=False
        )

        # Project on loadings
        saturated_param_ = torch.matmul(saturated_param_, self.saturated_loadings_)
        saturated_param_ = torch.matmul(saturated_param_, torch.linalg.pinv(self.saturated_loadings_))
        if with_reconstruction_intercept:
            saturated_param_ = saturated_param_ + self.reconstruction_intercept_

        # if self.family.lower() in ['negative_binomial', 'nb']:
        #     saturated_param_ = saturated_param_.clip(-np.inf,-1e-7)
        return saturated_param_.clone().detach()


    def compute_saturated_params(self, X, with_intercept=True, exp_family_params=None, save_family_params=False):
        if self.family.lower() in ['negative_binomial', 'nb']:
            # Load parameter if needed
            if exp_family_params is not None and 'r' in exp_family_params:
                r_coef = exp_family_params['r'].clone()
                gene_filter = exp_family_params['gene_filter'].clone()
            else:
                r_coef = torch.Tensor(compute_dispersion(pd.DataFrame(X.detach().numpy())).values).flatten()
                gene_filter = torch.where((r_coef > 0.1) & (~torch.isnan(r_coef)))[0]

            # Save parameters if required
            if save_family_params:
                if self.exp_family_params is None:
                    self.exp_family_params = {}
                self.exp_family_params['r'] = torch.Tensor(r_coef)
                self.exp_family_params['gene_filter'] = torch.where(r_coef > 0.1)[0]

            # Filter genes
            r_coef = r_coef[gene_filter]
            X_data = X[:,gene_filter]
            exp_family_params = {'r': r_coef, 'gene_filter': gene_filter}
            saturated_param_ = g_invertfun(self.family)(X_data, exp_family_params)
            # Reparametrization code
            # saturated_param_ = - torch.log(r_coef / X_data + 1)
            # saturated_param_ = saturated_param_.clip(-20)
            # saturated_param_ = - torch.log(torch.exp(-saturated_param_)-1)
            # saturated_param_ = torch.log(X_data).clip(-self.max_param, self.max_param)

        elif self.family.lower() in ['beta']:
            beta_parameters = [
                beta_dst.fit(X_feat, floc=0, fscale=1)[1]
                for X_feat in X.T
            ]
            
            if save_family_params:
                if self.exp_family_params is None:
                    self.exp_family_params = {}
                self.exp_family_params['beta'] = torch.Tensor(beta_parameters)
                self.exp_family_params['n_jobs'] = 10 #self.n_jobs

            exp_family_params = {'beta': torch.Tensor(beta_parameters), 'n_jobs': 20}
            saturated_param_ = g_invertfun(self.family)(X, exp_family_params)

        else:
            # Compute saturated params
            saturated_param_ = g_invertfun(self.family)(X, self.exp_family_params)
        saturated_param_ = torch.clip(saturated_param_, -self.max_param, self.max_param)

        # Project on loadings
        if with_intercept:
            saturated_param_ = saturated_param_ - self.saturated_intercept_

        return saturated_param_.clone().detach()


    def project_low_rank(self, X):
        saturated_params = self.compute_saturated_params(
            X, 
            with_intercept=True,
            exp_family_params=self.exp_family_params
        )
        return saturated_params.matmul(self.saturated_loadings_)


    def project_cell_view(self, X):
        projected_saturated_param_ = self.compute_projected_saturated_params(
            X, 
            with_reconstruction_intercept=True,
            exp_family_params=self.exp_family_params
        )

        params = deepcopy(self.exp_family_params)
        if self.family.lower() in ['negative_binomial', 'nb']:
            params['r'] = params['r'][params['gene_filter']]
        return G_grad_fun(self.family)(projected_saturated_param_, params)


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
        glmpca_clf.exp_family_params = self.exp_family_params

        return glmpca_clf


    def _saturated_loading_iter(self, saturated_param, data, batch_size=128, return_train_likelihood=False):
        """
        Computes the loadings, i.e. orthogonal low-rank projection, which maximise the likelihood of the data.
        """

        # Set device for GPU usage
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        _optimizer, _cost, _loadings, _intercept, _lr_scheduler = _create_saturated_loading_optim(
            saturated_param.data.clone(),
            data,
            self.n_pc,
            self.family,
            self.learning_rate_,
            self.max_param,
            self.exp_family_params
        )

        _loadings = _loadings.to(device)
        _intercept = _intercept.to(device)
        self.loadings_elements_optim_ = [_optimizer, _cost, _loadings, _intercept, _lr_scheduler]
        
        self.loadings_learning_scores_.append([])
        self.loadings_learning_rates_.append([])
        previous_loadings = deepcopy(_loadings)
        previous_intercept = deepcopy(_intercept)

        data = data.to(device)
        saturated_param = saturated_param.to(device)
        train_data = TensorDataset(data, saturated_param.data.clone())
        train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
        train_loader = train_loader.to(device)

        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('\tSTART ITER %s'%(idx))
            loss_val = []
            for data_batch, param_batch in train_loader:
                cost_step = _cost(
                    X=_loadings, 
                    data=data_batch, 
                    parameters=param_batch, 
                    intercept=_intercept
                )
                # loss_val.append(cost_step.detach().numpy())
                self.loadings_learning_scores_[-1].append(cost_step.detach().numpy())
                cost_step.backward()
                _optimizer.step()
                _optimizer.zero_grad()
                self.loadings_learning_rates_[-1].append(_lr_scheduler.get_last_lr())
            _lr_scheduler.step()

            if np.isinf(self.loadings_learning_scores_[-1][-1]) or np.isnan(self.loadings_learning_scores_[-1][-1]):
                print('\tRESTART BECAUSE INF/NAN FOUND', flush=True)
                self.learning_rate_ = self.learning_rate_ * 0.9
                self.loadings_learning_scores_ = self.loadings_learning_scores_[:-1]
                self.loadings_learning_rates_ = self.loadings_learning_rates_[:-1]
                return self._saturated_loading_iter(
                    saturated_param=saturated_param,
                    data=data, 
                    batch_size=batch_size, 
                    return_train_likelihood=return_train_likelihood
                )

        print('\tEND OPTIMISATION\n')

        if return_train_likelihood:
            params = deepcopy(self.exp_family_params)
            if params is not None and self.family.lower() in ['negative_binomial', 'nb']:
                params['r'] = params['r'][self.exp_family_params['gene_filter']]
            _proj_params = saturated_param - _intercept
            _proj_params = _proj_params.matmul(_loadings).matmul(_loadings.T)
            _proj_params = _proj_params + _intercept
            _likelihood = torch.mean(natural_parameter_log_likelihood(
                self.family, 
                data, 
                _proj_params, 
                params=params
            ))

            return _loadings, _intercept, _likelihood

        # Reinitialize learning rate
        self.learning_rate_ = self.initial_learning_rate_
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
            self.max_param,
            self.exp_family_params
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
        
        # self.subrotation_learning_scores_ = []
        # self.subrotation_learning_rates_ = []
        self.subrotation_learning_scores_.append([])
        self.subrotation_learning_rates_.append([])
        for idx in range(self.maxiter):
            if idx % 100 == 0:
                print('START ITER %s'%(idx))
            cost_step = _cost(_subrotation, _intercept)
            self.subrotation_learning_scores_[-1].append(cost_step.detach().numpy())
            cost_step.backward()
            _optimizer.step()
            _optimizer.zero_grad()
            self.subrotation_learning_rates_[-1].append(_lr_scheduler.get_last_lr())
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