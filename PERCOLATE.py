import torch, os
import numpy as np
from copy import deepcopy
from joblib import Parallel, delayed
from pickle import load, dump
from .GLMPCA import GLMPCA
from .GLMJIVE import GLMJIVE

class PERCOLATE:

    def __init__(
        self, 
        predictive_key,
        robust_types_order,
        n_factors,
        families, 
        maxiter=1000,
        max_param=None,
        learning_rates=None,
        batch_size=None, 
        n_glmpca_init=None,
        n_jobs=1,
        joint_estimation_n_iter=5
        ):

        self.predictive_key = predictive_key
        self.robust_types_order = robust_types_order
        self.n_factors = n_factors
        self.families = families
        if type(maxiter) is int:
            self.maxiter = {k: maxiter for k in n_factors}
        else:
            self.maxiter = maxiter
        if type(max_param) is int:
            self.max_param = {k: max_param for k in n_factors}
        else:
            self.max_param = max_param if max_param is not None else {k:None for k in n_factors}
        self.learning_rates = learning_rates if learning_rates is not None else {k:0.01 for k in n_factors}
        self.batch_size = batch_size if batch_size is not None else {k:128 for k in n_factors}
        self.n_glmpca_init = n_glmpca_init if n_glmpca_init is not None else {k:1 for k in n_factors}
        self.joint_estimation_n_iter = joint_estimation_n_iter

        # For parallelization
        self.n_jobs = n_jobs


    def fit(self, data_df, exp_family_params=None):

        exp_family_params = exp_family_params if exp_family_params is not None else {dt:{} for dt in data_df}

        # Train individual GLMPCA instances
        print('START TRAINING GLMPCA INSTANCES', flush=True)
        print('\tSTART PREDICTIVE', flush=True)
        self.predictive_glmpca_clf = GLMPCA(**self._glmpca_parameters(self.predictive_key))
        self.predictive_glmpca_clf.compute_saturated_loadings(
            data_df[self.predictive_key], 
            exp_family_params=exp_family_params[self.predictive_key]
        )
        self.robust_glmpca_clf = {}
        for robust_key in self.robust_types_order:
            print('\tSTART %s'%(robust_key), flush=True)
            self.robust_glmpca_clf[robust_key] = GLMPCA(**self._glmpca_parameters(robust_key))
            self.robust_glmpca_clf[robust_key].compute_saturated_loadings(
                data_df[robust_key],
                exp_family_params=exp_family_params[robust_key]
            )


        # Compute orthogonal scores
        print('START COMPUTING ORTHOGONAL SCORES', flush=True)
        self.robust_orthogonal_scores = {}
        for robust_key in self.robust_glmpca_clf:
            print('START %s'%(robust_key))
            self.robust_orthogonal_scores[robust_key] = self.robust_glmpca_clf[robust_key].compute_saturated_orthogonal_scores(
                correct_loadings=False
            )
        self.predictive_orthogonal_scores = self.predictive_glmpca_clf.compute_saturated_orthogonal_scores(correct_loadings=False)

        # Align iteratively
        print('START PERCOLATION', flush=True)
        self.percolate_iter_clfs = {}
        self.predictive_scores_residuals_ = [deepcopy(self.predictive_orthogonal_scores.detach())]

        for iter_idx, robust_key in enumerate(self.robust_types_order):
            print('\tSTART %s'%(robust_key))
            self.percolate_iter_clfs[robust_key], residual_ge = self._iter_percolate(iter_idx, self.predictive_scores_residuals_[iter_idx], data_df)
            self.predictive_scores_residuals_.append(residual_ge)
            self.percolate_iter_clfs[robust_key].set_out_of_sample_extension(robust_key)
            
        return self


    def project_robust(self, data_df):
        X_predict_joint = [
            self.percolate_iter_clfs[dt].compute_joint_signal(data_df[dt])
            for dt in self.percolate_iter_clfs
        ]

        return torch.cat(X_predict_joint, axis=1)


    def _iter_percolate(self, iter_idx, predictive_scores, data_df):
        """
        One percolation iteration
        """
        # Set up GPU device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        percolate_iter_clf = GLMJIVE(
            n_factors={
                self.robust_types_order[iter_idx]: self.n_factors[self.robust_types_order[iter_idx]], 
                self.predictive_key: self.n_factors[self.predictive_key]
            },
            n_joint=-1,
            families={
                self.robust_types_order[iter_idx]: self.families[self.robust_types_order[iter_idx]], 
                self.predictive_key: self.families[self.predictive_key]
        })
        
        # Integrate into GLMJIVE framework
        percolate_iter_clf.factor_models = {
            self.robust_types_order[iter_idx]: self.robust_glmpca_clf[self.robust_types_order[iter_idx]], 
            self.predictive_key: self.predictive_glmpca_clf
        }

        # Perform alignment using the structure of GLMJIVE code
        percolate_iter_clf.orthogonal_scores = [
            self.robust_orthogonal_scores[self.robust_types_order[iter_idx]].to(device),
            predictive_scores.to(device)
        ]
        percolate_iter_clf.data_types = [self.robust_types_order[iter_idx], self.predictive_key]
        percolate_iter_clf._aggregate_scores()
        percolate_iter_clf._initialize_models({
            self.robust_types_order[iter_idx]: data_df[self.robust_types_order[iter_idx]],
            self.predictive_key: data_df[self.predictive_key]
        })
        percolate_iter_clf.n_joint = percolate_iter_clf.estimate_number_joint_components_random_matrix(
            n_iter=self.joint_estimation_n_iter, n_jobs=self.n_jobs, quantile_top_component=0.95
        )
        percolate_iter_clf._computation_joint_individual_factor_model(not_aligned_types=self.predictive_key)

        indiv_proj = torch.eye(percolate_iter_clf.joint_scores_.shape[0])
        indiv_proj -= percolate_iter_clf.joint_scores_.matmul(percolate_iter_clf.joint_scores_.T)
        remainder_predictive = indiv_proj.matmul(predictive_scores)
        remainder_predictive_scores = torch.linalg.svd(remainder_predictive, full_matrices=False)
        remainder_predictive_scores = remainder_predictive_scores[0][:,remainder_predictive_scores[1]>0.9]
        
        return percolate_iter_clf, remainder_predictive_scores


    def _glmpca_parameters(self, key):
        return {
            'n_pc': self.n_factors[key], 
            'max_param' : self.max_param[key],
            'learning_rate' : self.learning_rates[key],
            'family': self.families[key],
            'maxiter' :self.maxiter[key],
            'batch_size': self.batch_size[key],
            'n_init': self.n_glmpca_init[key],
            'n_jobs': self.n_jobs,    
        }

    def save(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        params = {
            'predictive_key': self.predictive_key,
            'robust_types_order': self.robust_types_order,
            'n_factors': self.n_factors,
            'families': self.families, 
            'maxiter': self.maxiter,
            'max_param': self.max_param,
            'learning_rates': self.learning_rates,
            'batch_size': self.batch_size, 
            'n_glmpca_init': self.n_glmpca_init,
            'n_jobs': self.n_jobs,
            'joint_estimation_n_iter': self.joint_estimation_n_iter
        }
        dump(params, open('%s/params.pkl'%(folder), 'wb'))

        for dt in self.percolate_iter_clfs:
            self.percolate_iter_clfs[dt].save('%s/instance_%s/'%(folder, dt))

        dump([x.cpu() for x in self.predictive_scores_residuals_], open('%s/predictive_residuals.pkl'%(folder), 'wb'))



    def load(folder):
        params = load(open('%s/params.pkl'%(folder), 'rb'))
        percolate_instance = PERCOLATE(**params)

        percolate_instance.percolate_iter_clfs = {}
        for dt in percolate_instance.robust_types_order:
            percolate_instance.percolate_iter_clfs[dt] = GLMJIVE.load('%s/instance_%s/'%(folder, dt))

        percolate_instance.predictive_scores_residuals_ = load(open('%s/predictive_residuals.pkl'%(folder), 'rb'))

        return percolate_instance

