import torch
import numpy as np
from copy import deepcopy
from .GLMPCA import GLMPCA
from .generalized_SVD import generalized_SVD

class GLMJIVE:

    def __init__(
        self, 
        n_factors,
        n_joint, 
        families, 
        maxiter=10,
        max_param=None,
        learning_rates=None
        ):
        """
         Method can be 'svd' or 'likelihood'.
        """

        self.n_factors = n_factors
        self.n_joint = n_joint
        self.families = families
        self.maxiter = maxiter
        self.max_param = max_param if max_param is not None else {k:None for k in n_factors}
        self.learning_rates = learning_rates if learning_rates is not None else {k:0.01 for k in n_factors}
        # self.with_intercept = with_intercept


    def fit(self, X, no_alignment=False):
        """
        X must be a dictionary of data with same keys than n_factors and families.

        - no_alignment: bool, default to False
            Whether joint and individual components must be computed. If set to yes, process stops to
            the computation of the matrix M.
        """

        # Train factor models
        self.factor_models = {}
        self.orthogonal_scores = []
        self.data_types = list(X.keys())
        for data_type in self.data_types:
            print('START TYPE %s'%(data_type))
            self.factor_models[data_type] = GLMPCA(
                self.n_factors[data_type], 
                family=self.families[data_type], 
                maxiter=self.maxiter, 
                max_param=self.max_param[data_type],
                learning_rate=self.learning_rates[data_type]
            )

            self.factor_models[data_type].compute_saturated_loadings(X[data_type])
            self.orthogonal_scores.append(
                self.factor_models[data_type].compute_saturated_orthogonal_scores(X[data_type], correct_loadings=True)
            )

        # Align by computing joint scores
        # self.M_ = torch.cat(self.orthogonal_scores, axis=1)
        # self.M_svd_ = list(torch.linalg.svd(self.M_, full_matrices=False))
        self.M_svd_ = generalized_SVD(
            self.orthogonal_scores[0].detach(),
            self.orthogonal_scores[1].detach(),
            return_tensor=True
        )

        if no_alignment:
            return True

        # self.joint_scores_ = self.M_svd_[0][:,:self.n_joint]
        self.joint_scores_ = self.M_svd_['Q'][:,:self.n_joint]
        self.individual_scores_ = self.M_svd_['Q'][:,self.n_joint:]

        # Compute associated factor loadings
        print('START JOINT MODEL', flush=True)
        self.joint_models = {k:self.factor_models[k].clone_empty_GLMPCA() for k in X}
        # Multiply by sqrt of two for having one norm (two SVD of unit norm vectors)

        self.joint_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.clone().detach()
        self.joint_models[self.data_types[0]].saturated_loadings_rot_ = self.M_svd_['U1'].matmul(torch.linalg.pinv(self.M_svd_['S1']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,:self.n_joint]
        self.joint_models[self.data_types[0]].saturated_loadings_ = 2 * self.joint_models[self.data_types[0]].saturated_loadings_.matmul(
            self.M_svd_['U1'].matmul(torch.linalg.pinv(self.M_svd_['S1']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,:self.n_joint]
        )
        
        self.joint_models[self.data_types[1]].saturated_loadings_ = self.factor_models[self.data_types[1]].saturated_loadings_.clone().detach()
        self.joint_models[self.data_types[1]].saturated_loadings_rot_ = self.M_svd_['U2'].matmul(torch.linalg.pinv(self.M_svd_['S2']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,:self.n_joint]       
        self.joint_models[self.data_types[1]].saturated_loadings_ = 2 * self.joint_models[self.data_types[1]].saturated_loadings_.matmul(
            self.M_svd_['U2'].matmul(torch.linalg.pinv(self.M_svd_['S2']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,:self.n_joint]
        )
        
        for data_type in X:
            self.joint_models[data_type].n_pc = self.n_joint
            self.joint_models[data_type].compute_reconstructed_data(
                X[data_type], 
                self.joint_scores_
            )

        # Compute associated individual factors
        print('START INDIVIDUAL MODEL', flush=True)
        self.individual_models = {k:self.factor_models[k].clone_empty_GLMPCA() for k in X}
        self.individual_models[self.data_types[0]].saturated_loadings_ = 2 * self.factor_models[self.data_types[0]].saturated_loadings_.matmul(
            self.M_svd_['U1'].matmul(torch.linalg.pinv(self.M_svd_['S1']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint]
        )
        self.individual_models[self.data_types[1]].saturated_loadings_ = 2 * self.factor_models[self.data_types[1]].saturated_loadings_.matmul(
            self.M_svd_['U2'].matmul(torch.linalg.pinv(self.M_svd_['S2']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint:]
        )
        for data_type in X:
            self.individual_models[data_type].n_pc = self.individual_models[data_type].n_pc - self.n_joint
            self.individual_models[data_type].compute_reconstructed_data(
                X[data_type], 
                self.individual_scores_
            )

        return True


    def project_low_rank(self, X, data_source, data_type):
        """
        - data_source: name in training data to align (e.g. mutations).
        - data_type: individual or joint.
        """
        if data_type == 'individual':
            return self.individual_models[data_source].project_low_rank(X)
        elif data_type == 'joint':
            return self.joint_models[data_source].project_low_rank(X)


    def project_cell_view(self, X, data_source, data_type):
        """
        - data_source: name in training data to align (e.g. mutations).
        - data_type: individual or joint.
        """
        if data_type == 'individual':
            return self.individual_models[data_source].project_cell_view(X)
        elif data_type == 'joint':
            return self.joint_models[data_source].project_cell_view(X)
        elif data_type == 'noise':
            return X - self.project_cell_view(X, data_source, 'joint') - self.project_cell_view(X, data_source, 'individual')