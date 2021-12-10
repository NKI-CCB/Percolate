import torch
import numpy as np
from copy import deepcopy
from .GLMPCA import GLMPCA
from .difference_GLMPCA import difference_GLMPCA
from .residualGLMPCA import ResidualGLMPCA
from .generalized_SVD import generalized_SVD

class GLMJIVE:

    def __init__(
        self, 
        n_factors,
        n_joint, 
        families, 
        maxiter=10,
        max_param=None,
        learning_rates=None,
        alignment_method='svd'
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
        self.alignment_method = alignment_method
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
                self.factor_models[data_type].compute_saturated_orthogonal_scores(X[data_type], correct_loadings=False)
            )

        # Align by computing joint scores
        if self.alignment_method in ['svd', 'SVD']:
            self.M_ = torch.cat(self.orthogonal_scores, axis=1)
            self.M_svd_ = list(torch.linalg.svd(self.M_, full_matrices=False))
        elif self.alignment_method in ['gsvd', 'generalized', 'GSVD']:
            self.M_svd_ = generalized_SVD(
                self.orthogonal_scores[0].detach().T,
                self.orthogonal_scores[1].detach().T,
                return_tensor=True
            )

        # Stop if required
        if no_alignment:
            return True

        # Initialize models
        self.joint_models = {k:self.factor_models[k].clone_empty_GLMPCA() for k in X}
        self.individual_models = {k:difference_GLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}
        self.noise_models = {k:ResidualGLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}

        if self.alignment_method in ['svd', 'SVD']:
            # Compute joint scores like in AJIVE
            self.joint_scores_ = self.M_svd_[0][:,:self.n_joint]
            self.individual_scores_ = torch.diag(torch.Tensor([1]*self.joint_scores_.shape[0])) - self.joint_scores_.matmul(self.joint_scores_.T)

            _, S_M, V_M = self.M_svd_
            self.V_M_ = V_M.T
            self.joint_proj_ = {
                self.data_types[0]: self.V_M_.T[:,:self.n_factors[self.data_types[0]]].T,
                self.data_types[1]: self.V_M_.T[:,self.n_factors[self.data_types[0]]:].T
            }

            # Compute rotation matrices per data-type
            print('START JOINT MODEL', flush=True)
            self.joint_scores_contribution_ = {}       
            for d in self.data_types:
                self.joint_models[d].n_pc = self.n_joint
                rotation = torch.linalg.svd(self.factor_models[d].saturated_scores_.T.matmul(self.joint_scores_), full_matrices=False)
                self.joint_models[d].saturated_loadings_ = self.factor_models[d].saturated_loadings_.matmul(
                    self.factor_models[d].saturated_scores_.T.matmul(self.joint_scores_)
                )
                # Compute the joint models as indicated in Methods
                self.joint_models[d].saturated_loadings_ = self.factor_models[d].saturated_loadings_.matmul(
                    self.factor_models[d].saturated_loadings_.T
                ).matmul(
                    self.factor_models[d].projected_orthogonal_scores_svd_[2]
                ).matmul(
                    torch.diag(1/self.factor_models[d].projected_orthogonal_scores_svd_[1])
                ).matmul(self.joint_proj_[d]).matmul(torch.diag(1/S_M)[:,:self.n_joint])
                self.joint_models[d].compute_reconstructed_data(
                    X[d], 
                    self.joint_scores_
                )

                # Compute the contribution to the joint scores
                self.joint_scores_contribution_[d] = self.factor_models[d].saturated_param_.matmul(
                    self.joint_models[d].saturated_loadings_
                )

            # Set up individual
            print('START INDIVIDUAL MODEL', flush=True)
            for d in self.data_types:
                indiv_matrix = torch.Tensor(np.identity(self.joint_scores_.shape[0])) 
                indiv_matrix = indiv_matrix - self.joint_scores_.matmul(self.joint_scores_.T)
                indiv_matrix, _, _ = torch.linalg.svd(indiv_matrix)
                indiv_matrix = indiv_matrix[:,:self.factor_models[d].n_pc - self.n_joint]
                self.individual_models[d].saturated_loadings_ = self.factor_models[d].saturated_loadings_.matmul(
                    self.factor_models[d].saturated_scores_.T.matmul(indiv_matrix)
                )

                self.individual_models[d].fill_GLMPCA_instances(
                    self.factor_models[d], 
                    self.joint_models[d]
                )
                del indiv_matrix


            # Set up individual
            print('START NOISE MODEL', flush=True)
            for d in self.data_types:
                noise_matrix = torch.Tensor(np.identity(self.factor_models[d].saturated_loadings_.shape[0])) 
                noise_matrix = noise_matrix - self.factor_models[d].saturated_loadings_.matmul(self.factor_models[d].saturated_loadings_.T)
                noise_matrix, _, _ = torch.linalg.svd(noise_matrix)
                noise_matrix = noise_matrix[:,self.factor_models[d].n_pc:]
                self.noise_models[d].saturated_loadings_ = noise_matrix

                self.noise_models[d].fill_GLMPCA_instances(
                    self.factor_models[d]
                )
                del noise_matrix

        elif self.alignment_method in ['gsvd', 'generalized', 'GSVD']:
            self.joint_scores_ = self.M_svd_['Q'][:,:self.n_joint]
            self.individual_scores_ = self.M_svd_['Q'][:,self.n_joint:]

            # Compute rotation matrices per data-type
            print('START JOINT MODEL', flush=True)
            self.joint_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.clone().detach()
            self.joint_models[self.data_types[0]].saturated_loadings_rot_ = torch.matmul(
                torch.linalg.pinv(self.M_svd_['D']),
                self.M_svd_['W']
            ).matmul(
                torch.linalg.pinv(self.M_svd_['S1'])
            ).matmul(
                self.M_svd_['U1'].T
            )[:self.n_joint,:]
            self.joint_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.matmul(
                self.joint_models[self.data_types[0]].saturated_loadings_rot_.T
            )

            self.joint_models[self.data_types[1]].saturated_loadings_ = self.factor_models[self.data_types[1]].saturated_loadings_.clone().detach()
            self.joint_models[self.data_types[1]].saturated_loadings_rot_ = torch.matmul(
                torch.linalg.pinv(self.M_svd_['D']),
                self.M_svd_['W']
            ).matmul(
                torch.linalg.pinv(self.M_svd_['S2'])
            ).matmul(
                self.M_svd_['U2'].T
            )[:self.n_joint,:]
            self.joint_models[self.data_types[1]].saturated_loadings_ = self.joint_models[self.data_types[1]].saturated_loadings_.matmul(
                self.joint_models[self.data_types[1]].saturated_loadings_rot_.T
            )

            # Set up individual
            pass
            print('START INDIVIDUAL MODEL', flush=True)
            self.individual_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.matmul(
                self.M_svd_['U1'].matmul(torch.linalg.pinv(self.M_svd_['S1']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint:]
            )
            self.individual_models[self.data_types[1]].saturated_loadings_ = self.factor_models[self.data_types[1]].saturated_loadings_.matmul(
                self.M_svd_['U2'].matmul(torch.linalg.pinv(self.M_svd_['S2']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint:]
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
        elif data_type == 'noise':
            return self.noise_models[data_source].project_low_rank(X)


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
            return self.noise_models[data_source].project_cell_view(X)