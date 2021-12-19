import torch
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
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
        maxiter=1000,
        max_param=None,
        learning_rates=None,
        batch_size=128, 
        n_glmpca_init=1
        ):
        """
         Method can be 'svd' or 'likelihood'.
        """

        self.n_factors = n_factors
        self.n_joint = n_joint
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
        self.batch_size = batch_size
        self.n_glmpca_init = n_glmpca_init
        # self.with_intercept = with_intercept


    def fit(self, X, no_alignment=False):
        """
        X must be a dictionary of data with same keys than n_factors and families.

        - no_alignment: bool, default to False
            Whether joint and individual components must be computed. If set to yes, process stops to
            the computation of the matrix M.
        """

        # Train GLM-PCA instances.
        self._train_glmpca_instances(X)

        # Compute the matrix M and decompose it by SVD.
        self._aggregate_scores()

        # Stop if required
        if no_alignment:
            return True

        # Initialize models
        self.joint_models = {k:self.factor_models[k].clone_empty_GLMPCA() for k in X}
        self.individual_models = {k:difference_GLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}
        self.noise_models = {k:ResidualGLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}

        self._computation_joint_individual_factor_model(X)

        return True


    def set_out_of_sample_extension(self, known_data_type, cv=10, n_jobs=1):
        """
        Set up the out-of-sample computation by training kNN regression models from the 
        known data type to the other unknown type.

        known_data_type: str
            Data-type to regress on.
        """

        # Set known and unknown data-type
        self.known_data_type = known_data_type
        self.unknown_data_type = [e for e in self.data_types if e != self.known_data_type]
        assert len(self.unknown_data_type) == 1
        self.unknown_data_type = self.unknown_data_type[0]

        # Train regression model
        self.trans_type_regressors_ = {
            joint_factor_idx: self._train_trans_type_regression_model(joint_factor_idx, cv=cv, n_jobs=n_jobs)
            for joint_factor_idx in range(self.n_joint)
        }

        return True


    def _train_glmpca_instances(self, X):
        """
        Train the GLM-PCA instances needed for the GLM-JIVE.
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
                maxiter=self.maxiter[data_type], 
                max_param=self.max_param[data_type],
                learning_rate=self.learning_rates[data_type]
            )

            self.factor_models[data_type].compute_saturated_loadings(
                X[data_type],
                batch_size=self.batch_size, 
                n_init=self.n_glmpca_init
            )
            self.orthogonal_scores.append(
                self.factor_models[data_type].compute_saturated_orthogonal_scores(X[data_type], correct_loadings=False)
            )

        return True


    def _aggregate_scores(self):
        """
        Compute the matrix M alongside its SVD decomposition.
        """
        self.M_ = torch.cat(self.orthogonal_scores, axis=1)
        self.M_svd_ = list(torch.linalg.svd(self.M_, full_matrices=False))

        return True

    def _computation_joint_individual_factor_model(self, X):
        # Compute joint scores like in AJIVE
        self.joint_scores_ = self.M_svd_[0][:,:self.n_joint]
        self.individual_scores_ = torch.diag(torch.Tensor([1]*self.joint_scores_.shape[0])) - self.joint_scores_.matmul(self.joint_scores_.T)

        _, S_M, V_M = self.M_svd_
        self.V_M_ = V_M.T
        self.V_M_decomposition_ = {
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
            ).matmul(self.V_M_decomposition_[d]).matmul(torch.diag(1/S_M)[:,:self.n_joint])
            self.joint_models[d].compute_reconstructed_data(
                X[d], 
                self.joint_scores_
            )

            # Compute the contribution to the joint scores
            self.joint_scores_contribution_[d] = self.factor_models[d].saturated_param_.matmul(
                self.joint_models[d].saturated_loadings_
            )

        # Set up individual by computing the difference
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

        return True


    def compute_joint_signal(self, X, return_decomposition=False):
        """
        Given a sample of self.known_data:
            - Project X on the joint factor.
            - Predict the unknown_data type.
            - Sum two contributions.
        """

        # Project data
        U_known = self.joint_models[self.known_data_type].project_low_rank(X)

        # Predict unknown_data
        U_unknown = torch.Tensor([
            self.trans_type_regressors_[joint_factor_idx].predict(U_known.detach().numpy())
            for joint_factor_idx in range(self.n_joint)
        ]).T

        if return_decomposition:
            return U_known , U_unknown
        return U_known + U_unknown
        

    def _train_trans_type_regression_model(self, unknown_factor_idx, cv=10, n_jobs=1):
        """
        Train a kNN regression model from the known data-type to the unknown data-type.
        """

        X_known = self.joint_scores_contribution_[self.known_data_type].detach().numpy()
        X_unknown = self.joint_scores_contribution_[self.unknown_data_type][:,unknown_factor_idx].detach().numpy()

        param_grid = {
            'regression__n_neighbors': np.linspace(2,20,19).astype(int),
            'regression__weights': ['uniform', 'distance']
        }

        # GridSearch by cross-validation
        imputation_model_ = GridSearchCV(
            Pipeline([
                ('regression', KNeighborsRegressor())
            ]),
            cv=cv,
            n_jobs=n_jobs,
            pre_dispatch='1.2*n_jobs',
            param_grid=param_grid,
            verbose=1,
            scoring='neg_mean_squared_error'
        )

        return imputation_model_.fit(X_known, X_unknown)


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


    def estimate_number_joint_components(self, X, n_perm=20):
        """
        max_joint: int or float
            If float, proportion of the minimum number of components.
        """
        self.data_types = list(X.keys())
        self.permuted_M_ = []
        self.permuted_M_svd_ = []

        for perm_idx in range(n_perm):
            # Permute data
            source_idx = np.arange(X[self.data_types[0]].shape[0])
            np.random.shuffle(source_idx)
            perm_data = deepcopy(X)
            perm_data[self.data_types[0]] = X[self.data_types[0]][source_idx]

            # Train instance
            self.fit(perm_data, no_alignment=True)

            # Save resulting M
            self.permuted_M_.append(self.M_)
            self.permuted_M_svd_.append(self.M_svd_)

        # Train instance
        self._train_glmpca_instances(X)
        self._aggregate_scores()

        return True


    def clone_GLM_JIVE(self):
        return deepcopy(self)

