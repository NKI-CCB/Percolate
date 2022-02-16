import torch, os
import numpy as np
from copy import deepcopy
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from scipy.stats import ortho_group
from joblib import Parallel, delayed
from pickle import load, dump
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
        batch_size=None, 
        n_glmpca_init=None,
        n_jobs=1
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
        self.batch_size = batch_size if batch_size is not None else {k:128 for k in n_factors}
        self.n_glmpca_init = n_glmpca_init if n_glmpca_init is not None else {k:1 for k in n_factors}
        # self.with_intercept = with_intercept

        self.factor_models = {}
        self.joint_models = {}

        # For parallelization
        self.n_jobs = n_jobs


    def fit(self, X, no_alignment=False, exp_family_params=None):
        """
        X must be a dictionary of data with same keys than n_factors and families.

        - no_alignment: bool, default to False
            Whether joint and individual components must be computed. If set to yes, process stops to
            the computation of the matrix M.
        """

        # Train GLM-PCA instances.
        self._train_glmpca_instances(X, exp_family_params=exp_family_params)

        # Compute the matrix M and decompose it by SVD.
        self._aggregate_scores()

        # Stop if required
        if no_alignment:
            return True

        # Initialize models
        self._initialize_models(X)
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


    def _train_glmpca_instances(self, X, exp_family_params=None):
        """
        Train the GLM-PCA instances needed for the GLM-JIVE.
        """
        # Train factor models
        self.factor_models = {}
        self.orthogonal_scores = []
        self.data_types = list(X.keys())
        exp_family_params = exp_family_params if exp_family_params is not None else {data_type:None for data_type in X}
        
        # self.factor_models = dict(Parallel(n_jobs=2, verbose=1)(
        #     delayed(self._train_one_glmpca_instance)(data_type, X, exp_family_params)
        #     for data_type in self.data_types
        # ))
        self.factor_models = dict([
            self._train_one_glmpca_instance(data_type, X, exp_family_params)
            for data_type in self.data_types
        ])

        self.orthogonal_scores = [
            self.factor_models[data_type].compute_saturated_orthogonal_scores(X[data_type], correct_loadings=False)
            for data_type in self.factor_models
        ]

        return True

    def _train_one_glmpca_instance(self, data_type, X, exp_family_params):

        glmpca_clf = GLMPCA(
            self.n_factors[data_type], 
            family=self.families[data_type], 
            maxiter=self.maxiter[data_type], 
            max_param=self.max_param[data_type],
            learning_rate=self.learning_rates[data_type],
            n_jobs=self.n_jobs
        )

        glmpca_clf.compute_saturated_loadings(
            X[data_type],
            batch_size=self.batch_size[data_type], 
            n_init=self.n_glmpca_init[data_type],
            exp_family_params=exp_family_params[data_type]
        )

        return (data_type, glmpca_clf)


    def _aggregate_scores(self):
        """
        Compute the matrix M alongside its SVD decomposition.
        """
        self.M_ = torch.cat(self.orthogonal_scores, axis=1)
        self.M_svd_ = list(torch.linalg.svd(self.M_, full_matrices=False))

        return True

    def _initialize_models(self, X):
        self.joint_models = {k:self.factor_models[k].clone_empty_GLMPCA() for k in X}
        self.individual_models = {k:difference_GLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}
        self.noise_models = {k:ResidualGLMPCA.clone_from_GLMPCA(self.factor_models[k]) for k in X}

    def _computation_joint_individual_factor_model(self, X=None, not_aligned_types=[]):
        # Set up GPU device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Compute joint scores like in AJIVE
        self.joint_scores_ = self.M_svd_[0][:,:self.n_joint]

        # Compute individual scores by taking the rest of the glmpca signal
        joint_proj = self.joint_scores_.matmul(self.joint_scores_.T)
        individual_proj = torch.eye(self.joint_scores_.shape[0]).to(device) - joint_proj.to(device)
        self.individual_scores_ = {}
        for score, dt in zip(self.orthogonal_scores, self.data_types):
            individual_svd = torch.linalg.svd(individual_proj.matmul(score))
            self.individual_scores_[dt] = individual_svd[0][:,:-self.n_joint]

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
            if d in not_aligned_types:
                continue
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

            # Compute the contribution to the joint scores
            self.joint_scores_contribution_[d] = self.factor_models[d].saturated_param_.matmul(
                self.joint_models[d].saturated_loadings_
            )

        # Set up individual by computing the difference
        print('START INDIVIDUAL MODEL', flush=True)
        for d in self.data_types:
            if d in not_aligned_types:
                continue
            indiv_matrix = torch.Tensor(np.identity(self.joint_scores_.shape[0])).to(device)
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
            if d in not_aligned_types:
                continue
            noise_matrix = torch.Tensor(np.identity(self.factor_models[d].saturated_loadings_.shape[0])).to(device)
            noise_matrix = noise_matrix - self.factor_models[d].saturated_loadings_.matmul(self.factor_models[d].saturated_loadings_.T)
            try:
                noise_matrix, _, _ = torch.linalg.svd(noise_matrix)
                noise_matrix = noise_matrix[:,self.factor_models[d].n_pc:]
            except:
                print('NOISE MODEL CANNOT BE COMPUTED: SVD DID NOT CONVERGE')
                #raise ValueError('NOISE MODEL CANNOT BE COMPUTED: SVD DID NOT CONVERGE')
            self.noise_models[d].saturated_loadings_ = noise_matrix

            self.noise_models[d].fill_GLMPCA_instances(
                self.factor_models[d]
            )
            del noise_matrix

        return True


    def compute_joint_signal_from_saturated_params(self, saturated_params, return_decomposition=False):

        # Project data
        if type(saturated_params) is dict:
            U_known = self.joint_models[self.known_data_type].project_low_rank_from_saturated_parameters(
                saturated_params[self.known_data_type]
            )
        else:
            U_known = self.joint_models[self.known_data_type].project_low_rank_from_saturated_parameters(saturated_params)

        # Predict unknown_data
        U_unknown = torch.Tensor([
            self.trans_type_regressors_[joint_factor_idx].predict(U_known.detach().cpu().numpy())
            for joint_factor_idx in range(self.n_joint)
        ]).T

        if return_decomposition:
            return U_known , U_unknown
        return U_known + U_unknown


    def compute_joint_signal(self, X, return_decomposition=False):
        """
        Given a sample of self.known_data:
            - Project X on the joint factor.
            - Predict the unknown_data type.
            - Sum two contributions.
        """

        # Project data
        if type(X) is dict:
            U_known = self.joint_models[self.known_data_type].project_low_rank(X[self.known_data_type])
        else:
            U_known = self.joint_models[self.known_data_type].project_low_rank(X)

        # Predict unknown_data
        U_unknown = torch.Tensor([
            self.trans_type_regressors_[joint_factor_idx].predict(U_known.detach().cpu().numpy())
            for joint_factor_idx in range(self.n_joint)
        ]).T

        if return_decomposition:
            return U_known , U_unknown
        return U_known + U_unknown
        

    def _train_trans_type_regression_model(self, unknown_factor_idx, cv=10, n_jobs=1):
        """
        Train a kNN regression model from the known data-type to the unknown data-type.
        """

        X_known = self.joint_scores_contribution_[self.known_data_type].detach().cpu().numpy()

        # If the unknown data-type has not been aligned, then look at the difference.
        if self.unknown_data_type in self.joint_scores_contribution_:
            X_unknown = self.joint_scores_contribution_[self.unknown_data_type][:,unknown_factor_idx].detach().cpu().numpy()
        else:
            X_unknown = (self.joint_scores_ - self.joint_scores_contribution_[self.known_data_type]).detach().cpu().numpy()[:,unknown_factor_idx]

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


    def estimate_number_joint_components_random_matrix(self, n_iter=20, quantile_top_component=0.95, n_jobs=1):
        # Generate random orthogonal matrices
        random_state = np.random.randint(1,10**6,size=2)
        random_orth_mat = np.array(Parallel(n_jobs=min(n_jobs,2), verbose=1)(
            delayed(ortho_group.rvs)(np.max(score.shape), n_iter, random_state=seed)
            for score, seed in zip(self.orthogonal_scores, random_state)
        )).transpose(1,0,2,3)

        # Restrict to the shape of orthogonal scores (previous matrices are squared)
        random_orth_mat = [
            [
                torch.Tensor(m[:score.shape[0],:score.shape[1]])
                for m, score in zip(mat, self.orthogonal_scores)
            ]
            for mat in random_orth_mat
        ]
        # Verifies that resulting matrices are orthogonal
        for mat in random_orth_mat:
            for m in mat:
                torch.testing.assert_allclose(m.T.matmul(m), torch.eye(m.shape[1]))

        # Compute resulting top singular values
        random_svd_value = np.array([
            torch.linalg.svd(torch.cat(mat, axis=1))[1].detach().cpu().numpy()
            for mat in random_orth_mat
        ])

        # Compute number of joint components as the components above the 95% top random singular values
        number_joint = torch.sum(torch.linalg.svd(self.M_)[1] > np.quantile(random_svd_value[:,0],quantile_top_component))
        number_joint = number_joint.detach().cpu().numpy()
        return int(number_joint)


    def estimate_number_joint_components_permutation(self, n_perm=20, quantile_top_component=0.95):
        """
        max_joint: int or float
            If float, proportion of the minimum number of components.
        """
        self.permuted_M_ = []
        self.permuted_M_svd_ = []

        for perm_idx in range(n_perm):
            # Permute data
            source_idx = np.arange(self.M_.shape[0])
            target_idx = np.arange(self.M_.shape[0])
            np.random.shuffle(source_idx)
            np.random.shuffle(target_idx)

            # Train instance
            self.permuted_M_svd_.append(torch.cat([
                self.orthogonal_scores[0][source_idx],
                self.orthogonal_scores[1][target_idx]
            ], axis=1))
            self.permuted_M_svd_[-1] = torch.linalg.svd(self.permuted_M_svd_[-1])[1][0]


        self.permuted_M_svd_ = torch.Tensor(self.permuted_M_svd_)
        number_joint = torch.sum(torch.linalg.svd(self.M_)[1] > np.quantile(self.permuted_M_svd_, quantile_top_component))
        number_joint = number_joint.detach().cpu().numpy()
        return int(number_joint)

        #     self.fit(perm_data, no_alignment=True)

        #     # Save resulting M
        #     self.permuted_M_.append(self.M_)
        #     self.permuted_M_svd_.append(self.M_svd_)

        # # Train instance
        # self._train_glmpca_instances(X)
        # self._aggregate_scores()

        return True


    def clone_GLM_JIVE(self):
        return deepcopy(self)

    def save(self, folder):
        if not os.path.isdir(folder):
            os.mkdir(folder)

        # Save parameters
        GLMJIVE_params = {
            'n_factors': self.n_factors,
            'n_joint': self.n_joint,
            'families': self.families,
            'maxiter': self.maxiter,
            'max_param': self.max_param,
            'learning_rates': self.learning_rates,
            'batch_size': self.batch_size,
            'n_glmpca_init': self.n_glmpca_init
        }       
        dump(GLMJIVE_params, open('%s/params.pkl'%(folder), 'wb'))

        # Save factor models
        for data_type in self.data_types:
            if data_type in self.factor_models:
                self.factor_models[data_type].save('%s/factor_model_%s'%(folder, data_type))
            if data_type in self.joint_models:
                self.joint_models[data_type].save('%s/joint_model_%s'%(folder, data_type))

        # Save alignment
        dump(
            [e.cpu() for e in self.orthogonal_scores], 
            open('%s/orthogonal_scores.pkl'%(folder), 'wb')
        )
        dump(
            self.data_types,
            open('%s/data_types.pkl'%(folder), 'wb')
        )
        torch.save(self.M_.cpu(), '%s/M.pt'%(folder))

        if hasattr(self, 'known_data_type') and hasattr(self, 'unknown_data_type'):
            dump(
                {'known_data_type': self.known_data_type, 'unknown_data_type':self.unknown_data_type},
                open('%s/out_of_samples_data_types.pkl'%(folder), 'wb')
            )
        if hasattr(self, 'trans_type_regressors_'):
            dump(self.trans_type_regressors_, open('%s/out_of_samples_regression.pkl'%(folder), 'wb'))
        


    def load(folder):
        """
        Load a GLMJIVE instance saved in folder.
        """
        GLMJIVE_params = load(open('%s/params.pkl'%(folder), 'rb'))
        instance = GLMJIVE(**GLMJIVE_params)

        # Load factor models
        instance.data_types = load(open('%s/data_types.pkl'%(folder), 'rb'))
        for data_type in instance.data_types:
            if 'factor_model_%s'%(data_type) in os.listdir(folder):
                instance.factor_models[data_type] = GLMPCA.load('%s/factor_model_%s'%(folder, data_type))
            if 'joint_model_%s'%(data_type) in os.listdir(folder):
                instance.joint_models[data_type] = GLMPCA.load('%s/joint_model_%s'%(folder, data_type))

        # Load alignment
        instance.orthogonal_scores = load(open('%s/orthogonal_scores.pkl'%(folder), 'rb'))
        instance.M_ = torch.load('%s/M.pt'%(folder))

        if 'out_of_samples_data_types.pkl' in os.listdir(folder):
            dt = load(open('%s/out_of_samples_data_types.pkl'%(folder), 'rb'))
            instance.known_data_type = dt['known_data_type']
            instance.unknown_data_type = dt['unknown_data_type']
        if 'out_of_samples_regression.pkl' in os.listdir(folder):
            instance.trans_type_regressors_ = load(open('%s/out_of_samples_regression.pkl'%(folder), 'rb'))

        return instance