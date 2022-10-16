"""
Returns the data itself minored by the projection of the reconstruction of a PCA.
"""


import numpy as np
import torch
import torch.optim
from copy import deepcopy
from joblib import Parallel, delayed
from .exponential_family import *
import mctorch.nn as mnn
import mctorch.optim as moptim


class ResidualGLMPCA:
    """
    Compute the difference between two GLMPCA on the same space
    """
    def __init__(
        self, 
        n_pc, 
        family, 
        max_param = 10,
        ):

        self.n_pc = n_pc
        self.family = family
        self.max_param = np.abs(max_param)

        self.saturated_loadings_ = None
        # saturated_intercept_: before projecting
        self.saturated_intercept_ = None
        # reconstruction_intercept: after projecting
        self.reconstruction_intercept_ = None

        # Whether to perform sample or gene projection
        self.sample_projection = False


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
        glmpca_views = self.clf_contrastive.project_cell_view(X)

        return X - glmpca_views


    def fill_GLMPCA_instances(self, clf_contrastive):
        self.clf_contrastive = clf_contrastive

        self.saturated_intercept_ = clf_contrastive.saturated_intercept_
        self.reconstruction_intercept_ = 0.

        return True


    def clone_from_GLMPCA(clf):
        glmpca_clf = ResidualGLMPCA( 
            clf.n_pc, 
            clf.family,
            max_param=clf.max_param,
        )

        return glmpca_clf