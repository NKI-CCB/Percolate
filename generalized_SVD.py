import numpy as np
import torch

def generalized_SVD(A, B, return_tensor=True):
    P, D, Q = np.linalg.svd(np.concatenate([A.T,B.T]), full_matrices=True)
    S = np.block([np.diag(D), np.zeros(shape=(D.shape[0], Q.shape[0]-D.shape[0]))])
    D = np.diag(D)
    Q = Q.T

    np.testing.assert_almost_equal(P.dot(S).dot(Q.T), np.concatenate([A.T,B.T]))
    
    k = D.shape[0]
    P1 = P[:,:k]
    P2 = P[:,k:]
    
    P11 = P1[:A.shape[1]]
    P21 = P1[A.shape[1]:]
    
    U1, S1, W = np.linalg.svd(P11)
    S1 = np.block([np.diag(S1), np.zeros(shape=(S1.shape[0], W.shape[0] - S1.shape[0]))])
    W = W.T
    
    U2, S2 = np.linalg.qr(P21.dot(W))
    

    return_values = {'U1': U1, 'U2': U2, 'Q': Q, 'W':W, 'S1':S1, 'S2': S2, 'D': D, 'A': A, 'B': B}
    if return_tensor:
        return {k:torch.Tensor(return_values[k]) for k in return_values}
    else:
        return return_values


def reconstruct_generalized_SVD(gsvd_results):
    A_reconstruct = np.block([
        gsvd_results['W'].T.dot(gsvd_results['D']), 
        np.zeros(shape=(gsvd_results['S1'].shape[1], gsvd_results['A'].shape[0]-gsvd_results['S1'].shape[1]))
    ])
    A_reconstruct = gsvd_results['U1'].dot(gsvd_results['S1']).dot(A_reconstruct).dot(gsvd_results['Q'].T)
    
    B_reconstruct = np.block([
        gsvd_results['W'].T.dot(gsvd_results['D']), 
        np.zeros(shape=(gsvd_results['S2'].shape[1], gsvd_results['B'].shape[0]-gsvd_results['S2'].shape[1]))
    ])
    B_reconstruct = gsvd_results['U2'].dot(gsvd_results['S2']).dot(B_reconstruct).dot(gsvd_results['Q'].T)
    
    np.testing.assert_array_almost_equal(A_reconstruct, gsvd_results['A'].T)
    np.testing.assert_array_almost_equal(B_reconstruct, gsvd_results['B'].T)
    
    return A_reconstruct, B_reconstruct