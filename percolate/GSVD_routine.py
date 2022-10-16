# self.joint_scores_ = self.M_svd_['Q'][:,:self.n_joint]
# self.individual_scores_ = self.M_svd_['Q'][:,self.n_joint:]

# # Compute rotation matrices per data-type
# print('START JOINT MODEL', flush=True)
# self.joint_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.clone().detach()
# self.joint_models[self.data_types[0]].saturated_loadings_rot_ = torch.matmul(
#     torch.linalg.pinv(self.M_svd_['D']),
#     self.M_svd_['W']
# ).matmul(
#     torch.linalg.pinv(self.M_svd_['S1'])
# ).matmul(
#     self.M_svd_['U1'].T
# )[:self.n_joint,:]
# self.joint_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.matmul(
#     self.joint_models[self.data_types[0]].saturated_loadings_rot_.T
# )

# self.joint_models[self.data_types[1]].saturated_loadings_ = self.factor_models[self.data_types[1]].saturated_loadings_.clone().detach()
# self.joint_models[self.data_types[1]].saturated_loadings_rot_ = torch.matmul(
#     torch.linalg.pinv(self.M_svd_['D']),
#     self.M_svd_['W']
# ).matmul(
#     torch.linalg.pinv(self.M_svd_['S2'])
# ).matmul(
#     self.M_svd_['U2'].T
# )[:self.n_joint,:]
# self.joint_models[self.data_types[1]].saturated_loadings_ = self.joint_models[self.data_types[1]].saturated_loadings_.matmul(
#     self.joint_models[self.data_types[1]].saturated_loadings_rot_.T
# )

# # Set up individual
# pass
# print('START INDIVIDUAL MODEL', flush=True)
# self.individual_models[self.data_types[0]].saturated_loadings_ = self.factor_models[self.data_types[0]].saturated_loadings_.matmul(
#     self.M_svd_['U1'].matmul(torch.linalg.pinv(self.M_svd_['S1']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint:]
# )
# self.individual_models[self.data_types[1]].saturated_loadings_ = self.factor_models[self.data_types[1]].saturated_loadings_.matmul(
#     self.M_svd_['U2'].matmul(torch.linalg.pinv(self.M_svd_['S2']).T).matmul(self.M_svd_['W'].T).matmul(torch.linalg.pinv(self.M_svd_['D']).T)[:,self.n_joint:]
# )