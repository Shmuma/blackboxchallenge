import numpy as np

n_features = 36
n_actions = 4

coefs = np.loadtxt("reg_coefs.txt").reshape(n_actions, n_features + 1)
reg_coefs = coefs[:,:-1]
free_coefs = coefs[:,-1]

print reg_coefs.tolist()
print free_coefs.tolist()