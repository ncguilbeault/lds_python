import sys
import numpy as np
# import ssm.neural_latents.utils
import ssm.neural_latents.utils


def test_orthogonalizeMeansAndCovs(tol=1e-6):
    M = 10
    N = 100
    T = 200

    # generate random means, covs and Z
    means = np.random.normal(loc=0, scale=1, size=(M, 1, T))
    random_matrices = np.random.normal(loc=0, scale=1, size=(T, M, M))
    L = np.tril(random_matrices)
    LT = np.transpose(L, axes=[0, 2, 1])
    covs_reshaped = L @ LT
    covs = covs_reshaped.reshape(M, M, T)
    Z = np.random.normal(loc=0, scale=1, size=(N, M))
    #

    o_means, o_covs = ssm.neural_latents.utils.ortogonalizeMeansAndCovs(
        means=means, covs=covs, Z=Z)
    o_means_l, o_covs_l = ssm.neural_latents.utils.ortogonalizeMeansAndCovsWithLoop(
        means=means, covs=covs, Z=Z)
    o_means_error = ((o_means - o_means_l)**2).mean()
    assert(o_means_error < tol)
    o_covs_error = ((o_covs - o_covs_l)**2).mean()
    assert(o_covs_error < tol)


if __name__ == "__main__":
    test_orthogonalizeMeansAndCovs()
