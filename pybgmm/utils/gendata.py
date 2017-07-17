import numpy as np

def gendata_1d(N, W=[0.35, 0.4, 0.25], MU=[0., 2., 5.], SIGMA=[0.5, 0.5, 1.], seed = 12345):
    """

    :param N: number of samples
    :param W: Weight vector
    :param MU: mean vector
    :param SIGMA: standard deviation
    :param seed: random generation seed
    :return: data X and component y
    """

    np.random.seed(seed)

    W = np.array(W)
    assert np.sum(W)==1, "weight vector should sum to 1"

    MU = np.array(MU)
    SIGMA = np.array(SIGMA)
    y = np.random.choice(MU.size, size=N, p=W)
    X = np.random.normal(MU[y], SIGMA[y], size=N)

    X = X.reshape((X.shape[0], 1))

    return MU, X, y