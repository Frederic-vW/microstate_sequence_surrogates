import numpy as np
from scipy.stats import chi2


def embedded_jump(x):
    switch_indices = 1 + np.where(np.diff(x) != 0)[0]
    switch_indices = np.hstack(([0], switch_indices))
    return x[switch_indices]


def mc(p, T, n):
    p_sum = np.cumsum(p)
    T_sum = np.cumsum(T, axis=1)
    y = np.zeros(n, dtype=int)
    y[0] = np.min(np.argwhere(np.random.rand() < p_sum))
    for i in range(1,n):
        y[i] = np.min(np.argwhere(np.random.rand() < T_sum[y[i-1],:]))
    return y


def p_equilibrium(T):
    '''
    get equilibrium distribution from transition matrix:
    lambda = 1 - (left) eigenvector
    '''
    evals, evecs = np.linalg.eig(T.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0] # locate max eigenval.
    p_eq = np.abs(evecs[:,i]) # make eigenvec. to max. eigenval. non-negative
    p_eq /= p_eq.sum() # normalized eigenvec. to max. eigenval.
    return p_eq # stationary distribution


def pmf(x, K):
    """
    Empirical symbol distribution

    Args:
        x: numpy.array, size = length of microstate sequence
        K: number of microstate clusters
    Returns:
        p: empirical distribution
    """
    p = np.zeros(K)
    n = len(x)
    for i in range(n):
        p[x[i]] += 1
    p /= n
    return p


def test_markov0(x, K):
    """
    Test zero-order Markovianity of symbolic sequence x with ns symbols.
    Null hypothesis: zero-order MC (iid) <=>
    p(X[t]), p(X[t+1]) independent
    cf. Kullback, Technometrics (1962)

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        K: number of symbols
    Returns:
        p: p-value of the Chi2 test
    
    """
    n = len(x)
    f_ij = np.zeros((K,K))
    f_i = np.zeros(K)
    f_j = np.zeros(K)
    # calculate f_ij p( x[t]=i, p( x[t+1]=j ) )
    for t in range(n-1):
        i = x[t]
        j = x[t+1]
        f_ij[i,j] += 1
        f_i[i] += 1
        f_j[j] += 1
    g = 0.0 # test statistic
    for i, j in np.ndindex(f_ij.shape):
        f = f_ij[i,j]*f_i[i]*f_j[j]
        if (f > 0):
            num_ = n*f_ij[i,j]
            den_ = f_i[i]*f_j[j]
            g += (f_ij[i,j] * np.log(num_/den_))
    g *= 2.0
    df = (K-1) * (K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def test_markov1(x, K):
    """
    Test first-order Markovianity of symbolic sequence X with ns symbols.
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        K: number of symbols
    Returns:
        p: p-value of the Chi2 test
    
    """
    n = len(x)
    f_ijk = np.zeros((K,K,K))
    f_ij = np.zeros((K,K))
    f_jk = np.zeros((K,K))
    f_j = np.zeros(K)
    for t in range(n-2):
        i = x[t]
        j = x[t+1]
        k = x[t+2]
        f_ijk[i,j,k] += 1
        f_ij[i,j] += 1
        f_jk[j,k] += 1
        f_j[j] += 1
    g = 0.0 # test statistic
    for i, j, k in np.ndindex(f_ijk.shape):
        f = f_ijk[i,j,k]*f_j[j]*f_ij[i,j]*f_jk[j,k]
        if (f > 0):
            num_ = f_ijk[i,j,k]*f_j[j]
            den_ = f_ij[i,j]*f_jk[j,k]
            g += (f_ijk[i,j,k]*np.log(num_/den_))
    g *= 2.0
    df = K*(K-1)*(K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def test_transition_matrix(x, T):
    """
    Test whether the transition matrix of the sequence x is equal to T
    Null hypothesis:
    first-order MC <=>
    p(X[t+1] | X[t]) = p(X[t+1] | X[t], X[t-1])
    cf. Kullback, Technometrics (1962), Tables 8.1, 8.2, 8.6.

    Args:
        x: symbolic sequence, symbols = [0, 1, 2, ...]
        T: reference transition matrix
    Returns:
        p: p-value of the Chi2 test
    
    """
    # initial checks
    assert T.ndim == 2 # T is two-dimensional
    assert T.shape[0] == T.shape[1] # T is square
    assert T.shape[0] == len(np.unique(x)) # T has shape (K,K)
    n = len(x)
    K = T.shape[0]
    f_ij = np.zeros((K,K))
    f_i = np.zeros(K)
    for t in range(n-1):
        i = x[t]
        j = x[t+1]
        f_ij[i,j] += 1
        f_i[i] += 1
    g = 0.0 # test statistic
    for i, j in np.ndindex(f_ij.shape):
        den_ = f_i[i]*T[i,j]
        if (den_ > 0):
            g += (f_ij[i,j]*np.log(f_ij[i,j]/den_))
    g *= 2.0
    df = K*(K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def tpm(x, K):
    """
    Empirical transition matrix

    Args:
        data: numpy.array, size = length of microstate sequence
        K: number of microstate clusters
    Returns:
        T: empirical transition matrix
    """
    T = np.zeros((K, K))
    n = len(x)
    for i in range(n-1):
        T[x[i], x[i+1]] += 1
    p_row = np.sum(T, axis=1)
    for i in range(K):
        if (p_row[i] != 0.0):
            for j in range(K):
                T[i,j] /= p_row[i]  # normalize row sums to 1.0
    return T