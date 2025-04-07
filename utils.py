import numpy as np
from scipy.stats import chi2


def embedded_jump(x):
    """
    Extract the embedded jump sequence (y) from the input sequence (x)

    """
    switch_indices = 1 + np.where(np.diff(x) != 0)[0]
    switch_indices = np.hstack(([0], switch_indices)) # include starting point
    return x[switch_indices]


def mc(p, T, n):
    """
    Generate a Markov chain sample from 
    - initial distribution p
    - transition matrix T (conditional transition probabilities)
    - length: n samples

    NOTE:
    - Doesn't include type or shape checks for p, T, n
      Users are responsible for the soundness of arguments
    
    """
    p_sum = np.cumsum(p)
    T_sum = np.cumsum(T, axis=1)
    y = np.zeros(n, dtype=int)
    y[0] = np.min(np.argwhere(np.random.rand() < p_sum))
    for i in range(1,n):
        y[i] = np.min(np.argwhere(np.random.rand() < T_sum[y[i-1],:]))
    return y


def p_equilibrium(T):
    """
    Compute equilibrium distribution (p_eq) from a conditional transition 
    probability matrix (T). p_eq is the normalized (sum p_eq = 1) left 
    eigenvector to the eigenvalue lambda = 1 (Perron-Frobenius)
    
    """
    evals, evecs = np.linalg.eig(T.transpose())
    i = np.where(np.isclose(evals, 1.0, atol=1e-6))[0][0] # locate max eigenval.
    p_eq = np.abs(evecs[:,i]) # assure eigenvector is non-negative
    p_eq /= p_eq.sum() # normalize
    return p_eq


def pmf(x, K):
    """
    pmf: probability mass function (normalized histogram)
    Used to calculate microstate distribution

    NOTE: Can be computed faster. This form is in analogy to tpm and the Markov 
          test functions below.

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
    Test zero-order Markovianity of symbolic sequence x with K symbols.
    Null hypothesis: zero-order MC (iid) <=>
    Pr(X[t]), Pr(X[t+1]) independent
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
    g = 0.0 # G-test statistic
    for i, j in np.ndindex(f_ij.shape):
        num_ = n*f_ij[i,j]
        den_ = f_i[i]*f_j[j]
        if (num_*den_ > 0):
            g += (f_ij[i,j] * np.log(num_/den_))
    g *= 2
    df = (K-1) * (K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def test_markov1(x, K):
    """
    Test first-order Markovianity of symbolic sequence x with K symbols.
    Null hypothesis:
    first-order MC <=>
    Pr(X[t+1] | X[t]) = Pr(X[t+1] | X[t], X[t-1])
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
    g = 0.0 # G-test statistic
    for i, j, k in np.ndindex(f_ijk.shape):
        num_ = f_ijk[i,j,k]*f_j[j]
        den_ = f_ij[i,j]*f_jk[j,k]
        if (num_*den_ > 0):
            g += (f_ijk[i,j,k]*np.log(num_/den_))
    g *= 2
    df = K*(K-1)*(K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def test_markov2(x, K):
    """
    Test second-order Markovianity of symbolic sequence x with K symbols.

    Parameters
    ----------
        x : (N,) array_like
            A 1-D array of integer values
        K : int
            number of symbols
    
    Returns
    -------
        p : float
            test result (p-value)

    Notes
    -----
    Null hypothesis:
    first-order MC <=>
    Pr(X[t+1] | X[t], X[t-1]) = Pr(X[t+1] | X[t], X[t-1], X[t-2])

    Examples
    ---------

    References
    ----------
    cf. Kullback, Technometrics (1962), Table 10.2.

    """
    n = len(x)
    f_ijkl = np.zeros((K,K,K,K))
    f_ijk = np.zeros((K,K,K))
    f_jkl = np.zeros((K,K,K))
    f_jk = np.zeros((K,K))
    for t in range(n-3):
        i = x[t]
        j = x[t+1]
        k = x[t+2]
        l = x[t+3]
        f_ijkl[i,j,k,l] += 1
        f_ijk[i,j,k] += 1
        f_jkl[j,k,l] += 1
        f_jk[j,k] += 1
    g = 0.0 # G-test statistic
    for i, j, k, l in np.ndindex(f_ijkl.shape):
        num_ = f_ijkl[i,j,k,l]*f_jk[j,k]
        den_ = f_ijk[i,j,k]*f_jkl[j,k,l]
        if (num_*den_ > 0):
            g += (f_ijkl[i,j,k,l]*np.log(num_/den_))
    g *= 2
    df = K*K*(K-1)*(K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def test_transition_matrix(x, T):
    """
    Test whether the transition matrix of the sequence x is equal to T
    Null hypothesis:
    Pr(X[t+1] = j | X[t] = i) ~ T[i,j]
    cf. Kullback, Technometrics (1962), Table 7.2

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
    g = 0.0 # G-test statistic
    for i, j in np.ndindex(f_ij.shape):
        num_ = f_ij[i,j]
        den_ = f_i[i]*T[i,j]
        if (num_*den_ > 0):
            g += (f_ij[i,j]*np.log(num_/den_))
    g *= 2.0
    df = K*(K-1)
    p = chi2.sf(g, df, loc=0, scale=1)
    return p


def tpm(x, K):
    """
    tpm: transition probability matrix (T)
    T_{ij} = P(X[t+1]=j | X[t]=i) (conditional transition probabilities)

    Args:
        x: numpy.array, size = length of microstate sequence
        K: number of microstate clusters
    Returns:
        T: transition probability matrix
    
    """
    T = np.zeros((K, K))
    n = len(x)
    for i in range(n-1):
        T[x[i], x[i+1]] += 1
    p_row = T.sum(axis=1, keepdims=True) # row sums
    p_row[p_row==0] = 1. # dodgy but correct, avoid division by zero
    T /= p_row # make row sums = 1
    return T