"""AMI with the same chance model and grouped equal cluster-size terms."""
import math

from numba import njit
import numpy as np
from scipy.special import gammaln
from sklearn.metrics import mutual_info_score
from sklearn.metrics.cluster import contingency_matrix


@njit(cache=True)
def expected_info(n, a_sizes, a_counts, b_sizes, b_counts, log_factorial):
    # The hypergeometric null depends on the two marginals only through their
    # sizes. Sum each distinct pair once, weighted by its multiplicity.
    expectation = 0.
    ln_n = math.log(n)
    for i, a in enumerate(a_sizes):
        for j, b in enumerate(b_sizes):
            log_denominator = math.log(a) + math.log(b)
            constant = (log_factorial[a] + log_factorial[b] + log_factorial[n-a]
                        + log_factorial[n-b] - log_factorial[n])
            weight = a_counts[i] * b_counts[j]
            for overlap in range(max(1, a+b-n), min(a,b) + 1):
                log_probability = constant - (
                    log_factorial[overlap] + log_factorial[a-overlap]
                    + log_factorial[b-overlap] + log_factorial[n-a-b+overlap]
                )
                term = (overlap / n) * (ln_n + math.log(overlap) - log_denominator)
                expectation += term * math.exp(log_probability) * weight
    return expectation


def entropy(counts):
    mass = counts[counts > 0].astype(np.float64)
    return float(-np.sum((mass / mass.sum()) * np.log(mass / mass.sum())))


def adjusted_info(y, pred):
    c = contingency_matrix(y, pred, sparse=True)
    a, b = np.asarray(c.sum(axis=1)).ravel(), np.asarray(c.sum(axis=0)).ravel()
    if (len(a) == len(b) == 1) or (len(a) == len(b) == len(y)):
        return 1.
    if len(a) == 1 or len(b) == 1:
        return 0.
    n = len(y)
    a_sizes, a_counts = np.unique(a, return_counts=True)
    b_sizes, b_counts = np.unique(b, return_counts=True)
    expected = expected_info(n,a_sizes,a_counts,b_sizes,b_counts,gammaln(np.arange(n+1,dtype=float)+1))
    mi = mutual_info_score(None,None,contingency=c)
    normalizer = (entropy(a) + entropy(b)) / 2
    eps = np.finfo(np.float64).eps
    num, den = mi-expected, normalizer-expected
    return float(math.copysign(max(abs(num),eps),num) / math.copysign(max(abs(den),eps),den))
