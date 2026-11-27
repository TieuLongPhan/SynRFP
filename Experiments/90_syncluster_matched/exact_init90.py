"""Speed up k-means++ initialization for integer matrices using exact distances."""
from contextlib import contextmanager

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.cluster import _kmeans


@contextmanager
def integer_init(X):
    """Keep the sklearn RNG and update path, accelerating only initialization dot products.

    Every possible squared distance is an integer below 2**24. The norm and dot
    arithmetic are therefore exact in float32 under the checked bound.
    """
    bound = 4*np.einsum("ij,ij->i",X,X,dtype=np.float64).max(initial=0)
    if bound >= 2**24 or not np.equal(np.rint(X), X).all():
        raise ValueError("Integer distance arithmetic is not provably exact")
    original = _kmeans._euclidean_distances
    cached_y = None
    transposed = None
    norm_y = None

    def distances(a, b, X_norm_squared=None, Y_norm_squared=None, squared=False):
        nonlocal cached_y, transposed, norm_y
        if not squared or not isinstance(a,np.ndarray) or not isinstance(b,np.ndarray):
            return original(a,b,X_norm_squared,Y_norm_squared,squared=squared)
        assert a.dtype == b.dtype == np.float32
        if b is not cached_y:
            cached_y = b
            transposed=csr_matrix(b).T.tocsr()
            norm_y=np.einsum("ij,ij->i",b,b)
            if Y_norm_squared is not None:
                np.testing.assert_array_equal(np.asarray(Y_norm_squared).ravel(),norm_y)
            probe=a[:min(3,len(a))]
            direct=original(probe,b,Y_norm_squared=Y_norm_squared,squared=True)
            exact=np.einsum("ij,ij->i",probe,probe)[:,None]+norm_y[None,:]
            exact-=2*(csr_matrix(probe)@transposed).toarray()
            np.testing.assert_array_equal(direct,exact)
        norm_x=np.einsum("ij,ij->i",a,a) if X_norm_squared is None else np.asarray(X_norm_squared).ravel()
        return norm_x[:,None]+norm_y[None,:]-2*(csr_matrix(a)@transposed).toarray()

    _kmeans._euclidean_distances = distances
    try:
        yield
    finally:
        _kmeans._euclidean_distances = original
