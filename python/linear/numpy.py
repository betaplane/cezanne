import numpy as np

def lsq(X, Y, b0=True):
    """Least squares solution directly from numpy.

    :param X: Feature matrix with features in columns, examples (N) in rows.
    :param Y: Label vector (or matrix)
    :param b0: If ``True``, include intercept, otherwise don't.
    :type b0: :obj:`bool`
    :returns: Dictonary labeled b\* where \* is 0 for intercept and 1...N for coefficients.
    :rtype: :obj:`dict`

    """
    if b0:
        b = np.linalg.lstsq(np.r_['1,2', np.ones((X.shape[0], 1)), X], Y)[0]
    else:
        b = np.linalg.lstsq(X, Y)[0]
    return dict(
        zip(['b{}'.format(i) for i in np.arange(b.shape[0]) + (1 - b0)], b))
