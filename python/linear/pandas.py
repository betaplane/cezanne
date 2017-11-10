import pandas as pd
from . numpy import lsq

def lsq(*df, b0=True):
    """
    least squares regression: predictor in first column/argument, response in second
    returns [intercept (b0), slope (b1)]
    if b0=False, assumes zero intercept and returns [slope]

    See also :func:`.numpy.lsq`

    """
    df = pd.concat(df, axis=1, join='inner')
    X = df.dropna().as_matrix()
    if len(X):
        return lsq(X[:, :-1], X[:, -1])
    else:
        print("no overlap")
        return np.nan
