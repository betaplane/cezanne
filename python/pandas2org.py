from IPython.display import publish_display_data
from pandas import MultiIndex

def table(df):
    """Format a small :class:`~pandas.DataFrame` as an `org-mode table<https://orgmode.org/manual/Tables.html>`_.

    :param df: input DataFrame
    :type df: :class:`~pandas.DataFrame`
    :returns: org-mode table as IPython display string with 'text/org' MIME type

    """
    def index(idx):
        if isinstance(idx, MultiIndex):
            x = list(idx)
            return [x[0]] +[[' ' if x[i][j] == z else z for j, z in enumerate(y)]
                            for i, y in enumerate(x[1:])]
        else:
            return [[i] for i in idx]

    idx = index(df.index)
    cols = index(df.columns)
    M = df.as_matrix()
    s = '|\n|'.join('|'.join(' ' for _ in range(len(idx[0]))) + '|' + \
                          '|'.join(c[i] for c in cols) for i in range(len(cols[0]))) + \
        '|\n|' + '|'.join('-' for _ in range(len(idx[0]) + len(M[0]))) + '|\n|' + \
        '|\n|'.join('|'.join(str(i) for j in z for i in j) for z in zip(idx, M))
    return publish_display_data({'text/org': '|' + s + '|'})
