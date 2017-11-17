def test1(file, n_iter=2000, n_seed=10):
    import pca
    d = data().toy()

    config = pca.probPCA.configure()
    for i, kv in enumerate([
            {'W': 'prior'},
            {'Z': 'prior'},
            {'W': 'prior', 'Z': 'prior'},
            {'W': 'all'},
            {'Z': 'all'},
            {'W': 'all', 'Z': 'all'}
    ]):
        for j, conf in enumerate([
                [False, tf.ones_initializer],
                [True, tf.ones_initializer],
                [True, tf.random_normal_initializer],
        ]):
            c = config.copy()
            for k, v in kv.items():
                c.loc[('prior', k, 'scale'), :] = conf
            for s in range(n_seed):
                p = pca.probPCA(d.x1, config=c, seed=s, covariance=i, initialization=j, **kv)
                p.run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp1'] = p.losses.replace('None', np.nan)

def test2(file, n_iter=2000, n_seed=10):
    import pca
    d = data().toy()

    config = pca.probPCA.configure()
    for i, mu in [(1, 'full')]: #enumerate(['none', 'full']):
        for j, mu_loc in enumerate({
            'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
            'full': [
                [False, 'data_mean'], # prior mean set to data mean
                [True, 'data_mean']   # prior mean a hyperparamter
            ]
        }[mu][1:]):
            j = 1
            for k, mu_scale in enumerate({
                    'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                    'full': [
                        [False, tf.ones_initializer],
                        [True, tf.ones_initializer],
                        [True, tf.random_normal_initializer]
                    ]
            }[mu]):
                for l, tau in enumerate(['none', 'full']):
                    for m, tau_loc in enumerate({
                            'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                            'full': [
                                [False, tf.zeros_initializer],
                                [True, tf.zeros_initializer],
                                [True, tf.random_normal_initializer]
                            ]
                    }[tau]):
                        for n, tau_scale in enumerate({
                                'none': [[None, None]], # irrelevant since the 'posterior' value is used in config
                                'full': [
                                    [False, tf.ones_initializer],
                                    [True, tf.ones_initializer],
                                    [True, tf.random_normal_initializer]
                                ]
                        }[tau]):
                            c = config.copy()
                            c.loc[('prior', 'mu', 'loc'), :] = mu_loc
                            c.loc[('prior', 'mu', 'scale'), :] = mu_scale
                            c.loc[('prior', 'tau', 'loc'), :] = tau_loc
                            c.loc[('prior', 'tau', 'scale'), :] = tau_scale

                            for s in range(n_seed):
                                p = pca.probPCA(d.x1, seed=s, mu=mu, tau=tau, config=c, i=i, j=j, k=k, l=l, m=m, n=n)
                                p.run(n_iter).critique(d)

                            with pd.HDFStore(file) as S:
                                S['exp2'] = p.losses.replace('None', np.nan)

def test3(file, n_iter=20000, n_seed=30):
    import pca
    d = data().toy()

    for s in range(n_seed):
        p = pca.probPCA(d.x1, seed=s, covariance=0).run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp3'] = p.losses.replace('None', np.nan)

    c = pca.probPCA.configure()
    c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
    c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]

    for s in range(n_seed):
        p = pca.probPCA(d.x1, seed=s, config=c, W='all', Z='all', covariance=1)
        p.run(n_iter).critique(d)

    with pd.HDFStore(file) as S:
        S['exp3'] = p.losses.replace('None', np.nan)

def test4(file, n_iter=20000, n_seed=30):
    import pca

    for s in range(n_seed):
        d = data().toy()
        c = pca.probpca.configure()

        p = pca.probpca(d.x1, covariance=0, config=c).run(n_iter).critique(d)

        c.loc[('prior', 'w', 'scale'), :] = [true, tf.random_normal_initializer]
        c.loc[('prior', 'z', 'scale'), :] = [true, tf.random_normal_initializer]

        p = pca.probpca(d.x1, config=c, w='all', z='all', covariance=1)
        p.run(n_iter).critique(d)

        with pd.hdfstore(file) as s:
            s['exp4'] = p.losses.replace('none', np.nan)

def test5(file='test5.h5', n_iter=20000, n_seed=10, n_data=10):
    import pca

    for i in range(n_data):
        d = data().toy()
        for s in range(n_seed):
            for conv in ['data', 'ed']:
                c = pca.probPCA.configure()

                p = pca.probPCA(d.x1, covariance=0, config=c, seed=s, conv=conv)
                p.run(n_iter, conv).critique(d)

                c.loc[('prior', 'W', 'scale'), :] = [True, tf.random_normal_initializer]
                c.loc[('prior', 'Z', 'scale'), :] = [True, tf.random_normal_initializer]

                p = pca.probPCA(d.x1, config=c, W='all', Z='all', seed=s, covariance=1, conv=conv)
                p.run(n_iter, conv).critique(d)

                with pd.HDFStore(file) as S:
                    s['test5'] = p.losses.replace('none', np.nan)
