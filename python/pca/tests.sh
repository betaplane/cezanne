#!/bin/bash

# python -c 'import tests; tests.data_loss_vs_elbo()'

c=0
while [ $c == 0 ]; do
    c=`python -c 'import tests; tests.Test("convergence.h5", "data_loss_vs_elbo").run()'`
done
