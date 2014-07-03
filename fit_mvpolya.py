#!/usr/bin/env python

## fit_nbinom
# Copyright (C) 2014 Gokcen Eraslan
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import numpy as np
from scipy.special import gammaln
from scipy.optimize import fmin_l_bfgs_b as optim


def fit_mvpolya(X, initial_params=None):
    infinitesimal = np.finfo(np.float).eps

    def log_likelihood(params, *args):
        alpha = params
        X = args[0]

        res = np.sum([np.sum(gammaln(row+alpha)) \
                - np.sum(gammaln(alpha)) \
                + gammaln(np.sum(alpha)) \
                - gammaln(np.sum(row + alpha)) \
                + gammaln(np.sum(row)+1) \
                - np.sum(gammaln(row+1)) for row in X])

        return -res

    if initial_params is None:
        #initial_params = np.zeros(X.shape[1]) + 1.0
        initial_params = np.mean(X, 0) + infinitesimal

    bounds = [(infinitesimal, None)] * X.shape[1]
    optimres = optim(log_likelihood,
                     x0=initial_params,
                     args=(X,),
                     approx_grad=1,
                     bounds=bounds)

    params = optimres[0]
    return {'params': params}


if __name__ == '__main__':
    dir_params = np.array([1, 5, 3]*2)
    mult_params = np.random.dirichlet(dir_params, 1000)
    X = np.array([np.random.multinomial(10, x, size=1)[0] for x in mult_params])
    print("X row means:")
    print(np.mean(X, 0))
    print("Real params:")
    print(dir_params)
    print("Estimated params:")
    print(fit_mvpolya(X))
