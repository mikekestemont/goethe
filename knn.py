#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright © 2017 CLiPS, University of Antwerp
#

from collections import Counter
from operator import itemgetter

import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class KNN(object):
    """K-nearest Neighbours classifier.

    Parameters
    ----------
    k : int, optional, default 3
        Number of neighbours to use for classification.

    metric : str, optional, default 'cosine'
        Metric used to calculate the distances.

    Returns
    -------
    y_pred : list
        List with predicted class labels.

    Examples
    --------
    >>> knn = KNN(k=1, metric='cosine')
    >>> knn.fit(X, y)
    >>> res = knn.predict(Xi)
    """

    def __init__(self, k=3, metric='cosine'):
        """Set class parameters."""
        self.k = k
        self.metric = metric

        self._fit_X = None
        self._classes = None

    def _check_space(self, X):
        """Make sure we unsparsify X and optionally calculate centroids."""
        if not isinstance(X, np.ndarray):
            try:
                X = np.array(X, dtype='float64')
            except ValueError as e:
                print(e)
                X = X.toarray()

        return X

    def fit(self, X, y):
        """Fit classifier distances with optional centroids."""
        X = self._check_space(X)
        self._fit_X = X
        self._classes = y
        return self

    def predict(self, Xi):
        """Predict label y using matrix distances up to some k neighbours."""
        Xi = self._check_space(Xi)
        dists = cdist(Xi, self._fit_X, metric=self.metric)

        if self.k > 1:
            nn_idxs = dists.argsort(axis=1)[:, :self.k]
            y_pred = [Counter([self._classes[idx] for idx in nn]  # maj vote
                              ).most_common(1)[0][0] for nn in nn_idxs]

        else:  # assume k = 1
            nn_idxs = np.argmin(dists, axis=1)
            y_pred = [self._classes[idx] for idx in nn_idxs]

        return y_pred

    def neighborhood(self, Xi):
        Xi = self._check_space(Xi)
        dists = cdist(Xi, self._fit_X, metric=self.metric)

        nn_idxs = dists.argsort(axis=1)[:, :self.k]
        nn_dists = np.sort(dists, axis=1)[:, :self.k]

        return nn_idxs, nn_dists

    def all_distances(self, Xi):
        Xi = self._check_space(Xi)
        dists = cdist(Xi, self._fit_X, metric=self.metric)
        return dists
