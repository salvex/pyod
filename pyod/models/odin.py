# -*- coding: utf-8 -*-
"""ODIN (Outlier Detection with Indegree Number using k-Nearest Neighbors Graph)
"""
# Author: Salvatore Lucio Auria <salvo.auria@gmail.com> (code author)
#         Ville Hautamäki <villeh@cs.joensuu.fi> (paper author)
#         Ismo Kärkkäinen <iak@cs.joensuu.fi> (paper author)
#         Pasi Fränti <franti@cs.joensuu.fi> (paper author)
#         
# License: BSD 2 clause

from __future__ import division
from __future__ import print_function

from warnings import warn

import numpy as np
from numpy import percentile
from sklearn.neighbors import BallTree
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted


from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

#TODO: implement MeanDIST variant
#TODO: implmenet KDIST variant

from .base import BaseDetector


class ODIN(BaseDetector):
    """ODIN (Outlier Detection using Indegree Number).
    First, a kNN graph is created for dataset (X parameter). Then, if vertex has an indegree of T (contamination parameter) or 
    less, mark it as an outlier and otherwise mark it as an inlier. 
    
    the algorithm used to compute the nearest neighbors is the BallTree method
    See https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.BallTree.html#sklearn.neighbors.BallTree for details
    
    See :cite:`hautamäki2004odin` for details.
    
    
    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set,
        i.e. the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function. In this specific algorithm, contamination
        is the T indegree parameter
        
    n_neighbors : int, optional (default = 5)
        Number of neighbors to use by default for k neighbors queries.
    
    mode : str (default='auto')
        {'auto','manual'}
        
        - 'auto' : calculates the indegree threshold T using 'T = max(L(X[i])-L(X[i-1])) * t' formula. See :cite:`hautamäki2004odin` for details.
        - 'manual': uses user-defined indegree threshold T (indegree_t)
        
    indegree_t : int (default='1')
        if mode is 'manual' the algorithm uses the user-defined indegree_t
    
    method : str, optional (default='largest')
        {'largest', 'mean', 'median'}

        - 'largest': use the distance to the kth neighbor 
        - 'mean': use the average of all k neighbors 
        - 'median': use the median of the distance to k neighbors
    
    radius : float, optional (default=1.0)
        Range of parameter space to use by default for radius_neighbors queries.

    leaf_size: int, optional (default=30)
        Leaf size passed to BallTree or KDTree. This can affect the speed of the construction and query, as well as 
        the memory required to store the tree. 
        The optimal value depends on the nature of the problem.
        
        The amount of memory needed to store the tree scales as approximately ``n_samples / leaf_size``. 
        For a specified ``leaf_size``, a leaf node is guaranteed to satisfy 
        ``leaf_size <= n_points <= 2 * leaf_size``, except in the case that n_samples < leaf_size.
        

    metric: str or callable, default= ’minkowski’
        Metric to use for distance computation. Default is “minkowski”, which results in the standard Euclidean distance when p = 2. 
        
        See the documentation of scipy.spatial.distance and the metrics listed in distance_metrics for valid metric values.
        
        If metric is “precomputed”, X is assumed to be a distance matrix and must be square during fit.
        
        X may be a sparse graph, in which case only “nonzero” elements may be considered neighbors.
        
        If metric is a callable function, it takes two arrays representing 1D vectors as inputs and must return one value indicating the distance between those vectors. This works for Scipy’s metrics, but is less efficient than passing the metric name as a string.

        Distance matrices are not supported.

        Valid values for metric are:

        - from scikit-learn: ['cityblock', 'euclidean', 'l1', 'l2',
          'manhattan']

        - from scipy.spatial.distance: ['braycurtis', 'canberra', 'chebyshev',
          'correlation', 'dice', 'hamming', 'jaccard', 'kulsinski',
          'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto',
          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath',
          'sqeuclidean', 'yule']

        See the documentation for scipy.spatial.distance for details on these
        metrics.
        
        
    p: float (positive), default=2
        Parameter for the Minkowski metric from sklearn.metrics.pairwise.pairwise_distances. 
        When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. For arbitrary p, minkowski_distance (l_p) is used.
        
        See http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances
   
    metric_params: dict, default=None
        Additional keyword arguments for the metric function.

    n_jobs: int, default=None
        The number of parallel jobs to run for neighbors search. None means 1 unless in a joblib.parallel_backend context. -1 means using all processors. See Glossary for more details.
    
    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.    
    """

    def __init__(self, contamination=0.1, n_neighbors=10, mode='auto', indegree_t=1, method='largest',
                radius=1.0,
                leaf_size=30, metric='minkowski', p=2, metric_params=None, 
                n_jobs=1, **kwargs):
        super(ODIN,self).__init__(contamination=contamination)
        self.n_neighbors=n_neighbors
        self.mode = mode
        self.indegree_t = indegree_t
        self.method = method
        self.radius = radius 
        self.leaf_size = leaf_size
        self.metric = metric 
        self.p = p
        self.t = 1 - self.contamination
        self.metric_params = metric_params
        self.n_jobs = n_jobs
        
        if self.mode != 'auto' or self.mode != 'manual': 
            warn('There are only "auto" and "manual" mode'
                 ' in this run, "auto" will be used.',
                 RuntimeWarning)
        
    
        self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                    radius=self.radius,
                                    algorithm='ball_tree',
                                    leaf_size=self.leaf_size,
                                    metric=self.metric,
                                    p=self.p,
                                    metric_params=self.metric_params,
                                    n_jobs=self.n_jobs,
                                    **kwargs)
        
        
    def fit(self, X, y=None):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        
        
        self._set_n_classes(y)
        
        self.tree_ = BallTree(X)
        
        if self.metric_params is not None:
            self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                      metric=self.metric,
                                      **self.metric_params)
        else:
            self.tree_ = BallTree(X, leaf_size=self.leaf_size,
                                    metric=self.metric)
        
        
        self.neigh_.fit(X)
        
        #Use kneighbors algorithm to find the distances across all datapoints
        
        all_dist, _ = self.neigh_.kneighbors(n_neighbors=self.n_neighbors,
                                             return_distance=True)
        
        #Apply the selected distance method (largest,mean or median) to obtain l-dist value
        l_dist = self._get_dist_by_method(all_dist)   
        
        self.l_dist = l_dist
        
        #print(np.mean(l_dist))
        
        indegree_scores = np.zeros([X.shape[0]])
        
        for i in range(all_dist.shape[0]):
            dist_point_features = np.asarray(all_dist[i]).reshape(1, all_dist[i].shape[0])
            #dist_point = self._get_dist_by_method(dist_point_features)
            indegree_score = np.sum(dist_point_features > np.mean(l_dist), axis=1)
            
            #print(indegree_score)
            
            
            indegree_scores[i] = indegree_score
      
        
        self.decision_scores_ = indegree_scores.ravel()
        
        self._process_decision_scores()
                    
        return self
    
    def _calc_threshold(self):
        """Internal function to calculate indegree threshold T
        
        T = max(L(X[i])-L(X[i-1])) * t 
        
        where
        - L is the distance function (largest,mean or median) applied on the i-eth data point
        - t is the indegree threshold importance (1 - contamination)
        
        (L(X) is self.tot_dist,the value obtained by applying _gest_dist_by_method 
        on the distances found by kNN algorithm)
        
        Parameters
        ----------
        
        - l_dist: distance vector of samples with L function applied 
        - t: indegree threshold importance hyperparameter
        
        Returns 
        ------- 
        indegree_t: indegree_threshold
        
        """
        #TODO: complete method
        
        if self.l_dist.shape[0] < 2: 
            raise ValueError("The number of data points is too small.")
            
        differences = [self.l_dist[i] - self.l_dist[i-1]
                       for i in range(1,self.l_dist.shape[0])]
            
        indegree_t = np.amax(differences) * self.t
        
        return indegree_t
        
    
    
    def _process_decision_scores(self):
        """Internal function to calculate key attributes:
        
        Lightly modified in order to achieve correct threshold value: 
        
        T = max(L(X[i])-L(X[i-1])) * contamination 
        
        where L is the distance function (largest,mean or median) applied on the i-eth data point
        
        (L(X) is self.l_dist,the value obtained by applying _gest_dist_by_method 
        on the distances found by kNN algorithm)
        
        Parameters
        ----------
        
        - threshold_: used to decide the binary label
        - labels_: binary labels of training data

        Returns
        -------
        self
        """

        if isinstance(self.contamination, (float, int)):
            
            self.threshold_ = 0
            
            if self.mode == 'auto': #auto mode invokes the internal method _calc_threshold
                self.threshold_ = self._calc_threshold()
                #TODO: remove (for debug reasons)
                #print(self.threshold_)
                self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                    'int').ravel()
            else:
                self.threshold_ = self.indegree_t
                #TODO: remove (for debug reasons)
                #print(self.threshold_)
                self.labels_ = (self.decision_scores_ > self.threshold_).astype(
                    'int').ravel()
        # if this is a PyThresh object
        else:
            self.labels_ = self.contamination.eval(self.decision_scores_)
            self.threshold_ = self.contamination.thresh_
            if not self.threshold_:
                self.threshold_ = np.sum(self.labels_) / len(self.labels_)

        # calculate for predict_proba()

        self._mu = np.mean(self.decision_scores_)
        self._sigma = np.std(self.decision_scores_)

        return self
    
    
    
    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['tree_', 'decision_scores_',
                               'threshold_', 'labels_'])

        X = check_array(X)
        
        #Find the kneighbors to calculate the distances across all datapoints
        
        all_dist, _ = self.tree_.query(X,k=self.n_neighbors)
        
        #Apply the selected distance method
        l_dist = self._get_dist_by_method(all_dist)        
        #dist = self._get_dist_by_method(dist_arr)
        #TODO: remove, only for testing reason
        #print(np.mean(l_dist))
        #update the l-value distance vector
        self.l_dist = l_dist
        
        #update the threshold
        self.threshold_ = self._calc_threshold()
        
        #Calculate the in-degree for each data point
        indegree_scores = np.zeros([X.shape[0]])
        
        # initialize the in_degree score
        for i in range(all_dist.shape[0]):
            dist_point_features = np.asarray(all_dist[i]).reshape(1, all_dist[i].shape[0])
            #dist_point = self._get_dist_by_method(dist_point_features)
            indegree_score = np.sum(dist_point_features > np.mean(l_dist), axis=1)
            #TODO: remove, only for testing reasons
            #print(indegree_score)
            indegree_scores[i] = indegree_score
        

        return indegree_scores.ravel()
    
         
    def _get_dist_by_method(self, dist_arr):
        """Internal function to decide how to process passed in distance array

        Parameters
        ----------
        dist_arr : numpy array of shape (n_samples, n_neighbors)
            Distance matrix.

        Returns
        -------
        dist : numpy array of shape (n_samples,)
            The outlier scores by distance.
        """

        if self.method == 'largest':
            return dist_arr[:, -1]
        elif self.method == 'mean':
            return np.mean(dist_arr, axis=1)
        elif self.method == 'median':
            return np.median(dist_arr, axis=1)
    
    
   
    
         
        