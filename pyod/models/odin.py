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
from sklearn.metrics import DistanceMetric
from sklearn.decomposition import PCA
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
        contributes 
        
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
                n_jobs=1):
        
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
        self.neigh_ = None
        self.tree_ = None
        
        if self.mode != 'auto' or self.mode != 'manual': 
            warn('There are only "auto" and "manual" mode'
                 ' in this run, "auto" will be used.',
                 RuntimeWarning)
            
        # covariance matrix V of X (set to None, if "mahalanobis distance" is used)
        if self.metric == 'mahalanobis':
            self.V = None
            # custom metric (used to calculate Mahalanobis distance)
            self.c_metric = None
        
        
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
        
        #TODO: DON'T USE MAHALANOBIS DISTANCE
        if self.metric == 'mahalanobis':
            #calculates the transposed covariance matrix for X
            self.V = np.cov(X, rowvar=False)
            #create the metric container for mahalanobis
            self.c_metric = DistanceMetric.get_metric(metric=self.metric, V=self.V)
            
            self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                        radius=self.radius,
                                        algorithm='brute',
                                        leaf_size=self.leaf_size,
                                        metric=self.c_metric,
                                        p=self.p,
                                        metric_params=self.metric_params,
                                        n_jobs=self.n_jobs,
                                        )
        else:
            
            self.neigh_ = NearestNeighbors(n_neighbors=self.n_neighbors,
                                        radius=self.radius,
                                        algorithm='ball_tree',
                                        leaf_size=self.leaf_size,
                                        metric=self.metric,
                                        p=self.p,
                                        metric_params=self.metric_params,
                                        n_jobs=self.n_jobs,
                                        )
            
            
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
        
        #Apply the selected distance method and update the l-value distance vector
        self.l_dist = self._get_dist_by_method(all_dist)
        
        self.threshold_ = self._calc_threshold()
        
        #Calculate the indegree score: the functionality is inverse than the original "indegree score"
        #due tu API consistency (higher the indegree score, higher the outlierness probability in pyOD)
        #indegree_ij = 1 if dist_i > mean(l_dist) else 0 
        #indegree_i = sum(j=i to dim(X)) indegree_ij 
        
        self.decision_scores_ = self._calc_indegree(all_dist)
        
        self._process_decision_scores()
                    
        return self
    
    def _norm_vector(self,v):
        
        min_val = np.min(v)
        max_val = np.max(v)
        
        return (v-min_val)/(max_val - min_val)
    
    def _calc_threshold(self):
        """Internal function to calculate indegree threshold T
        
        T = max(L(X[i])-L(X[i-1])) * t 
        
        where
        - L is the distance function (largest,mean or median) applied on the i-eth data point
        - t is the indegree threshold importance (1 - contamination)
        
        (L(X) is self.l_dist,the value obtained by applying _gest_dist_by_method 
        on the distances found by kNN algorithm)
        
        Parameters
        ----------
        
        - l_dist: distance vector of samples with L function applied 
        - t: indegree threshold importance hyperparameter
        
        Returns 
        ------- 
        indegree_t: indegree_threshold
        
        """        
        if self.l_dist.shape[0] < 2: 
            raise ValueError("The number of data points is too small.")
            
        differences = [self.l_dist[i] - self.l_dist[i-1]
                       for i in range(1,self.l_dist.shape[0])]
            
        indegree_t = np.amax(differences) * self.t
    
        return indegree_t
   
    
        
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
        
        if self.metric != 'mahalanobis':
            #Find the kneighbors to calculate the distances across all datapoints
            all_dist, _ = self.tree_.query(X,k=self.n_neighbors)
            
            #Apply the selected distance method and update the l-value distance vector
            self.l_dist = self._get_dist_by_method(all_dist)
            
            #update the threshold
            self.threshold_ = self._calc_threshold()
            
            return self._calc_indegree(all_dist).ravel()
            
        else:
            #Find the kneighbors to calculate the distances across all datapoints
            all_dist, _ = self.neigh_.kneighbors(X,n_neighbors=self.n_neighbors,return_distance=True)
       
            #Apply the selected distance method and update the l-value distance vector
            self.l_dist = self._get_dist_by_method(all_dist)
            
            #update the threshold
            self.threshold_ = self._calc_threshold()
            
            return self._calc_indegree(all_dist).ravel()


           
    def _calc_indegree(self,d_matrix):
        """ #TODO: comment
        
        Parameters
        ----------
        d_matrix : numpy array with shape (n_samples,n_neighbors)
        
        Distance Matrix
        
        Returns
        -------
        indegree_scores : indegree score for each data point (the calculation is modified in order to get 
        API consistency: "the higher outlierness is represented by high score for each data point")
        
        """
        indegree_scores = np.zeros([d_matrix.shape[0]])
        
        #print("mean distance: " + str(np.mean(self.l_dist)))
        
        for i in range(d_matrix.shape[0]):
            dist_point_features = np.asarray(d_matrix[i]).reshape(1, d_matrix[i].shape[0])
            #the indegree number functionality is the opp
            indegree_number = np.sum(dist_point_features > self.threshold_, axis=1)
            #TODO: remove
            #print(dist_point_features)
            #TODO: experimental
            
            #use the scores as "weights" of distances 
            #E.G: in(i)=w_i*mean(dist_features[i])
            
            if indegree_number == 0:
                indegree_scores[i] = self.contamination * np.mean(dist_point_features)
            else:
                indegree_scores[i] = indegree_number * np.mean(dist_point_features)
             
        return (indegree_scores).ravel()
         
    def _get_dist_by_method(self, dist_arr):
        """Internal function to decide how to process passed in distance array

        Parameters
        ----------
        dist_arr : numpy array of two possible shapes (n_samples, n_neighbors) or (n_samples,)
            Distance matrix or Distance vector

        Returns
        -------
        dist : numpy array of shape (n_samples,) or float64 value 
            The outlier scores by distance.
        """
        if len(dist_arr.shape) == 2:
            if self.method == 'largest':
                return dist_arr[:, -1]
            elif self.method == 'mean':
                return np.mean(dist_arr, axis=1)
            elif self.method == 'median':
                return np.median(dist_arr, axis=1)
        elif len(dist_arr.shape) == 1:
            if self.method == 'largest':
                return dist_arr[-1]
            elif self.method == 'mean':
                return np.mean(dist_arr, axis=0)
            elif self.method == 'median':
                return np.median(dist_arr, axis=0)
            
    
    
   
    
         
        