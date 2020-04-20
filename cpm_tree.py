# -*- coding: utf-8 -*-
"""
Module to implement the clusterwise predictive modelling algorithm using
decision trees to model the clusters, and the kernel smoothing method to model
the cluster probability functions.
"""

# Import libraries.
import numpy as np
from scipy.spatial import distance_matrix
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import KFold


# Define kernel functions.
def K_Gauss(dist, lab):
    return np.exp(-(dist / lab)**2) 

def K_block(dist, lab):
    return np.array(dist <= lab, dtype = "int")

def K_Epan(dist, lab):
    K = np.zeros((len(dist), len(dist[0])))
    values = dist[dist <= lab]
    K[dist <= lab] = 1 - (values / lab)**2
    return K 


# Define cluster probabillity function estimate.
def h_est_fun(K, p):
    with np.errstate(divide='ignore',invalid='ignore'):
        return np.dot(K, p) / np.sum(K, axis=1)[:,None] 

class lin_mix:
    """
    Main class, fit a clusterwise predictive model to a training data set,
    and make predictions on a new data set.
    """
 
    # Define dictionary of kernel functions.
    kernels = {"Epanechnikov" : K_Epan, "Gauss" : K_Gauss, "block" : K_block}

    
    def __init__(self,
              k=2,
              kernel="Epanechnikov",
              labda=1,
              max_iter=100,
              pre_cluster=False,
              cluster_method=None,
              fuzzy=False,
              auto_param=False,
              c=0.9,
              gamma=1,
              max_leaf_nodes=5,
              min_samples_leaf=1):
        self.k = k
        self.kernel= kernel
        self.lab = labda
        self.fitted = False
        self.max_iter = max_iter
        self.pre_cluster = pre_cluster
        self.cluster_method = cluster_method
        self.fuzzy = fuzzy
        self.lab_array = False
        self.auto_param = auto_param       
        self.c = c
        self.gamma = gamma
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        
    def fit(self, x, y):
        
        # Find the number of rows in the dataset.
        self.n = len(x)
        
        #Find the number of collumns in the dataset.
        if x.ndim == 1:
            self.m = 1
        else:
            self.m = np.shape(x)[1]
            
        # Turn x and y into 2-d arrays.
        x_arr = x.reshape((self.n,self.m))
        y_arr = y.reshape((self.n,1))
        
        # Check if user provided a correct kernel function.
        if self.kernel not in self.kernels:
            raise ValueError("Not a valid kernel")
            return self  
        
        # Compute distances between points in training set.
        dist = distance_matrix(x_arr, x_arr)
        
        # Set initial labda value(s).
        if  hasattr(self.lab, "__len__"):
            test_labs = self.lab
            self.lab_array = True
        elif self.auto_param:
            self.lab_array = True
            lab_opt = self.lab
            size_opt = self.max_leaf_nodes
            same = 0
        else:    
            K = self.kernels[self.kernel](dist, self.lab)
            max_leaf = self.max_leaf_nodes
            min_samples = self.min_samples_leaf
        
        # Define placeholder loss array
        loss = np.zeros((self.n,self.k))

        # Set initial values of p_est, either using a cluster method or
        # set them equel plus a small perturbation.  
        if self.pre_cluster == True:
            clust = self.cluster_method.fit(x_arr)
            if self.fuzzy:
                p_est = clust.u
            else:
                for i in range(self.k):
                    if i == 0:
                        p_est = (clust.labels_ == i).reshape((self.n, 1))
                    else:
                        p_est = np.append(p_est, (clust.labels_ == i).reshape((self.n, 1)), axis=1)
        else:
            p_est = np.ones((self.n,self.k-1)) / self.k + np.random.uniform(low=-0.01, high=0.01, size=(self.n,self.k-1))
            p_est = np.append(p_est, (1 - np.sum(p_est, axis=1)).reshape((self.n,1)), axis=1)
            
        # Define k-fold object.    
        kf = KFold(n_splits=5, shuffle=True, random_state=42)  
        
        # Set initial total loss 
        total_loss_new = float('Inf')
        
        # Main loop.
        for i in range(self.max_iter):
                
            # Cross validation.
            
            # Define labda values to be tested.
            test_labs = np.array([lab_opt / (1 + self.gamma * self.c**same),
                                  lab_opt,
                                  lab_opt * (1 + self.gamma * self.c**same)])
            
            # Define number of nodes to be tested.
            test_size = np.array([max(2,size_opt - 1), size_opt, size_opt + 1], dtype=int)
            
            # Create array for cv scores.
            mses = np.array([])
            
            # For each labda and size value, calcuate the cv score.            
            for labi in test_labs:
                mses_size = np.array([])
                for size in test_size:
                    mse_arr = []
                    for train_index, test_index in kf.split(x_arr, y_arr):
                        x_train, x_test = x_arr[train_index], x_arr[test_index]
                        y_train, y_test = y_arr[train_index], y_arr[test_index]
                        f_cv = np.zeros((len(x_test), self.k))
                        for j in range(self.k):
                            tree_cv = DecisionTreeRegressor(max_leaf_nodes=size, random_state=0)
                            tree_cv.fit(x_train, y_train, sample_weight=p_est[:,j][train_index])
                            f_cv[:,j] = tree_cv.predict(x_test)
                        dist_cv = dist[test_index[:,None], train_index]
                        K_cv = K_Gauss(dist_cv, labi)
                        h_cv = h_est_fun(K_cv, p_est[train_index,:])    
                        pred_cv = np.sum(f_cv*h_cv, axis=1)
                        mse_cv = ((y_test.reshape((len(y_test),)) - pred_cv)**2).mean()
                        mse_arr.append(mse_cv)
                    mse_mean = np.array(mse_arr).mean()
                    mses_size = np.append(mses_size, mse_mean)
                size_opt = test_size[mses_size.argmin()]
                mses = np.append(mses, mses_size.min())

            if lab_opt == test_labs[mses.argmin()]:
                same += 1
            else:
                same -= 0.5   
                lab_opt = test_labs[mses.argmin()]     

            # Calculate new K matrix.    
            K_opt = K_Gauss(dist, lab_opt)
            
            # Calculate new estimates of h.
            h_est = h_est_fun(K_opt, p_est)
                
            # M step: find the k trees.
            trees = []
            for j in range(self.k):
                tree = DecisionTreeRegressor(max_leaf_nodes=size_opt, random_state=0)
                tree.fit(x_arr, y_arr, sample_weight=p_est[:,j])
                preds = tree.predict(x_arr)
                loss[:,j] = (y.reshape((self.n,)) - preds)**2
                trees.append(tree)
                
            # E step: calculate estimates of p.
            prop = h_est  * np.exp(-loss)
            prop_sum = np.sum(prop, axis=1)
            p_est = prop / prop_sum[:,None]
            p_est[np.isnan(p_est)] = 1 / self.k
                       
            # Save old and calculate new total loss.
            total_loss_old = total_loss_new
            total_loss_new = np.sum(p_est * loss) - np.sum(np.log(h_est + 1e-15) * p_est)

            # Stop if decrease in total loss is below threshold.
            if total_loss_new - total_loss_old < 1e-5 and total_loss_new - total_loss_old > -1e-5 and\
                i >= 10:
                break
            
        # Assign self parameters.               
        self.iters = i+1
        self.p_est = p_est
        self.h_est = h_est
        self.x = x
        self.y = y
        self.trees = trees
        if self.lab_array:
            self.lab = lab_opt
            self.size = size_opt
        self.fitted = True

        
    def predict(self, x):
        
                # Make sure the dataset has the right shape.
        if not hasattr(x, "__len__"):
            x_pred = np.array([x]).reshape((1,1))
        else:
            x_pred = x
            if x_pred.ndim == 1:
                x_pred = x_pred.reshape((1,len(x_pred)))
                
        # If the dataset has the right shape and the model has been trained, make predictions.                 
        if self.fitted:
            if x_pred.shape[1] == self.m:       
                dist_pred = distance_matrix(x_pred, self.x)
                K_pred = self.kernels[self.kernel](dist_pred, self.lab)
                h_final = h_est_fun(K_pred, self.p_est)
                h_final[np.isnan(h_final)] = 1 / self.k
                f_est = np.zeros((len(x_pred),self.k))
                for i in range(self.k):   
                    f_est[:,i] = self.trees[i].predict(x_pred)
                pred = np.sum(f_est*h_final, axis=1)
                return pred
            else:
                raise ValueError("Incorrect number of dimensions.")
        else:
            raise ValueError("Train the model first.")

