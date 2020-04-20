# -*- coding: utf-8 -*-
"""
Module to implement the clusterwise predictive modelling algorithm using
linear regression to model the clusters, and the kernel smoothing method to model
the cluster probability functions.
"""

# Import libraries.
import numpy as np
import math
from scipy.spatial import distance_matrix


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
    with np.errstate(divide='ignore', invalid='ignore'):
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
              auto_labda=False,
              c=0.9,
              gamma=1,
              random_seed=None):
        self.k = k
        self.kernel= kernel
        self.lab = labda
        self.fitted = False
        self.max_iter = max_iter
        self.pre_cluster = pre_cluster
        self.cluster_method = cluster_method
        self.fuzzy = fuzzy
        self.lab_array = False
        self.auto_labda = auto_labda        
        self.c = c
        self.gamma = gamma
        if isinstance(random_seed, int):
            self.random_seed = random_seed
        else:
            self.random_seed = np.random.randint(0,10000)
        
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
        elif self.auto_labda:
            self.lab_array = True
            lab_opt = self.lab
            same = 0
        else:
            # In case labda is considered a constant, define the matrix K.
            K = self.kernels[self.kernel](dist, self.lab)
            
        # Create matrix A.
        A = np.concatenate((np.ones((self.n,1)), x_arr), axis = 1)
        
        # Create placeholder array for the estimates of beta.
        beta_est = np.zeros((self.m+1 ,self.k))
        
        # Set random seed.
        np.random.seed(self.random_seed)  
 
        # Set initial values of p_est, either using a cluster method or
        # set them equel plus a small perturbation.       
        if self.pre_cluster == True:
            clust = self.cluster_method.fit(x_arr)
            # Check if the cluster method performs fuzzy clustering.
            if self.fuzzy:
                p_est = clust.u
            else:
                for i in range(self.k):
                    if i == 0:
                        p_est = (clust.labels_ == i).reshape((self.n, 1))
                    else:
                            p_est = np.append(p_est, (clust.labels_ == i).reshape((self.n, 1)), axis=1)
        else:
            p_est = np.random.uniform(size=(self.n,self.k))
            p_est /= np.sum(p_est, axis=1)[:,None]
            
        # Set initial log likelihood.    
        log_likeli_new = -float('Inf')
        
        # Main loop.
        for i in range(self.max_iter):
            
            # M step: calculate estimates for beta, sigma and h.
            for j in range(self.k):
                beta_est[:,j] = np.linalg.multi_dot([np.linalg.inv(np.dot(A.T, p_est[:,j][:,None] * A)),
                        A.T, p_est[:,j][:,None] * y_arr]).reshape((self.m+1,))
                
            sig_est = np.sqrt(np.sum(p_est * np.square(y_arr - np.dot(A, beta_est)), axis=0) / np.sum(p_est, axis=0))
            
            # loocv via L matrices.
            if self.lab_array:
                
                # Define labda values to be tested.
                if self.auto_labda:
                    test_labs = np.array([lab_opt / (1 + self.gamma * self.c**same),
                                          lab_opt,
                                          lab_opt * (1 + self.gamma * self.c**same)])
                    
                # Create array for cv scores.
                mses = np.array([])
                
                # For each labda value, calcuate the loocv score.
                for labi in test_labs:
                    H = []
                    H_diag = []
                    for j in range(self.k):
                        H_j = np.linalg.multi_dot([A, np.linalg.inv(np.dot(A.T, p_est[:,j][:,None] * A)),
                                               p_est[:,j][:,None].T * A.T])
                        H_diag_j = np.diag(H_j)
                        H.append(H_j)
                        H_diag.append(H_diag_j)
                    K = self.kernels[self.kernel](dist, labi)
                    self.testk = K
                    K_L = K / np.sum(K, axis=1)[:,None]
                    h_reg = np.diag(K_L)
                    preds = np.array([])
                    for s in range(self.n):
                        L_loo = 0
                        h_loo = np.dot(np.delete(K_L[s,:], s)[None,:],
                             np.delete(p_est, s, 0))
                        for j in range(self.k):
                            H_j_loo = np.dot(np.delete(H[j][s,:], s)[None,:],
                                 np.delete(y_arr, s, 0))[0,0] / (1 - H_diag[j][s])
                            L_j_loo = H_j_loo * h_loo[0,j]
                            L_loo += L_j_loo
                        pred = L_loo  / (1 - h_reg[s])
                        preds = np.append(preds, pred)
                    mse = ((preds - y)**2).mean()
                    mses = np.append(mses, mse)
                mses[np.isnan(mses)] = float('Inf')
                if lab_opt == test_labs[mses.argmin()]:
                    same += 1
                else:
                    same -= 0.5   
                    lab_opt = test_labs[mses.argmin()] 
                    
                # Calculate new K matrix.    
                K_opt = K_Gauss(dist, lab_opt)
                
                # Calculate new estimates of h.
                h_est = h_est_fun(K_opt, p_est)
            else:
                h_est = h_est_fun(K, p_est)
                
            # E step: calculate estimates of p.
            prop = h_est / sig_est * np.exp(-np.square(y_arr - np.dot(A, beta_est)) / (2 * sig_est**2))
            prop_sum = np.sum(prop, axis=1)
            p_est = prop / prop_sum[:,None]
            
            # Save old and calculate new log likelihood.
            log_likeli_old = log_likeli_new
            log_likeli_new = np.sum(-np.sum(p_est, axis=0) * np.log(2 * math.pi * sig_est**2) -\
                            np.sum(p_est * np.square(y_arr - np.dot(A, beta_est)), axis=0) /\
                            (2 * sig_est**2)) + np.sum(np.log(h_est + 1e-15) * p_est)

            # Stop if increase in likelihood is below threshold.
            if log_likeli_new - log_likeli_old < 1e-5 and log_likeli_new - log_likeli_old > -1e-5 and\
                i >= 10:
                break
            
        # Assign self parameters.        
        self.iters = i+1
        self.beta_est = beta_est
        self.p_est = p_est
        self.sig_est = sig_est
        self.h_est = h_est
        self.x = x
        self.y = y
        if self.lab_array:
            self.lab = lab_opt
            self.mse = mses.min()
            self.mses = mses
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
                A_uni = np.concatenate((np.ones((len(x_pred),1)),
                                        x_pred.reshape((len(x_pred),self.m))), axis = 1) 
                f_est =  np.dot(A_uni, self.beta_est)
                pred = np.sum(f_est*h_final, axis=1)
                return pred
            else:
                raise ValueError("Incorrect number of dimensions.")
        else:
            raise ValueError("Train the model first.")
