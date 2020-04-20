# -*- coding: utf-8 -*-
"""
Module to implement the clusterwise predictive modelling algorithm using
linear regression to model the clusters, and logistic regression to model
the cluster probability functions.
"""

# Import libraries.
import numpy as np
import math


# Define cluster probabillity function estimate.
def h_est_inc(gamma, A):
    # Pick minimum to prevent overflow errors 
    exp_mat = np.exp(np.minimum(np.dot(A, gamma), 700)) 
    return exp_mat / (1 + np.sum(exp_mat, axis = 1)[:,None])

class lin_mix:
    """
    Main class, fit a clusterwise predictive model to a training data set,
    and make predictions on a new data set.
    """
    
    def __init__(self,
              k=2,
              max_iter=100,
              max_logistic=20,
              pre_cluster=False,
              cluster_method=None,
              fuzzy=False,
              random_seed=None):
        self.k = k
        self.fitted = False
        self.max_iter = max_iter
        self.max_logistic = max_logistic
        self.pre_cluster = pre_cluster
        self.cluster_method = cluster_method
        self.fuzzy = fuzzy   
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
                p_inc = np.delete(p_est, -1, 1)
            else:
                for i in range(self.k):
                    if i == 0:
                        p_est = (clust.labels_ == i).reshape((self.n, 1))
                    else:
                            p_est = np.append(p_est, (clust.labels_ == i).reshape((self.n, 1)), axis=1)
                p_inc = np.delete(p_est, -1, 1)
        else:
            p_inc = np.ones((self.n,self.k-1)) / self.k + np.random.uniform(low=-0.01, high=0.01, size=(self.n,self.k-1))
            p_est = np.append(p_est, (1 - np.sum(p_est, axis=1)).reshape((self.n,1)), axis=1)

        # Set initial estimtes of gamma.
        gamma = np.zeros((self.m+1, self.k-1))
        
        # Create a 1-d array of gamma by appending each collumn below the first.
        gamma_vec = gamma.reshape(((self.m+1) * (self.k-1),))[:,None]
        
        # Set initial log likelihood.
        log_likeli_new = -float('Inf')
        
        # Main loop.
        for i in range(self.max_iter):
            
            # M step: calculate estimates for beta, sigma and h.
            for j in range(self.k):
                beta_est[:,j] = np.linalg.multi_dot([np.linalg.inv(np.dot(A.T, p_est[:,j][:,None] * A)),
                        A.T, p_est[:,j][:,None] * y_arr]).reshape((self.m+1,))
                
            sig_est = np.sqrt(np.sum(p_est * np.square(y_arr - np.dot(A, beta_est)), axis=0) / np.sum(p_est, axis=0))
            
            h_inc = h_est_inc(gamma, A)
            for l in range(self.max_logistic):
                dl1 = np.dot(A.T, p_inc - h_inc)
                dl1_vec = dl1.reshape(((self.m+1) * (self.k-1),), order="F")[:,None]
                dl2 = np.zeros(((self.m+1) * (self.k-1), (self.m+1) * (self.k-1)))
                for a in range(self.k-1):
                    dl2[a*(self.m+1):self.m+1 + a*(self.m+1), a*(self.m+1):self.m+1 + a*(self.m+1)] = -1 *\
                    np.dot(A.T, h_inc[:,a][:,None] * (1 - h_inc[:,a][:,None]) * A)
                    for b in range(a+1, self.k-1):
                        dl2[ b*(self.m+1):self.m+1 + b*(self.m+1), a*(self.m+1):self.m+1 + a*(self.m+1)] =\
                        dl2[a*(self.m+1) :self.m+1 + a*(self.m+1),  b*(self.m+1):self.m+1 + b*(self.m+1)] =\
                        np.dot(A.T, h_inc[:,a][:,None] * h_inc[:,b][:,None] * A)
                try:
                    gamma_vec = np.dot(np.linalg.inv(dl2), np.dot(dl2, gamma_vec) - dl1_vec)
                except:
                    print("singular matrix")
                    break
                gamma_test = gamma_vec.reshape(((self.m+1), (self.k-1)), order = "F")
                grad_norm = np.linalg.norm(dl1, ord=np.inf)
                if ((np.dot(A, gamma_test) > 700).sum(axis=1) > 0).sum() == 0 and grad_norm > 1e-5:
                    gamma = gamma_test
                    h_inc = h_est_inc(gamma, A)
                    h_est = np.concatenate((h_inc, 1 - np.sum(h_inc, axis=1)[:,None]), axis=1)
                else:
                    break
            h_est = np.concatenate((h_inc, 1 - np.sum(h_inc, axis=1)[:,None]), axis=1)
            
            # E step: calculate estimates of p.
            prop = h_est / sig_est * np.exp(-np.square(y_arr - np.dot(A, beta_est)) / (2 * sig_est**2))
            prop_sum = np.sum(prop, axis=1)
            p_est = prop / prop_sum[:,None]
            p_inc = np.delete(p_est, -1, 1)
            
            # Save old and calculate new log likelihood.
            log_likeli_old = log_likeli_new
            log_likeli_new = np.sum(-np.sum(p_est, axis=0) * np.log(2 * math.pi * sig_est**2) -\
                            np.sum(p_est * np.square(y_arr - np.dot(A, beta_est)), axis=0) /\
                            (2 * sig_est**2)) + np.sum(np.log(h_est + 1e-15) * p_est)
                
            # Stop if increase in likelihood is below threshold.
            if log_likeli_new - log_likeli_old < 1e-3 and log_likeli_new - log_likeli_old > -1e-3 and\
                i >= 10:
                break

        # Assign self parameters.    
        self.iters = i+1
        self.beta_est = beta_est
        self.p_est = p_est
        self.sig_est = sig_est
        self.h_est = h_est
        self.gamma = gamma
        self.x = x
        self.y = y
        self.likelihood = log_likeli_new
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
                A_uni = np.concatenate((np.ones((len(x_pred),1)),
                                        x_pred.reshape((len(x_pred),self.m))), axis = 1)
                h_final_inc = h_est_inc(self.gamma, A_uni)
                h_final = np.concatenate((h_final_inc, 1 - np.sum(h_final_inc, axis=1)[:,None]), axis=1)
                h_final[np.isnan(h_final)] = 1 / self.k
                f_est =  np.dot(A_uni, self.beta_est)
                pred = np.sum(f_est*h_final, axis=1)
                return pred
            else:
                raise ValueError("Incorrect number of dimensions.")
        else:
            raise ValueError("Train the model first.")
            


