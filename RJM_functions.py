import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LinearRegression, lasso_path

import matplotlib.backends.backend_pdf
import warnings
import covar

from scipy import sparse
from scipy import stats

from scipy.optimize import linear_sum_assignment

import time
import sys

from sklearn.covariance import graphical_lasso, OAS, shrunk_covariance#, LedoitWolf, OAS, 

import random 

#from glmnet import ElasticNet
from sklearn.linear_model import ElasticNet as skl_ElasticNet

from sklearn.cluster import KMeans, SpectralClustering

import os

from multiprocessing import Pool, cpu_count

import platform

from sklearn.datasets import make_sparse_spd_matrix
from sklearn.preprocessing import StandardScaler

from sklearn.mixture import GaussianMixture

import itertools

# rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
utils = importr('utils')
base = importr('base')
# the MoE packages
importr("flexmix")
importr("mixtools")
importr('huge')
# no glmnet in python for windows: use R instead
if platform.system() == 'Windows':
    importr('glmnet')
else:
    from glmnet import ElasticNet
    
    
################################################ core things ################################################

# useful class to avoid printing
# taken from: https://stackoverflow.com/questions/8391411/how-to-block-calls-to-print
class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout

###################################### chunks of code that became too large to stay in the main script ######################################
    
# list of all followed metrics, given the provided simulation settings  
def get_metrics_list(evaluate_oracle_error):
    optimisation_metrics = ['neg_ll_0', 'neg_ll_true', 'neg_ll_T', 'neg_ll_T_no_penalty']
    classification_metrics = ['soft_error', 'hard_error', 'rand_index']
    prediction_metrics = ['neg_ll_X', 'mse_Y']
    runtime_metrics = ['projection_learning_duration', 'clustering_duration']
    oracle_metrics = evaluate_oracle_error*['P', 'N', 'TP', 'TN', 
                                            'soft_error_oracle', 'hard_error_oracle', 
                                            'rand_index_oracle', 'rand_index_vs_oracle']

    metric_list = optimisation_metrics + classification_metrics + oracle_metrics + prediction_metrics + runtime_metrics

    return metric_list

# update a slice of a large data array with the content of a smaller one
def fill_da(small_da, big_da, slice_dict):
    # extend the dims of the small_da, with duplicated entries where needed, to match the slice of the big_da
    med_da = small_da.broadcast_like(big_da.sel(slice_dict)) 
    # update 
    big_da.loc[slice_dict] = med_da
    
# update the global results data array after one run of the chosen algorithm, with or without ROC metrics
def update_results_da(
    # the main data array to update
    results_da,
    # the sparsistency metrics data array to update if desired
    sparsistency_roc_da,
    # the dictionary containing the results of one run of the algo
    output, 
    # the current settings for this run
    settings_dictionary, 
    # do we evaluate the sparsistency metrics?
    evaluate_sparsistency_roc=False):

    # extract the desired results
    algo_results_da = output['results_da']

    # dictionary with the relevant metrics added
    settings_dictionary_ = dict(settings_dictionary, **dict(metric=algo_results_da.metric.values))
    
    # update results
    fill_da(algo_results_da, results_da, settings_dictionary_)

    # if we also collect the ROC metrics
    if evaluate_sparsistency_roc:
        # extract the ROC results
        algo_sparsistency_roc_da = output['sparsistency_roc_da']
        
        # dictionary with the relevant metrics added 
        settings_dictionary_ = dict(settings_dictionary, **dict(metric=algo_sparsistency_roc_da.metric.values))
        # (should never be necessary, but here for consistancy and insurance)
        
        # remove any reference to K_tried from the dict (otherwise glitches)
        if 'K_tried_JCR' in settings_dictionary_.keys():
            settings_dictionary_.pop('K_tried_JCR')
    
        # update results
        fill_da(algo_sparsistency_roc_da, sparsistency_roc_da, settings_dictionary_)
              
# wrapper, generate synthetic data from mixture parameters, all data distributions
def generate_synthetic_data(mu, Sigma, tau, n, synthetic_data_distribution):
    # get dimension
    p = len(mu.component_p)
    K = len(mu.label)

    # generate synthetic data from mixture parameters
    if (synthetic_data_distribution == 'normal') or (synthetic_data_distribution == 'gaussian'):
        # generate gaussian data 
        X_full, labels = generate_synthetic_gaussian_data(mu, Sigma, tau, n)

    elif synthetic_data_distribution == 'mixed_binary':
        # fraction of binary fearures
        fraction_binary = 0.2
        # total number of binary features
        d_binary = int(fraction_binary*p)
        # thresholds for binary features
        C = xr.DataArray(coords=[range(K), range(d_binary)], dims = ['label', 'component_p'])
        # generate them as realisations of the normal distribution, this way, future feature values can reasonably be either smaller or larger.
        for k in range(K):
            C.loc[dict(label=k)] = np.random.multivariate_normal(mu.sel(component_p=range(d_binary), label=k), 
                                                                 Sigma.sel(component_p=range(d_binary), component_pT=range(d_binary), label=k))
        # generate mixed_binary data 
        X_full, labels = generate_synthetic_mixed_data(mu, Sigma, tau, C, n)

    elif synthetic_data_distribution == 'student':
        # student degrees of freedom
        df = 10
        # generate student data 
        X_full, labels = generate_synthetic_student_data(mu, Sigma, tau, df, n)

    else:
        print('invalid synthetic data distribution')
        return

    return X_full, labels
                  
################################################ projection  #######################################################

# get projection vectors from data
def get_projection(X, q, method = 'mixture-PCA projection'):
    p = X.shape[1]
        
    if method == 'mixture-PCA projection':
        # get the eigendeomposition
        # take just XTX
        Sigma_empirical = X.cov()
        # PCA
        eig_vals_mixture, eig_vectors_mixture = np.linalg.eigh(Sigma_empirical)
        # get the mixture projection
        mixture_order = np.argsort(eig_vals_mixture)[::-1]
        projection = eig_vectors_mixture[:, mixture_order][:, :q]
        
    if method == 'shrunk mixture-PCA projection':
        # get the eigendeomposition
        # shrink the covariance before doing the PCA
        Sigma_empirical = pd.DataFrame(covar.cov_shrink_ss(np.float_(np.array(X ,order='C')))[0])
        # PCA
        eig_vals_mixture, eig_vectors_mixture = np.linalg.eigh(Sigma_empirical)
        # get the mixture projection
        mixture_order = np.argsort(eig_vals_mixture)[::-1]
        projection = eig_vectors_mixture[:, mixture_order][:, :q]
        
    if method == 'random projection':
        projection = np.random.normal(0, 1, (p, q))
        projection = projection/np.sqrt(np.sum(projection**2, axis = 0))

    if method == 'sparse random projection':
        # add a sparse random projection
        s = p**0.5
        count = 0
        U = np.random.uniform(0, 1, (p, q))
        projection = -s**0.5*(U < 1/(2*s)) + s**0.5*(U >= 1- 1/(2*s))
        while (count<100) & (np.linalg.matrix_rank(projection)<q):
            U = np.random.uniform(0, 1, (p, q))
            projection = -s**0.5*(U < 1/(2*s)) + s**0.5*(U >= 1- 1/(2*s))
            count += 1
        if count==100:
            print(f'p = {p}, q = {q}, failure of random sparse')
            projection = np.zeros((p, q))
            projection[:q, :] = np.eye(q)
        
    # used to test and debug
    if method == 'truncation':
        projection = np.zeros((p, q))
        projection[:q, :] = np.eye(q)        
        
    return np.real(projection)



# for matrix shrinkage
def gamma_rblw(S, n):
    # regularise n
    n = int(n)+1
    
    p = len(S)
    a = (n-2)/(n*(n+2))
    b = ( (p+1)*n -2 ) / (n*(n+2))

    U = p*np.trace(S.dot(S))/np.trace(S)**2 - 1
    

    return np.minimum(a + b/ U, 1)



####################################################### data ############################################################################



def subsample_rows(X, labels, n):
    # sample here the number of rows
    X_subsampled = X.sample(n, replace=False, axis='index')
    labels_subsampled = labels.loc[X_subsampled.index]
    return X_subsampled, labels_subsampled

# randomly partition a list into K lists
def partition(list_in, K):
    random.shuffle(list_in)
    return [list_in[i::K] for i in range(K)]

# If we want to enforce the same covariance matrix in all classes: sub-sample X among just one class
def subsample_one_class(X_da, labels_da, K_target=None, sampling_class=None):
    X_processed_da = X_da.copy()
    # get the initial number of classes
    K = pd.Series(labels_da).nunique()

    
    # if the user does not provide a number of artificial classes to build:
    if K_target is None:
        # keep the original amount of classess
        #K_target = K
        # take a fixed, low (we reduce the data by a factor of almost K) number of new classes
        K_target = 3
        
    # if the user does not specify from which class to subsample: take the largest one
    if sampling_class is None:
        sampling_class = -1
        n_sampling_class = 0
        for k in range(K):
            n_k = int((labels_da==k).sum())
            if n_sampling_class < n_k:
                sampling_class = k
                n_sampling_class = n_k
    else:
        # get the number of observation in the desired class
        n_sampling_class = int((labels_da==sampling_class).sum())


    # restrict yourself to this class
    X_processed_da = X_processed_da[labels_da==sampling_class]
    # reset observation index
    X_processed_da = X_processed_da.assign_coords({'observation':range(n_sampling_class)})

    # then, partition the remaining indices into K different groups to make K new, artificial, classes
    indices_list = partition(list(range(n_sampling_class)), K_target)

    # create the new label da
    labels_processed_da = xr.DataArray(
        #data = -np.ones(n_sampling_class), 
        coords = [range(n_sampling_class)], 
        dims = ['observation'])

    # put the new, artificial, labels into it
    for k in range(K_target):
        labels_processed_da.loc[dict(observation = indices_list[k])] = k
        
    return X_processed_da, labels_processed_da

# dataframe version of the function subsample_one_class
def subsample_one_class_df(X, labels, K_target=None, sampling_class=None):
    # get parameters
    n, p = X.shape
    
    # convert to Data array
    X_da = xr.DataArray(
        data = X, 
        coords = [range(n) , range(p)], 
        dims = ['observation', 'component_p'])

    labels_da =  xr.DataArray(
        data = labels, 
        coords = [range(n)], 
        dims = ['observation'])

    # run the data array version of the function
    X_da, labels_da = subsample_one_class(X_da, labels_da, K_target=K_target, sampling_class=sampling_class)

    # convert back to data frame
    X = pd.DataFrame(X_da.values)
    labels = pd.Series(labels_da.values, dtype=int)
    return X, labels

# increase the distance between mu 
def separate_mu(mu_empirical, delta=1.):
    
    # barycenter of the mus
    mu_average = mu_empirical.mean(dim='label')

    # the initial sum of squared norms between the different mus
    # Keep in mind that we have: MSE(mu_k-mu_avg) = MSE(mu_k-m_l)_{k<l} / K
    initial_norm = float(((mu_empirical - mu_average)**2).mean())**(0.5)

    mu_empirical_separated = mu_empirical + (delta/initial_norm-1)*(mu_empirical - mu_average)

    # reached_norm should be = delta
    reached_norm = float(((mu_empirical_separated - mu_empirical_separated.mean(dim='label'))**2).mean())**(0.5)
    if np.abs(reached_norm-delta)>1e-5:
        print('mu badly separated')
        
    return mu_empirical_separated

# enforce a certain distance between the empirical mu of each class (works even with K>2)
def fix_empirical_mu(X_da, labels_da, delta_mu=None):
    X_processed_da = X_da.copy()
    # if no deltu_mu specified, keep the native class averages from the real data
    mu_translation = 0
    # otherwise, apply the requiered separation between the class averages
    if not(delta_mu is None):
        # get the parameters
        K = pd.Series(labels_da).nunique()
        p = len(X_da.component_p)
        # get the empirical mu
        mu_empirical = xr.DataArray(
            coords = [range(K), range(p)], 
            dims = ['label', 'component_p'])
        for k in range(K):
            X_k = X_processed_da[labels_da==k]
            mu_empirical.loc[dict(label=k)] = X_k.mean(dim='observation')

        # separated mus
        mu_empirical_separated = separate_mu(mu_empirical, delta_mu)
        # the translation to apply to get the desired delta_mu
        mu_translation = mu_empirical_separated - mu_empirical
        # apply the separation to the data
        for k in range(K):
            # enforce the new mus on the data
            X_processed_da[labels_da==k] += mu_translation.sel(label=k)
            
    # also return the translation to update mu_true outside the function
    return X_processed_da, mu_translation
    

# sample from a thresholded gaussian
def sample_thresholded_normal(loc=0., scale=1., threshold=1., size=1.):
    # initialise the list the contains the results
    sample_list = []
    # while the list does not have the requiered size, keep adding values to it
    sample_size = 10*size
    while len(sample_list)<size:
        new_sample = np.random.normal(loc, scale, sample_size)
        accepted_values = new_sample[np.where(np.abs(new_sample)>threshold)[0]]
        sample_list+=list(accepted_values)
    # truncate the list to have exactly the right size
    sample_list = sample_list[:size]
    
    return np.array(sample_list)

# generate beta with a fixed coefficient overlap between classes
def generate_beta_with_overlap(p, K, n_non_zeros = 20, n_common = 2, 
                               # sparse or dense beta
                               sparse_beta=True,
                               # are all the non zero coefficient in the same place, even if they are not equal?
                               common_support=False,
                               # are the (absolute) value of the non zero coefficients fixed or random?
                               fixed_non_zero_value = True,
                               # if gaussian non zero, minimum threshold for the gaussian behind the sparse beta
                               threshold=1., 
                               # if fixed non zero, absolute value of all non zero coefficient,
                               non_zero_abs_value=2., 
                               # gaussian parameters for the dense beta
                               zero_scale = 0.2,
                               non_zero_mean = 2.,
                               non_zero_scale = 0.7):
    if n_common>n_non_zeros:
        print('Invalid number of non-zeros betas: n_common>n_non_zeros')
        return
    
    n_different = n_non_zeros-n_common

    # do all classes share the same non-zero support?
    if common_support:
        # randomly sample the non-zeros indices
        non_zero_indices = np.random.choice(range(p), n_non_zeros, replace=False)
        # randomly sample the common indices
        common_values_indices = np.random.choice(non_zero_indices, n_common, replace=False)
        # get the indices where the values are different
        unique_values_indices = [idx for idx in non_zero_indices if not idx in common_values_indices]
    
    else:
        # randomly sample the common indices
        common_values_indices = np.random.choice(range(p), n_common, replace=False)
        # get the remaining available indices 
        remaining_available_indices = [idx for idx in range(p) if not idx in common_values_indices]

    # initialise sparse beta
    if sparse_beta:
        # initialise beta at 0
        beta = xr.DataArray(
            data=np.zeros((K, p)),
            coords = [range(K), range(p)], 
            dims = ['label', 'component_p'])
    # initialise dense beta        
    else:
        # initialise dense beta: small coefficients everywhere
        beta = xr.DataArray(
            data=np.random.normal(loc=0., scale = zero_scale, size=(K, p)),
            coords = [range(K), range(p)], 
            dims = ['label', 'component_p'])

    ## fill in these indices with values 
    # fixed absolute values
    if fixed_non_zero_value:
        # fixed absolute values, random signs
        beta.loc[dict(component_p=common_values_indices)] = non_zero_abs_value* (2*(np.random.rand(n_common)>0.5)-1)
    # random, sparse beta
    elif sparse_beta:
        # random, gaussian thresholded, values
        beta.loc[dict(component_p=common_values_indices)] = sample_thresholded_normal(loc=0., scale=1., threshold=threshold, size=n_common)
    # random, dense beta
    else:
        # bimodal gaussian (gaussian mixture)
        beta.loc[dict(component_p=common_values_indices)] = (2*(np.random.rand(n_common)>0.5)-1)*np.random.normal(non_zero_mean, non_zero_scale, n_common)

    # sample different values on these indices
    for k in range(K):
        # If they have to be different, sample here the support of the class-unique coefficients
        if not common_support:
            # randomly sample the different non-zeros, non-common indices
            unique_values_indices = np.random.choice(remaining_available_indices, n_different, replace=False)
            # remove the selected indices
            remaining_available_indices = [idx for idx in remaining_available_indices if not idx in unique_values_indices]

        ## get the class-specific coefficients
        # fixed absolute values
        if fixed_non_zero_value:
            # fixed absolute values, random signs
            beta.loc[dict(label=k, component_p=unique_values_indices)] = non_zero_abs_value* (2*(np.random.rand(n_different)>0.5)-1)
        # random, sparse beta
        elif sparse_beta:
            # random, gaussian thresholded, values
            beta.loc[dict(label=k, component_p=unique_values_indices)] = sample_thresholded_normal(loc=0., scale=1., threshold=threshold, size=n_different)
        # random, dense beta
        else:
            # bimodal gaussian (gaussian mixture)
            beta.loc[dict(label=k, component_p=unique_values_indices)] = (2*(np.random.rand(n_different)>0.5)-1)*np.random.normal(non_zero_mean, non_zero_scale, n_different)
            
    return beta


# generate beta with commun support of constant non-zero coefficients, but where the signs of some of them are flipped between classes
def generate_beta_with_flipped_signs(p, K, n_non_zeros = 20, 
                                     # number of coefficients with flipped signs from the reference
                                     # each pair of beta are different by 2*sign_flip_window coefficients
                                     sign_flip_window = 2, 
                                     # sparse or dense beta
                                     sparse_beta=True,
                                     # absolute value of all non zero coefficient
                                     non_zero_abs_value=1., 
                                     # if dense beta, scale of the "zero" coefficients
                                     zero_scale = 0.2):
    
    # we have to be creative to generalise this idea to the case K>2
    if sign_flip_window*K>n_non_zeros:
        print('Invalid number of non-zeros betas: sign_flip_window*K>n_non_zeros')
        return
    
    # create a reference beta with a sign
    beta_base = non_zero_abs_value*np.ones(n_non_zeros)*(2*(np.random.rand(n_non_zeros)<0.5)-1)
    
    # each class-beta differs from the base by sign_flip_window flipped coefficients
    # these flipped coefficient are unique for each class-beta, hence the difference between any two class-beta is 2*sign_flip_window
    beta_k = dict()
    for k in range(K):
        # first copy beta base
        beta_k[k] = beta_base.copy()
        # then flip the kth window of coefficients
        beta_k[k][k*sign_flip_window: (k+1)*sign_flip_window] = -1*beta_k[k][k*sign_flip_window: (k+1)*sign_flip_window]

    # now that we have the non-zeros values, insert them into a p dimesional vector, where the rest are either zeros or low values
    
    # all classes share the same non-zero support
    # randomly sample indices where to put these non-zeros values
    non_zero_indices = np.random.choice(range(p), n_non_zeros, replace=False)

    # sparse beta
    if sparse_beta:
        # initialise beta at 0
        beta = xr.DataArray(
            data=np.zeros((K, p)),
            coords = [range(K), range(p)], 
            dims = ['label', 'component_p'])

        for k in range(K):
            # fill in these indices with the computed values 
            beta.loc[dict(label=k, component_p=non_zero_indices)] = beta_k[k]
    
    # dense beta        
    else:
        # initialise dense beta: small coefficients everywhere
        beta = xr.DataArray(
            data=np.random.normal(loc=0., scale = zero_scale, size=(K, p)),
            coords = [range(K), range(p)], 
            dims = ['label', 'component_p'])
        
        for k in range(K):
            # fill in these indices with the computed values 
            beta.loc[dict(label=k, component_p=non_zero_indices)] = beta_k[k]
            
    return beta

# a general function that calls the desired one
def generate_beta_wrapper(p, K, 
                          # name of the method
                          beta_method = 'overlap', #'flipped_signs'
                          n_non_zeros = 20, n_common = 2, 
                          # sparse or dense beta
                          sparse_beta=True,
                          # are all the non zero coefficient in the same place, even if they are not equal?
                          common_support=True,
                          # are the (absolute) value of the non zero coefficients fixed or random?
                          fixed_non_zero_value = True,
                          # minimum threshold for the gaussian behind the sparse beta
                          threshold=1., 
                          # gaussian parameters for the dense beta
                          zero_scale = 0.2,
                          non_zero_mean = 2.,
                          non_zero_scale = 0.7,
                          
                          # specific to 'flipped_signs':
                          
                          # number of coefficients with flipped signs from the reference
                          # each pair of beta are different by 2*sign_flip_window coefficients
                          sign_flip_window = 2,  
                          # absolute value of all non zero coefficient
                          non_zero_abs_value=1.):
    
    
    if beta_method == 'overlap':
        beta = generate_beta_with_overlap(p, K, 
                                          n_non_zeros = n_non_zeros, n_common = n_common, 
                                          # sparse or dense beta
                                          sparse_beta=sparse_beta,
                                          # are all the non zero coefficient in the same place, even if they are not equal?
                                          common_support=common_support,
                                          # are the (absolute) value of the non zero coefficients fixed or random?
                                          fixed_non_zero_value = fixed_non_zero_value,
                                          # if gaussian non zero, minimum threshold for the gaussian behind the sparse beta
                                          threshold=threshold, 
                                          # if fixed non zero, absolute value of all non zero coefficient,
                                          non_zero_abs_value=non_zero_abs_value, 
                                          # gaussian parameters for the dense beta
                                          zero_scale = zero_scale,
                                          non_zero_mean = non_zero_mean,
                                          non_zero_scale = non_zero_scale)
        
        
    elif beta_method == 'flipped_signs':
        beta = generate_beta_with_flipped_signs(p, K, 
                                                n_non_zeros = n_non_zeros, 
                                                # number of coefficients with flipped signs from the reference
                                                # each pair of beta are different by 2*sign_flip_window coefficients
                                                sign_flip_window = sign_flip_window, 
                                                # sparse or dense beta
                                                sparse_beta=sparse_beta,
                                                # absolute value of all non zero coefficient
                                                non_zero_abs_value=non_zero_abs_value, 
                                                # if dense beta, scale of the "zero" coefficients
                                                zero_scale = zero_scale)
        
    else:
        print(f'invalid beta generation method name: {beta_method}')
        return
    
    return beta

# generate an artificial Y from a given X and beta
def generate_Y(X, labels, beta, sigma):
    K = pd.Series(labels).nunique()
    n = len(labels.observation)
    # intitialise Y data array
    Y = xr.DataArray(
        data=np.random.normal(0, sigma, n),
        coords=labels.coords, 
        dims=labels.dims)
    
    for k in range(K):
        # Xs belonging to class k
        X_k = X[labels==k]
        # E[Y|X]
        dotproduct = xr.dot(beta.sel(label=k), X_k, dims='component_p')

        # Y_k
        Y[labels==k] += dotproduct
    return Y


# split data into a train data set and a validation data set
def split_train_val_sets(Y_da, X_da, labels_da, train_set_size=0.7):

    # total number of observations
    n = len(Y_da.observation)
    # size of the train data set
    n_train = int(train_set_size*n)
    # sample the train observation
    train_indices = np.random.choice(range(n), size = n_train, replace = False)
    # the complementary observations make up the validation set
    val_indices = [idx for idx in range(n) if not idx in train_indices]

    # get the train observations
    labels_train = labels_da.sel(observation=train_indices).copy()
    X_train = X_da.sel(observation=train_indices).copy()
    Y_train = Y_da.sel(observation=train_indices).copy()

    # reset the observation IDs
    labels_train = labels_train.assign_coords({'observation':range(n_train)})
    X_train = X_train.assign_coords({'observation':range(n_train)})
    Y_train = Y_train.assign_coords({'observation':range(n_train)})

    # get the validation observations
    labels_val = labels_da.sel(observation=val_indices).copy()
    X_val = X_da.sel(observation=val_indices).copy()
    Y_val = Y_da.sel(observation=val_indices).copy()

    # reset the observation IDs
    labels_val = labels_val.assign_coords({'observation':range(n-n_train)})
    X_val = X_val.assign_coords({'observation':range(n-n_train)})
    Y_val = Y_val.assign_coords({'observation':range(n-n_train)})
    
    return Y_train, X_train, labels_train, Y_val, X_val, labels_val

# initialise model parameters, works for ambient and embedding space
def initialise_parameters(K, p, q=None):
    # beta
    beta_0 = sparse.random(K, p+1, density=0.3, format='array')
    # all values are in [0,1] by default, also allow negative values
    beta_0 = (1-2*beta_0)*(beta_0!=0)
    beta_0 = xr.DataArray(
        data = beta_0, 
        coords = [range(K), range(p+1)], 
        dims = ['label', 'component_p'])

    # variance Y
    #sigma_0 = np.random.chisquare(2, size=K)
    # chisqquare: too much variance
    sigma_0 = np.ones(K)
    sigma_0 = xr.DataArray(
        data = sigma_0, 
        coords=[range(K)], 
        dims=['label'])
    
    # ambient space or embedding space?
    if q is None:
        # name of the component dimension
        component_name = 'component_p'
        # size of the component dimension
        component_size = p
    else:
        # name of the component dimension
        component_name = 'component_q'
        # size of the component dimension
        component_size = q

    # mu 
    mu_0 = np.random.multivariate_normal(np.zeros(component_size), np.eye(component_size), K)
    mu_0 = xr.DataArray(
        data = mu_0, 
        coords=[range(K), range(component_size)], 
        dims=['label', component_name])

    # Sigma 
    Sigma_0 = stats.invwishart.rvs(df=component_size, scale = np.eye(component_size), size = K)/component_size
    if component_size==1:
        # to avoid exceptions
        Sigma_0 = Sigma_0.reshape(K, 1, 1)
    if K == 1:
        # to avoid exceptions
        Sigma_0 = Sigma_0.reshape(1, component_size, component_size)
    Sigma_0 = xr.DataArray(
        data = Sigma_0, 
        coords=[range(K), range(component_size), range(component_size)], 
        dims=['label', component_name, f'{component_name}T'])

    # balanced classes
    tau_0 = np.ones(K)/K
    tau_0 = xr.DataArray(
        data = tau_0, 
        coords=[range(K)], 
        dims=['label'])
    return beta_0, sigma_0, mu_0, Sigma_0, tau_0

# convert label from R^n with entries in {1, ..., K} to R^{n x K} with entries in {0, 1}
def convert_label_to_dummies(labels):
    n = len(labels)
    K = pd.Series(labels).nunique()

    # labels data array
    labels_da =  xr.DataArray(
        data = np.zeros((n, K)), 
        coords= [range(n), range(K)], 
        dims=['observation', 'label'])

    for k in range(K):
        idx_k = np.where(labels==k)[0]
        labels_da.loc[dict(observation= idx_k, label = k)] = 1
    return labels_da



####################################################### EM #######################################################

# log likelihood of each datea point for each class, weighted by class weight
# Y = f(X), lasso between Y and X
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def individual_log_likelihood(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da=None, tempering_exponent=1):
    n = len(Y_da.observation)
    K = len(mu_da.label)
    
    ## log phi WT.X
    log_phi = np.zeros((n, K))
    # ambient space or embedding space?
    if WT_X_da is None:
        for k in range(K):
            log_phi[:, k] = stats.multivariate_normal.logpdf(X_da, mean=mu_da.sel(label=k), cov=Sigma_da.sel(label=k), allow_singular=True)
    else:
        for k in range(K):
            log_phi[:, k] = stats.multivariate_normal.logpdf(WT_X_da, mean=mu_da.sel(label=k), cov=Sigma_da.sel(label=k), allow_singular=True)
    
    log_phi_da = xr.DataArray(
        data = log_phi, 
        coords=[range(n), range(K)], 
        dims=['observation', 'label'])
    
    # 'temper' the X bloc to afjust its effect on the class weights
    log_phi_da = tempering_exponent*log_phi_da

    ## log phi Y | X
    # compute E[Y|X]
    X_beta_da = compute_conditional_expectation(X_da, beta_da)
    
    log_phi_Y = np.zeros((n, K))
    for k in range(K):
        log_phi_Y[:, k] = stats.norm.logpdf(Y_da, loc=X_beta_da.sel(label=k), scale=sigma_da.sel(label=k))

    log_phi_Y_da = xr.DataArray(
        data = log_phi_Y, 
        coords=[range(n), range(K)], 
        dims=['observation', 'label'])

    ## log phi WT.X + log phi tau
    log_p = log_phi_da+log_phi_Y_da+np.log(tau_da)
    return log_p

# E step
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def E_step_weights(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da = None,
                   tempering_exponent=1, stop_at_vanishing_cluster=False, regularise_E_step=True):
    # log likelihood of each datea point for each class, weighted by class weight
    log_p = individual_log_likelihood(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da=WT_X_da, tempering_exponent=tempering_exponent)

    # E step weights
    p_kt = np.exp(log_p - log_p.max(dim='label')) / np.exp(log_p - log_p.max(dim='label')).sum(dim = 'label')
    
    #print(p_kt.sum(dim='observation').values)
    # relace Na by zeros if there are any
    if bool(p_kt.isnull().any()):
        print('fillna')
        p_kt = p_kt.fillna(0)
    
    # If an entire cluster starts to vanish, return it as is, this will stop the EM
    if stop_at_vanishing_cluster & bool((p_kt.sum(dim='observation')<1).any()):
        print('vanishing cluster')
        return p_kt
    
    # should we avoid vanishing clusters by increasing slightly the E step probability?
    if regularise_E_step:
        n_low_threshold=5
        # If an entire cluster is almost empty but salvageable, we can regularise to avoid vanishing cluster
        if bool((p_kt.sum(dim='observation')<n_low_threshold).any()):
            print('increasing low weights')
            # for the regularity of the GLMnet model, have at least n_low_threshold points per class
            p_kt += n_low_threshold/(len(p_kt.observation)-n_low_threshold*len(p_kt.label))
            print(p_kt.sum(dim='observation').values)
            p_kt /= p_kt.sum(dim='label')
            print(p_kt.sum(dim='observation').values)
            if bool((p_kt.sum(dim='observation')==0).any()):
                print(p_kt)

    return p_kt

# M step
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def M_step_lasso(Y_da, X_da, p_kt, penalty_lasso, penalty_sigma = 0, penalty_tau = 0, WT_X_da = None, shrinkage = 'oas'):
    n = len(Y_da.observation)
    K = len(p_kt.label)
    p = len(X_da.component_p)

    # penalty -penalty_tau*ln(tau_k) on the class weights tau
    tau_t1 = (p_kt.sum(dim='observation') + penalty_tau)/(n + K*penalty_tau)
    
    
    # ambient space or embedding space?
    if WT_X_da is None:
        # mu
        mu_t1 = (p_kt * X_da).sum(dim='observation')/p_kt.sum(dim='observation')
        # Sigma
        Sigma_t1 = estimate_Sigma(X_da, mu_t1, p_kt, shrinkage=shrinkage) 
    else:
        # mu 
        mu_t1 = (p_kt * WT_X_da).sum(dim='observation')/p_kt.sum(dim='observation')
        # Sigma
        if shrinkage:
            Sigma_t1 = estimate_Sigma(X_da, mu_t1, p_kt, shrinkage=shrinkage)
        else:
            Sigma_t1 = xr.dot( (p_kt*(WT_X_da - mu_t1)), (WT_X_da - mu_t1).rename({'component_q':'component_qT'}), dims='observation')/p_kt.sum(dim='observation')

    # beta hat
    beta_t1 = xr.DataArray(coords=[range(K),  range(p+1)], dims=['label', 'component_p'])
    for k in range(K):        
        # lasso penalty
        # this alpha is set so that the glmnet of sklearn will optimise the correct loss
        alpha = float(penalty_lasso.sel(label=k)/p_kt.sel(label=k).sum())

        try:
            lasso_model = skl_ElasticNet(
                # pure lasso: l1_ratio=1, 
                l1_ratio=1,
                alpha=alpha,
                fit_intercept=True, 
                max_iter=5000)
                #warm_start=True, tol=1e-6)

            lasso_model.fit(X_da, Y_da, sample_weight=p_kt.sel(label=k))
            
        # if this does not work, regularise the regression by making it a ridge regression
        except:
            print('Lasso failure: trying ElasicNet instead')
            lasso_model = skl_ElasticNet(
                # EN: l1_ratio=0.5, 
                l1_ratio=0.5,
                alpha=alpha,
                fit_intercept=True, 
                max_iter=5000)
                #warm_start=True)

            lasso_model.fit(X_da, Y_da, sample_weight=p_kt.sel(label=k))
            
        beta_t1.loc[dict(label =k, component_p=range(p))] = lasso_model.coef_
        beta_t1.loc[dict(label =k, component_p=p)]= lasso_model.intercept_
    
    # dotproduct E[Y|X] = X^T beta + beta_0
    X_beta_t1 = compute_conditional_expectation(X_da, beta_t1)
    
    # sigma hat
    # do not penalise the intercept in this formula!
    sigma_t1 = np.sqrt(
        ((p_kt * (Y_da - X_beta_t1)**2).sum(dim='observation')/2. + penalty_lasso*np.abs(beta_t1.sel(component_p=range(p))).sum(dim='component_p') )/
        (p_kt.sum(dim='observation')/2.+penalty_sigma))
    
    return beta_t1, sigma_t1, mu_t1, Sigma_t1, tau_t1

# compute observed (Mixture) likelihood
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def mixture_negative_log_likelihood(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da=None, tempering_exponent=1):
    log_p = individual_log_likelihood(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da, tempering_exponent)
    #return -float(np.log(np.exp(log_p).sum(dim='label')).sum()) # -np.log(float(np.exp(log_p).sum()))
    log_max = log_p.max(dim='label')
    mixture_log_likelihood = float((np.log(np.exp(log_p - log_max).sum(dim='label')) + log_max).sum())
    return -mixture_log_likelihood


# compute the term pen(theta) that penalises the likelihood optimised by the EM
def penalty_EM(beta_da, sigma_da, tau_da, penalty_lasso = 0, penalty_sigma=0, penalty_tau = 0):
    penalty = compute_penalty_beta(beta_da, sigma_da, penalty_lasso) + compute_penalty_sigma(sigma_da, penalty_sigma)+compute_penalty_tau(tau_da, penalty_tau)
    return penalty

def compute_penalty_beta(beta_da, sigma_da, penalty_lasso):
    # return the scaled l_1 norm of beta
    
    # do not penalise the intercept: take only component_p=range(p) and not component_p=p
    p = len(beta_da.component_p)-1
    return float((penalty_lasso*np.abs(beta_da.sel(component_p=range(p))).sum(dim='component_p')/sigma_da**2).sum(dim='label'))

def compute_penalty_sigma(sigma_da, penalty_sigma):
    # return the weighted log of sigma
    return float(penalty_sigma*np.log(sigma_da).sum(dim='label'))

def compute_penalty_tau(tau_da, penalty_tau):
    # return the weighted log of tau
    return float(-1*penalty_tau*np.log(tau_da).sum(dim='label'))

# this is the function that the EM minimises
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def EM_objective_function(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da=None,
                          tempering_exponent=1, penalty_lasso = 0, penalty_sigma=0, penalty_tau=0):
    # penalty
    penalty = penalty_EM(beta_da, sigma_da, tau_da, penalty_lasso = penalty_lasso, penalty_sigma=penalty_sigma, penalty_tau=penalty_tau)
    
    # dotproduct E[Y|X] = X^T beta
    #X_beta_da = compute_conditional_expectation(X_da, beta_da)
    # negative log-likelihodd
    neg_ll = mixture_negative_log_likelihood(Y_da, X_da, beta_da, sigma_da, mu_da, Sigma_da, tau_da, WT_X_da=WT_X_da, tempering_exponent=tempering_exponent)
    
    return neg_ll + penalty

# get X_regression for the dot product X_beta = E[Y|X]
def add_column_of_1s(X_da):
    # add a column of 1s to X
    n = len(X_da.observation)
    p = len(X_da.component_p)

    X_regression = xr.DataArray(
            coords=[range(n), range(p+1)], 
            dims=['observation', 'component_p'])
    # put X on the first components
    X_regression.loc[dict(component_p = range(p))] = X_da
    X_regression.loc[dict(component_p = p)] = 1
    return X_regression

# dotproduct E[Y|X] = X^T beta
def compute_conditional_expectation(X_da, beta_da):
    #X_regression = add_column_of_1s(X_da)
    #X_beta_da = xr.dot(X_regression, beta_da, dims='component_p')
    
    # more effecient than computing X_regression each time:
    p = len(X_da.component_p)
    X_beta_da = xr.dot(X_da, beta_da.sel(component_p=range(p)), dims='component_p')
    X_beta_da += beta_da.sel(component_p=p)
    return X_beta_da

# full EM with the JCR model
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def EM_JCR_lasso(Y_da, X_da, 
                 # model parameters
                 beta_0, sigma_0, mu_0, Sigma_0, tau_0, 
                 # projected data
                 WT_X_da = None,
                 # stopping criteria
                 neg_ll_ratio_threshold = 1e-5, 
                 E_step_difference_threshold = 1e-3,
                 max_steps = 100, 
                 min_steps = 10,
                 # do we regularise the estimated covariance matrices?
                 shrinkage = 'oas',
                 # exponent of the X-bloc in the E step
                 tempering_exponent=1, 
                 # penalty intensity in the lasso estimation of beta 
                 penalty_lasso = 1, 
                 penalty_sigma = 0,
                 penalty_tau = 0):
            
    # intial values
    beta_t, sigma_t, mu_t, Sigma_t, tau_t = beta_0.copy(), sigma_0.copy(), mu_0.copy(), Sigma_0.copy(), tau_0.copy()
    
    # The lasso penalty is updated once during this EM. Keep track of whether or not this has already happened
    penalty_lasso_version = 0
    
    # initialise this at 1 for the first evaluation of the difference in E step (the computed mean difference is then = (K-1)/2)
    p_kt0 = 1.
    
    # stoping criteria initialisation to enter the loop
    neg_ll_ratio = 2.*neg_ll_ratio_threshold
    E_step_difference = 2.*E_step_difference_threshold
    t = 0
    print('neg_ll_ratio:')
    #print('E_step_difference:')
    while ((np.abs(neg_ll_ratio) > neg_ll_ratio_threshold) |  (E_step_difference>E_step_difference_threshold))& (t<max_steps):
        # get current (penalised) negative log likelihood
        neg_ll_t0 = EM_objective_function(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da,
                                          tempering_exponent, penalty_lasso, penalty_sigma, penalty_tau)
        # E step
        p_kt = E_step_weights(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da, tempering_exponent, 
                              stop_at_vanishing_cluster=True, regularise_E_step=True)

        # if cluster vanishes, stop
        if bool((p_kt.sum(dim='observation')<1).any()):
            if t < min_steps:
                print('vanishing cluster: signal received, too soon to abort EM computations')
                print('regularised E step')
                p_kt = E_step_weights(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da, tempering_exponent, 
                                      stop_at_vanishing_cluster=False, regularise_E_step=True)

            else:
                print('vanishing cluster: signal received, aborting EM computations')
                return beta_t, sigma_t, mu_t, Sigma_t, tau_t
        # compute the difference with the previous E step
        E_step_difference = float(np.abs(p_kt-p_kt0).mean(dim='observation').sum()/2.)
        
        # The first time that the classification does not change: re-assess the lambda
        # do not check for that on the first steps
        if t>=min_steps:
            if bool((p_kt.argmax(dim='label')==p_kt0.argmax(dim='label')).all()) & (penalty_lasso_version==0):
                print('update lasso penalty')
                if platform.system() == 'Windows':
                    penalty_lasso = estimate_penalty_lasso_CV_R(Y_da, X_da, p_kt)
                else:
                    penalty_lasso = estimate_penalty_lasso_CV(Y_da, X_da, p_kt)
                # put this flag to 1: we will not update lambda anymore afterwards
                penalty_lasso_version = 1
        
        # save the previous E step 
        p_kt0 = p_kt.copy()
        
        # M step        
        beta_t, sigma_t, mu_t, Sigma_t, tau_t = M_step_lasso(Y_da, X_da, p_kt, penalty_lasso=penalty_lasso, penalty_sigma = penalty_sigma, 
                                                             penalty_tau = penalty_tau,  WT_X_da=WT_X_da, shrinkage = shrinkage)

        # get new (penalised) negative log likelihood
        neg_ll_t1 = EM_objective_function(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da=WT_X_da,
                                          tempering_exponent=tempering_exponent, penalty_lasso = penalty_lasso, 
                                          penalty_sigma=penalty_sigma, penalty_tau=penalty_tau)
        # compute ratio
        neg_ll_ratio = (neg_ll_t0 - neg_ll_t1) / np.abs(neg_ll_t0)
        
        t+=1
        print(neg_ll_ratio)
        #print(E_step_difference)
    print(f'final sparsity beta = {float(np.mean(beta_t==0))}')
        
    return beta_t, sigma_t, mu_t, Sigma_t, tau_t

# estimate the best lasso parameters with a CV GLMNET estimation for each class
def estimate_penalty_lasso_CV(Y_da, X_da, weights):
    K = len(weights.label)
    penalty_lasso = xr.DataArray(coords=[range(K)], dims=['label'])
    print()
    for k in range(K):
        lasso_model = ElasticNet(
                        # pure lasso: alpha=1, 
                        alpha=1, 
                        fit_intercept=True,
                        # multicore? try that
                        n_jobs=3)
        lasso_model.fit(X_da, Y_da, sample_weight=weights.sel(label=k))
        print(f'n_{k} = {float(weights.sel(label=k).sum())}')
        #penalty_lasso.loc[dict(label =k)] = float(weights.sel(label=k).sum(dim='observation'))*float(lasso_model.lambda_best_)
        # glmnet uses it own conventions for how lambda is normalised: 2 x my_lambda = N x lambda_glmnet
        new_penalty = float(0.5*len(weights.observation))*float(lasso_model.lambda_best_)
        penalty_lasso.loc[dict(label =k)] = new_penalty
        print(f'new_lambda_{k} = {new_penalty}')
    print()
        
    return penalty_lasso


# estimate the best lasso parameters with a CV GLMNET estimation for each class (R version)
def estimate_penalty_lasso_CV_R(Y_da, X_da, weights):
    K = len(weights.label)
    penalty_lasso = xr.DataArray(coords=[range(K)], dims=['label'])
    print()
    
    # declare "X" and 'Y' as global objects for the R environment
    robjects.globalenv["X"] = X_da.values
    robjects.globalenv["Y"] = Y_da.values
    
    for k in range(K):
        # declare these class weights
        robjects.globalenv["weights"] = weights.sel(label=k).values
        # run the cv lasso
        robjects.r(
        '''
        options(warn=-1)
        elasticnet_cv = cv.glmnet(as.matrix(X), as.vector(Y), family = "gaussian", alpha = 1,intercept = TRUE, weights =weights )
        ''')
        # glmnet uses it own conventions for how lambda is normalised: 2 x my_lambda = N x lambda_glmnet
        new_penalty = float(0.5*len(weights.observation))*float(np.array(robjects.r('elasticnet_cv$lambda.1se')))
        print(f'n_{k} = {float(weights.sel(label=k).sum())}')
        # update lambda
        penalty_lasso.loc[dict(label =k)] = new_penalty
        print(f'new_lambda_{k} = {new_penalty}')
    print()
        
    return penalty_lasso


####################################################### evaluate EM results #######################################################


########################## base

# compute the negative log-l for different sets of parameters (EM, "true" and initial)
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def compute_optimisation_metrics(Y_da, X_da,
                                 # EM estimated parameters
                                 beta_t, sigma_t, mu_t, Sigma_t, tau_t,
                                 # supervised estimated parameters
                                 beta=None, sigma=None, mu=None, Sigma=None, tau=None,
                                 # initial parameters
                                 beta_0=None, sigma_0=None, mu_0=None, Sigma_0=None, tau_0=None, 
                                 # projected data
                                 WT_X_da = None, 
                                 # tempering exponent
                                 tempering_exponent = 1.,
                                 # penalty intensities
                                 penalty_lasso = 1., 
                                 penalty_sigma=0.,
                                 penalty_tau=0.):

    ## we divide these by n for cleaner values
    n=len(Y_da)

    # reached unpenalised likelihood
    neg_ll_T_no_penalty = mixture_negative_log_likelihood(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da=WT_X_da,
                                                          tempering_exponent=tempering_exponent)
        
    # reached penalised likelihood    
    neg_ll_T = EM_objective_function(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da=WT_X_da, 
                                     tempering_exponent=tempering_exponent, penalty_lasso = penalty_lasso, 
                                     penalty_sigma=penalty_sigma, penalty_tau=penalty_tau)/n

    neg_ll_true = None
    if not (beta is None):
        # real empirical parameters 
        neg_ll_true = EM_objective_function(Y_da, X_da, beta, sigma, mu, Sigma, tau, WT_X_da=WT_X_da, 
                                         tempering_exponent=tempering_exponent, penalty_lasso = penalty_lasso, 
                                         penalty_sigma=penalty_sigma, penalty_tau=penalty_tau)/n

    neg_ll_0 = None
    if not (beta_0 is None):
        # likelihood with initial parameters
        neg_ll_0 = EM_objective_function(Y_da, X_da, beta_0, sigma_0, mu_0, Sigma_0, tau_0, WT_X_da=WT_X_da, 
                                         tempering_exponent=tempering_exponent, penalty_lasso = penalty_lasso, 
                                         penalty_sigma=penalty_sigma, penalty_tau=penalty_tau)/n
        
    return neg_ll_T_no_penalty, neg_ll_T, neg_ll_true, neg_ll_0

# for classification: we need to properly align the classes: hugarian algorithm.
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def get_optimal_label_permutation(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da = None,
                                  tempering_exponent=1., metric = 'soft_error'):

    # classification error

    # estimated class weights for each data point
    # use the correct E step function for the correct EM mode

    # estimated class probabilities
    p_kT = E_step_weights(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da = WT_X_da, 
                          tempering_exponent=tempering_exponent, stop_at_vanishing_cluster=False, regularise_E_step=True)


    # rename dimension
    p_kT = p_kT.rename({'label':'proposed_label'})
    
    if metric == 'soft_error':
        # difference table
        cost_table = np.abs(p_kT - labels_da).mean(dim='observation')/2

    if metric == 'hard_error':
        # estimated hard assignment
        hard_p_kT = 1*(p_kT == p_kT.max(dim='proposed_label'))
        # difference table
        cost_table = np.abs(hard_p_kT - labels_da).mean(dim='observation')/2

    # Solve the linear_sum_assignment problem
    _, optimal_permutation = linear_sum_assignment(cost_table)
    # table: proposed_label x true_label
    # hence, optimal_permutation: proposed_label -> true_label
    # this is directly the correct order
    
    return optimal_permutation

# reoder the label dimension of a data array given a permutation
def reorder_labels(da, permutation):
    return da.assign_coords({'label':permutation}).sortby('label')

# reoder the label dimension of all model parameters given a permutation
def reoder_labels_parameters(beta, sigma, mu, Sigma, tau, permutation):
    beta_reordered = reorder_labels(beta, permutation)
    sigma_reordered = reorder_labels(sigma, permutation)
    mu_reordered = reorder_labels(mu, permutation)
    Sigma_reordered = reorder_labels(Sigma, permutation)
    tau_reordered = reorder_labels(tau, permutation)
    return beta_reordered, sigma_reordered, mu_reordered, Sigma_reordered, tau_reordered

# compute the model based classification error
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def compute_classification_metrics(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                   # projected data
                                   WT_X_da = None,
                                   # exponent of block X
                                   tempering_exponent=1., 
                                   # should we return the estimated hard labels?
                                   return_labels = False):

    # estimated class weight by data point
    p_kT = E_step_weights(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da = WT_X_da, 
                          tempering_exponent=tempering_exponent, stop_at_vanishing_cluster=False, regularise_E_step=True)

    # soft assigment error
    soft_error = float(np.abs(p_kT - labels_da).mean()/2)

    # estimated hard assignment
    hard_p_kT = 1*(p_kT == p_kT.max(dim='label'))
    # hard assignment error
    hard_error = float(np.abs(hard_p_kT - labels_da).mean()/2)

    # also get the rand index
    rand_index = adjusted_rand_score(labels_da.argmax(dim='label'), hard_p_kT.argmax(dim='label'))
    
    if return_labels:
        return soft_error, hard_error, rand_index, hard_p_kT
        
    return soft_error, hard_error, rand_index

# have functions that compute the likelihood in X and Y separately, for OoS evaluation

# gaussian log likelihood (weighted by class probability "tau") of each data point X_i for each class k (in the embedding space)
def gaussian_log_likelihood_X(X_da, mu_da, Sigma_da, tau_da, tempering_exponent=1):
    n = len(X_da.observation)
    K = len(mu_da.label)
    
    ## log phi WT.X
    log_phi = np.zeros((n, K))
    for k in range(K):
        log_phi[:, k] = stats.multivariate_normal.logpdf(X_da, mean=mu_da.sel(label=k), cov=Sigma_da.sel(label=k), allow_singular=True)

    log_phi_da = xr.DataArray(
        data = log_phi, 
        coords=[range(n), range(K)], 
        dims=['observation', 'label'])
    
    # 'temper' the X bloc to afjust its effect on the class weights
    log_phi_da = tempering_exponent*log_phi_da
    ## log phi WT.X + log phi tau
    log_p = log_phi_da+np.log(tau_da)
    
    return log_p

'''
# total mixture neg-ll over all observations, for X only (no Y)
def mixture_negative_log_likelihood_X(X_da, mu_da, Sigma_da, tau_da, tempering_exponent=1):
    # log(p(X|z=k)*p(z=k)): gaussian log likelihood (weighted by class probability "tau") 
    log_p =  gaussian_log_likelihood_X(X_da, mu_da, Sigma_da, tau_da, tempering_exponent=tempering_exponent)
    # single out the maximum values to avoid computation errors
    log_max = log_p.max(dim='label')
    # log p(X) = sum_i sum_l(p(X|z=l)*p(z=l))
    mixture_log_likelihood = float((np.log(np.exp(log_p - log_max).sum(dim='label')) + log_max).sum())
    return -mixture_log_likelihood
'''

# prediction hat{Y} := E_{theta}[Y|X]
# also returns the total mixture neg-ll over all observations, for X only (no Y)
def prediction_Y(X_da, X_beta_da, mu_da, Sigma_da, tau_da, tempering_exponent=1, return_X_likelihood=False):
    
    ## compute p(z=k|X) = p(X|z=k)*p(z=k) / sum_l(p(X|z=l)*p(z=l))
    
    # log(p(X|z=k)*p(z=k)): gaussian log likelihood (weighted by class probability "tau") 
    log_p = gaussian_log_likelihood_X(X_da, mu_da, Sigma_da, tau_da, tempering_exponent=tempering_exponent)
    # single out the maximum values to avoid computation errors
    log_max = log_p.max(dim='label')

    # p(z=k|X)
    p_z_conditional_X = np.exp(log_p - log_max)/np.exp(log_p - log_max).sum(dim='label')
    
    ## for each data point: weighted average of the dotproduct X_beta_da over each class 
    Y_hat = (X_beta_da * p_z_conditional_X).sum(dim='label')
    
    if return_X_likelihood:
        # log p(X) = sum_i sum_l(p(X|z=l)*p(z=l))
        mixture_neg_log_likelihood = -float((np.log(np.exp(log_p - log_max).sum(dim='label')) + log_max).sum())
    
        return Y_hat, mixture_neg_log_likelihood
    return Y_hat

# compute the prediction error on Y and the negative mixture log-likelihood on X (meant to be mostly used on OoS data)
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def compute_prediction_metrics(Y_da, X_da, 
                               # EM estimated parameters
                               beta_t, sigma_t, mu_t, Sigma_t, tau_t,
                               # projected data
                               WT_X_da = None,
                               # tempering exponent
                               tempering_exponent = 1.,):
    
    n = len(Y_da.observation)
    
    #EM_mode = 'Y~X'
    X_beta_t = compute_conditional_expectation(X_da, beta_t)
    
    # ambient space or embedding space?
    if WT_X_da is None:
        # prediction hat{Y} := E_{theta}[Y|X] and total mixture neg-ll over all observations, for X only (no Y)
        Y_hat, neg_ll_X = prediction_Y(X_da, X_beta_t, mu_t, Sigma_t, tau_t,
                                       tempering_exponent=tempering_exponent, return_X_likelihood=True)        
    else:
        # prediction hat{Y} := E_{theta}[Y|X] and total mixture neg-ll over all observations, for X only (no Y)
        Y_hat, neg_ll_X = prediction_Y(WT_X_da, X_beta_t, mu_t, Sigma_t, tau_t,
                                       tempering_exponent=tempering_exponent, return_X_likelihood=True) 
    
    # normalise ll
    neg_ll_X = neg_ll_X/n
    # compute mse
    mse_Y = float(np.mean((Y_da-Y_hat)**2))
    
    return neg_ll_X, mse_Y

# metrics involving the oracle model based classifier
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def compute_oracle_metrics(Y_da, X_da, labels_da, 
                           beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                           beta_true, sigma_true, mu_true, Sigma_true, tau_true, 
                           # projected data
                           WT_X_da=None, 
                           # exponent of block X
                           tempering_exponent=1.):
    # Oracle Model Based Classifier Error
    soft_error_oracle, hard_error_oracle, rand_index_oracle, labels_oracle =\
        compute_classification_metrics(Y_da, X_da, labels_da, 
                                       beta_true, sigma_true, mu_true, Sigma_true, tau_true, 
                                       WT_X_da=WT_X_da, tempering_exponent=tempering_exponent, return_labels=True)

    # compute the classification errors with the labels estimated by the oracle model based classifier as reference
    _, _, rand_index_vs_oracle = compute_classification_metrics(Y_da, X_da, labels_oracle,  
                                                                beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                                                WT_X_da = WT_X_da, 
                                                                tempering_exponent=tempering_exponent)
    
    return soft_error_oracle, hard_error_oracle, rand_index_oracle, rand_index_vs_oracle




# optimisation metrics + classification metrics + prediction metrics (Y and X separated) + oracle related metrics
# works for both ambient space (phi(X)) and embedding space (phi(WtX))
def compute_metrics(Y_da, X_da, labels_da,
                    # EM estimated parameters 
                    beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                    # supervised estimated parameters
                    beta=None, sigma=None, mu=None, Sigma=None, tau=None,
                    # real precision matrix, for graph sparsistency
                    #Omega = None,
                    # initial parameters
                    beta_0=None, sigma_0=None, mu_0=None, Sigma_0=None, tau_0=None, 
                    # projected data
                    WT_X_da = None, 
                    # exponent of block X
                    tempering_exponent=1., 
                    # penalty intensities
                    penalty_lasso = 1., 
                    penalty_sigma = 0.,
                    penalty_tau = 0.,
                    # do we return the estimated labels?
                    return_labels = False, 
                    # do we skip all metrics except the optimisiation ones? (used when K=/=K_true)
                    only_optimisation_metrics = False):
    
    ### Estimate optimisation metrics

    # negative log likelihoods of each parameter sets
    neg_ll_T_no_penalty, neg_ll_T, neg_ll_true, neg_ll_0 =  compute_optimisation_metrics(Y_da, X_da,
                                                                    # EM estimated parameters
                                                                    beta_t, sigma_t, mu_t, Sigma_t, tau_t,
                                                                    # supervised estimated parameters
                                                                    beta, sigma, mu, Sigma, tau,
                                                                    # initial parameters
                                                                    beta_0, sigma_0, mu_0, Sigma_0, tau_0,
                                                                    # projected data
                                                                    WT_X_da=WT_X_da, 
                                                                    # tempering exponent
                                                                    tempering_exponent = tempering_exponent,
                                                                    # penalty intensities
                                                                    penalty_lasso = penalty_lasso, 
                                                                    penalty_sigma = penalty_sigma,
                                                                    penalty_tau = penalty_tau)
    # if K=/=K_true, we only compute optimisation metrics
    if only_optimisation_metrics:
        # the rand index can still be computed even with K=/=K_true
        
        # estimated class weight by data point
        p_kT = E_step_weights(Y_da, X_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da = WT_X_da, 
                              tempering_exponent=tempering_exponent, stop_at_vanishing_cluster=False, regularise_E_step=True)
        # estimated hard assignment
        hard_p_kT = 1*(p_kT == p_kT.max(dim='label'))
        # also get the rand index
        rand_index = adjusted_rand_score(labels_da.argmax(dim='label'), hard_p_kT.argmax(dim='label'))
        
        
        # output: list of metrics
        output = dict(neg_ll_T_no_penalty = neg_ll_T_no_penalty, neg_ll_T =  neg_ll_T, neg_ll_true = neg_ll_true, neg_ll_0 = neg_ll_0, rand_index=rand_index)
        return output

    ### Estimate classification metrics

    # get errors with the new ordering
    classification_metrics = compute_classification_metrics(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t,
                                                            # projected data
                                                            WT_X_da = WT_X_da,
                                                            # exponent of block X
                                                            tempering_exponent=tempering_exponent, 
                                                            # should we return the estimated hard labels?
                                                            return_labels = return_labels)
    
    soft_error, hard_error, rand_index = classification_metrics[:3]
    
    ### Estimate prediction metrics

    neg_ll_X, mse_Y = compute_prediction_metrics(Y_da,  X_da,
                                                 # EM estimated parameters
                                                 beta_t, sigma_t, mu_t, Sigma_t, tau_t,
                                                 # projected data
                                                 WT_X_da = WT_X_da,
                                                 # tempering exponent
                                                 tempering_exponent = tempering_exponent)
    # output: list of metrics
    output = dict(
        # optimisation metrics
        neg_ll_T_no_penalty = neg_ll_T_no_penalty, neg_ll_T =  neg_ll_T, neg_ll_true = neg_ll_true, neg_ll_0 = neg_ll_0, 
        # classification metrics
        soft_error = soft_error, hard_error = hard_error, rand_index = rand_index, 
        # prediction metrics
        neg_ll_X  = neg_ll_X, mse_Y = mse_Y)
        
    # if we evaluated the oracle metrics
    if not (mu is None):
        
        ### Estimate the sparsistency of beta 
        P, N, TP, TN, _, _, _, _ = compute_beta_confusion_metrics(beta_t, beta, sum_over_labels=True)
        # da to int
        P, N, TP, TN = int(P), int(N), int(TP), int(TN)

        ### Estimate the oracle model based classifier related metrics
        soft_error_oracle, hard_error_oracle, rand_index_oracle, rand_index_vs_oracle =\
            compute_oracle_metrics(Y_da, X_da, labels_da, 
                                   beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                   beta, sigma, mu, Sigma, tau, 
                                   # projected data
                                   WT_X_da=WT_X_da, 
                                   # exponent of block X
                                   tempering_exponent=tempering_exponent)
        output.update(dict(
            # spasistency beta
            P = P, N = N, TP = TP, TN = TN, 
            # classification metrics oracle
            soft_error_oracle = soft_error_oracle, hard_error_oracle = hard_error_oracle, rand_index_oracle = rand_index_oracle, 
            # classification metrics wrt oracle
            rand_index_vs_oracle = rand_index_vs_oracle))
        
    ### estimated labels if we need them
    if return_labels:
        output.update({'estimated labels':classification_metrics[-1]})

    return output


########################## Evaluate Classification Error with the true parameters

def get_true_parameters(X_da, labels_dummies_da, shrinkage='oas'):
    # real (empirical) mu 
    mu_true = (X_da*labels_dummies_da).sum(dim='observation') / labels_dummies_da.sum(dim='observation')
    # real (empirical) tau 
    tau_true = labels_dummies_da.mean(dim='observation')

    # Sigma is the problem, the empirical covariance is not good enough (not invertible)
    # shrink the empirical Sigma:
    Sigma_true = estimate_Sigma(X_da, mu_true, weights=labels_dummies_da, shrinkage='oas')
    
    return mu_true, Sigma_true, tau_true


####################### reorder estimated parameters with regards to labels_da
def optimaly_reorder_parameters(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                WT_X_da=None, tempering_exponent=1.):
    
    K = len(beta_t.label)
    
    # for classification: we need to properly align the classes: hugarian algorithm.
    optimal_permutation = get_optimal_label_permutation(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                                        WT_X_da=WT_X_da, tempering_exponent=tempering_exponent)

    # reorder the parameter labels accordingly
    beta_t, sigma_t, mu_t, Sigma_t, tau_t = reoder_labels_parameters(beta_t, sigma_t, mu_t, Sigma_t, tau_t, optimal_permutation)

    # check that it worked:
    optimal_permutation = get_optimal_label_permutation(Y_da, X_da, labels_da, beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                                        WT_X_da=WT_X_da, tempering_exponent=tempering_exponent)

    if (optimal_permutation != np.arange(K)).any():
        print('bug in the label re-ordering')

    return beta_t, sigma_t, mu_t, Sigma_t, tau_t

########################## sparsistency analysis



# get the T, P, TP, TN... quantities for the ROC analysis
def get_confusion_metrics(vector_estimated, vector_true, return_all_metrics=False, sum_over_labels = True):
    
    # we must have the dimension to sum along named "component..."
    component_name = [dim for dim in vector_true.dims if dim[:9]=='component'][0]
    # list of dims over which we take the sum
    dims_sum = [component_name]+['label']*sum_over_labels
    
    # real positive 
    P = (vector_true!=0).sum(dim=dims_sum)
    # real negative 
    N = (vector_true==0).sum(dim=dims_sum)

    # agreement on the sparsity
    same_sparsity = (vector_estimated!=0) == (vector_true!=0)

    # true positives
    TP = (same_sparsity*(vector_true!=0)).sum(dim=dims_sum)
    # true negatives
    TN = (same_sparsity*(vector_true==0)).sum(dim=dims_sum)
    
    # we also return the norms of the difference
    l0_sign = (np.sign(vector_estimated)!=np.sign(vector_true)).mean(dim=dims_sum) 
    l1_sign = np.abs(np.sign(vector_estimated)-np.sign(vector_true)).mean(dim=dims_sum)
    l1 = np.abs(vector_estimated-vector_true).mean(dim=dims_sum) / np.abs(vector_true).mean(dim=dims_sum)
    l2 = ((vector_estimated-vector_true)**2).mean(dim=dims_sum)**0.5 / (vector_true**2).mean(dim=dims_sum)**0.5 

    
    # do we want all the "redundant" metrics as well?
    if return_all_metrics:
        # false positives
        FP = N - TN
        # false negatives
        FN = P - TP

        # True Positive rate
        TPR = TP/P
        # False Positive rate
        FPR = FP/N
        # True Negative rate
        TNR = TN/N 
        # False Negative rate
        FNR = FN/P

        # precision or positive predictive value (PPV)
        PPV = TP/(TP + FP)
        # accuracy (ACC)
        ACC = (TP + TN)/(P+N)

        return P, N, TP, TN, FP, FN, TPR, FPR, TNR, FNR, PPV, ACC, l0_sign, l1_sign, l1, l2
    return P, N, TP, TN, l0_sign, l1_sign, l1, l2

# metrics involving the sparsistency of the beta estimation at the end of the EM
def compute_beta_confusion_metrics(beta_t, beta, return_all_metrics=False, sum_over_labels=True):
    p = len(beta.component_p) - 1
    # exclude the intercept
    beta_no_intercept = beta.sel(component_p=range(p)).copy()
    beta_no_intercept_t = beta_t.sel(component_p=range(p)).copy()
        
    confusion_metrics_beta = get_confusion_metrics(beta_no_intercept_t, beta_no_intercept, 
                                                   return_all_metrics=return_all_metrics, sum_over_labels=sum_over_labels)
    
    return confusion_metrics_beta

def get_off_diagonal_coefficients(M_da):
    # we must have the dimension to sum along named "component..."
    component_name = [dim for dim in M_da.dims if dim[:9]=='component'][0]
    p = len(M_da[component_name])
    K = len(M_da.label)
    
    if 'solution' in M_da.dims:
        n_solutions = len(M_da.solution)
        M_off_diagonal = xr.DataArray(coords = [range(K), range(p*(p-1)//2), range(n_solutions)], dims = ['label', 'component', 'solution'])
    else:
        M_off_diagonal = xr.DataArray(coords = [range(K), range(p*(p-1)//2)], dims = ['label', 'component'])

    for k in range(K):
        if 'solution' in M_da.dims:
            for solution in range(n_solutions):
                M_off_diagonal.loc[dict(label=k)] = M_da.sel(label=k).values[np.triu_indices(p, k=1)]
        else:
            M_off_diagonal.loc[dict(label=k)] = M_da.sel(label=k).values[np.triu_indices(p, k=1)]
        
    return M_off_diagonal

# metrics involving the sparsistency of the Omega estimation at the end of the EM
def compute_Omega_confusion_metrics(Omega_t, Omega, return_all_metrics=False, sum_over_labels=True):
    Omega_off_diagonal = get_off_diagonal_coefficients(Omega)
    Omega_off_diagonal_t = get_off_diagonal_coefficients(Omega_t)

    confusion_metrics_Omega = get_confusion_metrics(Omega_off_diagonal_t, Omega_off_diagonal,
                                                    return_all_metrics=return_all_metrics, sum_over_labels=sum_over_labels)

    return confusion_metrics_Omega 

# get the roc metric on a path of lasso solutions for beta, update the provided data array
def roc_path_beta(
    # data
    Y_da, X_da, labels_da, 
    # true parameters, 
    beta_true, 
    # data array to update
    local_sparsistency_roc_da):
    
    # get dimesions
    p = len(X_da.component_p)
    K = len(beta_true.label)
    # number of silution to compute
    n_solutions = len(local_sparsistency_roc_da.solution)

    # initialise estimated parameters
    beta_t = xr.DataArray(coords=[range(K),  range(p+1), range(n_solutions)], dims=['label', 'component_p', 'solution'])

    for k in range(K):
        # data truely belonging to this class
        Y_k = Y_da[labels_da==k]
        X_k = X_da[labels_da==k]
        
        # if empty class, just use all data 
        if len(X_k)==0:
            # when K=2, this is the same as having the same estimate for both classes
            Y_k = Y_da.copy()
            X_k = X_da.copy()
        
        # a statement "if len(labels_da==k)==0" is not enough to catch all exceptions
        try:
            # trying to scale X with an skl function should to catch low sample size exceptions
            myScaler = StandardScaler()
            # this does not update X_k, even if it works, it is purely a test line of code
            myScaler.fit_transform(X_k)
        except ValueError as e:
            if str(e)[:28] == 'Found array with 0 sample(s)':
                # if empty class, just use all data 
                # when K=2, this is the same as having the same estimate for both classes
                Y_k = Y_da.copy()
                X_k = X_da.copy()
            else:
                raise e
        
        ##################################################################
        
        # number of observations
        n_k = len(X_k)
        # proxy for the relevant penalty scale
        penalty_scale = float((0.5*np.log(p)/n_k)**0.5)*50
        # beta needs a higher penalty scale than Omega

        # list of penalties
        alpha_list = list(np.logspace(0.5, -2, num = n_solutions)*penalty_scale)
        
        # try the lowest penalty to see if we need even more regularisation
        try:
            lasso_model = skl_ElasticNet(l1_ratio=1, alpha=alpha_list[-1], fit_intercept=True, max_iter=5000, warm_start=True)
            lasso_model.fit(X_k, Y_k)   
        except FloatingPointError:
            print('increase penalty scale')
            # update list of penalties
            alpha_list *= 5
            
        # Lasso path
        for solution_id, alpha in enumerate(alpha_list):
            print(alpha)
            # still possible to get exceptions here
            try:
                # l1 penalised beta estimation (lasso)
                lasso_model = skl_ElasticNet(l1_ratio=1, alpha=alpha, fit_intercept=True,  
                                             max_iter=5000, warm_start=True)
                lasso_model.fit(X_k, Y_k)
                
                # update the parameter
                beta_t.loc[dict(label =k, component_p=range(p), solution = solution_id)] = lasso_model.coef_
                #beta_t.loc[dict(label =k, component_p=p, solution = solution_id)]= lasso_model.intercept_
                
                print(f'beta non zero entries : {(lasso_model.coef_!=0).sum()}')

            except FloatingPointError:
                print('Lasso not computable')
        
        ##################################################################
        
        # version with the function lasso_path (no control over the lambda list)
        # disabled for now
        if False:
            try:
                # lasso path
                _, coeffs, _ = lasso_path(X = X_k, y = Y_k, n_alphas=n_solutions, fit_intercept = True)
                # estimated coeff (without the intercept)
                beta_t.loc[dict(label =k, component_p=range(p))] = coeffs
            except ValueError as e:
                if str(e)[:28] == 'Found array with 0 sample(s)':
                    #print(str(e)[:28])
                    # if empty class, just use all data 
                    # when K=2, this is the same as having the same estimate for both classes

                    # lasso path
                    _, coeffs, _ = lasso_path(X = X_da, y = Y_da, n_alphas=n_solutions, fit_intercept = True)
                    # estimated coeff (without the intercept)
                    beta_t.loc[dict(label =k, component_p=range(p))] = coeffs
                else:
                    raise e

    # compute the confusion metrics on the path
    P, N, TP, TN, l0_sign, l1_sign, l1, l2 = compute_beta_confusion_metrics(beta_t, beta_true, sum_over_labels=True)
    # put the results in a dictionnary to properly name them
    output = dict(P=P, N=N, TP=TP, TN=TN, l0_sign=l0_sign, l1_sign=l1_sign, l1=l1, l2=l2)

    # update the data array
    for metric in local_sparsistency_roc_da.metric.values:
        local_sparsistency_roc_da.loc[dict(parameter = 'beta', metric = metric)] = output[metric]
        
# get the roc metric on a path of glasso solutions for Omega, update the provided data array
def roc_path_Omega_glasso(
    # data
    Y_da, X_da, labels_da, 
    # true parameters, 
    Omega_true, 
    # data array to update
    local_sparsistency_roc_da):
    
    # get dimesions
    p = len(X_da.component_p)
    K = len(Omega_true.label)
    # number of silution to compute
    n_solutions = len(local_sparsistency_roc_da.solution)

    # initialise estimated parameters
    Omega_t = xr.DataArray(coords=[range(K), range(p), range(p), range(n_solutions)], dims=['label', 'component_p', 'component_pT', 'solution'])

    for k in range(K):
        # data truely belonging to this class
        X_k = X_da[labels_da==k].copy()
        
        # if empty class, just use all data 
        if len(X_k)==0:
            # when K=2, this is the same as having the same estimate for both classes
            X_k = X_da.copy()
        
        # scale X beforhand to avoid computational errors
        myScaler = StandardScaler()
        try:
            X_k = myScaler.fit_transform(X_k)
        # this can fail even if n_k>0
        except ValueError as e:
            if str(e)[:28] == 'Found array with 0 sample(s)':
                # if empty class, just use all data 
                # when K=2, this is the same as having the same estimate for both classes
                X_k = X_da.copy()
                X_k = myScaler.fit_transform(X_k)
            else:
                raise e
        
        # l2 regularisation
        estimator = OAS()
        estimator.fit(X_k)
        emp_cov = estimator.covariance_

        # number of observations
        n_k = len(X_k)
        # proxy for the relevant penalty scale
        penalty_scale = float((0.5*np.log(p)/n_k)**0.5)
        #penalty_scale = np.maximum(float((0.5*np.log(p)/n_k)**0.5), 1)
        #penalty_scale = 1

        # list of penalties
        alpha_list = np.logspace(0.5, -2, num = n_solutions)*penalty_scale

        # warm start: use the previously computed matrix as initialisation
        cov_init = None
        
        # try the lowest penalty to see if we need even more regularisation
        try:
            cov, precision = graphical_lasso(emp_cov, alpha = alpha_list[-1], cov_init = cov_init, mode='cd')  
        except FloatingPointError:
            print('additional covariance shrinkage')
            emp_cov = shrunk_covariance(emp_cov, shrinkage=0.8)
            
        # Graphical Lasso path
        for solution_id, alpha in enumerate(alpha_list):
            print(alpha)
            # still possible to get exceptions here
            try:
                # l1 regularisation: Glasso
                cov, precision = graphical_lasso(emp_cov, alpha = alpha, cov_init = cov_init, mode='cd')                
                # estimated precision
                Omega_t.loc[dict(label =k, solution = solution_id)] = precision
                print(f'Omega glasso non zero entries : {((precision!=0).sum()-p)//2}')
                # update estimated covariance for warm start
                cov_init = cov
            except FloatingPointError:
                print('Graphical Lasso not computable')
                
        
    # compute the confusion metrics on the path
    P, N, TP, TN, l0_sign, l1_sign, l1, l2 = compute_Omega_confusion_metrics(Omega_t, Omega_true, sum_over_labels=True)
    # put the results in a dictionnary to properly name them
    output = dict(P=P, N=N, TP=TP, TN=TN, l0_sign=l0_sign, l1_sign=l1_sign, l1=l1, l2=l2)

    # update the data array
    for metric in local_sparsistency_roc_da.metric.values:
        local_sparsistency_roc_da.loc[dict(parameter = 'Omega glasso', metric = metric)] = output[metric]
        
# get the roc metric on a path of glasso solutions for Omega, update the provided data array
def roc_path_Omega_mixed_binary(
    # data
    Y_da, X_da, labels_da, 
    # true parameters, 
    Omega_true, 
    # data array to update
    local_sparsistency_roc_da):
    
    # get dimesions
    p = len(X_da.component_p)
    K = len(Omega_true.label)
    # number of silution to compute
    n_solutions = len(local_sparsistency_roc_da.solution)

    # initialise estimated parameters
    Omega_t = xr.DataArray(coords=[range(K), range(p), range(p), range(n_solutions)], dims=['label', 'component_p', 'component_pT', 'solution'])

    for k in range(K):
        # data truely belonging to this class
        X_k = X_da[labels_da==k].copy()
        
        # if empty class, just use all data 
        if len(X_k)==0:
            # when K=2, this is the same as having the same estimate for both classes
            X_k = X_da.copy()
        
        # we do not scale X beforhand for this estimator
        myScaler = StandardScaler()
        try:
            # this is just to test that n_k is large enough
            myScaler.fit_transform(X_k)
        except ValueError as e:
            if str(e)[:28] == 'Found array with 0 sample(s)':
                # if empty class, just use all data 
                # when K=2, this is the same as having the same estimate for both classes
                X_k = X_da.copy()
            else:
                raise e
        
        ## get the mixed rank based estimator of the covariance matrix
        
        # get a dataframe with the right colomns name (for ordering)
        X_k = pd.DataFrame(X_k.values, columns = range(p))
        # put the categorical variable together on the first bloc
        X_reordered = isolate_categorical_columns(X_k, categorical_threshold=0.25)
        # compute the rank based estimator of the empirical covariance
        R_hat = compute_rank_based_estimator(X_reordered)
        # permutation to go back to to the initial column ordering
        columns_order = np.argsort(X_reordered.columns)
        # reorder the columns of the estimator to match the original ordering
        emp_cov = R_hat[:, columns_order][columns_order, :]

        # l2 regularisation
        #estimator = OAS()
        #estimator.fit(X_k)
        #emp_cov = estimator.covariance_

        # number of observations
        n_k = len(X_k)
        # proxy for the relevant penalty scale
        penalty_scale = 2*float((0.5*np.log(p)/n_k)**0.5)
        #penalty_scale = np.maximum(float((0.5*np.log(p)/n_k)**0.5), 1)
        #penalty_scale = 1

        # list of penalties
        alpha_list = np.logspace(0.5, -2, num = n_solutions)*penalty_scale

        # warm start: use the previously computed matrix as initialisation
        cov_init = None
        
        # try the lowest penalty to see if we need even more regularisation
        try:
            cov, precision = graphical_lasso(emp_cov, alpha = alpha_list[-1], cov_init = cov_init, mode='cd')  
        except FloatingPointError:
            print('additional covariance shrinkage')
            emp_cov = shrunk_covariance(emp_cov, shrinkage=0.8)
            
        # Graphical Lasso path
        for solution_id, alpha in enumerate(alpha_list):
            print(alpha)
            # still possible to get exceptions here
            try:
                # l1 regularisation: Glasso
                cov, precision = graphical_lasso(emp_cov, alpha = alpha, cov_init = cov_init, mode='cd')                
                # estimated precision
                Omega_t.loc[dict(label =k, solution = solution_id)] = precision
                print(f'Omega mixed_binary non zero entries : {((precision!=0).sum()-p)//2}')
                # update estimated covariance for warm start
                cov_init = cov
            except FloatingPointError:
                print('Graphical Lasso not computable')
                
        
    # compute the confusion metrics on the path
    P, N, TP, TN, l0_sign, l1_sign, l1, l2 = compute_Omega_confusion_metrics(Omega_t, Omega_true, sum_over_labels=True)
    # put the results in a dictionnary to properly name them
    output = dict(P=P, N=N, TP=TP, TN=TN, l0_sign=l0_sign, l1_sign=l1_sign, l1=l1, l2=l2)

    # update the data array
    for metric in local_sparsistency_roc_da.metric.values:
        local_sparsistency_roc_da.loc[dict(parameter = 'Omega mixed_binary', metric = metric)] = output[metric]
        
# get the roc metric on a path of glasso solutions for Omega, update the provided data array
def roc_path_Omega_non_paranormal(
    # data
    Y_da, X_da, labels_da, 
    # true parameters, 
    Omega_true, 
    # data array to update
    local_sparsistency_roc_da):
    
    # get dimesions
    p = len(X_da.component_p)
    K = len(Omega_true.label)
    # number of silution to compute
    n_solutions = len(local_sparsistency_roc_da.solution)

    # initialise estimated parameters
    Omega_t = xr.DataArray(coords=[range(K), range(p), range(p), range(n_solutions)], dims=['label', 'component_p', 'component_pT', 'solution'])

    for k in range(K):
        # data truely belonging to this class
        X_k = X_da[labels_da==k].copy()
        
        # if empty class, just use all data 
        if len(X_k)==0:
            # when K=2, this is the same as having the same estimate for both classes
            X_k = X_da.copy()
        
        # we do not scale X beforhand for this estimator
        myScaler = StandardScaler()
        try:
            # this is just to test that n_k is large enough
            myScaler.fit_transform(X_k)
        except ValueError as e:
            if str(e)[:28] == 'Found array with 0 sample(s)':
                # if empty class, just use all data 
                # when K=2, this is the same as having the same estimate for both classes
                X_k = X_da.copy()
            else:
                raise e
        
        ## get the non_paranormal estimator of the covariance matrix
        
        # declare "X" as a global object for the R environment
        robjects.globalenv["X"] = X_k.values
        # run the npn transformation of X
        robjects.r(
        '''
        options(warn=-1)
        df <- data.frame(X)
        X_npn <- huge.npn(df, npn.func = "shrinkage", npn.thresh = NULL,verbose = TRUE)
        ''')
        # get the transformed data from R 
        print('get the transformed data from R')
        X_k_npn = pd.DataFrame(np.array(robjects.r('X_npn')))
        # compute the empirical covariance
        print('compute the empirical covariance')
        emp_cov = np.array(X_k_npn.cov())

        # l2 regularisation
        #estimator = OAS()
        #estimator.fit(X_k)
        #emp_cov = estimator.covariance_

        # number of observations
        n_k = len(X_k)
        # proxy for the relevant penalty scale
        penalty_scale = 2*float((0.5*np.log(p)/n_k)**0.5)
        #penalty_scale = np.maximum(float((0.5*np.log(p)/n_k)**0.5), 1)
        #penalty_scale = 1

        # list of penalties
        alpha_list = np.logspace(0.5, -2, num = n_solutions)*penalty_scale

        # warm start: use the previously computed matrix as initialisation
        cov_init = None
        
        # try the lowest penalty to see if we need even more regularisation
        print('try the lowest penalty to see if we need even more regularisation')
        try:
            cov, precision = graphical_lasso(emp_cov, alpha = alpha_list[-1], cov_init = cov_init, mode='cd')  
        except FloatingPointError:
            print('additional covariance shrinkage')
            emp_cov = shrunk_covariance(emp_cov, shrinkage=0.8)
            
        # Graphical Lasso path
        for solution_id, alpha in enumerate(alpha_list):
            print(alpha)
            # still possible to get exceptions here
            try:
                # l1 regularisation: Glasso
                cov, precision = graphical_lasso(emp_cov, alpha = alpha, cov_init = cov_init, mode='cd')                
                # estimated precision
                Omega_t.loc[dict(label =k, solution = solution_id)] = precision
                print(f'Omega non_paranormal non zero entries : {((precision!=0).sum()-p)//2}')
                # update estimated covariance for warm start
                cov_init = cov
            except FloatingPointError:
                print('Graphical Lasso not computable')
                
        
    # compute the confusion metrics on the path
    P, N, TP, TN, l0_sign, l1_sign, l1, l2 = compute_Omega_confusion_metrics(Omega_t, Omega_true, sum_over_labels=True)
    # put the results in a dictionnary to properly name them
    output = dict(P=P, N=N, TP=TP, TN=TN, l0_sign=l0_sign, l1_sign=l1_sign, l1=l1, l2=l2)

    # update the data array
    for metric in local_sparsistency_roc_da.metric.values:
        local_sparsistency_roc_da.loc[dict(parameter = 'Omega non_paranormal', metric = metric)] = output[metric]
        
# compute the ROC metrics (beta and Omega) for the provided labels
def compute_roc_path(Y_da, X_da, 
                     # labels (could be anything, estimated or real)
                     labels_da, 
                     # true parameters, 
                     beta_true, Omega_true,
                     # number of solutions to compute for the ROC curve
                     n_solutions, 
                     # parameters to estimate (with estimator variants)
                     parameter_list=['beta', 'Omega glasso']):    
    
    # initialise roc results dataarray
    roc_metric_list = ['P', 'N', 'TP', 'TN', 'l0_sign', 'l1_sign', 'l1', 'l2']
    #parameter_list = ['beta', 'Omega']
    
    # local sparsistency_roc_da
    local_sparsistency_roc_da = xr.DataArray(
            coords = [roc_metric_list,
                      parameter_list,
                      range(n_solutions),
            ], 
            dims = ['metric',
                    'parameter', 
                    'solution',
                    ])
    
    # we can only run this if all classes have at least on data point
    #if bool((labels_da.sum(dim='observation')>1).all()):

    # get the roc metric on a path of lasso solutions for beta, update the provided data array
    roc_path_beta(
        # data
        Y_da, X_da, labels_da, 
        # true parameters, 
        beta_true, 
        # data array to update
        local_sparsistency_roc_da)
    
    # Omega solutions
    for parameter in parameter_list:
        # get the roc metric on a path of glasso solutions for Omega, update the provided data array
        if parameter=='Omega glasso':
            roc_path_Omega_glasso(
                # data
                Y_da, X_da, labels_da, 
                # true parameters, 
                Omega_true, 
                # data array to update
                local_sparsistency_roc_da)
            
        # get the roc metric on a path of non_paranormal solutions for Omega, update the provided data array
        if parameter=='Omega non_paranormal':         
            roc_path_Omega_non_paranormal(
                # data
                Y_da, X_da, labels_da, 
                # true parameters, 
                Omega_true, 
                # data array to update
                local_sparsistency_roc_da)
            
        # get the roc metric on a path of mixed/binary solutions for Omega, update the provided data array
        if parameter=='Omega mixed_binary':        
            roc_path_Omega_mixed_binary(
                # data
                Y_da, X_da, labels_da, 
                # true parameters, 
                Omega_true, 
                # data array to update
                local_sparsistency_roc_da)

    return local_sparsistency_roc_da

# roc path with both true and estimated labels
def compute_roc_path_all_labels(Y_da, X_da, 
                                # true labels 
                                labels_da, 
                                # estimated labels
                                labels_t, 
                                # true parameters, 
                                beta_true, Omega_true,
                                # number of solutions to compute for the ROC curve
                                n_solutions, 
                                # parameters to estimate (with estimator variants)
                                parameter_list = ['beta', 'Omega glasso']):
    
    # initialise roc results dataarray
    roc_metric_list = ['P', 'N', 'TP', 'TN', 'l0_sign', 'l1_sign', 'l1', 'l2']
    #parameter_list = ['beta', 'Omega']
    label_type_list = ['estimated', 'true']
    
    # get the roc for both the true and the estimated labels
    local_sparsistency_roc_da = xr.DataArray(
        coords = [
            roc_metric_list,
            parameter_list,
            range(n_solutions),
            label_type_list
        ], 
        dims = [
            'metric', 
            'parameter', 
            'solution',
            'label_type'
        ])

    # roc with estimated labels
    local_sparsistency_roc_da.loc[dict(label_type = 'estimated')] =\
        compute_roc_path(Y_da, X_da, 
                         # labels (could be anything, estimated or real)
                         labels_t, 
                         # true parameters, 
                         beta_true, Omega_true,
                         # number of solutions to compute for the ROC curve
                         n_solutions,
                         # parameters to estimate (with estimator variants)
                         parameter_list)

    # roc with true labels
    local_sparsistency_roc_da.loc[dict(label_type = 'true')] =\
        compute_roc_path(Y_da, X_da, 
                         # labels (could be anything, estimated or real)
                         labels_da, 
                         # true parameters, 
                         beta_true, Omega_true,
                         # number of solutions to compute for the ROC curve
                         n_solutions, 
                         # parameters to estimate (with estimator variants)
                         parameter_list)
    
    return local_sparsistency_roc_da        
            
# the roc da total to return at the end of the "evaluate EM" function 
def initialise_roc_da(n_solutions, 
                      # EM : has final tempering, general : does not
                      mode = 'EM',
                      # settings
                      remove_final_tempering=False, 
                      uniformely_temper_metrics=False, 
                      evaluate_OoS_error=False, 
                      parameter_list=['beta', 'Omega glasso']):
    
    roc_metric_list = ['P', 'N', 'TP', 'TN', 'l0_sign', 'l1_sign', 'l1', 'l2']
    # sparse parameter to reconstruct
    parameter_list = parameter_list
    # are we looking at the estimated parameters with the estimated or the true labels?
    label_type_list = ['estimated', 'true']
    # data set used, train or validation
    dataset_type_list = ['IS']+['OoS']*evaluate_OoS_error
    # three possibilities: same as the EM, no tmp (T=1) or uniform at 1/x_dim
    final_tempering_list = ["same_as_EM"]+["none"]*remove_final_tempering+["1/x_dim"]*uniformely_temper_metrics

    coords_list = [roc_metric_list,
                   parameter_list,
                   range(n_solutions),
                   label_type_list,
                   dataset_type_list]
    
    dims_list = ['metric',
                 'parameter', 
                 'solution',
                 'label_type',
                 'dataset_type']
    
    # EM : has final tempering, general : does not
    if mode == 'EM':
        coords_list +=[final_tempering_list]
        dims_list += ['final_tempering']

    sparsistency_roc_da = xr.DataArray(coords = coords_list, dims = dims_list)
    
    return sparsistency_roc_da     


########################## utility for the large function

# compute the final tempering exponent from its name
def compute_final_tmp_exponent(final_tempering, EM_tempering_exponent, x_dim):
    if final_tempering == 'none':
        return 1.
    elif final_tempering == 'same_as_EM':
        return EM_tempering_exponent
    elif final_tempering == '1/x_dim':
        return 1./x_dim

########################## large function

# With a given dataset and projection, run the EM and compute all the success metrics along the way 
def evaluate_EM_JCR(
    # train data
    Y_train, X_train, labels_train, 
    # provide the projection
    projection_da = None,
    # or directly the encoded data (not necesarily linear encoding),
    WT_X_da = None,
    # initialisation
    beta_0 = None, sigma_0 = None, mu_0 = None, Sigma_0 = None, tau_0 = None,
    # real values of the simulated parameters (for Oracle loss)
    beta_true = None, sigma_true = None, mu_true = None, Sigma_true = None, tau_true = None, 
    # real precision matrix, for graph sparsistency
    Omega_true = None,
    # validation data (if desired)
    Y_val = None, X_val = None, labels_val = None,
    # EM stopping criteria
    neg_ll_ratio_threshold = 1e-5, 
    E_step_difference_threshold = 1e-3,
    max_steps = 100, 
    min_steps = 10,
    # do we regularise the estimated covariance matrices in the M step?
    shrinkage = 'oas',
    # exponent of bloc X
    tempering_exponent = 1.,
    # penalty intensities
    penalty_lasso = 1, 
    penalty_sigma = 0,
    penalty_tau = 0,
    # should we remove the tempering on X in the final class weigh estimation as well?
    remove_final_tempering=False,
    # should we estimate the final likelihood with a UNIFORMELY (1/q) tempered/normalised bloc X?
    uniformely_temper_metrics=False, 
    # do we evaluate the oracle error with the true parameters?
    evaluate_oracle_error=False, 
    # do we evaluate the ROC curve of the sparsistency in beta and Omega?
    evaluate_sparsistency_roc=False,
    # sparse parameters to estimate
    parameter_list=['beta', 'Omega glasso'],
    # number of solutions to compute for the ROC curve
    n_solutions = 20, 
    # do we skip all metrics except the optimisiation ones? (used when K=/=K_true)
    only_optimisation_metrics = False, 
    # desired K, if different from the real one
    K = None):
    
    # do we evaluate the OoS error?
    evaluate_OoS_error = not (Y_val is None)
    
    # get dimensions
    p = len(X_train.component_p)
    q = None
    # by default, we simply use the real K
    if K is None:
        K = len(labels_train.label)
    # the dimension of X in the space of interest (ambient by default)
    x_dim = p

    ############################## treat train data ##############################
    print('treat train data')

    # use generic names
    Y_da, X_da, labels_da = Y_train, X_train, labels_train
    
    # if embedding space
    if (not (WT_X_da is None)) or (not (projection_da is None)):
        # if only the projection is provided, need to project here
        if WT_X_da is None:
            # project X train
            print('project X train')
            # generic name (not "WT_X_train")
            WT_X_da = xr.dot(X_train, projection_da, dims = 'component_p')
        # get dim
        q = len(WT_X_da.component_q)
        # the dimension of X in the space of interest (embedding in this case)
        x_dim = q
        # if the true parameters are provided: create their projected version
        if not (mu_true is None):
            mu_true = xr.dot(mu_true, projection_da, dims='component_p')
            WT_Sigma_true = xr.dot(Sigma_true, projection_da, dims='component_p').rename({'component_q':'component_qT'})
            Sigma_true = xr.dot(WT_Sigma_true, projection_da.rename({'component_p':'component_pT'}), dims='component_pT')


    # If not provided, make a random initialisation
    if beta_0 is None:
        print('get a random initialisation')       
        beta_0, sigma_0, mu_0, Sigma_0, tau_0 = initialise_parameters(K, p, q)
        
    
    ############################### Run the EM ##############################
    print('EM')
    
    start_time = time.time()
    # run EM
    beta_t, sigma_t, mu_t, Sigma_t, tau_t = EM_JCR_lasso(Y_da, X_da, 
                                                         # model parameters
                                                         beta_0, sigma_0, mu_0, Sigma_0, tau_0, 
                                                         # projected data
                                                         WT_X_da = WT_X_da,
                                                         # stopping criteria
                                                         neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                                         E_step_difference_threshold = E_step_difference_threshold,
                                                         max_steps = max_steps, 
                                                         min_steps = min_steps,
                                                         # do we regularise the estimated covariance matrices?
                                                         shrinkage = shrinkage,
                                                         # exponent of the X-bloc in the E step
                                                         tempering_exponent=tempering_exponent, 
                                                         # penalty intensities
                                                         penalty_lasso = penalty_lasso, 
                                                         penalty_sigma = penalty_sigma,
                                                         penalty_tau = penalty_tau)
    
    end_time = time.time()
    # save the duration of the EM
    EM_duration = end_time - start_time
    
    # initialise metrics dictionary to save the results in
    #metrics_output={}
    #metrics_output['runtime'] = EM_duration
    
    
    ################################# data array for results evaluation ######################################
    
    # data set used, train or validation
    dataset_type_list = ['IS']+['OoS']*evaluate_OoS_error
        
    # three possibilities: same as the EM , no tmp (T=1) or uniform at 1/x_dim
    final_tempering_list = ["same_as_EM"]+["none"]*remove_final_tempering+["1/x_dim"]*uniformely_temper_metrics

    # list of all followed metrics, given the provided simulation settings  
    metric_list = get_metrics_list(evaluate_oracle_error)

    # initialise local results dataarray
    results_da = xr.DataArray(
        coords = [
            metric_list,
            final_tempering_list,
            dataset_type_list
        ], 
        dims = [
            'metric', 
            'final_tempering',
            'dataset_type'
        ])
        
    # first, update the clustering duration
    results_da.loc[dict(metric='clustering_duration')] = EM_duration
    
    # do we have ROC sparsistency metrics to compute and return?
    if evaluate_sparsistency_roc:
        sparsistency_roc_da = initialise_roc_da(n_solutions=n_solutions, 
                                                # EM : has final tempering, general : does not
                                                mode = 'EM',
                                                # settings
                                                remove_final_tempering=remove_final_tempering, 
                                                uniformely_temper_metrics=uniformely_temper_metrics, 
                                                evaluate_OoS_error=evaluate_OoS_error,
                                                parameter_list=parameter_list)
    
    ################################# beginning of results evaluation ######################################

    
    for dataset_type in dataset_type_list:
        # if we evaluate the OoS results, we need to use the OoS data
        if dataset_type == 'OoS':
            # Validation data set 
            print('OoS perfs')
            ### treat validation data
            # use generic names
            Y_da, X_da, labels_da = Y_val, X_val, labels_val
            # if embedding space
            if not (projection_da is None):
                # project X train
                # generic name (not "WT_X_val")
                WT_X_da = xr.dot(X_val, projection_da, dims = 'component_p')
            
        ################################# True parameters for the Oracle Model Based Classifier Error ######################################
        if evaluate_oracle_error:
            # check that we have the true beta and sigma
            if (beta_true is None) | (sigma_true is None):
                print('error: beta_true or sigma_true not provided')
                return
            print('Oracle perfs')
            # if the true X parameters are not provided (X is real data), then make a supervised estimation
            if (mu_true is None) | (Sigma_true is None) | (tau_true is None):
                print('get supervised empirical parameters')
                # get the X parameters empirically
                if WT_X_da is None:
                    mu_true, Sigma_true, tau_true = get_true_parameters(X_da, labels_da, shrinkage='oas')
                else:
                    mu_true, Sigma_true, tau_true = get_true_parameters(WT_X_da, labels_da, shrinkage='oas')
        
        # we follow the final tmp exponent: if it is the same twice in a row, then no need to redo anything
        # init at an impossible value
        final_tempering_exponent_0 = -1.
        
        for final_tempering in final_tempering_list:
            # get the final tempering as a float
            final_tempering_exponent = compute_final_tmp_exponent(final_tempering, tempering_exponent, x_dim)
            # get the name of these settings 
            #settings_name = f'{dataset_type}, tmp {final_tempering}' 
            
            # dictionnary that describes the setting
            settings_dictionary = dict(final_tempering=final_tempering, dataset_type=dataset_type)
            
            # we only need to do the work if this exponent is not the same as previously
            if (final_tempering_exponent != final_tempering_exponent_0):

                ############################### Reorder estimated labels to properly estimate classification errors ###########################
                print('reorder')

                # we cannot do that if K =/= K_true (and we don't need to if we only want optimisation metrics)
                if not only_optimisation_metrics:
                    # reorder estimated parameters with regards to labels_da
                    beta_t, sigma_t, mu_t, Sigma_t, tau_t = optimaly_reorder_parameters(Y_da, X_da, labels_da, 
                                                                                        beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                                                                        WT_X_da=WT_X_da, tempering_exponent=final_tempering_exponent)

                ############################################### compute performances #################################################
                print('perfs')

                # compute all metrics on the train data
                # we save the list since we can potentially re-use it later if tmp=1
                metrics = compute_metrics(Y_da, X_da, labels_da,
                                          # EM estimated parameters 
                                          beta_t, sigma_t, mu_t, Sigma_t, tau_t, 
                                          # supervised estimated parameters 
                                          beta_true, sigma_true, mu_true, Sigma_true, tau_true, 
                                          # initial parameters 
                                          beta_0, sigma_0, mu_0, Sigma_0, tau_0, 
                                          # projected data
                                          WT_X_da = WT_X_da,
                                          # exponent of block X
                                          tempering_exponent=final_tempering_exponent, 
                                          # penalty intensities
                                          penalty_lasso = penalty_lasso, 
                                          penalty_sigma = penalty_sigma,
                                          penalty_tau = penalty_tau, 
                                          # if we evaluate the ROC curve, we need the estimated labels
                                          return_labels = evaluate_sparsistency_roc, 
                                          # do we skip all metrics except the optimisiation ones? (used when K=/=K_true)
                                          only_optimisation_metrics = only_optimisation_metrics)
                                          
                #################################### compute ROC performances ####################################   
                if evaluate_sparsistency_roc & (not only_optimisation_metrics):
                    # recover the estimated labels
                    labels_t = metrics['estimated labels']
                    # convert dummies to labels
                    labels_t = labels_t.argmax(dim='label')
                    
                    # get the roc metrics for these settings as a data array
                    local_sparsistency_roc_da = compute_roc_path_all_labels(Y_da, X_da, 
                                                                            # true labels 
                                                                            labels_da.argmax(dim='label'), 
                                                                            # estimated labels
                                                                            labels_t, 
                                                                            # true parameters, 
                                                                            beta_true, Omega_true,
                                                                            # number of solutions to compute for the ROC curve
                                                                            n_solutions,
                                                                            # parameters to estimate (with estimator variants)
                                                                            parameter_list)
                    # update data array
                    sparsistency_roc_da.loc[settings_dictionary] = local_sparsistency_roc_da
            
            ############################################### update da  #################################################
            
            for idx, metric_name in enumerate(metrics.keys()):
                # prevent an exception if the estimated labels are present in metrics
                if not (metric_name == 'estimated labels'):

                    # dictionary with the current metric
                    settings_dictionary_ = dict(settings_dictionary, **dict(metric=metric_name))
                    # update
                    results_da.loc[settings_dictionary_] = metrics[metric_name]
            
            # update the final tmp exponent
            final_tempering_exponent_0 = final_tempering_exponent
    
    # create an easy to unwrap output
    output = {'results_da' : results_da}
    
    if evaluate_sparsistency_roc:
        output.update({'sparsistency_roc_da' : sparsistency_roc_da}) 
    
    return output


###################################################### Other algorithms as benchmark ######################################################

# large function containing all the other algorithms
def estimate_cluster_labels(Y_da, X_da, 
                            # algorithm to use
                            algorithm,
                            # parameters for each algorithm, indexed by name
                            parameters,
                            # validation data if provided
                            Y_val=None, X_val=None):
    
    # do we evaluate the OoS error?
    evaluate_OoS_error = not (Y_val is None)
    
    # by default, we assume that the method cannot be run on OoS data
    labels_val_estimated = None
    
    # convert to pandas
    X = pd.DataFrame(X_da.values)
    Y = pd.Series(Y_da.values)
    
    # by default, no validation set
    X_val_df, Y_val_df = None, None
    if evaluate_OoS_error:
        # convert to pandas
        X_val_df = pd.DataFrame(X_val.values)
        Y_val_df = pd.Series(Y_val.values)
        
    # if we are asked a clustering on X and Y jointly
    if algorithm.split('_')[-1] == 'XY':
        # join X and Y 
        XY = X.copy()
        XY['Y'] = Y
        # default:
        XY_val = None
        # if we provide validation data
        if evaluate_OoS_error:
            # join X and Y 
            XY_val = X_val_df.copy()
            XY_val['Y'] = Y_val_df
    
    # KMeans on X
    if algorithm == 'KMeans_X':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(X, 'KMeans', parameters, X_val = X_val_df)
            
    # KMeans on [X, Y]
    if algorithm == 'KMeans_XY':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(XY, 'KMeans', parameters, X_val = XY_val) 
        
    # SpectralClustering on X
    if algorithm == 'SpectralClustering_X':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(X, 'SpectralClustering', parameters, X_val = X_val_df)
            
    # SpectralClustering on [X, Y]
    if algorithm == 'SpectralClustering_XY':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(XY, 'SpectralClustering', parameters, X_val = XY_val) 
        
    # GaussianMixture on X
    if algorithm == 'GaussianMixture_X':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(X, 'GaussianMixture', parameters, X_val = X_val_df)
            
    # GaussianMixture on [X, Y]
    if algorithm == 'GaussianMixture_XY':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(XY, 'GaussianMixture', parameters, X_val = XY_val) 
        
    # MixtureOfExperts on [Y~X] (by design, there is no MoE on X alone)
    if algorithm == 'MixtureOfExperts_XY':
        duration, labels_estimated, labels_val_estimated = run_clustering_algorithm(XY, 'MixtureOfExperts', parameters, X_val = XY_val) 
        
    return duration, labels_estimated, labels_val_estimated

# template for the algorithms that fall in the "clustering" category, returns labels and execution time
def run_clustering_algorithm(X, algorithm, parameters, X_val = None):
    
    # dictionnary of possible methods
    algorithm_dict = {'KMeans':KMeans, 'SpectralClustering':SpectralClustering, 
                      'GaussianMixture':GaussianMixture, 'MixtureOfExperts':MixtureOfExperts}
    
    # by default, we assume that the method cannot be run on OoS data
    labels_val_estimated = None
    
    # initialise the method
    clustering_algorithm_model = algorithm_dict[algorithm](**parameters)

    # train the algorithm and get the labels
    start_time = time.time()
    labels_estimated = clustering_algorithm_model.fit_predict(X)
    end_time = time.time()
    
    # save the duration of the training and clustering
    duration = end_time - start_time
    
    # if we provide validation data (and the clustering algo has a "predict" method)
    if (not (X_val is None)) & ('predict' in dir(algorithm_dict[algorithm])):
        # get the labels
        labels_val_estimated = clustering_algorithm_model.predict(X_val)
    
    return duration, labels_estimated, labels_val_estimated

# reorder estimated dummy labels with regards to dummy labels_da
def optimaly_reorder_labels(labels_da, labels_estimated):
    
    # difference table
    cost_table = np.abs(labels_estimated.rename({'label':'proposed_label'}) - labels_da).mean(dim='observation')/2

    # Solve the linear_sum_assignment problem
    _, optimal_permutation = linear_sum_assignment(cost_table)

    # reorder
    labels_reordered = reorder_labels(labels_estimated, optimal_permutation)
    
    return labels_reordered

# compute error and rand index from the estimated labels, method usable on anyone
def compute_classification_metrics_general(labels_da, labels_estimated):

    # reorder the estimated labels to best match the reference ones
    labels_reordered = optimaly_reorder_labels(labels_da, labels_estimated)

    # hard assignment error
    hard_error = float(np.abs(labels_reordered - labels_da).mean()/2)

    # also get the rand index
    rand_index = adjusted_rand_score(labels_da.argmax(dim='label'), labels_reordered.argmax(dim='label'))
    
    return hard_error, rand_index


########### Mixtures of Experts class (with implementation from R)

# class that has the functions "predict" and "fit_predict" to respect the conventions of the other clustering methods
class MixtureOfExperts:
    def __init__(self, n_clusters, library = 'flexmix'):
        self.K = n_clusters
        self.library = library

    def fit(self, XY):
        ## get dimensions
        n = len(XY.index)
        p = len(XY.columns)-1
        
        ## convert to R

        # reorder the columns to put 'Y' in the first position
        data = XY[['Y']+list(range(p))]
        # convert to r matrix
        data_r = base.matrix(data.values.astype('float'), nrow=n, ncol=p+1)
        # declare "data_r" as a global object for the R environment
        robjects.globalenv["data_r"] = np.array(data_r)
        # likewise, set the desired number of classes K in R
        robjects.globalenv["K"] = self.K

        ## run the MoE and get the estimated regression parameters

        # train the MoE model (with the desired MoE library)
        if self.library == 'flexmix':
            # run the flexmix R script
            robjects.r(
                '''
                options(warn=-1)
                df <- data.frame(data_r)
                model_MoE <- flexmix(X1 ~ ., data = df, k = K)
                ''')
            
            # put the intercept at the end and not at the beginning of beta
            beta_reordered = np.array(robjects.r('parameters(model_MoE)'))[list(range(1, p+1))+[0] , :].T
            # it seems that this package can sometimes return just 1 instead of K betas (vanishing cluster)
            if beta_reordered.shape[0] == 1:
                # in this case, copy the unique recovered beta for all classes
                beta_reordered = np.array([list(beta_reordered) for k in range(self.K)])
            # get the estimated parameters
            beta_t = xr.DataArray(data = beta_reordered, coords=[range(self.K), range(p+1)], dims=['label', 'component_p'])
            sigma_t = xr.DataArray(data = np.array(robjects.r('parameters(model_MoE)'))[p+1, :], coords=[range(self.K)], dims=['label'])
            pi_t = xr.DataArray(data = np.array(robjects.r('model_MoE@prior')), coords=[range(self.K)], dims=['label'])
        
        if self.library == 'mixtools':
            # run the mixtools R script
            robjects.r(
                '''
                options(warn=-1)
                df <- data.frame(data_r)
                model_MoE <- regmixEM(y=df$X1, x= as.matrix(df[ , ! names(df) =="X1"]), k = K)
                ''')
                        
            # put the intercept at the end and not at the beginning of beta
            beta_reordered = np.array(robjects.r('model_MoE$beta'))[list(range(1, p+1))+[0] , :].T
            # maybe this package can also sometimes return just 1 instead of K betas (vanishing cluster)
            if beta_reordered.shape[0] == 1:
                # just to be covered, we also consider this possibility, in which case copy the unique recovered beta for all classes
                beta_reordered = np.array([list(beta_reordered) for k in range(self.K)])
            # get the estimated parameters
            beta_t = xr.DataArray(data = beta_reordered, coords=[range(self.K), range(p+1)], dims=['label', 'component_p'])
            sigma_t = xr.DataArray(data=np.array(robjects.r('model_MoE$sigma')), coords=[range(self.K)], dims=['label'])
            pi_t = xr.DataArray(data=np.array(robjects.r('model_MoE$lambda')), coords=[range(self.K)], dims=['label'])
        
        # update the model parameters of this instance of the method
        self.beta = beta_t
        self.sigma = sigma_t
        self.pi = pi_t
        
        # remark: we could use, for train data,
        #self.labels_estimated = np.array(robjects.r('model_MoE@cluster'))-1
      
    # only usable if we have already trained the method beforehand
    def predict(self, XY):
        # get dimensions
        n = len(XY.index)
        p = len(XY.columns)-1
        
        # convert joint dataframe back into two data arrays
        X_da = xr.DataArray(
            data = XY.loc[:, range(p)], 
            coords = [range(n), range(p)], 
            dims = ['observation', 'component_p'])

        Y_da = xr.DataArray(
            data = XY.loc[:, 'Y'], 
            coords = [range(n)], 
            dims = ['observation'])

        ## classify with conditional gaussian likelihood y|X 
        
        # dotproduct E[Y|X] = X^T beta
        neg_ll_Y_cond_X = (Y_da - compute_conditional_expectation(X_da, self.beta))**2/(2*self.sigma**2) + np.log(self.sigma) - np.log(self.pi)

        # get the estimated labels
        labels_estimated = neg_ll_Y_cond_X.argmin(dim='label')
        
        return labels_estimated.values
    
    # fit and predict on train data
    def fit_predict(self, XY):
        # fit the data
        self.fit(XY)
        # get the estimated labels
        labels_estimated = self.predict(XY)
        
        return labels_estimated


########################## large function

# function to run and evaluate any algorithm other than JCR_EM
def evaluate_algorithm(
    # train data
    Y_train, X_train, labels_train, 
    # algorithm used
    algorithm,
    # parameters for the algorithm
    algorithm_parameters,
    # projection
    projection_da = None,
    # real beta, for sparsistency
    beta_true = None,  
    # real precision matrix, for graph sparsistency
    Omega_true = None,
    # validation data (if desired)
    Y_val = None, X_val = None, labels_val = None,
    # if provided, the labels estimated by an oracle method to compare oneself to
    labels_train_oracle = None, 
    labels_val_oracle = None,
    # do we use the model based oracle labels as reference for the other algorithms?
    use_oracle_as_reference=False,
    # do we evaluate the ROC curve of the sparsistency in beta and Omega?
    evaluate_sparsistency_roc=False, 
    # sparse parameters to estimate
    parameter_list=['beta', 'Omega glasso'],
    # number of solutions to compute for the ROC curve
    n_solutions = 20):
    
    # do we evaluate the OoS error?
    evaluate_OoS_error = not (Y_val is None)
    
    # we must have the dimension to sum along named "component..."
    #component_name = [dim for dim in X_train.dims if dim[:9]=='component'][0]
    
    # get dimensions
    p = len(X_train.component_p)
    K = len(labels_train.label)
    q = None
    
    ############################################# treat data ###############################################

    # if ambient space
    if projection_da is None:
        # we use the 'WT_{...}' for the data used in the clustering (even if W=I_p)
        WT_X_train = X_train
        WT_X_val = None
        if evaluate_OoS_error:
            WT_X_val = X_val
    # if embedding space
    else:
        q = len(projection_da.component_q)
        # project X train
        print('project X train')
        WT_X_train = xr.dot(X_train, projection_da, dims = 'component_p')
        WT_X_val = None
        if evaluate_OoS_error:
            # project X val
            print('project X val')
            WT_X_val = xr.dot(X_val, projection_da, dims = 'component_p')
        
    
    ######################################### initialise data array ########################################

    # data set used, train or validation
    dataset_type_list = ['IS']+['OoS']*evaluate_OoS_error

    # list of all followed metrics, given the provided simulation settings  
    metric_list = ['clustering_duration', 'hard_error', 'rand_index'] + [
        'hard_error_oracle', 'rand_index_oracle']*use_oracle_as_reference

    # initialise local results dataarray
    results_da = xr.DataArray(
        coords = [
            metric_list,
            dataset_type_list
        ], 
        dims = [
            'metric', 
            'dataset_type'
        ])
    
    # initialise local sparsistency results dataarray for ROC  
    if evaluate_sparsistency_roc: 
        sparsistency_roc_da = initialise_roc_da(n_solutions=n_solutions, mode = 'general', 
                                                evaluate_OoS_error=evaluate_OoS_error,
                                                parameter_list=parameter_list)
    
    
    ######################################### Run the algo ########################################

    # get the estimated labels
    duration, labels_estimated, labels_val_estimated = estimate_cluster_labels(Y_train, WT_X_train, 
                                                                               # algorithm to use
                                                                               algorithm = algorithm,
                                                                               # parameters for each algorithm, indexed by name
                                                                               parameters=algorithm_parameters,
                                                                               # validation data if provided
                                                                               Y_val=Y_val, X_val=WT_X_val)

    # first, update the clustering duration
    results_da.loc[dict(metric='clustering_duration')] = duration

    # get dummy version of the estimated labels
    labels_dummies_estimated = convert_label_to_dummies(labels_estimated)
    # get the errors
    hard_error, rand_index = compute_classification_metrics_general(labels_train, labels_dummies_estimated)
    # update the IS error
    results_da.loc[dict(dataset_type='IS', metric=['hard_error', 'rand_index'])] = [hard_error, rand_index]

    if not (labels_train_oracle is None):
        # get the errors
        _, rand_index = compute_classification_metrics_general(labels_train_oracle, labels_dummies_estimated)
        # update the OoS error
        results_da.loc[dict(dataset_type='IS', metric=['rand_index_vs_oracle'])] = rand_index
        
    # ROC analysis 
    #(only if we gave X and not just W^tX to the method)
    if evaluate_sparsistency_roc: 
        # reorder the estimated labels to best match the reference ones
        labels_reordered = optimaly_reorder_labels(labels_train, labels_dummies_estimated)
        # get the roc metrics for these settings as a data array
        local_sparsistency_roc_da = compute_roc_path_all_labels(Y_train, X_train, 
                                                                # true labels 
                                                                labels_train.argmax(dim='label'), 
                                                                # estimated labels
                                                                labels_reordered.argmax(dim='label'), 
                                                                # true parameters, 
                                                                beta_true, Omega_true,
                                                                # number of solutions to compute for the ROC curve
                                                                n_solutions,
                                                                # parameters to estimate (with estimator variants)
                                                                parameter_list)
        # update data array
        sparsistency_roc_da.loc[dict(dataset_type='IS')] = local_sparsistency_roc_da

    if not (labels_val_estimated is None):
        # get dummy version of the estimated labels
        labels_dummies_estimated = convert_label_to_dummies(labels_val_estimated)
        # get the errors
        hard_error, rand_index = compute_classification_metrics_general(labels_val, labels_dummies_estimated)
        # update the OoS error
        results_da.loc[dict(dataset_type='OoS', metric=['hard_error', 'rand_index'])] = [hard_error, rand_index]

        if not (labels_val_oracle is None):
            # get the errors
            _, rand_index = compute_classification_metrics_general(labels_val_oracle, labels_dummies_estimated)
            # update the OoS error
            results_da.loc[dict(dataset_type='OoS', metric=['rand_index_vs_oracle'])] = rand_index 
        
        # ROC analysis 
        #(only if we gave X and not just W^tX to the method)
        if evaluate_sparsistency_roc: 
            # reorder the estimated labels to best match the reference ones
            labels_reordered = optimaly_reorder_labels(labels_val, labels_dummies_estimated)
            # get the roc metrics for these settings as a data array
            local_sparsistency_roc_da = compute_roc_path_all_labels(Y_val, X_val, 
                                                                    # true labels 
                                                                    labels_val.argmax(dim='label'), 
                                                                    # estimated labels
                                                                    labels_reordered.argmax(dim='label'), 
                                                                    # true parameters, 
                                                                    beta_true, Omega_true,
                                                                    # number of solutions to compute for the ROC curve
                                                                    n_solutions,
                                                                    # parameters to estimate (with estimator variants)
                                                                    parameter_list)
            # update data array
            sparsistency_roc_da.loc[dict(dataset_type='OoS')] = local_sparsistency_roc_da
    
    # create an easy to unwrap output
    output = {'results_da' : results_da}
    
    if evaluate_sparsistency_roc:
        output.update({'sparsistency_roc_da' : sparsistency_roc_da}) 
    
    return output


########################################################## Stability metrics ##########################################################


# from data, projection and given subsample size, subsample once, run the EM and get the estimated labels
def get_subsampled_class_assignment(Y_da, X_da, projection_da, 
                                    # fraction of the total data in each subsample data set
                                    subsampled_fraction=0.75, 
                                    ## EM parameters
                                    # initialisation parameters
                                    beta_0=None, sigma_0=None, mu_0=None, Sigma_0=None, tau_0=None, 
                                    # stopping criteria
                                    neg_ll_ratio_threshold = 1e-5, 
                                    E_step_difference_threshold = 1e-3,
                                    max_steps = 100, 
                                    min_steps = 10,
                                    # do we regularise the estimated covariance matrices?
                                    shrinkage = False,
                                    # exponent of the X-bloc in the E step
                                    tempering_exponent=1, 
                                    # penalty intensities
                                    penalty_lasso = 1, 
                                    penalty_sigma = 0,
                                    penalty_tau = 0):
    
    n = len(Y_da.observation)
    # size of subsampled data set
    size_subsample_fold = int(subsampled_fraction*n)
    # generate a subsample of observation
    subsampled_indices = np.random.choice(range(n), size = size_subsample_fold, replace = False)
    # subsample the observations
    X_subsampled = X_da.sel(observation=subsampled_indices).copy()
    Y_subsampled = Y_da.sel(observation=subsampled_indices).copy()

    # reset the observation IDs
    X_subsampled = X_subsampled.assign_coords({'observation':range(size_subsample_fold)})
    Y_subsampled = Y_subsampled.assign_coords({'observation':range(size_subsample_fold)})

    # project X train
    WT_X_subsampled = xr.dot(X_subsampled, projection_da, dims = 'component_p')

    # run EM
    beta_t, sigma_t, mu_t, Sigma_t, tau_t = EM_JCR_lasso(Y_subsampled, X_subsampled, 
                                                         # model parameters
                                                         beta_0, sigma_0, mu_0, Sigma_0, tau_0, 
                                                         # projected data
                                                         WT_X_da = WT_X_subsampled,
                                                         # stopping criteria
                                                         neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                                         E_step_difference_threshold = E_step_difference_threshold,
                                                         max_steps = max_steps, 
                                                         min_steps = min_steps,
                                                         # do we regularise the estimated covariance matrices?
                                                         shrinkage = shrinkage,
                                                         # exponent of the X-bloc in the E step
                                                         tempering_exponent=tempering_exponent, 
                                                         # penalty intensities
                                                         penalty_lasso = penalty_lasso, 
                                                         penalty_sigma = penalty_sigma,
                                                         penalty_tau = penalty_tau)

    ## evaluate the rand index of this EM

    # estimated class weight by data point
    p_kT = E_step_weights(Y_subsampled, X_subsampled, beta_t, sigma_t, mu_t, Sigma_t, tau_t, WT_X_da= WT_X_subsampled,
                          tempering_exponent=tempering_exponent, stop_at_vanishing_cluster=False, regularise_E_step=True)

    # estimated hard assignment (label format)
    hard_assignment = p_kT.argmax(dim='label')
    
    return hard_assignment, subsampled_indices

# from data, projection and given subsample size, build several subsampling folds, get the estimated labels for each of them
def get_several_subsampled_class_assignments(Y_da, X_da, projection_da, 
                                             # number of folds
                                             n_stability_folds=5,
                                             # fraction of the total data in each subsample data set
                                             subsampled_fraction=0.75, 
                                             ## EM parameters
                                             # initialisation parameters
                                             beta_0=None, sigma_0=None, mu_0=None, Sigma_0=None, tau_0=None, 
                                             # stopping criteria
                                             neg_ll_ratio_threshold = 1e-5, 
                                             E_step_difference_threshold = 1e-3,
                                             max_steps = 100, 
                                             min_steps = 10,
                                             # do we regularise the estimated covariance matrices?
                                             shrinkage = False,
                                             # exponent of the X-bloc in the E step
                                             tempering_exponent=1, 
                                             # penalty intensities
                                             penalty_lasso = 1, 
                                             penalty_sigma = 0,
                                             penalty_tau = 0):
    
    n = len(Y_da.observation)
    
    # size of each subsampled data set
    size_subsample_fold = int(subsampled_fraction*n)

    # initialise estimated labels storage data array
    hard_assignment_da = xr.DataArray(
        coords=[
            range(n_stability_folds), 
            range(size_subsample_fold)
        ], 
        dims=[
            'subsample_fold', 
            'observation'
        ])
    
    # we must also save the name of each subsampled index
    subsampled_indices_da = xr.DataArray(
        coords=[
            range(n_stability_folds), 
            range(size_subsample_fold)
        ], 
        dims=[
            'subsample_fold', 
            'observation'
        ])

    # get the estimated labels (not re-orderd) over the different subsampling
    print()
    print('stability subsampling:')
    for subsample_fold in range(n_stability_folds):
        print(f'fold: {subsample_fold}')
        with HiddenPrints():
            hard_assignment, subsampled_indices = get_subsampled_class_assignment(Y_da, X_da, projection_da, 
                                                              # fraction of the total data in each subsample data set
                                                              subsampled_fraction = subsampled_fraction, 
                                                              ## EM parameters
                                                              # initialisation parameters
                                                              beta_0=beta_0, sigma_0=sigma_0, mu_0=mu_0, Sigma_0=Sigma_0, tau_0=tau_0, 
                                                              # stopping criteria
                                                              neg_ll_ratio_threshold = neg_ll_ratio_threshold,
                                                              E_step_difference_threshold = E_step_difference_threshold,
                                                              max_steps = max_steps, 
                                                              min_steps = min_steps,
                                                              # do we regularise the estimated covariance matrices?
                                                              shrinkage = shrinkage,
                                                              # exponent of the X-bloc in the E step
                                                              tempering_exponent=tempering_exponent, 
                                                              # penalty intensities
                                                              penalty_lasso = penalty_lasso, 
                                                              penalty_sigma = penalty_sigma,
                                                              penalty_tau = penalty_tau)

        hard_assignment_da.loc[dict(subsample_fold=subsample_fold)]=hard_assignment
        subsampled_indices_da.loc[dict(subsample_fold=subsample_fold)]=subsampled_indices


    return hard_assignment_da, subsampled_indices_da

# from several class assignments with overlap on the labelled observations, compute the pairwise rand_index 
def compute_pairwise_rand_index(hard_assignment_da, subsampled_indices_da):
    rand_index_list = []
    n_stability_folds = len(hard_assignment_da.subsample_fold)
    for fold1 in range(n_stability_folds):
        for fold2 in range(fold1):
            
            ## get the rand index between the common labels of evey pair of folds

            # indices of the observations in fold1 
            fold1_indices = np.int_(subsampled_indices_da.sel(subsample_fold=fold1).values)
            # indices of the observations in fold2
            fold2_indices = np.int_(subsampled_indices_da.sel(subsample_fold=fold2).values)
            # their intersection 
            folds_intersection = set(fold1_indices).intersection(set(fold2_indices))

            ## select only the indices in the intersection and reorder them

            # rename the observations to their original idx
            hard_assignment_fold1 = hard_assignment_da.sel(subsample_fold=fold1).assign_coords({'observation':fold1_indices}).copy()
            hard_assignment_fold2 = hard_assignment_da.sel(subsample_fold=fold2).assign_coords({'observation':fold2_indices}).copy()

            # restrict the labels to the intersection only (and reorder them)
            hard_assignment_fold1 = hard_assignment_fold1.sel(observation=list(folds_intersection))
            hard_assignment_fold2 = hard_assignment_fold2.sel(observation=list(folds_intersection))

            # get the rand index
            rand_index = adjusted_rand_score(hard_assignment_fold1, hard_assignment_fold2)

            # update the list
            rand_index_list +=[rand_index]
    return rand_index_list


#######################################################  shrinkage of sigma  #######################################################

# ledoit_wolf shrinkage
def ledoit_wolf_shrinkage(X_da, mu_da, weights):
    # generic way of getting the correct component name (either q or p)
    component_name = [dim for dim in X_da.dims if dim[:9]=='component'][0]
    
    # get dimesions
    p = len(X_da[component_name])
    K = len(weights.label)

    # remove estimated mean
    X_centred = X_da-mu_da

    # rename the component locally in the function to simplify
    X_centred = X_centred.rename({component_name:'component'})

    # empirical covariance
    S = xr.dot(weights*X_centred, X_centred.rename({'component':'componentT'}), dims='observation')/weights.sum(dim='observation')

    # trace
    m = np.trace(S.transpose('component', 'componentT','label'))/p

    # get the data array version
    m_da = xr.DataArray(m, coords=[range(K)], dims=['label'])
    I_da = xr.DataArray(np.eye(p), coords=[range(p), range(p)], dims=['component', 'componentT'])

    # distance squared between S and m*Id
    d2 = ((S-m_da*I_da)**2).sum(dim=['component', 'componentT'])/p

    # distance squared between S and each xxT
    d2_xx_S = ((X_centred*X_centred.rename({'component':'componentT'}) - S)**2).sum(dim=['component', 'componentT'])/p

    # square of average distance between S and each xxT
    bbar2 = (weights*d2_xx_S).sum(dim='observation')/(weights.sum(dim='observation'))**2

    # lw shrinkage
    alpha = (bbar2/d2)**((bbar2/d2)<1)

    # lw estimator
    S_lw = alpha*m_da*I_da + (1-alpha)*S
    # put the proper names back
    S_lw = S_lw.rename({'component':component_name, 'componentT':f'{component_name}T'})
    return S_lw

# theoretical improvement upon LW
def rao_blackwell_ledoit_wolf_shrinkage(X_da, mu_da, weights):
    # generic way of getting the correct component name (either q or p)
    component_name = [dim for dim in X_da.dims if dim[:9]=='component'][0]
        
    # get dimesions
    p = len(X_da[component_name])
    K = len(weights.label)

    # remove estimated mean
    X_centred = X_da-mu_da

    # rename the component locally in the function to simplify
    X_centred = X_centred.rename({component_name:'component'})
    
    # "number" of observations per class
    n_k = weights.sum(dim='observation')

    # empirical covariance
    S = xr.dot(weights*X_centred, X_centred.rename({'component':'componentT'}), dims='observation')/n_k

    # trace of S
    tr_S = np.trace(S.transpose('component', 'componentT','label'))

    # get the data array version
    tr_S = xr.DataArray(tr_S, coords=[range(K)], dims=['label'])

    # trace of S^2 
    tr_S2 = (S**2).sum(dim=['component', 'componentT'])

    # rblw shrinkage
    alpha = ((n_k-2)/n_k * tr_S2 + tr_S**2) / ((n_k+2)*(tr_S2 - tr_S**2/p))
    alpha = alpha**(alpha<1)

    # data array version of Id
    I_da = xr.DataArray(np.eye(p), coords=[range(p), range(p)], dims=['component', 'componentT'])

    # lw estimator
    S_rblw = alpha*tr_S*I_da/p + (1-alpha)*S
    # put the proper names back
    S_rblw = S_rblw.rename({'component':component_name, 'componentT':f'{component_name}T'})
    return S_rblw

# theoretical improvement upon RBLW
def oracle_approximating_shrinkage(X_da, mu_da, weights):
    # generic way of getting the correct component name (either q or p)
    component_name = [dim for dim in X_da.dims if dim[:9]=='component'][0]
        
    # get dimesions
    p = len(X_da[component_name])
    K = len(weights.label)

    # remove estimated mean
    X_centred = X_da-mu_da
    
    # rename the component locally in the function to simplify
    X_centred = X_centred.rename({component_name:'component'})

    # "number" of observations per class
    n_k = weights.sum(dim='observation')

    # empirical covariance
    S = xr.dot(weights*X_centred, X_centred.rename({'component':'componentT'}), dims='observation')/n_k

    # trace of S
    tr_S = np.trace(S.transpose('component', 'componentT','label'))

    # get the data array version
    tr_S = xr.DataArray(tr_S, coords=[range(K)], dims=['label'])

    # trace of S^2 
    tr_S2 = (S**2).sum(dim=['component', 'componentT'])

    # oa shrinkage
    alpha = ((1-2)/p * tr_S2 + tr_S**2) / ((n_k-1)/p * (tr_S2 - tr_S**2/p))
    alpha = alpha**(alpha<1)

    # data array version of Id
    I_da = xr.DataArray(np.eye(p), coords=[range(p), range(p)], dims=['component', 'componentT'])

    # lw estimator
    S_oa = alpha*tr_S*I_da/p + (1-alpha)*S
    
    # put the proper names back
    S_oa = S_oa.rename({'component':component_name, 'componentT':f'{component_name}T'})
    return S_oa

# estimate a shrunk Sigma with the desired method
def estimate_Sigma(X_da, mu_da, weights, shrinkage='oas'):
    # ledoit_wolf shrinkage
    if shrinkage == 'lw':
        return ledoit_wolf_shrinkage(X_da, mu_da, weights)
    # rao_blackwell ledoit_wolf shrinkage
    elif shrinkage == 'rblw':
        return rao_blackwell_ledoit_wolf_shrinkage(X_da, mu_da, weights)
    # oracle approximating shrinkage
    elif shrinkage=='oas':
        return oracle_approximating_shrinkage(X_da, mu_da, weights)
    else:
        print(f'invalid shrinkage: {shrinkage}')
        return

    
############################################################# synthetic data generation #############################################################

# generare K sparse SPD matrices
def generate_sparse_precision_matrix(K, p, same_covariance = False):
    Omega = xr.DataArray(
            coords=[range(K), range(p), range(p)], 
            dims=['label', 'component_p', f'component_pT'])

    for k in range(K):
        # if same covariance matrix across classes, no need to generate more than one matrix
        if (k==0) or (not same_covariance):
            # generate base sparse SPD matrix
            precision_matrix = make_sparse_spd_matrix(dim=p, alpha=1-10./p, norm_diag=True)

            # current lower eigenvalue (possibly negative after transformation)
            lambda_min = np.linalg.eigvalsh(precision_matrix).min()
            # new lower eigenvalue
            lambda_min_target = np.random.chisquare(1)/p

            # make the matrix positive again
            precision_matrix = precision_matrix + (lambda_min<lambda_min_target)*(lambda_min_target-lambda_min)*np.eye(p)

            # normalise the matrix by its diagonal
            diagonal = np.sqrt(np.diag(precision_matrix))
            precision_matrix /= diagonal
            precision_matrix /= diagonal.reshape(1, -1)

        # update Omega
        Omega.loc[dict(label = k)] = precision_matrix
        
    return Omega

# generate synthetic mixture parameters
def generate_synthetic_parameters(K, p, same_covariance = False):
    # mu 
    mu_true = xr.DataArray(data = np.random.multivariate_normal(np.zeros(p), np.eye(p), K), coords=[range(K), range(p)], dims=['label', 'component_p'])
    # Omega: precision
    Omega_true = generate_sparse_precision_matrix(K, p, same_covariance)
    # Sigma: covariance
    Sigma_true = xr.DataArray(data = np.linalg.inv(Omega_true), coords = Omega_true.coords, dims = Omega_true.dims)
    # tau
    tau_true = xr.DataArray(data = np.ones(K)/K, coords=[range(K)], dims=['label'])
    
    return mu_true, Omega_true, Sigma_true, tau_true

# generate synthetic data from mixture parameters
def generate_synthetic_gaussian_data(mu, Sigma, tau, n_total):
    # get parameters
    p = len(mu.component_p)
    K = len(mu.label)
    
    # empty initialisation
    X_full = pd.DataFrame(columns=range(p))
    labels = pd.Series(dtype=int)

    # generate X for each class
    for k in range(K):
        # first K-1 classes
        if k<K-1:
            n_k = np.random.binomial(n_total, p=tau.sel(label=k))
        # last class
        else:
            n_k = n_total - len(X_full)
        # generate data
        X_k = np.random.multivariate_normal(mean = mu.sel(label=k), cov = Sigma.sel(label=k), size = n_k)
        # update data frame
        X_full = pd.concat([X_full, pd.DataFrame(X_k)])

        # update labels Series
        labels = pd.concat([labels, pd.Series([k]*n_k)])
        
    return X_full.reset_index(drop=True), labels.reset_index(drop=True)


############################################################# Mixed data functions #############################################################

# generate synthetic data from mixture parameters
def generate_synthetic_mixed_data(mu, Sigma, tau, C, n_total):
    # get parameters
    p = len(mu.component_p)
    K = len(mu.label)
    # number of binary features among the p features
    d_binary = len(C.component_p)
    
    # empty initialisation
    X_full = pd.DataFrame(columns=range(p))
    labels = pd.Series(dtype=int)

    # generate X for each class
    for k in range(K):
        # first K-1 classes
        if k<K-1:
            n_k = np.random.binomial(n_total, p=tau.sel(label=k))
        # last class
        else:
            n_k = n_total - len(X_full)
        # generate data
        X_k = np.random.multivariate_normal(mean = mu.sel(label=k), cov = Sigma.sel(label=k), size = n_k)
        # threshold data
        X_k[:, range(d_binary)] = 1*(X_k[:, range(d_binary)]>C.sel(label=k).values)
        # update data frame
        X_full = pd.concat([X_full, pd.DataFrame(X_k)])

        # update labels Series
        labels = pd.concat([labels, pd.Series([k]*n_k)])
        
    return X_full.reset_index(drop=True), labels.reset_index(drop=True)

# function "H" of the mixed data paper
def H(t, delta_j):
    return 4*stats.multivariate_normal.cdf([delta_j, 0], mean=[0, 0], cov = [[1, t/np.sqrt(2)], [t/np.sqrt(2), 1]]) - 2*stats.norm.cdf(delta_j)

# function "F" of the mixed data paper
def F(t, delta_j, delta_k):
    x = [delta_j, delta_k]
    return 2*(stats.multivariate_normal.cdf(x, mean=[0, 0], cov = [[1, t], [t, 1]]) - stats.norm.cdf(delta_j)*stats.norm.cdf(delta_k))

# invert any function of 0<=t<1 by grid search
def invert_function(func, target_value, tol = 1e-3, **args):
    # initial list
    t_list = np.linspace(-1+1e-5, 1-1e-5)
    idx_t = 0
    
    # deal with exceptions when kendall's tau is close to the bounds
    if target_value<func(t_list[0], **args):
        return -1
    elif target_value>func(t_list[-1], **args):
        return 1
    
    while True:   
        t = t_list[idx_t]
        #print(func(t, **args)-target_value)
        # if we go above target_value by more than tol: refine the grid
        if func(t, **args)-target_value>=tol:
            # new grid 
            t_list = np.linspace(t_list[idx_t-1], t_list[idx_t])
            idx_t = -1
        # if we go close enough to target: stop
        if np.abs(func(t, **args)-target_value)<tol:
            break
        idx_t +=1
        
    return t

# count categorical columns in the data (if any)
def count_categorical_columns(X, categorical_threshold=0.25):
    return ((X.nunique()/len(X))<categorical_threshold).sum()

# put the categorical columns at the beginning of the data set
def isolate_categorical_columns(X, categorical_threshold=0.25):
    # automatically detect categorical data: columns with less than 25% unique values
    categorical_columns = (X.nunique()/len(X))<categorical_threshold
    continuous_columns = (X.nunique()/len(X))>=categorical_threshold

    # count the number of each column type
    #d_categorical = categorical_columns.sum()
    #d_continuous = continuous_columns.sum()

    # reorder the columns, categorical first
    X_reordered = pd.concat([X.loc[:, categorical_columns], X.loc[:, continuous_columns]], axis = 1)

    return X_reordered#, d_categorical, d_continuous

# computes the rank based estimator of mixed data
def compute_rank_based_estimator(X):
    # get dimension
    p = len(X.columns)
    n = len(X)
    # count categorical columns, 
    d1 = count_categorical_columns(X, categorical_threshold=0.25)
    # continuous columns
    d2 = p - d1
    
    # matrix used in rank based estimation of Sigma
    Kendall_tau = np.zeros((p, p))
    for i1 in range(n):
        for i2 in range(i1):
            # difference between these two observations
            observation_difference = np.array(X.loc[i2]- X.loc[i1])
            # matrix of the products of these differences, for each pair of features
            Kendall_tau+=observation_difference.reshape(-1, 1)*observation_difference.reshape(1, -1)
    # normalise
    Kendall_tau *= 2/(n*(n-1)) 

    # rank based estimator: initialisation
    R_hat = np.zeros((p, p))

    delta_hat = stats.norm.ppf(1-np.array(X.mean())[range(d1)])
    # binary-binary part
    for j in range(d1):
        for k in range(j):
            x = [delta_hat[j], delta_hat[k]]
            #print(j, k)
            R_hat[j, k] = invert_function(F, target_value=Kendall_tau[j, k], delta_j=delta_hat[j], delta_k=delta_hat[k])

    # continuous - continuous
    for j in range(d1, p):
        for k in range(d1, j):
            R_hat[j, k] = np.sin(Kendall_tau[j, k]*np.pi/2)

    # binary - continuous
    for j in range(d1):
        for k in range(d1, p):
            R_hat[j, k] = invert_function(H, target_value=Kendall_tau[j, k], delta_j=delta_hat[j])

    # add the missing triangle to this matrx as well as the diagonal
    R_hat = R_hat+R_hat.T+np.eye(p)

    # nearest PD matrix
    R_hat_PD = nearestPD(R_hat)
    
    return R_hat_PD

def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False
    
    
############################################################# non gaussian/student data functions #############################################################

# generate synthetic data as mixture of students
def generate_synthetic_student_data(mu, Sigma, tau, df, n_total):
    # get parameters
    p = len(mu.component_p)
    K = len(mu.label)
    
    # empty initialisation
    X_full = pd.DataFrame(columns=range(p))
    labels = pd.Series(dtype=int)

    # generate X for each class
    for k in range(K):
        # first K-1 classes
        if k<K-1:
            n_k = np.random.binomial(n_total, p=tau.sel(label=k))
        # last class
        else:
            n_k = n_total - len(X_full)
        # generate data
        X_k = np.random.multivariate_normal(mean = np.zeros(p), cov = Sigma.sel(label=k), size = n_k)
        chi = np.random.chisquare(df = df, size = n_k).reshape(-1, 1)
        # student variable
        X_k = X_k/np.sqrt(chi/df) + mu.sel(label=k).values.reshape(1, -1)

        # update data frame
        X_full = pd.concat([X_full, pd.DataFrame(X_k)])

        # update labels Series
        labels = pd.concat([labels, pd.Series([k]*n_k)])
        
    return X_full.reset_index(drop=True), labels.reset_index(drop=True)