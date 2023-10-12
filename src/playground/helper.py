import numpy as np
from scipy.special import kl_div
from properscoring import crps_ensemble
from scipy.stats import multivariate_normal
from scipy.spatial.distance import jensenshannon
import scipy

def CRPS(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast) 
    b_normalized = [max(0, x / sum_b) for x in forecast]

    return crps_ensemble(a_normalized, b_normalized).mean()

def js_div_(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    # Not sure if I converted the disitrbution in a correct way
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast)
    b_normalized = [max(0, x / sum_b) for x in forecast]
    return jensenshannon(a_normalized, b_normalized)

def CRPS_chatgpt(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast) 
    b_normalized = [max(0, x / sum_b) for x in forecast]

    cdf_a = np.cumsum(a_normalized)
    cdf_b = np.cumsum(b_normalized)
    return np.trapz(np.abs(cdf_a - cdf_b))

def kl_div_(observed, forecast):
    
    observed = observed.reshape(-1)
    forecast = forecast.reshape(-1)
    # Not sure if I converted the disitrbution in a correct way
    sum_a = sum(observed)
    a_normalized = [max(0, x / sum_a) for x in observed]
    sum_b = sum(forecast)
    b_normalized = [max(0, x / sum_b) for x in forecast]
    return sum(kl_div(a_normalized, b_normalized))

def crps_nrg(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    diff = y_pred[1:] - y_pred[:-1]
    weight = np.arange(1, num_samples) * np.arange(num_samples - 1, 0, -1)
    weight = np.expand_dims(weight, -1)

    per_obs_crps = absolute_error - np.sum(diff * weight, axis=0) / num_samples**2
    return np.average(per_obs_crps, weights=sample_weight)


def crps_pwm(y_true, y_pred, sample_weight=None):
    num_samples = y_pred.shape[0]
    absolute_error = np.mean(np.abs(y_pred - y_true), axis=0)

    if num_samples == 1:
        return np.average(absolute_error, weights=sample_weight)

    y_pred = np.sort(y_pred, axis=0)
    b0 = y_pred.mean(axis=0)
    b1_values = y_pred * np.arange(num_samples).reshape((num_samples, 1))
    b1 = b1_values.mean(axis=0) / num_samples

    per_obs_crps = absolute_error + b0 - 2 * b1
    return np.average(per_obs_crps, weights=sample_weight)

import numpy as np

def findMeanVectors(A):
    """
    A : a matrix of K x N 
    """
    return np.mean(A, axis=0)  

def findCovMatrix(A):
    return np.cov(A, rowvar=False) 

def kl_mvn(m_to, S_to, m_fr, S_fr):
    """Calculate `KL(to||fr)`, where `to` and `fr` are pairs of means and covariance matrices"""
    
    d = m_fr - m_to
    
    c, lower = scipy.linalg.cho_factor(S_fr)
    def solve(B):
        return scipy.linalg.cho_solve((c, lower), B)
    
    def logdet(S):
        return np.linalg.slogdet(S)[1]

    term1 = np.trace(solve(S_to))
    term2 = logdet(S_fr) - logdet(S_to)
    term3 = d.T @ solve(d)
    return (term1 + term2 + term3 - len(d))/2.

def kl_between_two_dist(A, B):
    """
    A : forecasted values for K houses and N data points, K x N matrix 
    B : Actual values for K houses and N data points, K x N matrix 
    """

    mean_A = findMeanVectors(A)
    cov_A = findCovMatrix(A)

    mean_B = findMeanVectors(B)
    cov_B = findCovMatrix(B)
    # Ensure that the covariance matrices are positive definite
    cov_A = cov_A + np.eye(cov_A.shape[0]) * 1e-6
    cov_B = cov_B + np.eye(cov_B.shape[0]) * 1e-6
    return kl_mvn(mean_A, cov_A, mean_B, cov_B)
    

def l1_distances(A, B):
    res = []
    for i in range(A.shape[0]):
        res.append(np.linalg.norm((A[i] - B[i]), ord=1))
    
    return res 

def l1_distances_mean(A, B):
    res = []
    for i in range(A.shape[0]):
        res.append(np.linalg.norm((A[i] - B[i]), ord=1))
    return np.mean(res) 
        
# A = np.array([[0.1, 0.4, 0.5], [0.1, 0.5, 0.4]])
# B = np.array([[0.01, 0.06, 0.93], [0.05, 0.05, 0.9]])

# print(kl_between_two_dist(A, B))   
# Define the means and covariance matrices for A and B
# mu_A = np.array([0.1, 0.4, 0.5])
# cov_A = np.array([[0.05, 0.0, 0.0],
#                   [0.0, 0.05, 0.0],
#                   [0.0, 0.0, 0.05]])

# mu_B = np.array([0.1, 0.3, 0.7])
# cov_B = np.array([[0.01, 0.02, 0.0],
#                   [0.02, 0.09, 0.0],
#                   [0.0, 0.0, 0.09]])

# Create multivariate normal distributions for A and B
# dist_A = multivariate_normal(mean=mu_A, cov=cov_A)
# dist_B = multivariate_normal(mean=mu_B, cov=cov_B)

