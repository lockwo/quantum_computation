import numpy as np
import random

def rotosolve(f, num_params, iter=100):
    theta = np.random.uniform(-np.pi, np.pi, num_params)
    results = dict()
    for _ in range(iter):
        for d in range(num_params):
            phi = random.uniform(-np.pi, np.pi)
            theta_temp = np.copy(theta)
            theta_temp[d] = phi
            m_phi = f(theta_temp)
            theta_temp[d] = phi + np.pi/2
            m_phi1 = f(theta_temp)
            theta_temp[d] = phi - np.pi/2
            m_phi2 = f(theta_temp)
            theta[d] = phi - np.pi/2 - np.arctan2(2 * m_phi - m_phi1 - m_phi2, m_phi1 - m_phi2)
    
    results["fun"] = f(theta)
    results["params"] = theta
    return results
