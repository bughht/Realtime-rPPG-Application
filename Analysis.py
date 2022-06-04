'''
Author: Bughht
Date: 2022-01-09 13:36:23
LastEditors: Harryhht
LastEditTime: 2022-02-21 13:58:20
Description:
'''

import numpy as np
import scipy.io as scio
import pyCompare
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import *
from scipy.stats import kendalltau

sns.set()


def icc(data, icc_type="icc2"):
    """Calculate intraclass correlation coefficient for data within
        Brain_Data class
    ICC Formulas are based on:
    Shrout, P. E., & Fleiss, J. L. (1979). Intraclass correlations: uses in
    assessing rater reliability. Psychological bulletin, 86(2), 420.
    icc1:  x_ij = mu + beta_j + w_ij
    icc2/3:  x_ij = mu + alpha_i + beta_j + (ab)_ij + epsilon_ij
    Code modifed from nipype algorithms.icc
    https://github.com/nipy/nipype/blob/master/nipype/algorithms/icc.py
    Args:
        icc_type: type of icc to calculate (icc: voxel random effect,
                icc2: voxel and column random effect, icc3: voxel and
                column fixed effect)
    Returns:
        ICC: (np.array) intraclass correlation coefficient
    """

    Y = data.T
    [n, k] = Y.shape

    # Degrees of Freedom
    dfc = k - 1
    dfe = (n - 1) * (k - 1)
    dfr = n - 1

    # Sum Square Total
    mean_Y = np.mean(Y)
    SST = ((Y - mean_Y) ** 2).sum()

    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()

    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    MSC = SSC / dfc / n

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc1":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        # ICC = (MSR - MSRW) / (MSR + (k-1) * MSRW)
        NotImplementedError("This method isn't implemented yet.")

    elif icc_type == "icc2":
        # ICC(2,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error +
        # k*(mean square columns - mean square error)/n)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)

    elif icc_type == "icc3":
        # ICC(3,1) = (mean square subject - mean square error) /
        # (mean square subject + (k-1)*mean square error)
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)

    return ICC


data = scio.loadmat('data/recorded_data.mat')

x = data['x'][0]
y = data['y'][0]
# x = data['three3'][0]
# y = data['pbv'][0]

check_x = np.linspace(60, 150)
check_y = np.linspace(60, 150)

print(x, y)
plt.figure(figsize=(6, 5))
sns.scatterplot(x, y)
plt.plot(check_x, check_y, c='r')
plt.xlim((60, 150))
plt.ylim((60, 150))
plt.xlabel('GOLD Criteria')
plt.ylabel('Test Result: PBV')
plt.show()
print(icc(np.array([x, y])))
k, p = kendalltau(x, y)
print(k, p)
pyCompare.blandAltman(x, y,
                      percentage=False,
                      title='Bland-Altman Plot',
                      limitOfAgreement=1.96)
plt.show()
