'''
Author: Harryhht 
Date: 2022-02-01 13:33:41
LastEditors: Harryhht
LastEditTime: 2022-02-01 14:14:57
Description: Methods for extracting BVP from signal based on CUPY
'''

import cupy as cp
import numpy as np
from scipy.signal import butter, filtfilt, savgol_filter
from scipy import stats
import scipy


'''
description: Detrending filter for RGB signal and BVP signal. 
param {*} X
param {int} detLambda
return {*}
'''
def sig_detrend(X,**kargs):
    if 'detLambda' not in kargs:
        kargs['detLambda'] = 10

    X = np.swapaxes(X, 1, 2)
    result = np.zeros_like(X)
    i = 0
    t = X.shape[1]
    l = t/kargs['detLambda']  # lambda
    I = np.identity(t)
    D2 = scipy.sparse.diags([1, -2, 1], [0, 1, 2], shape=(t-2, t)).toarray()
    Hinv = np.linalg.inv(I+l**2*(np.transpose(D2).dot(D2)))
    for est in X:
        detrendedX = (I - Hinv).dot(est)
        result[i] = detrendedX
        i += 1
    result = np.swapaxes(result, 1, 2)
    return result.astype(np.float32)

    
'''
description: Z-score filter
param {*}
return {*}
'''
def zscore(sig):
    x=np.array(np.swapaxes(sig,1,2))
    y=stats.zscore(x,axis=2)
    y=np.swapaxes(y,1,2)
    return y



'''
description: CHROM method
param {*} signal
return {*}
'''
def cupy_CHROM(signal):
    X = signal
    Xcomp = 3*X[:, 0] - 2*X[:, 1]
    Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
    sX = cp.std(Xcomp, axis=1)
    sY = cp.std(Ycomp, axis=1)
    alpha = (sX/sY).reshape(-1, 1)
    alpha = cp.repeat(alpha, Xcomp.shape[1], 1)
    bvp = Xcomp - cp.multiply(alpha, Ycomp)
    return bvp



'''
description: POS method
param {*} signal
param {object} kargs
return {*}
'''
def cupy_POS(signal, **kargs):
    """
    POS method on GPU using Cupy.

    The dictionary parameters are: {'fps':float}.

    Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491. 
    """
    # Run the pos algorithm on the RGB color signal c with sliding window length wlen
    # Recommended value for wlen is 32 for a 20 fps camera (1.6 s)
    eps = 10**-9
    X = signal
    fps = cp.float32(kargs['fps'])
    e, c, f = X.shape            # e = #estimators, c = 3 rgb ch., f = #frames
    w = int(1.6 * fps)   # window length

    # stack e times fixed mat P
    P = cp.array([[0, 1, -1], [-2, 1, 1]])
    Q = cp.stack([P for _ in range(e)], axis=0)

    # Initialize (1)
    H = cp.zeros((e, f))
    for n in cp.arange(w, f):
        # Start index of sliding window (4)
        m = n - w + 1
        # Temporal normalization (5)
        Cn = X[:, :, m:(n + 1)]
        M = 1.0 / (cp.mean(Cn, axis=2)+eps)
        M = cp.expand_dims(M, axis=2)  # shape [e, c, w]
        Cn = cp.multiply(M, Cn)

        # Projection (6)
        S = cp.dot(Q, Cn)
        S = S[0, :, :, :]
        S = cp.swapaxes(S, 0, 1)    # remove 3-th dim

        # Tuning (7)
        S1 = S[:, 0, :]
        S2 = S[:, 1, :]
        alpha = cp.std(S1, axis=1) / (eps + cp.std(S2, axis=1))
        alpha = cp.expand_dims(alpha, axis=1)
        Hn = cp.add(S1, alpha * S2)
        Hnm = Hn - cp.expand_dims(cp.mean(Hn, axis=1), axis=1)
        # Overlap-adding (8)
        H[:, m:(n + 1)] = cp.add(H[:, m:(n + 1)], Hnm)

    return H

