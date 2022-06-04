'''
Author: Harryhht
Date: 2022-01-06 10:28:39
LastEditors: Harryhht
LastEditTime: 2022-02-20 16:31:34
Description:
'''
import copy
from obspy.signal.detrend import polynomial, spline
from rsa import sign
from scipy import signal
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from face2series import CAM2FACE
import threading
from queue import Queue
import numpy as np
import seaborn as sns
import time
sns.set()


class Series2rPPG():
    def __init__(self) -> None:
        # load hist series from CAM
        self.series_class = CAM2FACE()
        self.Ongoing = True

    # Start Processes
    def PROCESS_start(self):
        self.series_class.PROCESS_start()

    def Signal_Preprocessing_single(self, sig):
        return polynomial(sig, order=2)

    def Signal_Preprocessing(self, rgbsig):
        data = np.array(rgbsig)
        data_r = polynomial(data[:, 0], order=2)
        data_g = polynomial(data[:, 1], order=2)
        data_b = polynomial(data[:, 2], order=2)

        return np.array([data_r, data_g, data_b]).T

    def PBV(self, signal):
        sig_mean = np.mean(signal, axis=1)

        sig_norm_r = signal[:, 0]/sig_mean[0]
        sig_norm_g = signal[:, 1]/sig_mean[1]
        sig_norm_b = signal[:, 2]/sig_mean[2]

        pbv_n = np.array(
            [np.std(sig_norm_r), np.std(sig_norm_g), np.std(sig_norm_b)])
        pbv_d = np.sqrt(
            np.var(sig_norm_r) + np.var(sig_norm_g) + np.var(sig_norm_b))
        pbv = pbv_n/pbv_d

        C = np.array([sig_norm_r, sig_norm_g, sig_norm_b])
        print(C.shape)
        Ct = C.T
        Q = np.matmul(C, Ct)
        W = np.linalg.solve(Q, pbv)

        A = np.matmul(Ct, W)
        B = np.matmul(pbv.T, W)
        bvp = A/B
        return bvp

    def CHROM(self, signal):
        X = signal
        Xcomp = 3*X[:, 0] - 2*X[:, 1]
        Ycomp = (1.5*X[:, 0])+X[:, 1]-(1.5*X[:, 2])
        sX = np.std(Xcomp)
        sY = np.std(Ycomp)
        alpha = sX/sY
        bvp = Xcomp - alpha * Ycomp
        return bvp

    def PCA(self, signal):
        bvp = []
        for i in range(signal.shape[0]):
            X = signal[i]
            pca = PCA(n_components=3)
            pca.fit(X)
            bvp.append(pca.components_[0] * pca.explained_variance_[0])
            bvp.append(pca.components_[1] * pca.explained_variance_[1])
        bvp = np.array(bvp)
        return bvp

    def GREEN(self, signal):
        return signal[:, 1]

    def GREEN_RED(self, signal):
        return signal[:, 1]-signal[:, 0]

    def cal_bpm(self, pre_bpm, spec, fps):
        return pre_bpm*0.95+np.argmax(spec[:int(len(spec)/2)])/len(spec)*fps*60*0.05

    # Deconstruction

    def __del__(self):
        self.Ongoing = False
        self.series_class.__del__()


if __name__ == "__main__":
    processor = Series2rPPG()
    processor.PROCESS_start()
