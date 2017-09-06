import os
import glob
import librosa
import resampy
import numpy as np
import soundfile as sf
import progressbar as pb
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.metrics import (confusion_matrix, classification_report,
                                roc_curve, auc)
from sklearn.preprocessing import StandardScaler, label_binarize

from spatial import *
import datatools

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class MultiGMMClassifier:

    def __init__( self, n_components=10 ):

        self.gmms = OrderedDict() # initialise dictionary for gmms
        self.n_components = n_components

    def fit( self, X, y ):

        for label in np.unique(y):

            # make a GMM for the data class
            self.gmms[label] = GaussianMixture(self.n_components)

            # extract class data from large array
            label_data = X[y==label]

            self.gmms[label].fit(label_data) # train GMMs


    def score( self, X ):

        y_score = np.array([gmm.score_samples(X)
                    for _, gmm in self.gmms.items()]).swapaxes(0, 1)

        return y_score



def audio_clip_clsfy( X, y, info, indeces ):

    for entry in info:
        # find indeces of data from specific audio file
        start, end = indeces[entry]

        # slice data from large arrays
        X_to_eval = X[start:end,:]

        # slice class label from large array
        y_to_eval = y[start].reshape(-1)
        y_to_eval = label_binarize(y_to_eval, classes=[0,1,2,3,4,5,6,7])
        # above line is clunky - will fix in rewrite (hopefully!)

        # build y_test array - clunky but necessary
        if 'y_test' not in locals():
            y_test = y_to_eval
        else:
            y_test = np.append(y_test, y_to_eval, axis=0)

        # calculate scores for each frame and sum across clip
        this_score = np.sum(ed_svm_clsfr.decision_function(X_to_eval), 0).reshape(1,-1)

        # save scores in an array
        if 'y_score' not in locals():
            y_score = this_score
        else:
            y_score = np.append(y_score, this_score, axis=0)

    return y_test, y_score
