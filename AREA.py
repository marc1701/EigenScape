import librosa
import numpy as np
import soundfile as sf
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

from spatial_funcs import *

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class BasicAudioClassifier:
    ''' Basic GMM-MFCC audio classifier along the lines of the baseline model
    described in DCASE 2015 '''

    # @classmethod
    # def init_from_textfile( cls, info_file, dataset_directory='' ):
    #
    #     info = extract_info(info_file)
    #
    #     return cls(info, dataset_directory)


    def __init__( self, dataset_directory='' ):

        self._label_list = [] # set up list of class labels
        self._gmms = {} # initialise dictionary for GMMs

        # set dataset_directory or class will assume current working directory
        self.dataset_directory = dataset_directory

        # data scaler for normalisation - remembers mean and var of input data
        self._scaler = StandardScaler()
        # we can apply the same transform later to test data using these values


    def train( self, info ):

        info = extract_info(info)
        # make list of unique labels in training data
        self._label_list = sorted(set(labels
                                    for examples, labels in info.items()))
        data, indeces = self._extract_features(info)

        # fit scaler and scale training data (exclude target_numbers column)
        data[:,:-1] = self._scaler.fit_transform(data[:,:-1])
        self._fit_gmms(data)
        results = self._test_input(data, info, indeces)

        # find overall training accuracy percentage
        self.train_acc = int(plot_confusion_matrix(train_info, results)[2]*100)
        print('Training complete. Classifier is ' + str(self.train_acc) +
        ' % accurate in labelling the training data.')
        return results


    def classify( self, info ):

        # call this function to classify new data after training
        info = extract_info(info)
        # generate audio features
        data, indeces = self._extract_features(info)
        # scale data using pre-calculated mean and var
        data[:,:-1] = self._scaler.transform(data[:,:-1])
        # _test_input function gets scores from GMM set
        results = self._test_input(data, info, indeces)

        self.test_acc = int(plot_confusion_matrix(test_info, results)[2]*100)
        print('Testing complete. Classifier is ' + str(self.test_acc) +
        ' % accurate in labelling the training data.')


############################# 'Private' methods: ###############################

    def _extract_features( self, info ):

        indeces = {}

        for filepath, label in info.items():

            target = self._label_list.index(label) # numerical class indicator
            # load audio file
            # note librosa collapses to mono and resamples @ 22050 Hz
            audio, fs = librosa.load(self.dataset_directory + filepath)

            # calculate MFCC values
            mfccs = librosa.feature.mfcc(audio, fs).T
            # swap axes so feature vectors are horizontal (time runs downwards)

            # append a targets column at the end of the mfcc array
            data_to_add = np.hstack((mfccs, np.array([[target]] * len(mfccs))))

            if 'data' not in locals(): # does this variable exist yet?
                indeces[filepath] = [0, len(data_to_add)]
                data = data_to_add
            else:
                indeces[filepath] = [len(data), len(data) + len(data_to_add)]

                # add indeces for mfccs from current file to dictionary
                data = np.vstack((data, data_to_add))
                print('Added ' + filepath + ' features to the dataset.')
                # this allows for testing classification using mfccs
                # from specific examples without having to reload audio

        print('Feature extraction complete.')
        return data, indeces


    def _fit_gmms( self, data ):
        for label in self._label_list:
            print('Training GMM for ' + label)
            self._gmms[label] = GaussianMixture(n_components=10)
            label_num = self._label_list.index(label)
            # extract class data from training matrix
            label_data = data[data[:,-1] == label_num,:-1]
            self._gmms[label].fit(label_data) # train GMMs


    def _test_input( self, data, info, indeces ):

        # import pdb; pdb.set_trace()
        results = OrderedDict()
        scores = {} # initialise dictionary for scores

        for entry in info:
            # find indeces of data from specific audio file
            start, end = indeces[entry][0], indeces[entry][1]
            print('Testing ' + entry)

            # slice data from large array
            data_to_evaluate = data[start:end,:-1]

            for label, gmm in self._gmms.items():
                scores[label] = np.sum(gmm.score_samples(data_to_evaluate))

            # find label with highest score and store result in dictionary
            results[entry] = max(scores, key = scores.get)

        return results


################################################################################
################################################################################
class MultiFoldClassifier(BasicAudioClassifier):

    def __init__(self, dataset_info, **kwargs):
        super(MultiFoldClassifier, self).__init__(**kwargs)

        # to speed up multifold testing, all audio is loaded and features
        # calculated at the start (saves reloading all the data for each fold)
        dataset_info = extract_info(dataset_info)

        # make list of unique labels in data
        self._label_list = sorted(set(labels for x, labels in dataset_info.items()))

        self.data, self.indeces = self._extract_features(dataset_info)


    def train(self, train_info):

        # put info from text file into OrderedDict
        train_info = extract_info(train_info)

        # get all indeces of training data points and put into numpy array
        train_indeces = np.array([np.r_[
                        self.indeces[file][0]:self.indeces[file][1]]
                        for file in train_info]).reshape(-1)

        # slice training data from main data array
        train_data = np.copy(self.data[train_indeces])
        # fit the scaler to training data only
        train_data[:,:-1] = self._scaler.fit_transform(train_data[:,:-1])

        # apply scaler to dataset copy (overwritten on each fold pass)
        # original data array left unchanged
        self.fold_data = np.copy(self.data)
        self.fold_data[:,:-1] = self._scaler.transform(self.fold_data[:,:-1])

        # fit GMMs to training data (GMMs overwritten on each fold pass)
        self._fit_gmms(train_data)
        results = self._test_input(self.fold_data, train_info, self.indeces)

        # find overall training accuracy percentage
        self.train_acc = int(plot_confusion_matrix(train_info, results)[2]*100)
        print('Training complete. Classifier is ' + str(self.train_acc) +
        ' % accurate in labelling the training data.')


    def classify(self, test_info):

        # put info from text file into OrderedDict
        test_info = extract_info(test_info)

        results = self._test_input(self.fold_data, test_info, self.indeces)

        self.test_acc = int(plot_confusion_matrix(test_info, results)[2]*100)
        print('Testing complete. Classifier is ' + str(self.test_acc) +
        ' % accurate in labelling the training data.')


################################################################################
################################################################################

class DiracSpatialClassifier(MultiFoldClassifier):
    """docstring for SpatialClassifier.MultiFoldClassifier"""

    def __init__(self, hi_freq=20000, n_bands=42, filt_taps=128, **kwargs):

        self.hi_freq = hi_freq
        self.n_bands = n_bands
        self.filt_taps = filt_taps

        super(DiracSpatialClassifier, self).__init__(**kwargs)


################################################################################

    def _extract_features( self, info ):

        indeces = {}

        for filepath, label in info.items():

            target = self._label_list.index(label) # numerical class indicator
            # load audio file
            audio, fs = sf.read(self.dataset_directory + filepath)
            print('Reading in ' + filepath + ' ...')

            # hi_freq provided to limit frequency range we are interested in
            # (low frequcies usually of interest). filt_taps can probably be
            # fixed in the future after some testing
            azi, elev, psi = extract_spatial_features(audio,fs,
                                hi_freq=self.hi_freq,n_bands=self.n_bands,
                                filt_taps=self.filt_taps)
            features = np.hstack((azi,elev,psi))

            # append a targets column at the end of the mfcc array
            data_to_add = np.hstack((features, np.array([[target]]
                                        * len(features))))

            if 'data' not in locals(): # does this variable exist yet?
                indeces[filepath] = [0, len(data_to_add)]
                data = data_to_add
            else:
                indeces[filepath] = [len(data), len(data) + len(data_to_add)]

                # add indeces for features from current file to dictionary
                data = np.vstack((data, data_to_add))
                # this allows for testing classification using features
                # from specific examples without having to reload audio

        print('Feature extraction complete.')
        return data, indeces


################################################################################
################################################################################


# class SpatialClassifier(BasicAudioClassifier):
#
#     def _extract_features( self, info ):
#
#         indeces = {}
#
#         for filepath, label in info.items():
#
#             target = self._label_list.index(label) # numerical class indicator
#             # load audio file
#             # note librosa collapses to mono and resamples @ 22050 Hz
#             audio, fs = sf.read(self.dataset_directory + filepath)
#             print('Reading in ' + filepath + ' ...')
#
#             azi, elev, psi = extract_spatial_features(audio,fs)
#             features = np.hstack((azi,elev,psi))
#
#             # append a targets column at the end of the mfcc array
#             data_to_add = np.hstack((features, np.array([[target]]
#                                         * len(features))))
#
#             if 'data' not in locals(): # does this variable exist yet?
#                 indeces[filepath] = [0, len(data_to_add)]
#                 data = data_to_add
#             else:
#                 indeces[filepath] = [len(data), len(data) + len(data_to_add)]
#
#                 # add indeces for features from current file to dictionary
#                 data = np.vstack((data, data_to_add))
#                 # this allows for testing classification using features
#                 # from specific examples without having to reload audio
#
#         print('Feature extraction complete.')
#         return data, indeces
################################################################################
################################################################################


def extract_info( file_to_read ):

    with open(file_to_read) as info_file:
        info = OrderedDict(line.split() for line in info_file)

    return info # info is a dictionary with filenames and class labels
    # and is the preferred input format for BasicAudioClassifier


# def extract_info(file_to_read):
#
#     with open(file_to_read) as info_file:
#         info = OrderedDict([(line, line[:line.find('-')]) for line in info_file])
#
# new version of extract_info function to work with info files containing only
# file names (file names will contain labels)

def plot_confusion_matrix( info, results ):
# this function extracts lists of classes from OrderedDicts passed to it
# is this doing too much now

    true = [label for entry, label in info.items()]
    predictions = [label for entry, label in results.items()]

    label_list = sorted(set(true + predictions))
    report = classification_report(true, predictions, label_list)

    confmat = confusion_matrix(true, predictions)
    accuracies = confmat.diagonal() / confmat.sum(axis=1)
    class_accuracies = dict(zip(label_list, accuracies))
    total_accuracy = confmat.diagonal().sum() / confmat.sum()

    dataframe_confmat = pd.DataFrame(confmat, label_list, label_list)
    plt.figure(figsize = (10,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.show()

    return confmat, class_accuracies, total_accuracy, report
