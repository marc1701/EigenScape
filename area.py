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


class BasicAudioClassifier:
    ''' Basic GMM-MFCC audio classifier along the lines of the baseline
    model described in DCASE 2015 '''

    def __init__( self, dataset_directory='', n_gaussians=10 ):

        # self._label_list = [] # set up list of class labels
        self._gmms = OrderedDict() # initialise dictionary for GMMs

        # set dataset_directory or class will assume current working directory
        self.dataset_directory = dataset_directory

        # data scaler for normalisation - remembers mean and var of input data
        self._scaler = StandardScaler()
        # we can apply the same transform later to test data using these values

        self._n_gaussians = n_gaussians

    def train( self, info ):

        info = extract_info(info)

        # make list of unique labels in training data
        self._label_list = sorted(set(labels for x, labels in info.items()))
        data, indeces = self._build_dataset(info)

        # fit scaler and scale training data (exclude target_numbers column)
        data[:,:-1] = self._scaler.fit_transform(data[:,:-1])
        self._fit_gmms(data)
        y_test, y_score = self._test_input(data, info, indeces)

        # find overall training accuracy percentage
        self.train_acc = int(plot_confusion_matrix(y_test, y_score,
                            self._label_list)[2]*100)
        print('Training complete. Classifier is ' + str(self.train_acc) +
        ' % accurate in labelling the training data.')


    def test( self, info ):

        # call this function to classify new data after training
        info = extract_info(info)
        # generate audio features
        data, indeces = self._build_dataset(info)
        # scale data using pre-calculated mean and var
        data[:,:-1] = self._scaler.transform(data[:,:-1])
        # _test_input function gets scores from GMM set
        y_test, y_score = self._test_input(data, info, indeces)

        self.test_acc = int(plot_confusion_matrix(y_test, y_score,
                            self._label_list)[2]*100)
        print('Testing complete. Classifier is ' + str(self.test_acc) +
        ' % accurate in labelling the test data.')


############################# 'Private' methods: ###############################

    def _build_dataset( self, info ):
        # print('Generating feature dataset from audio files...')

        indeces = {}

        progbar = pb.ProgressBar(max_value=len(info))
        progbar.start()

        for n, (filepath, label) in enumerate(info.items()):

            progbar.update(n)

            target = self._label_list.index(label) # numerical class indicator

            features = self._extract_features(filepath)

            # append a targets column at the end of the mfcc array
            data_to_add = np.hstack((
                            features, np.array([[target]] * len(features))))

            if 'data' not in locals(): # does this variable exist yet?
                indeces[filepath] = [0, len(data_to_add)]
                data = data_to_add
            else:
                indeces[filepath] = [len(data), len(data) + len(data_to_add)]

                # add indeces for features from current file to dictionary
                data = np.vstack((data, data_to_add))
                # this allows for testing classification using features
                # from specific examples without having to reload audio

        progbar.finish()
        # print('Feature extraction complete.')
        return data, indeces


    def _extract_features(self, filepath):
        # load audio file
        # note librosa collapses to mono - auto resample not used
        audio, fs = librosa.load(self.dataset_directory + filepath, sr=None)

        # resampling here @ to fs/2
        audio = librosa.core.resample(audio, fs, fs/2)
        fs = fs/2

        # calculate MFCC values
        features = librosa.feature.mfcc(audio, fs).T
        # swap axes so feature vectors are horizontal (time runs downwards)

        return features


    def _fit_gmms( self, data ):

        # print('Fitting GMMs to data classes...')
        progbar = pb.ProgressBar(max_value=len(self._label_list))
        progbar.start()

        for n, label in enumerate(self._label_list):

            progbar.update(n)

            self._gmms[label] = GaussianMixture(self._n_gaussians)
            label_num = self._label_list.index(label)
            # extract class data from training matrix
            label_data = data[data[:,-1] == label_num,:-1]
            self._gmms[label].fit(label_data) # train GMMs

        progbar.finish()


    def _test_input( self, data, info, indeces ):

        for entry in info:
            # find indeces of data from specific audio file
            start, end = indeces[entry]

            # slice data from large arrays
            data_to_evaluate = data[start:end,:-1]

            # extract class label from data array
            data_class = data[start,-1].reshape(-1)
            data_class = label_binarize(data_class, classes = [0,1,2,3,4,5,6,7])
            # this feels clunky but more elaborate would need rewrite

            # build y_test array - clunky but necessary
            if 'y_test' not in locals():
                y_test = data_class
            else:
                y_test = np.append(y_test, data_class, axis=0)

            this_score = np.array([np.sum(gmm.score_samples(data_to_evaluate))
                           for _, gmm in self._gmms.items()]).reshape(1,-1)

            # scale scores between 0 and 1 (ready for ROC)
            # norm_score = (this_score / np.max(np.abs(this_score))) + 1

            # save previous scores in an array
            if 'y_score' not in locals():
                y_score = this_score
            else:
                y_score = np.append(y_score, this_score, axis=0)

        return y_test, y_score


################################################################################
################################################################################
class MultiFoldClassifier(BasicAudioClassifier):

    def __init__(self, **kwargs):
        super(MultiFoldClassifier, self).__init__(**kwargs)

        # to speed up multifold testing, all audio is loaded and features
        # calculated at the start (saves reloading all the data for each fold)
        dataset_files = [os.path.basename(x)
                            for x in glob.glob(
                            self.dataset_directory + '*.wav')]

        dataset_files.sort()
        # put files in alphabetical / numeric order

        # finding labels from filenames removes the need for a dedicated
        # 'full set' text file, but this will only work if filenames contain
        # the labels

        dataset_info = OrderedDict([line, line[:line.find('.')]]
                            for line in dataset_files)

        # make list of unique labels in data
        self._label_list = sorted(set(labels
                            for x, labels in dataset_info.items()))

        self.data, self.indeces = self._build_dataset(dataset_info)


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
        y_test, y_score = self._test_input(
                                    self.fold_data, train_info, self.indeces)

        # find overall training accuracy percentage
        self.train_acc = int(plot_confusion_matrix(y_test, y_score,
                            self._label_list)[2]*100)
        print('Training complete. Classifier is ' + str(self.train_acc)
              + ' % accurate in labelling the training data.')

        return y_test, y_score


    def test(self, test_info):

        # put info from text file into OrderedDict
        test_info = extract_info(test_info)

        y_test, y_score = self._test_input(
                                    self.fold_data, test_info, self.indeces)

        self.test_acc = int(plot_confusion_matrix(y_test, y_score,
                            self._label_list)[2]*100)
        print('Testing complete. Classifier is ' + str(self.test_acc) +
        ' % accurate in labelling the test data.')

        return y_test, y_score


    def save_data(self, filename):
        # write out file with sensible number formatting (minimises file size)
        np.savetxt(filename + '_data.txt', self.data, fmt='%1.4f')
        # not sure now about this formatting - might be an idea to save out
        # multiple resolutions and see where there stops being changes

        self.save_metadata(filename)


    def save_metadata(self, filename):

        # write out dictionary of file clip indeces (readable back using eval)
        with open(filename + '_indeces.txt','w') as out_file:
            out_file.write('{')
            for entry, vals in self.indeces.items():
                out_file.write("'"+entry+"'"+':'+str(vals)+', ')
            out_file.write('}')

        # write out list of labels
        with open(filename + '_labels.txt','w') as out_file:
            for label in self._label_list:
                out_file.write(label + '\n')

################################################################################
################################################################################

class DiracSpatialClassifier(MultiFoldClassifier):
    """docstring for SpatialClassifier.MultiFoldClassifier"""

    def __init__(self, hi_freq=None, n_bands=20, filt_taps=2048, **kwargs):

        self.hi_freq = hi_freq
        self.n_bands = n_bands
        self.filt_taps = filt_taps

        super(DiracSpatialClassifier, self).__init__(**kwargs)


    def save_data(self, filename):

        # write out file with sensible number formatting (minimises file size)
        np.savetxt(filename+ '_data.txt', self.data,
                   fmt=['%d']*(self.n_bands*2)
                   + ['%1.3f']*(self.n_bands) + ['%d'])

        self.save_metadata(filename)

################################################################################

    def _extract_features(self, filepath):

        audio, fs = sf.read(self.dataset_directory + filepath)
        audio = resampy.resample(audio, fs, fs/2, axis=0)
        fs = fs/2 # update fs after resampling

        # hi_freq provided to limit frequency range we are interested in
        # (low frequcies usually of interest). filt_taps can probably be
        # fixed in the future after some testing
        azi, elev, psi = extract_spatial_features(audio,fs,hi_freq=self.hi_freq,
                            n_bands=self.n_bands,filt_taps=self.filt_taps)

        features = np.hstack((azi,elev,psi))

        return features


################################################################################
################################################################################

class ExternalDataClassifier(MultiFoldClassifier):

    def __init__(self, ext_data, indeces, labels_file, **kwargs):

        self.data = np.loadtxt(ext_data)
        self.indeces = eval(open(indeces,'r').read())

        with open(labels_file,'r') as labels:
            self._label_list = [line.rstrip() for line in labels.readlines()]

        BasicAudioClassifier.__init__(self)


################################################################################
################################################################################


def extract_info( file_to_read ):

    with open(file_to_read) as info_file:
        info = OrderedDict([line.rstrip('\n'), line[:line.find('.')]]
                            for line in info_file)

    # with open(file_to_read) as info_file:
    #     info = OrderedDict(line.split() for line in info_file)

    return info # info is a dictionary with filenames and class labels
    # and is the input format for BasicAudioClassifier


def plot_confusion_matrix( y_test, y_score, label_list ):
# plot confusion matrix based on binarized y_test and y_score provided as output
# from the classifier objects

    true = np.argmax(y_test, 1)
    predictions = np.argmax(y_score, 1)

    label_abbr = [''.join(caps for caps in label if caps.isupper())
                    for label in label_list]
    report = classification_report(true, predictions, target_names=label_list)

    confmat = confusion_matrix(true, predictions)
    accuracies = confmat.diagonal() / confmat.sum(axis=1)
    class_accuracies = dict(zip(label_list, accuracies))
    total_accuracy = confmat.diagonal().sum() / confmat.sum()

    dataframe_confmat = pd.DataFrame(confmat, label_list, label_abbr)
    plt.figure(figsize = (10,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.show()

    return confmat, class_accuracies, total_accuracy, report


def calc_roc( y_test, y_score ):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y_test.shape[1]

    for i in range(n_classes):
        # compute ROC curves for each class
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr[i+1], tpr[i+1], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc[i+1] = auc(fpr[i+1], tpr[i+1])

    return fpr, tpr, roc_auc



def plot_roc( y_test, y_score, label_list ):
# plots a series of ROC curves for each class and the micro-average
# might be a little simplistic for my needs - manual plotting could be needed

    fpr, tpr, roc_auc = calc_roc(y_test, y_score)

    label_list.append('Micro-Average')

    for i in range(len(label_list)):

        plt.figure(i)
        lw = 2
        plt.plot(fpr[i], tpr[i], color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(
                'Receiver operating characteristic example - ' + label_list[i])
        plt.legend(loc="lower right")
        plt.show()
