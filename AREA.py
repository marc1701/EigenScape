import librosa
import numpy as np
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt


class BasicAudioClassifier:
    ''' Basic GMM-MFCC audio classifier along the lines of the baseline model
    described in DCASE 2015 '''

    def __init__( self, file_to_read='' ):

        # data scaler for normalisation - remembers mean and var of input data
        self.scaler = StandardScaler()
        # we can apply the same transform later to test data using these values

        self.label_list = [] # set up list of class labels
        self.gmms = {} # initialise dictionary for GMMs

        if file_to_read != '':
            self.train(file_to_read)


    def train( self, file_to_read ):
        data, info, indeces = self.gen_audio_features(file_to_read)
        # fit scaler and scale training data (exclude target_numbers column)
        data[:,:-1] = self.scaler.fit_transform(data[:,:-1])
        self.fit_gmms(data)
        results = self.test_input(data, info, indeces)

        correct = 0
        for entry in results:
            if results[entry] == info[entry]:
                correct += 1

        self.train_acc = (correct/len(info)*100)
        print('Training complete. Classifier is ' + str(self.train_acc) +
        ' % accurate in classifying the training data.')


    def classify( self, file_to_read ):
        # call this function to classify new data after training

        # generate audio features
        data, info, indeces = self.gen_audio_features(file_to_read)
        # save info of test data for confusion matrix plot
        self.last_info = info
        # scale data using pre-calculated mean and var
        data[:,:-1] = self.scaler.transform(data[:,:-1])
        # test_input function gets scores from GMM set
        results = self.test_input(data, info, indeces)
        # save latest results for confusion matrix plot
        self.last_results = results

        return results


    def gen_audio_features( self, file_to_read ):

        with open(file_to_read) as info_file:
            info = OrderedDict(line.split() for line in info_file)
        # info is a dictionary with filenames and class labels for training or testing data

        indeces = {}
        for filepath, label in info.items():
            if label not in self.label_list:
                self.label_list.append(label)
                print('Added ' + label + ' to the label list.')

            target = self.label_list.index(label) # numerical class indicator
            # load audio file
            # note librosa collapses to mono and resamples @ 22050 Hz
            audio, fs = librosa.load(
                        '../TUT-acoustic-scenes-2016-development/' + filepath)

            mfccs = librosa.feature.mfcc(audio, fs) # calculate MFCC values
            # swap axes so feature vectors are horizontal (time runs downwards)
            mfccs = mfccs.swapaxes(0, 1)

            # append a targets column at the end of the mfcc array
            data_to_add = np.hstack((mfccs, np.array([[target]] * len(mfccs))))

            if 'data' not in locals(): # does this variable exist yet?
                indeces[filepath] = [0, len(data_to_add)]
                data = data_to_add
            else:
                indeces[filepath] = [len(data), len(data) + len(data_to_add)]

                # add indeces for mfccs from current file to dictionary
                data = np.vstack((data, data_to_add))
                # this allows for testing classification using mfccs
                # from specific examples without having to reload audio

        print('Feature extraction complete.')
        return data, info, indeces


    def fit_gmms( self, data ):
        for label in self.label_list:
            # print('Training GMM for ' + label)
            self.gmms[label] = GaussianMixture(n_components=10)
            label_num = self.label_list.index(label)
            # extract class data from training matrix
            label_data = data[data[:,-1] == label_num,:-1]
            self.gmms[label].fit(label_data) # train GMMs


    def test_input( self, data, info, indeces ):

        results = OrderedDict()
        scores = {} # initialise dictionary for scores

        for entry in info:
            # find indeces of data from specific audio file
            start, end = indeces[entry][0], indeces[entry][1]
            print('Testing ' + entry)

            # slice data from large array
            data_to_evaluate = data[start:end,:-1]

            for label, gmm in self.gmms.items():
                scores[label] = np.sum(gmm.score_samples(data_to_evaluate))

            # find label with highest score and store result in dictionary
            results[entry] = max(scores, key = scores.get)

        # returns an OrderedDict of predictions corresponding to input file list
        return results


    def show_confusion( self ):
        plot_confusion_matrix(self.last_info, self.last_results, self.label_list)

################################################################################
def plot_confusion_matrix( info, results, label_list ):
# this function extracts lists of classes from OrderedDicts passed to it
# label list could be extracted from the info dict to make this more standalone

    true = [label for entry, label in info.items()]
    predictions = [label for entry, label in results.items()]

    confmat = confusion_matrix(true, predictions, label_list)

    dataframe_confmat = pd.DataFrame(confmat, label_list, label_list)
    plt.figure(figsize = (10,7))
    sn.heatmap(dataframe_confmat, annot=True)
    plt.show()
