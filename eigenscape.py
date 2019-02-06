import os
import glob
import librosa
import resampy
import numpy as np
import pandas as pd
import seaborn as sn
import soundfile as sf
import progressbar as pb
import matplotlib.pyplot as plt
from scipy import interp
from collections import OrderedDict
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import (confusion_matrix, classification_report,
                                roc_curve, auc)

from spatial import *
import datatools


class MultiGMMClassifier:
    # classifier using a bank of GMMs (rather than a single GMM with many
    # components). one GMM is trained per class label present in y. probability
    # scores from each GMM for test data are concatenated to form
    # decision_function output

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


    def decision_function( self, X ):
        # this is probably not technically a decision function, but this ensures
        # compatability with the other sklearn classifiers

        y_score = np.array([gmm.score_samples(X)
                    for _, gmm in self.gmms.items()]).swapaxes(0, 1)

        return y_score



def BOF_audio_classify( classifier, X, y, info, indices ):
    # provide a classifier object and this will sum its output for each
    # frame in order to classify an entire audio clip
    # entire dataset, info and indices are provided in order to minimise
    # recalculation of features

    for entry in info:
        # find indices of data from specific audio file
        start, end = indices[entry]

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
        this_score = np.sum(classifier.decision_function(
                                X_to_eval), 0).reshape(1,-1)

        # save scores in an array
        if 'y_score' not in locals():
            y_score = this_score
        else:
            y_score = np.append(y_score, this_score, axis=0)

    return y_test, y_score




########################################################################



def build_audio_featureset(feature_extractor, dataset_directory='', **kwargs):
    # constructs large numpy array containing features extracted from the audio
    # files present in dataset_directory. as features are extracted from short
    # frames of a larger audio file, this fucntion keeps track of which feature
    # vectors have been extracted from which larger file (indices dict)

    dataset_files = [os.path.basename(x) for x in glob.glob(
                        dataset_directory + '*.wav')]

    dataset_files.sort()
    # put files in alphabetical / numeric order

    # finding labels from filenames removes the need for a dedicated
    # 'full set' text file, but this will only work if filenames contain
    # the labels

    info = OrderedDict([line, line[:line.find('.')]]
                        for line in dataset_files)

    # make list of unique labels in data
    label_list = sorted(set(labels for x, labels in info.items()))

    indices = {}

    progbar = pb.ProgressBar(max_value=len(info))
    progbar.start()

    for n, (filepath, label) in enumerate(info.items()):

        progbar.update(n)

        target = label_list.index(label) # numerical class indicator

        features = feature_extractor(dataset_directory + filepath, **kwargs)

        # append a targets column at the end of the array
        data_to_add = np.hstack((
                        features, np.array([[target]] * len(features))))

        if 'data' not in locals(): # does this variable exist yet?
            indices[filepath] = [0, len(data_to_add)]
            data = data_to_add
        else:
            indices[filepath] = [len(data), len(data) + len(data_to_add)]

            # add indices for features from current file to dictionary
            data = np.vstack((data, data_to_add))
            # this allows for testing classification using features
            # from specific examples without having to reload audio

    progbar.finish()

    return data, indices, label_list


dirac_fmt = ['%d']*(20) + ['%d']*(20) + ['%1.3f']*(20) + ['%d']

def save_data(filename, data, indices, label_list, fmt='%1.3f'):
    # write out file with sensible number formatting (minimises file size)
    np.savetxt(filename + '_data.txt', data, fmt)

    # write out dictionary of file clip indices (readable back using eval)
    with open(filename + '_file_indices.txt','w') as out_file:
        out_file.write('{')
        for entry, vals in indices.items():
            out_file.write("'"+entry+"'"+':'+str(vals)+', ')
        out_file.write('}')

    # write out list of labels
    with open(filename + '_labels.txt','w') as out_file:
        for label in label_list:
            out_file.write(label + '\n')


def calculate_mfccs(filepath):
    # load audio file
    # note librosa collapses to mono - auto resample not used
    audio, fs = librosa.load(filepath, sr=None)

    # resampling here @ to fs/2
    audio = librosa.core.resample(audio, fs, fs/2)
    fs = fs/2

    # calculate MFCC values
    features = librosa.feature.mfcc(audio, fs).T
    # swap axes so feature vectors are horizontal (time runs downwards)

    return features


def calculate_dirac(filepath, hi_freq=None, n_bands=20, filt_taps=2048):
    # resamples audio and extracts features using directional audio coding
    # techniques

    audio, fs = sf.read(filepath)
    audio = resampy.resample(audio, fs, fs/2, axis=0)
    fs = fs/2 # update fs after resampling

    # hi_freq provided to limit frequency range we are interested in
    # (low frequcies usually of interest). filt_taps can probably be
    # fixed in the future after some testing
    azi, elev, psi = extract_spatial_features(audio, fs, hi_freq=hi_freq,
                        n_bands=n_bands, filt_taps=filt_taps)

    features = np.hstack((azi,elev,psi))

    return features



def extract_info( file_to_read ):
    # converts file lists into dictionaries with file names and class labels

    with open(file_to_read) as info_file:
        info = OrderedDict([line.rstrip('\n'), line[:line.find('.')]]
                            for line in info_file)

    return info


def vectorise_indices(info, indices):
    vector_indices = np.array([np.r_[
                     indices[file][0]:indices[file][1]]
                     for file in info]).reshape(-1)

    return vector_indices


def plot_confusion_matrix( y_test, y_score, label_list, plot=True ):
# plot confusion matrix based on binarized y_test and y_score provided as output
# from the classifier objects

    true = np.argmax(y_test, 1)
    predictions = np.argmax(y_score, 1)

    label_abbr = [''.join(caps for caps in label if caps.isupper())
                    for label in label_list]
    # report = classification_report(true, predictions, target_names=label_list)

    confmat = confusion_matrix(true, predictions)
    # accuracies = confmat.diagonal() / confmat.sum(axis=1)
    # class_accuracies = dict(zip(label_list, accuracies))
    # total_accuracy = confmat.diagonal().sum() / confmat.sum()

    if plot:
        dataframe_confmat = pd.DataFrame(confmat, label_list, label_abbr)
        plt.figure(figsize = (10,7))
        sn.heatmap(dataframe_confmat, annot=True)
        plt.show()

    return confmat# , class_accuracies, total_accuracy, report


def calc_roc( y_test, y_score ):

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    n_classes = y_test.shape[1]

    for i in range(n_classes):
        # compute ROC curves for each class
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and AUC
    fpr[i+1], tpr[i+1], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc[i+1] = auc(fpr[i+1], tpr[i+1])

    # Compute macro-average ROC curve and AUC
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr[i+2] = all_fpr
    tpr[i+2] = mean_tpr
    roc_auc[i+2] = auc(fpr[i+2], tpr[i+2])

    return fpr, tpr, roc_auc



def plot_roc( y_test, y_score, label_list ):
# plots a series of ROC curves for each class and the micro-average
# might be a little simplistic for my needs - manual plotting could be needed

    fpr, tpr, roc_auc = calc_roc(y_test, y_score)

    label_list.append('Micro-Average')
    label_list.append('Macro-Average')

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


def idx_sel(features, f_idx):
    # neat little function to integrate np.r_ with index dictionaries
    idx = np.array([np.r_[f_idx[feat][0]:f_idx[feat][1]]
                    for feat in features]).reshape(-1)
    return idx


def plot_multifold_roc( y_test_folds, y_score_folds, label_list ):
# in this case y_test_folds and y_score_folds are dictionaries containing y_test
# and y_score values across each calculated fold
    label_list.append('Micro-Average')
    label_list.append('Macro-Average')

    for j in range(len(label_list)):

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        for i in y_test_folds:
            fpr, tpr, roc_auc = calc_roc(y_test_folds[i], y_score_folds[i])
            tprs.append(interp(mean_fpr, fpr[j], tpr[j]))
            tprs[-1][0] = 0.0
            aucs.append(roc_auc[j])
            plt.plot(fpr[j], tpr[j], lw=1, alpha=0.3,
                     label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc[j]))

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
                 label='Luck', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic - ' + label_list[j])
        plt.legend(loc="lower right")
        plt.show()
