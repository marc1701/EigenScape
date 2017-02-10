# matplotlib.use('TkAgg')
# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

# from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture
import librosa
import numpy as np
from collections import OrderedDict
# import glob

# filepaths = glob.glob(
# '../TUT-acoustic-scenes-2016-development/evaluation_setup/fold*_train.txt')

# read audio file database text into script
with open(
'../TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_train.txt',
'r') as info_file:
    training_info = OrderedDict(line.split() for line in info_file)

##########################################################
####### CALCULATION OF MFCCS #######
# need to make the audio folder an input parameter
label_list = [] # make list of labels
training_indeces = {}
for filepath, label in training_info.items():
    if label not in label_list:
        label_list.append(label)
        print('Added ' + label + ' to the label list.')

    target = label_list.index(label) # numerical class indicator
    # load audio file - note librosa collapses to mono and resamples @ 22050 Hz
    audio, fs = librosa.load(
                '../TUT-acoustic-scenes-2016-development/' + filepath)

    mfccs = librosa.feature.mfcc(audio, fs) # calculate MFCC values
    # swap axes so feature vectors are horizontal (time runs downwards)
    mfccs = mfccs.swapaxes(0, 1)

    # append a targets column at the end of the mfcc array
    data_to_add = np.hstack((mfccs, np.array([[target]] * len(mfccs))))

    if 'training_data' not in locals(): # does this variable exist yet?
        training_indeces[filepath] = [0, len(data_to_add)]
        training_data = data_to_add
    else:
        training_indeces[filepath] = [len(training_data), len(training_data)
                                      + len(data_to_add)]

        # add indeces for mfccs from current file to dictionary
        training_data = np.vstack((training_data, data_to_add))
        # this allows for testing classification using mfccs
        # from specific examples without having to reload audio

print('Training data read complete.')
###### MIGHT BE AN IDEA TO TURN THIS INTO A FUCTION ######
##########################################################

# normalise data - scaler object remembers mean and var from training data
# can apply the same transform later to test data using these values
data_scaler = prp.StandardScaler()
# fit scaler and scale training data (exclude last column - target_numbers)
training_data[:,:-1] = data_scaler.fit_transform(training_data[:,:-1])

gmms = {} # initialise dictionary for GMMs
scores = {} # initialise dictionary for scores
for label in label_list:
    print('Training GMM for ' + label)
    gmms[label] = GaussianMixture(n_components=10)
    label_num = label_list.index(label)
    # extract class data from training matrix
    label_data = training_data[training_data[:,-1] == label_num,:-1]
    gmms[label].fit(label_data) # train GMMs

results = OrderedDict()
for entry in training_info:
    # find indeces of data from specific audio file
    start, end = training_indeces[entry][0], training_indeces[entry][1]
    print('Testing ' + entry)

    # slice data from large array
    data_to_evaluate = training_data[start:end,:-1]

    for label, gmm in gmms.items():
        scores[label] = np.sum(gmm.score_samples(data_to_evaluate))

    # find label with highest score and store result in dictionary
    results[entry] = max(scores, key = scores.get)

# correct = 0
# for entry in results:
#     if results[entry] == training_info[entry]:
#         correct += 1

true = [label for entry, label in training_info.items()]
predictions = [label for entry, label in results.items()]


def plot_confusion_matrix:
    dataframe_confmat = pd.DataFrame(confmat, label_list, label_list)
    plt.figure(figsize = (10,7))
    sn.heatmap(dataframe_confmat, annot=True)
    # plt.show()
