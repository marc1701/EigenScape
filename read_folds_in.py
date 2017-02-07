# matplotlib.use('TkAgg')
# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

# from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture
import librosa
import numpy as np
# import glob

# filepaths = glob.glob(
# '../TUT-acoustic-scenes-2016-development/evaluation_setup/fold*_train.txt')

# read audio file database text into script
with open(
'../TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_train.txt',
'r') as info_file:
    training_info = [line.split() for line in info_file]

##########################################################
####### CALCULATION OF MFCCS #######
# need to make the audio folder an input parameter
label_list = [] # make list of numbers for labels
# training_dict = {}
for entry in training_info:
    if entry[1] not in label_list:
        label_list.append(entry[1])

    target = label_list.index(entry[1]) # numerical class indicator
    # load audio file - note librosa collapses to mono and resamples @ 22050 Hz
    audio, fs = librosa.load(
                '../TUT-acoustic-scenes-2016-development/' + entry[0])
    mfccs = librosa.feature.mfcc(audio, fs) # calculate MFCC values
    # swap axes so feature vectors are horizontal (time runs downwards)
    mfccs = np.swapaxes(mfccs, 0, 1)
    new_data_size = len(mfccs)

    # append a targets column at the end of the mfcc array
    data_to_add = np.hstack((mfccs, np.array([[target]] * new_data_size)))

    if 'training_data' not in locals(): # does this variable exist yet?
        training_data = data_to_add
    else:
        training_data = np.vstack((training_data, data_to_add))
###### MIGHT BE AN IDEA TO TURN THIS INTO A FUCTION ######
##########################################################

# normalise data - scaler object remembers mean and var from training data
# can apply the same transform later to test data using these values
data_scaler = prp.StandardScaler()
# fit scaler and scale training data (exclude last colums - target_numbers)
scaled_data = data_scaler.fit_transform(training_data[:,:-1])
# stick target_numbers back on the end
scaled_data = np.hstack((scaled_data, training_data[:,[-1]]))

gmms = {} # initialise dictionary for GMMs
results = {} # initialise dictionary for result tallies
scores = {} # initialise dictionary for scores
for label in label_list:
    gmms[label] = GaussianMixture(n_components=10)
    label_num = label_list.index(label)
    # extract class data from training matrix
    label_data = scaled_data[scaled_data[:,-1] == label_num,:-1]
    gmms[label].fit(label_data) # train GMMs
    results[label] = [0,0] # set up results dictionary

# reload audio files individually to test accuracy of classifier
# (can very probably devise a better way of storing data from earlier to avoid
#  having to reload all the audio files!)
for entry in training_info:
    audio, fs = librosa.load('../TUT-acoustic-scenes-2016-development/' +
    entry[0])

    test_data = np.swapaxes(librosa.feature.mfcc(audio, fs), 0, 1)
    test_data = data_scaler.transform(mfccs)
    results[entry[1]][0] += 1

    for label, gmm in gmms.items():
        scores[label] = np.sum(gmm.score_samples(test_data))

    top_score = -9e99
    top_label = ''

    for label, score in scores.items():
        if top_label == '' or top_score < score:
            top_label = label
            top_score = score

    if top_label == entry[1]:
        results[entry[1]][1] += 1
