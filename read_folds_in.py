# matplotlib.use('TkAgg')
# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture
# import glob

# filepaths = glob.glob(
# '../TUT-acoustic-scenes-2016-development/evaluation_setup/fold*_train.txt')

# read audio file database text into script
with open('../TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_train.txt', 'r') as info_file:
    training_info = [line.split() for line in info_file]

# this is a massive fudge - need to make the audio folder an input parameter
for entry in training_info:
    entry[0] = '../TUT-acoustic-scenes-2016-development/' + entry[0]

# create audio objects (w/MFCCs) for all the specified audio files
n_mfccs = 20
training_set = [audio_sample(x[0], x[1], n_mfccs) for x in training_info]

label_list = [] # make list of numbers for labels
for example in training_set:
    if example.label not in label_list:
        label_list.append(example.label)

    target = label_list.index(example.label) # numerical class indicator
    new_data_size = len(example.mfccs)

    # append a target column at the end of the mfcc array
    data_to_add = np.hstack((example.mfccs, np.array([[target]] * new_data_size)))

    if 'training_data' not in locals(): # does this exist yet?
        training_data = data_to_add
    else:
        training_data = np.vstack((training_data, data_to_add))

training_data[:,:n_mfccs] = prp.scale(training_data[:,:n_mfccs]) # normalise data



# # read testing set into memory - REPEAT OF ABOVE LINES, SHOULD FUNCTIONISE THIS
# with open('../TUT-acoustic-scenes-2016-development/evaluation_setup/fold1_evaluate.txt', 'r') as info_file:
#     testing_info = [line.split() for line in info_file]
#
# for entry in testing_info:
#     entry[0] = '../TUT-acoustic-scenes-2016-development/' + entry[0]

# create audio objects (w/MFCCs) for all the specified audio files
# testing_set = [audio_sample(x[0], x[1]) for x in testing_info]

# build dictionary and hold accumulated feature vectors from each class label
# this is currently imcompatible with the accumulator code below
# training_data = {}
# for example in training_set:
#     if example.label not in training_data:
#         training_data[example.label] = example.mfccs
#     else:
#         training_data[example.label] = np.vstack((training_data[example.label],
#         example.mfccs))

gmms = {} # initialise dictionary for GMMs
results = {} # initialise dictionary for result tallies
scores = {} # initialise dictionary for scores
for label in label_list:
    gmms[label] = GaussianMixture(n_components=50)
    label_num = label_list.index(label)
    # extract class data from training matrix
    label_training_data = training_data[training_data[:,n_mfccs] == label_num, :n_mfccs]
    gmms[label].fit(label_training_data) # train GMMs
    results[label] = [0,0] # set up results dictionary

# for label in training_data:
#     # normalise feature data
#     scores[label] = 0
correct = 0
for example in training_set:
    # test_data = prp.scale(example.mfccs)
    test_data = example.mfccs
    results[example.label][0] += 1
    for label, gmm in gmms.items():
        scores[label] = np.sum(gmm.score_samples(test_data))

    top_score = -9e99
    top_label = ''
    for label, score in scores.items():
        if top_label == '' or top_score < score:
            top_label = label
            top_score = score

    if top_label == example.label:
        correct += 1
        results[example.label][1] += 1

# correct = 0
# for example in testing_set:
#     results[example.label][0] += 1 # add to example labels class count
#     for label, gmm in gmms.items():
#         scores[label] = np.sum(gmm.score_samples(example.mfccs))
#
#     top_score = 0
#     top_label = ''
#     for label, score in scores.items():
#         if top_label == '' or top_score < score:
#             top_label = label
#             top_score = score
#
#     if top_label == example.label:
#         results[example.label][1] += 1

# i = -1
# training_data = {"scene_labels":[], "target_numbers":np.array([], dtype="int"), "data":np.array([])}
# for example in training_set:
#     if training_data["data"].size == 0:
#         training_data["data"] = example.mfccs
#     else:
#         training_data["data"] = np.vstack((training_data["data"], example.mfccs))
#
#     if example.label not in training_data["scene_labels"]:
#         training_data["scene_labels"].append(example.label)
#         i += 1
#
#     training_data["target_numbers"] = np.append(training_data["target_numbers"], [i] * np.size(example.mfccs, 0))

# n_classes = len(training_data["scene_labels"])
# # calculate the means of each class
# means = np.array([training_data["data"][training_data["target_numbers"]==i].mean(0) for i in range(n_classes)])
#
# # create GMM object
# classifier = GaussianMixture(15, "full", means_init=means)
#
# # fit Gaussians to data
# classifier.fit(training_data["data"])
#
# # test model on training data
# predictions = classifier.predict(training_data["data"])
# # this hasn't worked very well at all so far
#
# accuracy = np.mean(predictions == training_data["target_numbers"]) * 100
