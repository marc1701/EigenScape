# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture

# read audio file database text into script
with open('evaluation_setup/fold1_train.txt', 'r') as info_file:
    training_info = [info_file.readline().split() for lines in info_file]

# create audio objects (w/MFCCs) for all the specified audio files
training_set = [audio_sample(x[0], x[1]) for x in training_info]


# read testing set into memory - REPEAT OF ABOVE LINES, SHOULD FUNCTIONISE THIS
with open('evaluation_setup/fold1_evaluate.txt', 'r') as info_file:
    testing_info = [info_file.readline().split() for lines in info_file]

# create audio objects (w/MFCCs) for all the specified audio files
testing_set = [audio_sample(x[0], x[1]) for x in testing_info]


# build dictionary and hold accumulated feature vectors from each class label
# this is currently imcompatible with the accumulator code below
training_data = {}
for example in training_set:
    if example.label not in training_data:
        training_data[example.label] = example.mfccs
    else:
        training_data[example.label] = np.vstack((training_data[example.label], example.mfccs))

gmms = {} # initialise dictionary for GMMs
for label in training_data:
    training_data[label] = prp.scale(training_data[label]) # normalise feature data
    gmms[label] = GaussianMixture(n_components=10)
    gmms[label].fit(training_data[label]) # train GMMs


# Test all files from testing data using the code currently in gmm_classify.py
# Total up how many were classified correctly against however many in class n
# print labels and percentages

correct = 0
for example in testing_set:
    for label, gmm in gmms.items():
        score = gmm.score_samples(example.mfccs)
        score = np.sum(score)
        if score > best_score:
            best_score = score
            best_label = label
        if example.label == best_label:
            correct += 1

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
