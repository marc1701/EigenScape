# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture

training_info = [] # initialise list for string import
with open("evaluation_setup/fold1_train.txt", "r") as info_file:
    for lines in info_file:
        training_info.append(info_file.readline().split())

training_set = [] # initialise list for audio objects
for x in training_info:
    training_set.append(audio_sample(x[0], x[1]))

# build dictionary and hold accumulated feature vectors from each class label
# training_data = {}
# for example in training_set:
#     if example.label not in training_data:
#         training_data[example.label] = example.mfccs
#     else:
#         training_data[example.label] = np.vstack((training_data[example.label], example.mfccs))

# for soundclass in training_data: # normalise feature data
#     training_data[soundclass] = prp.scale(training_data[soundclass])
i = -1
training_data = {"scene_labels":[], "target_numbers":np.array([], dtype="int"), "data":np.array([])}
for example in training_set:
    if training_data["data"].size == 0:
        training_data["data"] = example.mfccs
    else:
        training_data["data"] = np.vstack((training_data["data"], example.mfccs))

    if example.label not in training_data["scene_labels"]:
        training_data["scene_labels"].append(example.label)
        i += 1

    training_data["target_numbers"] = np.append(training_data["target_numbers"], [i] * np.size(example.mfccs, 0))

n_classes = len(training_data["scene_labels"])
# calculate the means of each class
means = np.array([training_data["data"][training_data["target_numbers"]==i].mean(0) for i in range(n_classes)])

# create GMM object
classifier = GaussianMixture(15, "full", means_init=means)

# fit Gaussians to data
classifier.fit(training_data["data"])

# test model on training data
predictions = classifier.predict(training_data["data"])
# this hasn't worked very well at all so far

accuracy = np.mean(predictions == training_data["target_numbers"]) * 100
