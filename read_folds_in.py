# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GMM

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
i = 0
training_data = {"scene_labels":[], "target_numbers":np.array([], dtype="int"), "data":np.array([])}
for example in training_set:
    if training_data["data"].size == 0:
        training_data["data"] = example.mfccs
    else:
        training_data["data"] = np.vstack((training_data["data"], example.mfccs))

    training_data["target_numbers"] = np.append(training_data["target_numbers"], [i] * np.size(example.mfccs, 0))
    # getting a weird thing at the moment where this ends up being longer than the data array

    if example.label not in training_data["scene_labels"]:
        training_data["scene_labels"].append(example.label)
        i += 1

# classifier = GMM(10, "full", )
