# train_text = open("evaluation_setup/fold1_train.txt", "r")
# training_files = train_text.readlines() # store lines of text file in a list

from mfcc_audio_class import *
import sklearn.preprocessing as prp
from sklearn.mixture import GaussianMixture as gmm

training_info = [] # initialise list for string import
with open("evaluation_setup/fold1_train.txt", "r") as info_file:
    for lines in info_file:
        training_info.append(info_file.readline().split())

training_set = [] # initialise list for audio objects
for x in training_info:
    training_set.append(audio_sample(x[0], x[1]))

for example in training_set: # normalise feature data
    example.mfccs = prp.scale(example.mfccs)
