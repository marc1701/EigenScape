import glob
from AREA import *

filepaths = glob.glob('../TUT-acoustic-scenes-2016-development/evaluation_setup/fold*_train.txt')

train_info = {}
for path in filepaths:
    train_info[path[-15:-4]] = extract_info(path)

classifiers = {}
train_results = {}
for info, data in train_info.items():
    classifiers[info[:5]] = BasicAudioClassifier('../TUT-acoustic-scenes-2016-development/')
    train_results[info] = classifiers[info[:5]].train(data)
