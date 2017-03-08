# read filenames of recordings in
#
# bformat_segment function creates filenames - could we use filename list output
# from this??
#
# randomly shuffle recordings within class
# divide each class set into 4 segments
# write text files with 3x segments for testing and the remaining for training
# for now include class label

import glob
import random
import os

# neat function for unpacking nested lists
# (makes some of the later list comprehensions more comprehensible)
def unpack_list(x):
    unpacked = [i for y in x for i in y]
    return unpacked

n_folds = 4
dataset_dirs = ['SiteA-24-11-16','SiteB-24-11-16','SiteC-24-11-16']
audio_directory = 'audio'
text_directory = 'evaluation_setup'

# retrieve list of files
filename_list = [glob.glob(folder + '/*') for folder in dataset_dirs]

# within-class shuffle of audio samples
for category in filename_list:
    random.shuffle(category)

# calculate number of samples per fold based on length of largest class
fold_len = max(len(category) for category in filename_list) // n_folds

# take slices of each class from filename_list
chunks = [[category[i*fold_len:(i+1)*fold_len] for category in filename_list]
            for i in range(n_folds)]

# unpack each chunk to make list of folds
test_folds = [unpack_list(x) for x in chunks]

# Reformat filename strings to feature filename followed by class label after a
# tab. Class label is taken from first part of directory name (before'-').
for fold in test_folds:
    for n, filepath in enumerate(fold):
        fold[n] = filepath[filepath.find('/')+1:] + '\t' + filepath[
                    0:filepath.find('-')] + '\n'

# make list of folds to use as training for each fold used as test data
train_folds = [unpack_list([x for x in test_folds if x not in [fold]])
                for fold in test_folds]

# alphabetise train_folds
for fold in train_folds:
    fold.sort()

# make a folder for evaluation text files
os.makedirs('evaluation_setup', exist_ok=True)

# write results out to text files
for n, fold in enumerate(test_folds):
    txtfile = open(text_directory + '/fold' + str(n+1) + '_test.txt','w')
    txtfile.writelines(fold)
    txtfile.close()

for n, fold in enumerate(train_folds):
    txtfile = open(text_directory + '/fold' + str(n+1) + '_train.txt','w')
    txtfile.writelines(fold)
    txtfile.close()

# make a folder to move audio database into
os.makedirs(audio_directory, exist_ok=True)

for filepath in unpack_list(filename_list):
    os.rename(filepath, audio_directory + '/' + filepath[filepath.find('/')+1:])
