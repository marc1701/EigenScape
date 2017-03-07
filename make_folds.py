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

n_folds = 4
dataset_dirs = ['SiteA','SiteB','SiteC']

filename_list = [glob.glob(folder + '/*') for folder in dataset_dirs]

for category in filename_list:
    random.shuffle(category)

fold_len = max(len(category) for category in filename_list) // n_folds
# chunk2 = [filename_list[1][i*fold_len:(i+1)*fold_len] for i in range(n_folds)]

chunks = [[category[i*fold_len:(i+1)*fold_len] for category in filename_list]
            for i in range(n_folds)]

chunks = [[filepath for category in chunk for filepath in category]
            for chunk in chunks]

fold1_test = chunks[3:]
fold1_train = [x for x in chunks if x not in fold1_test]
# fold1_train = [filepath for filepath in chunk for chunk in fold1_train]


def unpack_list(list_in):
    unpacked = [x for x in sublist for sublist in list_in]
    return unpacked
