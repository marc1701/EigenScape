import glob
import random
import os

# neat function for unpacking nested lists
# (makes some of the later list comprehensions more comprehensible)
def unpack_list(x):
    unpacked = [i for y in x for i in y]
    return unpacked

def write_fold_list(fold_list, name_format, directory):
    # make a directory if it doesn't already exist
    os.makedirs(directory, exist_ok=True)

    for n, fold in enumerate(fold_list):
        filename = '/' + name_format.replace('*', str(n+1))
        txtfile = open(directory + filename,'w')
        txtfile.writelines(fold)
        txtfile.close()

# dataset_dirs are expected to include a class name followed by a dash and a
# unique recording identifier (e.g. SiteA-001).
def segment_dataset(n_folds, dataset_dirs, output_text_dir,
                        output_audio_dir=None):
# n_folds = 4
# dataset_dirs = ['SiteA-24-11-16','SiteB-24-11-16','SiteC-24-11-16']
# output_audio_dir = 'audio'
# output_text_dir = 'evaluation_setup'

# retrieve list of files
    filename_list = [glob.glob(folder + '/*') for folder in dataset_dirs]

    # within-class shuffle of audio samples
    for category in filename_list:
        random.shuffle(category)

    # calculate number of samples per fold based on length of largest class
    fold_len = max(len(category) for category in filename_list) // n_folds

    # take slices of each class from filename_list
    chunks = [[category[i*fold_len:(i+1)*fold_len]
                for category in filename_list]
                for i in range(n_folds)]

    # unpack each chunk to make list of folds
    test_folds = [unpack_list(x) for x in chunks]

    # Reformat filename strings to feature filename followed by class label
    # after a tab. Class label is taken from directory name (before'-').
    for fold in test_folds:
        for n, filepath in enumerate(fold):
            fold[n] = filepath[filepath.find('/')+1:] + '\t' + filepath[
                        :filepath.find('-')] + '\n'

    # make list of folds to use as training for each fold used as test data
    train_folds = [unpack_list([x for x in test_folds if x not in [fold]])
                    for fold in test_folds]

    # alphabetise train_folds
    for fold in train_folds:
        fold.sort()

    # write results out to text files
    write_fold_list(test_folds, 'fold*_test.txt', output_text_dir)
    write_fold_list(train_folds, 'fold*_train.txt', output_text_dir)

    if output_audio_dir:
        # make a folder to move audio database into
        os.makedirs(output_audio_dir, exist_ok=True)

        # move audio to database folder
        for filepath in unpack_list(filename_list):
            os.rename(filepath, output_audio_dir + '/' + filepath[
                        filepath.find('/')+1:])

        # delete old audio folders
        for directory in dataset_dirs:
            os.rmdir(directory)
