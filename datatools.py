# datatools 0.2
# much better adapted to the file system used for the eigenScape recordings

import numpy as np
import soundfile as sf
import progressbar as pb
import glob
import os
import shutil
import random

def create_test_setup(dataset_dir, seg_length=30, **kwargs):
    # automates use of functions below
    # audio present in dataset_dir is split into segments of seg_length.
    # seg_length is set in seconds and length of full files must be exactly
    # divisible by this value. segments are then split into n_folds for training
    # and testing, making sure that segments from the same longer recording do
    # not cross over between sets

    split_audio_set(dataset_dir, seg_length)
    make_folds(**kwargs)


def split_audio_set(dataset_dir, seg_length):

    file_list = glob.glob(dataset_dir + '/*')

    progbar = pb.ProgressBar(max_value=len(file_list))
    progbar.start()

    for n, path in enumerate(file_list):
        progbar.update(n)

        audio, fs = sf.read(path)

        seg_samples = seg_length * fs # calculate samples in each segment
        n_segs = len(audio) // seg_samples # calculate total number of segments

        segs = np.split(audio, n_segs) # split audio into n equal chunks

        n_digits = len(str(len(segs))) # n_digits for use in filename
        filename = os.path.splitext(os.path.basename(path))[0] # strip extension

        # make hidden output directories
        output_dir = '.chop/' + filename[filename.find('.')+1]
        os.makedirs(output_dir, exist_ok=True)

        for n, seg in enumerate(segs, start=1):
            filepath = (output_dir + '/' + filename + '.'
                        + str(n).zfill(n_digits) + '.wav')

            sf.write(filepath, seg, fs, 'PCM_24')

    progbar.finish()

###############################################################################


def make_folds(n_folds=4, output_text_dir='fold_info',
                output_audio_dir='audio'):

    numbers = np.array([i for i in range(8)])
    random.shuffle(numbers) # generate random number sequence (1-8)

    test_folds = np.split(numbers, n_folds)
    train_folds = [[x for x in test_folds if all(x != y)] for y in test_folds]

    filepath_list = [glob.glob(folder + '/*')
                     for folder in glob.glob('.chop/*')]
    filename_list = [[os.path.basename(path) for path in fold]
                     for fold in filepath_list]
    filenames_to_write = [[os.path.basename(path) + '\n' for path in fold]
                     for fold in filepath_list]


    # make lists of files from imported list
    test_files = [unpack_list([filenames_to_write[n] for n in m])
                    for m in test_folds]
    train_files = [unpack_list([x for x in test_files if x not in [files]])
                    for files in test_files]

    # write results out to text files
    write_file_list(test_files, 'fold*_test.txt', output_text_dir)
    write_file_list(train_files, 'fold*_train.txt', output_text_dir)

    if output_audio_dir:
        # make a folder to move audio database into
        os.makedirs(output_audio_dir, exist_ok=True)

        # move audio to database folder
        for n, filepath in enumerate(unpack_list(filepath_list)):
            os.rename(filepath, output_audio_dir + '/'
                        + unpack_list(filename_list)[n])

        shutil.rmtree('.chop')


def write_file_list(file_list, name_format, directory):
    # make a directory if it doesn't already exist
    os.makedirs(directory, exist_ok=True)

    for n, fold in enumerate(file_list):
        filename = '/' + name_format.replace('*', str(n+1))
        txtfile = open(directory + filename,'w')
        txtfile.writelines(fold)
        txtfile.close()


# neat function for unpacking nested lists
# (makes some of the later list comprehensions more comprehensible)
def unpack_list(x):
    unpacked = [i for y in x for i in y]
    return unpacked
