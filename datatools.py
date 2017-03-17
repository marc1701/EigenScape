import numpy as np
import soundfile as sf
import librosa
import os
import glob
import random

# Splits input B-Format audio and splits into a number of segments of specified
# segment length. Takes a list of files as input - these can be of any number of
# channels as long as there are four channels in total (e.g. 2 x stereo files,
# 4 x mono files, or a single B-Format file). Audio is resampled to a specified
# sample rate (44100 default) before segmentation. Trims a length of time from
# the start and end of input files to avoid sounds made by the sound recordist.
# This is 20 seconds by default, but can be disabled by setting segment_length
# to 0.
#
# In the future this could be expanded to segment n-channel audio (e.g. for use
# with eigenmike recordings).

def bformat_segment(file_list, output_prefix, segment_length=30,
                    target_fs=44100, trim_length=20, filenum_start=1):

    # Save output from sf.read as list.
    sf_out = [sf.read(sound) for sound in file_list]

    # Extract the audio arrays to a new list.
    audio_list = [sound[0] for sound in sf_out]
    # Deal with one-dimensional numpy arrays in case of monaural audio
    for i, element in enumerate(audio_list):
        if element.ndim < 2:
            audio_list[i] = element.reshape(-1,1)

    # Unpack audio from list to numpy array.
    audio = np.hstack([sound for sound in audio_list])
    # Unpack fs values from list.
    fs = [fs for (audio,fs) in sf_out]

    # Check we have 4 channels of audio (B-Format).
    if audio.shape[-1] != 4:
        raise Exception('Total number of audio channels is greater than 4.')

    # Check sample rates of incoming audio files match.
    if fs.count(fs[0]) != len(fs):
        raise Exception('Sample rates of input audio files do not match.')
    else:
        # Save fs as single value.
        fs = fs[0]

    if fs != target_fs:
        # Resample function works with stereo maximum.
        WX = librosa.core.resample(audio[:,:2].T, fs, target_fs).T
        YZ = librosa.core.resample(audio[:,2:].T, fs, target_fs).T

        # Reconstitute 4-channel array.
        audio = np.hstack((WX, YZ))

    # Trim time from start and end of recording (avoids noise from recordists).
    n_trim = int(target_fs * trim_length)
    if n_trim:
        audio = audio[n_trim:-n_trim,:]

    # Calculate the number of samples in a segment.
    seg_samples = target_fs * segment_length
    # Calculate how many segments there will be.
    n_segments = np.floor(len(audio) / seg_samples)

    # Calculate number of excess samples.
    n_ex_samples = int(len(audio) - seg_samples * n_segments)
    # Trim excess samples from start of audio.
    audio = audio[n_ex_samples:,:]

    # Split audio array into segments of specified length.
    segments = np.split(audio, n_segments)

    # Make a directory to store output files if this doesn't already exist.
    os.makedirs(output_prefix, exist_ok=True)

    # Save segmented audio files to directory.
    n_digits = len(str(len(segments)))
    for n, segment in enumerate(segments, start=filenum_start):
         filepath = (output_prefix + '/' + output_prefix + '.'
                     + str(n).zfill(n_digits) + '.wav')
         sf.write(filepath, segment, target_fs, 'PCM_24')

################################################################################
################################################################################

# neat function for unpacking nested lists
# (makes some of the later list comprehensions more comprehensible)
def unpack_list(x):
    unpacked = [i for y in x for i in y]
    return unpacked

def write_file_list(file_list, name_format, directory):
    # make a directory if it doesn't already exist
    os.makedirs(directory, exist_ok=True)

    for n, fold in enumerate(file_list):
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
            fold[n] = (filepath[filepath.find('/')+1:] + '\t'
                       + filepath[:filepath.find('-')] + '\n')

    # make list of folds to use as training for each fold used as test data
    train_folds = [unpack_list([x for x in test_folds if x not in [fold]])
                    for fold in test_folds]

    # alphabetise train_folds
    for fold in train_folds:
        fold.sort()

    # write results out to text files
    write_file_list(test_folds, 'fold*_test.txt', output_text_dir)
    write_file_list(train_folds, 'fold*_train.txt', output_text_dir)

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

        # write a text file listing the whole dataset
        allfiles_list = [os.path.basename(x)
                         for x in glob.glob(output_audio_dir + '/*')]

        for n, filepath in enumerate(allfiles_list):
            allfiles_list[n] = (filepath[filepath.find('/')+1:] + '\t'
                                + filepath[:filepath.find('-')] + '\n')

        write_file_list([allfiles_list], 'full_dataset.txt', output_audio_dir)
