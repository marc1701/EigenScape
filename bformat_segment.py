import numpy as np
import soundfile as sf
import librosa
import os

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
