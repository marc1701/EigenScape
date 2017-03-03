import numpy as np
import soundfile as sf
import librosa

file_WX = '../Scarcroft Road B-Format/SiteA_111223_090_st12.wav'
file_YZ = '../Scarcroft Road B-Format/SiteA_111223_090_st34.wav'
target_fs = 44100
trim = True
output_prefix = 'SiteA'
# need to work out a way to give outputs an intelligent name
# probably just prefix_n.wav and user can specify starting n

audio_WX, fs = sf.read(file_WX)
audio_YZ = sf.read(file_YZ)[0]

if fs != target_fs:
    audio_WX = librosa.core.resample(audio_WX.T, fs, target_fs).T
    audio_YZ = librosa.core.resample(audio_YZ.T, fs, target_fs).T

audio_Bformat = np.hstack((audio_WX, audio_YZ))

if trim == True:
    # trim first and last 20 s from clip (avoids noise from recordists)
    # could make this value editable
    n_trim = target_fs * 20
    audio_Bformat = audio_Bformat[n_trim:-n_trim,:]

seg_samples = target_fs * 30
n_segments = np.floor(len(audio_Bformat) / seg_samples)
# 30 s clips - this could be a parameter

n_ex_samples = len(audio_Bformat) - seg_samples * n_segments
audio_Bformat = audio_Bformat[n_ex_samples:,:] # trim excess samples

segments = np.split(audio_Bformat, n_segments)

# sf.write('x.wav', x, target_fs, 'PCM_24')
