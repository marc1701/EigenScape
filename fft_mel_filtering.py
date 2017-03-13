import numpy as np
import soundfile as sf
import librosa


def find_nearest_index(array, value):
    diff = np.abs(array - value)
    idx = int(np.where(diff == np.min(diff))[0])

    return idx

n_bands = 42

test_audio, fs = sf.read('audio/Albion-.1.wav')
# W_channel = test_audio[:,0]
# W_spectrogram = librosa.core.stft(test_audio[:,0])
# X_spectrogram = librosa.core.stft(test_audio[:,1])
# Y_spectrogram = librosa.core.stft(test_audio[:,2])
# Z_spectrogram = librosa.core.stft(test_audio[:,3])
multi_spectrogram = np.array([librosa.core.stft(channel) for channel in test_audio.T])

# calculate FFT bin centre frequencies (default 2048-point FFT)
FFT_freqs = librosa.core.fft_frequencies(44100)

# calculate mel band centre frequencies
mel_freqs = librosa.core.mel_frequencies(n_mels=n_bands+2, fmax=22050)

# find nearest FFT bins to mel band frequencies
mel_fft_indeces = [find_nearest_index(FFT_freqs, mel) for mel in mel_freqs]

# find point where mel bins reach upper FFT bin and cut any repeats
cut_point = mel_fft_indeces.index(max(mel_fft_indeces))+1
# might not need this
mel_fft_indeces = mel_fft_indeces[:cut_point]

# W_spec_bands = np.dstack(([W_spectrogram]*n_bands))
# X_spec_bands = np.dstack(([X_spectrogram]*n_bands))
# Y_spec_bands = np.dstack(([Y_spectrogram]*n_bands))
# Z_spec_bands = np.dstack(([Z_spectrogram]*n_bands))
# # filter_bands = np.stack(([multi_spectrogram]*n_bands),3)

for i in range(n_bands):
     filter_bands[:,:mel_fft_indeces[i],:,i] = 0
     filter_bands[:,mel_fft_indeces[i+2]:,:,i] = 0

# for i in range(n_bands):
#     W_spec_bands[:mel_fft_indeces[i],:,i] = 0
#     W_spec_bands[mel_fft_indeces[i+2]:,:,i] = 0
#
#     X_spec_bands[:mel_fft_indeces[i],:,i] = 0
#     X_spec_bands[mel_fft_indeces[i+2]:,:,i] = 0
#
#     Y_spec_bands[:mel_fft_indeces[i],:,i] = 0
#     Y_spec_bands[mel_fft_indeces[i+2]:,:,i] = 0
#
#     Z_spec_bands[:mel_fft_indeces[i],:,i] = 0
#     Z_spec_bands[mel_fft_indeces[i+2]:,:,i] = 0 # [i+2]+1 for overlap
#     # need to look at this really

# W_filtered = np.array([librosa.core.istft(band)
#                         for band in np.rollaxis(W_spec_bands,2)])
# X_filtered = np.array([librosa.core.istft(band)
#                         for band in np.rollaxis(X_spec_bands,2)])
# Y_filtered = np.array([librosa.core.istft(band)
#                         for band in np.rollaxis(Y_spec_bands,2)])
# Z_filtered = np.array([librosa.core.istft(band)
#                         for band in np.rollaxis(Z_spec_bands,2)])

filter_bands = np.rollaxis(filter_bands,3).swapaxes(0,1)
filt_audio = np.array([[librosa.core.istft(band) for band in channel]
                        for channel in filter_bands])
# read in audio
# calculate spectrograms for each of the 4 channels
# work out mel frequencies and find nearest fft indeces
# remove end of nearest fft indeces list if/when highest bin starts repeating
# create n-1 mel frequencies copies of each channel's spectrogram
# for each mel band zero all frequency bands not within given mel band
# perform inverse fft on each spectrogram channel
# put the filtered time-domain audio into appropriate array (to match output of
# filters using the old method)
