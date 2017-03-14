import numpy as np
import soundfile as sf
import librosa
import matplotlib.pyplot as plt


def find_nearest_index(array, value):
    diff = np.abs(array - value)
    idx = int(np.where(diff == np.min(diff))[0])

    return idx

# A more computationally efficient way to filter audio into mel bands than
# 4096 tap FIR filters. This seems to produce similar eventual values for azi,
# elev and psi, but slightly phase-shifted to the right. I'm not sure about
# this technique, it feels like a fudge.
def freq_domain_melfilt(n_bands, audio, fs):

    # calculate spectrogram for each channel of input audio
    multi_spectrogram = np.array([librosa.core.stft(channel)
                                    for channel in audio.T])

    # calculate FFT bin centre frequencies (default 2048-point FFT)
    FFT_freqs = librosa.core.fft_frequencies(44100)

    # calculate mel band centre frequencies
    mel_freqs = librosa.core.mel_frequencies(n_mels=n_bands+2, fmax=22050)

    # find nearest FFT bins to mel band frequencies
    mel_fft_indeces = [find_nearest_index(FFT_freqs, mel) for mel in mel_freqs]

    # create n_bands copies of the spectrogram and stack in a numpy array
    filter_bands = np.stack(([multi_spectrogram]*n_bands),3)

    # zero all frequecy bins outside designated mel bandwidth
    for i in range(n_bands):
         filter_bands[:,:mel_fft_indeces[i],:,i] = 0
         filter_bands[:,mel_fft_indeces[i+2]:,:,i] = 0

    filter_bands = np.rollaxis(filter_bands,3).swapaxes(0,1)
    filt_audio = np.array([[librosa.core.istft(band) for band in channel]
                            for channel in filter_bands])

    filt_audio = filt_audio.T.swapaxes(0,1)

    return filt_audio
