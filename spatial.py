# A set of functions used to extract azimuth, elevation and diffuseness
# estimates from B-format audio recordings using techniques outlined in the
# DirAC system by researchers at the Helsinki University of Technology

import numpy as np
import soundfile as sf
import librosa
from scipy import signal

def multichannel_frame( audio_in, n_fft=2048, pad=False ):

    if pad == True:
        # optional padding for parity with the way librosa calculates MFCCs
        # n_fft = 2048 as standard in librosa - need to make this alterable
        # **** It's worth noting that this is essentially actually adding minute
        # extra amounts of time to the beginning and ending of the analysis -
        # since we're in the time domain this might be an issue ****
        audio_in = np.array([np.pad(channel, int(n_fft // 2), 'reflect')
                    for channel in audio_in])

    frames = np.array([librosa.util.frame(np.ascontiguousarray(channel))
                for channel in audio_in])

    frames = frames.swapaxes(0,2)

    return frames


def mel_spaced_filterbank( n_filts, low_freq, hi_freq, filt_taps, fs ):

    filter_band_freqs = librosa.core.mel_frequencies(n_filts+2,
                                                     low_freq, hi_freq)

    # move cutoff frequencies to avoid signal.firwin cutoff error
    if filter_band_freqs[0] == [0]:
        filter_band_freqs[0] = 1

    if filter_band_freqs[-1] >= fs / 2:
        filter_band_freqs[-1] -= 1

    filters = np.array([signal.firwin(filt_taps,[filter_band_freqs[i],
                        filter_band_freqs[i+2]], pass_zero=False,nyq=fs/2)
                        for i in range(n_filts)])

    return filters


def extract_spatial_features(audio, fs, low_freq=0, hi_freq=None, n_bands=20,
                                filt_taps=2048, Z_0=413.3, rho_0=1.2041,
                                c=343.21, ordering='ACN'):

    # constants to use in equations (these are approx. correct for air @ 20 C):
    # Z_0 = characteristic acoustic impedance of air
    # rho_0 = density of air
    # c = speed of sound

    if hi_freq == None:
        hi_freq = fs // 2

    filters = mel_spaced_filterbank(n_bands, low_freq, hi_freq, filt_taps, fs)

    filt_audio = np.array([signal.lfilter(filt, [1.0], audio, axis=0)
                           for filt in filters])
    # filt_audio is indexed [ freq : time : channels ]

    # multiband calculation of u (velocity):
    u = - filt_audio[:,:,1:] / (Z_0 * np.sqrt(2))

    # re-order ACN format to FuMa (YZX to XYZ)
    if ordering == 'ACN':
        u = np.roll(u,1)

    p = filt_audio[:,:,0] # p = sound pressure (W)

    # I = instantaneous sound field intensity
    I = p.T * u.T
    # calculate 3D matrix of I for each sample and frequency band

    # E = instantaneous sound field energy
    E = 0.5 * rho_0 * (p**2 / Z_0**2 + np.linalg.norm(u, axis=2)**2)
    # calculate 2D matrix of E for each sample and frequency band

    E_means = np.mean(multichannel_frame(E, pad=True), axis=1)
    # calculate mean value of E across time frames

    I_means = np.array([np.mean(multichannel_frame(freq_band.T, pad=True),
                        axis=1) for freq_band in I.T])
    # calculate mean value of I across time frames

    # calculate diffuseness (psi) for each frame
    psi = 1 - np.linalg.norm(I_means, axis=2).T / (c*E_means)
    # calculate psi across all frames and frequency bands
    # 1 = totally diffuse, 0 = totally directional

    DOA = - I_means
    # DOA estimation defined as opposite direction of intensity vector

    # astype(int) makes it easier to read these arrays for a human
    azi = np.degrees(np.arctan2(DOA[:,:,1], DOA[:,:,0])).astype(int).T
    # calculate azimuth angles for each frame and frequency band

    elev = np.degrees(np.arccos(DOA[:,:,2]
                      / np.linalg.norm(DOA, axis=2))).astype(int).T
    # calculate elevation angles for each frame and frequency band

    # azi and elev matrices have time frames on axis 0 and freq bands on axis 1

    return azi, elev, psi
