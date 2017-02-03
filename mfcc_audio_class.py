import numpy as np
import soundfile as sf
import scipy.fftpack as fft

class audio_sample:

    sample_count = 0
    # fs = 0
    # mel_bank = 0

    @classmethod
    def gen_mel_bank(cls, n_filterbanks=26, n_fft=1024, low_freq=200, high_freq=16000):
        # make n_fft accessible to gen_mfccs() instance method
        cls.n_fft = n_fft

        # calculate mel filterbank
        mel_lower = 1125 * np.log(1+low_freq/700)
        mel_upper = 1125 * np.log(1+high_freq/700) # upper and lower bounds mapped to mel scale
        filter_mel_centres = np.linspace(mel_lower, mel_upper, n_filterbanks+2) # n linearly spaced filters + 2 frequency points for calculation
        filter_freqs = 700*(np.exp(filter_mel_centres/1125)-1) # map mel frequencies back to standard scale
        filter_bins = np.floor((n_fft+1)*filter_freqs/cls.fs)

        cls.mel_bank = np.zeros([n_filterbanks, int((n_fft/2))]) # set up mel bank matrix (zeros)

        for m in range(n_filterbanks):
            for k in range(int(filter_bins[m]), int(filter_bins[m+1])): # between previous and present filter centre
                cls.mel_bank[m,k] = (k-filter_bins[m])/(filter_bins[m+1] - filter_bins[m])

            for k in range(int(filter_bins[m+1]), int(filter_bins[m+2])): # between present and next filter centre
                cls.mel_bank[m,k] = (filter_bins[m+2]-k)/(filter_bins[m+2] - filter_bins[m+1])


    def __init__(self, audio, label="unknown", n_mfccs=11): # not sure we need a label here
        self.sound, self.fs = sf.read(audio)
        self.label = label
        self.n_mfccs = n_mfccs
        # can we assign new labels to a new number here - class-wide dictionary
        # self.mfccs = gm.gen_mfccs(self.sound, self.fs)

        if audio_sample.sample_count == 0:
            audio_sample.fs = self.fs
            audio_sample.gen_mel_bank() # only calculate for the first instance - re-use later
        else:
            if self.fs != audio_sample.fs:
                raise Exception("Incoming sound has different fs than existing set.")
        #     audio_sample.mel_filters(self, n_filterbanks, n_mfccs, n_fft, low_freq, high_freq)

        self.gen_mfccs()

        audio_sample.sample_count += 1


    def gen_mfccs(self):
        # should be able to re-call this method to redo mfcc calculation later
        # I need to include provision for changing the n_fft also

        data = self.sound[:,0] # extract left channel (mono input)
        frame_length = int(np.floor(self.fs/1000) * 25) # calculate samples needed for 25s frame (20-40ms standard)
        step_length = int(np.floor(frame_length/2))
        n_frames = int(np.floor(len(data)/frame_length)) # calculate total number of frames in data

        # reallocate variables
        self.mfccs = np.zeros([n_frames, np.size(audio_sample.mel_bank,1)])
        filter_energies = np.zeros(np.size(audio_sample.mel_bank,1))

        # loop through audio
        for n in range(n_frames):
            frame = data[n*step_length:(n*step_length)+frame_length] # slice frame from full file
            frame = np.hanning(frame_length) * frame # apply hann window function to frame
            frame_fft = np.abs(np.real(fft.fft(frame,audio_sample.n_fft))) # frequency spectrum
            frame_fft = frame_fft[:int(len(frame_fft)/2)] # remove mirrored upper half

            for m in range(np.size(audio_sample.mel_bank,0)): # MFCC calculation loop
                filter_energies[m] = np.log(np.sum(frame_fft * audio_sample.mel_bank[m,:])) # sum energy in each bank

            self.mfccs[n,:] = fft.dct(filter_energies) # mfccs =  DCT of log energy
            # this should build up a matrix of MFCCs for the whole file

        self.mfccs = self.mfccs[:,:self.n_mfccs] # keep only specified number of mfccs
        # need an if statement here to deal with the situation where we want to keep all MFCCs
        # i.e. mfcc[0,:] should be kept
        del(self.sound) # actual audio no longer needed
