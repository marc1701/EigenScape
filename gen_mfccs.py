def gen_mfccs( data, fs, n_filterbanks=26, n_mfccs=11, n_fft=1024, low_freq=200, high_freq=16000 ):
# remember to  ^     ^  change this back (just 'audio')
    # need some error handling here to deal with e.g. n_mfccs > n_filterbanks etc.

    import numpy as np
    import soundfile as sf
    import scipy.fftpack as fft

    # import data and set up loop variables
    # data, fs = sf.read(audio)
    data = data[:,0] # extract left channel (mono input)
    frame_length = int(np.floor(fs/1000) * 25) # calculate samples needed for 25s frame (20-40ms standard)
    step_length = int(np.floor(frame_length/2))
    n_frames = int(np.floor(len(data)/frame_length)) # calculate total number of frames in data 

    # calculate mel filterbank
    mel_lower = 1125 * np.log(1+low_freq/700)
    mel_upper = 1125 * np.log(1+high_freq/700) # upper and lower bounds mapped to mel scale
    filter_mel_centres = np.linspace(mel_lower, mel_upper, n_filterbanks+2) # n linearly spaced filters + 2 frequency points for calculation
    filter_freqs = 700*(np.exp(filter_mel_centres/1125)-1) # map mel frequencies back to standard scale 
    filter_bins = np.floor((n_fft+1)*filter_freqs/fs)

    mel_bank = np.zeros([int((n_fft/2)), n_filterbanks]) # set up mel bank matrix (zeros)

    for m in range(n_filterbanks):
        for k in range(int(filter_bins[m]), int(filter_bins[m+1])): # between previous and present filter centre
            mel_bank[k,m] = (k-filter_bins[m])/(filter_bins[m+1] - filter_bins[m])

        for k in range(int(filter_bins[m+1]), int(filter_bins[m+2])): # between present and next filter centre
            mel_bank[k,m] = (filter_bins[m+2]-k)/(filter_bins[m+2] - filter_bins[m+1])

    # initialise variables
    mfccs = np.zeros([n_filterbanks, n_frames])
    filter_energies = np.zeros(n_filterbanks)

    # loop through input audio
    for n in range(n_frames):
        frame = data[n*step_length:(n*step_length)+frame_length] # slice frame from full file
        frame = np.hanning(frame_length) * frame # apply hann window function to frame
        frame_fft = np.abs(np.real(fft.fft(frame,n_fft))) # frequency spectrum
        frame_fft = frame_fft[:int(len(frame_fft)/2)] # remove mirrored upper half

        for m in range(n_filterbanks): # MFCC calculation loop
            filter_energies[m] = np.log(np.sum(frame_fft * mel_bank[:,m])) # sum energy in each bank

        mfccs[:,n] = fft.dct(filter_energies) # mfccs =  DCT of log energy
        # this should build up a matrix of MFCCs for the whole file
        
    mfccs = mfccs[1:n_mfccs+1,:] # keep only specified number of mfccs
    # need an if statement here to deal with the situation where we want to keep all MFCCs
    # i.e. mfcc[0,:] should be kept    
    
    return mfccs