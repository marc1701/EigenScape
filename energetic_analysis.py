# set constants to use in equations (these are approx. correct for air @ 20 C)
Z_0 = 413.3 # characteristic acoustic impedance of air
rho_0 = 1.2041 # density of air
c = 343.21 # speed of sound

# filt_audio is indexed [ time : freq : channels ]
# u = -(1 / (Z_0 * np.sqrt(2))) * filt_audio[10,:,1:].T # u = particle velocity (XYZ)

# multiband calculation of u:
# u = np.array([-(1 / (Z_0 * np.sqrt(2))) * band for band in filt_audio[:,:,1:]])
# u = -(1 / (Z_0 * np.sqrt(2))) * filt_audio[:,:,1:]
u = - filt_audio[:,:,1:] / (Z_0 * np.sqrt(2))

p = filt_audio[:,:,0] # p = sound pressure (W)

# I = instantaneous sound field intensity
# I = p * u # this will give an array with I at each sample point
# I = np.array([p.T * channel for channel in u.T]) # multiband I
I = p.T * u.T

# E = instantaneous sound field energy
# E = 0.5 * rho_0 * ( (p**2 / Z_0**2) + (np.linalg.norm(u, axis=0)**2) )

# multiband E
E = 0.5 * rho_0 * ( (p**2 / Z_0**2) + (np.linalg.norm(u, axis=2)**2) )

# this calculates E for each sample point - works accross all


# for parity with the way librosa calculates MFCCs:
# librosa.util.frame(np.pad(E, int(n_fft // 2), 'reflect')
# n_fft = 2048 by default - will have to incorporate a way to make this track
# whatever values happen to be being used for the MFCC calculation
# **** It's worth noting that this is essentially actually adding minute
# extra amounts of time to the beginning and ending of the analysis - since
# we're in the time domain this might be an issue ****

# old way of calculating I_means - this has no window overlap and no parity
# with MFCC calculation
# # slice up I into 20 ms windows and calculate means
# I_means = np.array([I_slice.mean(axis=1)
#                         for I_slice in np.split(I, len(E)/t_samples, axis=1)])
#
# # likewise for E
# E_means = np.array([E_slice.mean()
#                         for E_slice in np.split(E, len(E)/t_samples)])

# E_means = np.array([E_frame.mean()
#     for E_frame in librosa.util.frame(np.pad(E, int(2048 // 2), 'reflect')).T])

# E_means = np.mean(librosa.util.frame(np.pad(E, 1024, 'reflect')), axis=0) # better single-channel
# np.pad(E, ((0,0),(1024,1024)), 'reflect')

E_means = np.mean(multichannel_frame(E, pad=True), axis=1) # multiband E


# I_means = np.array([I_frame.mean(axis=0) for I_frame in multichannel_frame(I, pad=True)])
# I_means = np.mean(multichannel_frame(I, pad=True), axis=1) # better single band I
# multichannel_frame applies padding automatically

I_means = np.array([np.mean(multichannel_frame(band.T, pad=True), axis=1) for band in I.T])
# multiband I

# calculate diffuseness (psi) for each 20 ms slice
# psi = np.array([1 - (np.linalg.norm(I_mean) / (c * E_mean))
#                         for I_mean, E_mean in zip(I_means, E_means)])
# psi = 1 - np.linalg.norm(I_means,axis=1) / (c*E_means) # better single band psi

psi = 1 - np.linalg.norm(I_means, axis=2).T / (c*E_means) # multiband psi

DOA = - I_means

# astype(int) makes it easier to read off these arrays for a human
# not sure if including extra decimal precision will be helful for ML
# I think not but probably worth trying both ways
# azi = np.degrees(np.array([np.arctan2(y,x) for x, y, z in DOA])).astype(int)
azi = np.degrees(np.array([np.array([np.arctan2(y,x)
                    for x, y, z in DOA_band])
                        for DOA_band in DOA])).astype(int).T

# elev = np.degrees(np.array([np.arccos(z/np.linalg.norm([x,y,z])) for x, y, z in DOA])).astype(int)
elev = np.degrees(np.array([np.array([np.arccos(z/np.linalg.norm([x,y,z]))
                    for x, y, z in DOA_band])
                        for DOA_band in DOA])).astype(int).T


# it would probably be possible to use matrix multiplication to calculate all
# values for all frequencies - but I can't hold in my head how to do it and I
# think possibly doing it with a loop will make for clearer code anyways
