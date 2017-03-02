# set constants to use in equations (these are approx. correct for air @ 20 C)
Z_0 = 413.3 # characteristic acoustic impedance of air
rho_0 = 1.2041 # density of air
c = 343.21 # speed of sound

# filt_audio is indexed [ time : freq : channels ]
u = -(1 / (Z_0 * np.sqrt(2))) * filt_audio[:,10,1:].T # u = particle velocity (XYZ)
p = filt_audio[:,10,0] # p = sound pressure (W)

# I = instantaneous sound field intensity
I = p * u # this will give an array with I at each sample point

# E = instantaneous sound field energy
E = 0.5 * rho_0 * ( (p**2 / Z_0**2) + (np.linalg.norm(u, axis=0)**2) )
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

E_means = np.array([E_frame.mean()
    for E_frame in librosa.util.frame(np.pad(E, int(2048 // 2), 'reflect')).T])

I_means = np.array([I_frame.mean(axis=0) for I_frame in multichannel_frame(I)])
# multichannel_frame applies padding automatically

# calculate diffuseness (psi) for each 20 ms slice
psi = np.array([1 - (np.linalg.norm(I_mean) / (c * E_mean))
                        for I_mean, E_mean in zip(I_means, E_means)])

DOA = - I_means

azi = np.array([np.arctan2(y,x) for x, y, z in DOA])
elev = np.array([np.arccos(z/np.linalg.norm([x,y,z])) for x, y, z in DOA])


# it would probably be possible to use matrix multiplication to calculate all
# values for all frequencies - but I can't hold in my head how to do it and I
# think possibly doing it with a loop will make for clearer code anyways
