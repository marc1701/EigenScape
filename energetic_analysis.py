# set constants to use in equations (these are approx. correct for air @ 20 C)
win_len = 20 # 20 ms time-avergaging window
t_samples = fs/1000 * win_len

Z_0 = 413.3 # characteristic acoustic impedance of air
rho_0 = 1.2041 # density of air
c = 343.21 # speed of sound

# filt_audio is indexed [ time : freq : channels ]
u = -(1 / (Z_0 * np.sqrt(2))) * filt_audio[:,10,1:].T # u = particle velocity
p = filt_audio[:,10,0] # W-channel of B-format audio = p = sound pressure

# I = instantaneous sound field intensity
I = p * u # this will give an array with I at each sample point

# E = instantaneous sound field energy
E = 0.5 * rho_0 * ( (p**2 / Z_0**2) + (np.linalg.norm(u, axis=0)**2) )
# this calculates E for each sample point - works accross all

# slice up I into 20 ms windows and calculate means
I_means = np.array([I_slice.mean(axis=1)
                        for I_slice in np.split(I, len(E)/t_samples, axis=1)])

# likewise for E
E_means = np.array([E_slice.mean()
                        for E_slice in np.split(E, len(E)/t_samples)])

# calculate diffuseness (psi) for each 20 ms slice
psi = np.array([1 - (np.linalg.norm(I_mean) / (c * E_mean))
                        for I_mean, E_mean in zip(I_means, E_means)])

DOA = - I_means

azi = np.array([np.arctan2(y,x) for x, y, z in DOA])
elev = np.array([np.arccos(z/np.linalg.norm([x,y,z])) for x, y, z in DOA])

azi = np.arctan(y/x)
elev = np.arccos(z) # z/r, but we presume vectors are normed so r = 1

# psi = np.zeros(len(E) / t_samples) # this is indexing over only 1 second - wrong but using as test presently
# for i in range(int(len(E) / t_samples)):
#     frame_start = i * t_samples
#     frame_end = (i+1) * t_samples
#
#     I_mean = I[:,frame_start:frame_end].mean(axis=1) # might need this independently
#     E_mean = E[frame_start:frame_end].mean()
#
#     psi[i] = 1 - (np.linalg.norm(I_mean) / (c * E_mean))

# it would probably be possible to use matrix multiplication to calculate all
# values for all frequencies - but I can't hold in my head how to do it and I
# think possibly doing it with a loop will make for clearer code anyways
