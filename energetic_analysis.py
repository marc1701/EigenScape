# set constants to use in equations (these are approx. correct for air @ 20 C)
win_len = 20 # 20 ms time-avergaging window
t_samples = fs/1000 * win_len

Z_0 = 413.3 # characteristic acoustic impedance of air
rho_0 = 1.2041 # density of air
c = 343.21 # speed of sound

# filt_audio is indexed [ time : freq : channels ]
u = -(1 / (Z_0 * np.sqrt(2))) * filt_audio[:,10,1:].T
p = filt_audio[:,10,0] # W-channel of B-format audio

I = p * u # this will give an array with I at each sample point
I_ave = I[:,t:t2].mean(axis=1) # temporal averaging of I over t - t2 samples

E = 0.5 * rho_0 * ( (p**2 / Z_0**2) + (np.linalg.norm(u, axis=0)**2) )
# this calculates E for each sample point - works accross all

psi = 1 - (np.linalg.norm(I_ave) / (c * E)) # time averaging not added here
# filt_audio[f,t,1:].reshape(3,1) - this is how to get a vertical vector from
# B-format recordings including x, y and z. f = frequency, t = time

# it would probably be possible to use matrix multiplication to calculate all
# values for all frequencies - but I can't hold in my head how to do it and I
# think possibly doing it with a loop will make for clearer code anyways
