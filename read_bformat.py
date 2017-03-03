audio_W, fs = sf.read('../SiteA.W.wav')
audio_W = audio_W.reshape(-1,1)
audio_X = sf.read('../SiteA.X.wav')[0].reshape(-1,1)
audio_Y = sf.read('../SiteA.Y.wav')[0].reshape(-1,1)
audio_Z = sf.read('../SiteA.Z.wav')[0].reshape(-1,1)

audio = np.concatenate((audio_W, audio_X, audio_Y, audio_Z), axis=1)

audio = audio[:441000,:] # crop to first 10 seconds for speed (just for now)
