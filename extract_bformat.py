import numpy as np
import soundfile as sf
import glob
import os
filepaths = glob.glob('/home/mcg509/Documents/Ambisonic (4th Order)/*')

output_dir = '/home/mcg509/Documents/Ambisonic (1st Order)/'

for path in filepaths:
    audio, fs = sf.read(path)

    first_order_audio = audio[:, :4]

    filename = os.path.basename(path)

    out_path = output_dir + filename
    sf.write(out_path, first_order_audio, fs, 'PCM_24')
