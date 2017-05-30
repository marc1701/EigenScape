import numpy as np
import soundfile as sf
import glob

filepaths = glob.glob('/media/mcg509/MARC1TB/Eigenmike Audio/Ambisonic/*')

output_dir = '/home/mcg509/Documents/Ambisonic'

for path in filepaths:

    audio, fs = sf.read(path)
    target_length = fs * 60 * 10 # number of samples to keep

    n_trim_samples = len(audio) - target_length
    trimmed_audio = audio[n_trim_samples:] # trim from beginning of recording

    label_start_idx = path.find('c/') + 2
    label_end_idx = path.find('-')

    label = path[label_start_idx:label_end_idx]
    recording_num = path[label_end_idx + 2]

    out_fname = label + '-' + recording_num + '.wav' # output filename

    out_path = output_dir + '/' + out_fname

    sf.write(out_path, trimmed_audio, fs, 'PCM_24')
