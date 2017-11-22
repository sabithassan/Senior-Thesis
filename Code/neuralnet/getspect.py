import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys, os
from scipy import signal
from scipy.fftpack import fft, fftshift

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks

def graph_spectrogram(wav_file, spectofile):
    audio_pathdiff = wav_file.split('/')[-1][:-5]
    # print os.path.join(spectofile, audio_pathdiff+".png"), os.path.exists(os.path.join(spectofile, audio_pathdiff))
    if os.path.exists(os.path.join(spectofile, audio_pathdiff+".png")):
        print "Already DONE with:   ", audio_pathdiff
        return
    rate, data = get_wav_info(wav_file)
    # print shape(rate), shat
    fs = 1600
    f, t, Sxx = signal.spectrogram(data, fs=fs, window=signal.get_window("boxcar", fs*0.025),
                                    nperseg=int(fs*.025), noverlap=int(fs*.01), nfft=2048)
    plt.pcolormesh(t, f, Sxx)

    plt.axis('off');
    plt.savefig('tt.png',
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    return rate, data

def main(f):
    # print f
    s = f.split("/")
    s = s[:-1]
    directory = os.path.join('/'.join(s), 'spectrogram')
    print directory
    if not os.path.exists(directory):
        os.makedirs(directory)
    for audio_file in os.listdir(f):
        if audio_file[0] == "Y":
            # audio_file1 =  "_".join((audio_file[1:]).split("_")[:-2])
            audio_pathdiff = audio_file[:-5]

            graph_spectrogram(os.path.join(f, audio_file), directory)
            # print directory+'/'+audio_pathdiff+'.png'
            # plotstft(os.path.join(f, audio_file), plotpath=directory+'/'+audio_pathdiff+'.png')


main(sys.argv[1])
