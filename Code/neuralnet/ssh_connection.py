import paramiko
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys, os
from scipy import signal
from scipy.fftpack import fft, fftshift

USERNAME = 'sabith' # Put your AndrewID
PASSWORD = '@Summerof69' # Put you Andrew Passoword
MACHINE_NAME = 'srazak-01.qatar.cmu.local' # Name of the server being accessed
DIRECTORY_NAME = '/home/sshaar' # Main directory where the .wav files stored

# This is the arrary that will store roughly the 10,000 spectograms. Therefore,
#   when training the CNN, you can access the spectograms from here.
SPECTOGRAMS = []

## this contains all labels
LABELS = []

# This contains all the different folders names in the DIRECOTRY_NAME.
FILE_NAMES = []

# The number of folders taken into accunt in one go at a time. Therefore,
#   len(SPECTOGRAMS) <= 2,000 * STEPS
STEPS = 1

# This is to indicate which file we are at currently. This moves by STEPS.
CURRENT_INDEX = 0

# First I need to fill the FILE_NAMES. These commands are done to fill the FILE_NAMES.
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect(MACHINE_NAME, username=USERNAME, password=PASSWORD)
cmd = 'ls ' + DIRECTORY_NAME + ' | grep x'
stdin, stdout, stderr = client.exec_command(cmd)
FILE_NAMES = (stdout.read().decode('utf-8')).split("\n")[:-1]
client.close()
print (FILE_NAMES)
count = 0
for x in FILE_NAMES:
    if ("xaa" in x):
        count+=1
print (count)
FILE_NAMES = FILE_NAMES[26:]
print(FILE_NAMES)

# Call this function everytime you wanna read STEPS new folders from the server.
#   Therefore, when training the CNN call this every time you want to update the
#   Spectograms.
def get_step_spectograms():
    global CURRENT_INDEX


    # CLear the RAM.
    SPECTOGRAMS = []

    n = len(FILE_NAMES)
    # Keep openning the connection to make sure it is not lost.
    # Keep openning the connection to make sure it is not lost.
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(MACHINE_NAME, username=USERNAME, password=PASSWORD)
    sftp = client.open_sftp()
    # Read .wav files from STEPS folders form teh DIRECTORY_NAME.
    for i in range(CURRENT_INDEX, CURRENT_INDEX+STEPS):

        # Make sure that you dont go beyong the number of folders found in the
        #   DIRECOTRY_NAME.
        if (i >= n):
            break

        

        # The command is listing all the .wav files on the folder FILE_NAMES[i]
        cmd = ('ls ' + DIRECTORY_NAME + '/' + FILE_NAMES[i] + '/' + FILE_NAMES[i][:4] +
                '_audio_formatted_and_segmented_downloads/')
        stdin, stdout, stderr = client.exec_command(cmd)
        wav_files_in_directory = (stdout.read().decode('utf-8')).split("\n")[:-1]

        count = 0
        check = 10
        for wav_filename in wav_files_in_directory:
            # print count

            if (count == check):
                print ("Finished reading", count, ".wav files from", FILE_NAMES[i][:4])
                check += 10
                #count = 0

            

            # Read the audiofile
            wave_file = sftp.open('/home/sshaar/' + FILE_NAMES[i]+'/' + FILE_NAMES[i][:4] +
                    '_audio_formatted_and_segmented_downloads/' + wav_filename, 'r')

            rate, data = wavfile.read(wave_file)

            # Specification for the spectorgram. You may change this function if wanted.
            #   I think we need to modify this more.
##            if ("xaa" in FILE_NAMES[i]):
##                fs = 1600
##            else:
            fs = 16000
            #fs = 1600
            f, t, Sxx = signal.spectrogram(data, fs=fs, window=signal.get_window("boxcar", int(fs*0.025)),
                                            nperseg=int(fs*.025), noverlap=int(fs*.01), nfft=2048)

            # Add the spectogram of the audiofile to the list.
            SPECTOGRAMS.append(Sxx) # I am not sure anout this!
            #print (SPECTOGRAMS)
            print ("file", wav_filename)
            t = wav_filename.split("wav")
            t = t[0]
            t = t.split("@")[1]
            #print (t)
            LABELS.append(t)
            print ("TAG", t)
            print ("SPECTO SHAPE", Sxx.shape)
            #print( "ALL SHAPE", SPECTOGRAMS.shape)
            wave_file.close()

            count += 1
            print (count)

        print ('Finished reading all .wav files in ' + FILE_NAMES[i][:4])

        client.close()
    CURRENT_INDEX += STEPS

#print (SPECTOGRAMS)
CURRENT_INDEX = 0
get_step_spectograms()
#print (SPECTOGRAMS)

#get_step_spectograms()
#print (SPECTOGRAMS)
