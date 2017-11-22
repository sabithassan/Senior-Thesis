from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
#from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import numpy as np
import paramiko
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sys, os
from scipy import signal
from scipy.fftpack import fft, fftshift
np.random.seed(123)  # for reproducibility
 
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils


from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


USERNAME = 'sabith' # Put your AndrewID
PASSWORD = '@Summerof69' # Put you Andrew Passoword
MACHINE_NAME = 'srazak-01.qatar.cmu.local' # Name of the server being accessed
DIRECTORY_NAME = '/home/sshaar' # Main directory where the .wav files stored

# This is the arrary that will store roughly the 10,000 spectograms. Therefore,
#   when training the CNN, you can access the spectograms from here.
SPECTOGRAMS = []

## this contains all labels
LABELS = []
UNIQUETAGS = []
# This contains all the different folders names in the DIRECOTRY_NAME.
FILE_NAMES = []

# The number of folders taken into accunt in one go at a time. Therefore,
#   len(SPECTOGRAMS) <= 2,000 * STEPS
STEPS = 5

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
#print (FILE_NAMES)
count = 0
for x in FILE_NAMES:
    if ("xaa" in x):
        count+=1
#print (count)
FILE_NAMES = FILE_NAMES[26:]
#print(FILE_NAMES)

# Call this function everytime you wanna read STEPS new folders from the server.
#   Therefore, when training the CNN call this every time you want to update the
#   Spectograms.
def get_step_spectograms():
    global CURRENT_INDEX
    global SPECTOGRAMS, LABELS
    

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

            Sxx = np.resize(Sxx, (1025, 600))
            #Sxx = Sxx.reshape((1,) + Sxx.shape)
            print ("SPECTO SHAPE", Sxx.shape)
            # Add the spectogram of the audiofile to the list.
            SPECTOGRAMS.append(Sxx) # I am not sure anout this!
            #print (SPECTOGRAMS)
            print ("file", wav_filename)
            t = wav_filename.split(".wav")
            t = t[0]
            t = t.split("@")[1]
            if (t not in UNIQUETAGS):
                UNIQUETAGS.append(t)
            #print (t)
            LABELS.append(t)
            #print ("TAG", t)
            #print ("SPECTO SHAPE", Sxx.shape)
            #print( "ALL SHAPE", SPECTOGRAMS.shape)
            wave_file.close()

            count += 1
            print (count)
            print (len(SPECTOGRAMS))

            if (count == 50):
                break
            

        print ('Finished reading all .wav files in ' + FILE_NAMES[i][:4])

        client.close()
    CURRENT_INDEX += STEPS

def oneHot(data):
    values = array(data)
    print(values)
    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    print(integer_encoded)
    # binary encode
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    #print(onehot_encoded)
    # invert first example
    inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
    return onehot_encoded


#print (SPECTOGRAMS)
CURRENT_INDEX = 0
get_step_spectograms()
#print (SPECTOGRAMS)
n = len(SPECTOGRAMS)
print ("TOTAL POINTS", n)
##SPECTTEST = SPECTOGRAMS[int((n*3)/4):]
##LABELTEST = LABELS[int((n*3)/4):]
##SPECTOGRAMS = SPECTOGRAMS[:int((n*3)/4)]
##LABELS = LABELS[:int((n*3)/4)]

SPECTTEST = SPECTOGRAMS
LABELTEST = LABELS

SPECTOGRAMS = np.array(SPECTOGRAMS)
SPECTTEST = np.array(SPECTTEST)
LABELS = np.array(LABELS)
LABELTEST = np.array(LABELTEST)


SPECTOGRAMS = SPECTOGRAMS.reshape(SPECTOGRAMS.shape[0], 1, 1025, 600)
SPECTTEST = SPECTTEST.reshape(SPECTTEST.shape[0], 1, 1025, 600)

print ("FINAL SPEC SHAPE", SPECTOGRAMS.shape)
print ("FINAL TEST SPEC SHAPE", SPECTTEST.shape)

classes = len(UNIQUETAGS)
print ("POSSIBLE CLASSES:", classes, UNIQUETAGS)
LABELS = oneHot(LABELS)
LABELTEST = oneHot(LABELTEST)


print ("FINAL LABEL SHAPE", LABELS.shape)
print ("FINAL TEST LABEL SHAPE", LABELTEST.shape)



#print ("SPECTO SHAPE", SPECTOGRAMS.shape)
#print ("LABEL SHAPE", LABELS.shape)

model = Sequential()
 
model.add(Convolution2D(32, kernel_size=(4, 4), strides=(1, 1), data_format='channels_first', activation='relu', input_shape=(1,1025,600)))
model.add(Convolution2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
 
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(classes, activation='softmax'))

# 8. Compile model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
 
# 9. Fit model on training data
model.fit(SPECTOGRAMS, LABELS, 
          batch_size=2, epochs = 5, verbose=1)
 
# 10. Evaluate model on test data
score = model.evaluate(SPECTTEST, LABELTEST, verbose=0)
print ("SCORE :", score)
#print (SPECTOGRAMS)

#get_step_spectograms()
#print (SPECTOGRAMS)
