from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

import os
allImages = []
allTags = []
uniqueTags = []
files = os.listdir("/spectograms")
for file in files:
    print (file)
    img = load_img(file)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)
    print (x.shape)
    allImages.append(x)
    tag = file.split(".")
    tag = tag[0]
    tag = tag.split("@")
    tag = tag[0]
    if tag not in uniqueTags:
        uniqueTags.append(tag)
    allTags.append(tag)

allImages = np.array(allImages)
allTags = np.array(allTags)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Dense(len(uniqueTags))
model.add(Activation('sigmoid'))

model.fit(allImages, allTags,
          epochs=50,
          batch_size=batch_size,
          validation_data=(allImages, allTags))
