import glob
import cv2 as cv
import numpy as np
import random
import os
from keras.layers import Activation,Dropout,Conv2D,GlobalAveragePooling2D
from keras.models import Sequential
from keras.applications import MobileNetV2,VGG16
from keras.optimizers import SGD,Adam
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

PATH='C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/Data'

data=[]
train_X=[]
train_Y=[]

classes=['empty','rock','paper','scissors']

for sign in classes:

    paths=glob.glob(os.path.join(PATH,sign,'*.jpg'))
    print(f'[INFO] Converting {sign} images to list along with the target values....')
    for path in paths:
        t=0
        img = cv.imread(path)
        target=path.split('/')[-1][5:13]
        if 'paper' in target:
            t=0
        elif 'rock' in target:
            t=1
        elif 'scissors' in target:
            t=2
        else:
            t=3
        print(target)
        data.append((img,t))
print('[INFO] Training data acquired with the target values...... ')

print('[INFO] Shuffling the data.....')
random.shuffle(data)


print('[INFO] Extracting out the target values.....')
for case in data:
    (a,b)=case
    train_X.append(a)
    train_Y.append(b)

train_X = np.array(train_X, dtype="float") / 255.0
train_Y=np_utils.to_categorical(train_Y,4)
(trainX, testX, trainY, testY) = train_test_split(train_X, train_Y,test_size=0.10, random_state=42)

aug=ImageDataGenerator(rotation_range=20,zoom_range=0.15,width_shift_range=0.2,height_shift_range=0.2,shear_range=0.15,
                       horizontal_flip=True,fill_mode="nearest")
opt = Adam(3e-4)

checkpoint = ModelCheckpoint("C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/best_model.hdf5", monitor='loss', verbose=1,
    save_best_only=True, mode='auto', period=1)

model=Sequential([
    VGG16(weights='imagenet',include_top=False,input_shape=(128,128,3),classes=4),
    Dropout(0.5),
    Conv2D(4,(1,1),padding='valid'),
    Activation('relu'),
    GlobalAveragePooling2D(),
    Activation('softmax')
])
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
print("[INFO] training network...")
H = model.fit(
	x=aug.flow(trainX, trainY, batch_size=32),
	validation_data=(testX, testY),
	steps_per_epoch=len(trainX) // 32,
    callbacks=[checkpoint],
	epochs=10)
print('[INFO]Saving model......')
model.save(f'C:/Enlighten here/Essentials/Machine Learning/MyProjects/Kaggle-Paper Rock Scissors/MobileNetV2.hdf5')