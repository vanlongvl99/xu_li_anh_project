import numpy as np 
import cv2
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
import keras    
import os
from os import listdir
from sklearn.model_selection import train_test_split
import datetime

path = "dataset"

# set label names from PC
cnt = 0
label_names = {}
for forder_name in listdir(path):
    label_names[forder_name] = cnt
    cnt += 1
print(label_names)




#dictionary to label all traffic signs class.
# label_names = {'Speed_limit_(60km_h)': 0, 'ahead_only': 1, 'Speed_limit_(80km_h)': 2, 'Speed_limit_(30km_h)': 3, '40_km_h': 4, 'traffic_signal': 5, 'Speed_limit_(70km_h)': 6, 'road_work': 7, 'turn_left_ahead': 8, 'Speed_limit_(20km_h)': 9, 'duong_1_chieu': 10, 'stop': 11, 'no_entry': 12, 'turn_right_ahead': 13, 'Speed_limit_(50km_h)': 14}

y_labels = []
x_data = []
for forder_name in listdir(path):
    for file_name in listdir(path + "/" + forder_name):
        # pre-processing data
        image = cv2.imread(path + "/" + forder_name +"/" + file_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
        image = (image/255)           
        image = cv2.resize(image, (32, 32))
        y_labels.append(label_names[forder_name])
        x_data.append(image)

# prepare data
x_data = np.array(x_data)
y_labels = np.array(y_labels)
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.2)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
print("============")
from sklearn.preprocessing import LabelBinarizer
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.fit_transform(y_test)


batch_size = 64
epochs = 8
num_classes = 14
print(datetime.datetime.now())
model = Sequential()
model.add(Conv2D(32, kernel_size = (3,3), activation = 'relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(num_classes, activation='softmax'))
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

model.save_weights("model_weights_cnn_svm.h5")
model.save("model_cnn_svm.h5")
print(datetime.datetime.now())

