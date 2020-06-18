import numpy
import numpy as np
import cv2
from os import listdir
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import train_test_split
import joblib

filename = 'finalized_model_svm.sav'


label_names = {'Speed_limit_(60km_h)': 0, 'ahead_only': 1, 'Speed_limit_(80km_h)': 2, 'Speed_limit_(30km_h)': 3, '40_km_h': 4, 'traffic_signal': 5, 'Speed_limit_(70km_h)': 6, 'road_work': 7, 'turn_left_ahead': 8, 'Speed_limit_(20km_h)': 9, 'stop': 10, 'no_entry': 11, 'turn_right_ahead': 12, 'Speed_limit_(50km_h)': 13}
print(label_names)
y_labels = []
x_data = []

image = cv2.imread('../no_entry.png')
image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
image = (image/255)           
image = cv2.resize(image, (32, 32))
# y_labels.append(label_names[forder_name])
x_data.append(image)

x_data = np.array(x_data)
# y_labels = np.array(y_labels)

model_cnn = load_model('model_cnn_svm.h5')
model_cnn.load_weights('model_weights_cnn_svm.h5')
model_cnn.summary()

layer_name = 'flatten_1'

layer_dict = dict([(layer.name, layer) for layer in model_cnn.layers])

model_features = Model(inputs = model_cnn.inputs, outputs = layer_dict[layer_name].output)
# layer_dict = dict([(layer.name, layer) for layer in model_cnn.layers])
# model_feature = Model(inputs=model_cnn.inputs, outputs=layer_dict[layer_name].output)
model_features.summary()

x_feature = model_features.predict(x_data)
print(x_feature.shape)
# load the model from disk
loaded_model = joblib.load(filename)
pre = loaded_model.predict(x_feature)
print(pre)

# result = loaded_model.score(X_test, Y_test)
# print(result)