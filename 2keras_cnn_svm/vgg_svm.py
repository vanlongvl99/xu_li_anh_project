import numpy
import numpy as np
import cv2
from os import listdir
from keras.models import load_model
from keras.models import Model
from sklearn.model_selection import train_test_split
import joblib
from keras.applications.vgg16 import VGG16
 
#load label names
path = "../dataset"
cnt = 0
label_names = {}
for forder_name in listdir(path):
    label_names[forder_name] = cnt
    cnt += 1
print(label_names)


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
X_train, X_test, y_train, y_test = train_test_split(x_data, y_labels, test_size=0.1)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

model_vgg = VGG16()
model_vgg.summary()



layer_name = 'flatten'
layer_dict = dict([(layer.name, layer) for layer in model_vgg.layers])
model_feature = Model(inputs=model_vgg.inputs, outputs=layer_dict[layer_name].output)
model_feature.summary()
x_feature_train = model_feature.predict(X_train)
x_feature_test = model_feature.predict(X_test)
print(x_feature_train.shape)
# 
from sklearn import svm
# Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

# Train the model using the training sets
clf.fit(x_feature_train, y_train)
filename = 'model_linear_vgg_svm0.sav'
joblib.dump(clf, filename)
y_pred = clf.predict(x_feature_test[:100])
print(y_pred.shape)
from sklearn import metrics
# Model Accuracy: how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test[:100], y_pred))