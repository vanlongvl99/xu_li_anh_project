import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
from tkinter import *
from PIL import ImageTk, Image
import numpy
import numpy as np
import cv2
from os import listdir
import matplotlib
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import Model
import joblib



path = "dataset"
cnt = 0
label_names = {}
for forder_name in listdir(path):
    label_names[cnt] = forder_name
    cnt += 1

print(label_names)


#SVM
#load the trained model to classify sign
model_cnn_svm = load_model("./model_cnn_svm.h5")
model_cnn_svm.load_weights("./model_weights_cnn_svm.h5")
layer_name = 'flatten_1'
layer_dict = dict([(layer.name, layer) for layer in model_cnn_svm.layers])
model_feature = Model(inputs=model_cnn_svm.inputs, outputs=layer_dict[layer_name].output)
model_feature.summary()
filename = 'finalized_model_svm_1.sav'
loaded_model_SVM = joblib.load(filename)




#initialise GUI
top=tk.Tk()
top.geometry('800x600')
top.title('traffic sign Recognition')
top.configure(background='#CDCDCD')
label=Label(top,background='#CDCDCD', font=('arial',15,'bold'))
sign_image = Label(top)
def classify(file_path):
    global label_packed
    # print(file_path)
    #pre-processing data
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)   
    image = np.array(image)
    image = (image/255)     
    print(image.shape)        
    image = cv2.resize(image, (32, 32))
    image = [image]
    image = np.array(image)
    print(image.shape,"\n")
    #SVM
    x_feature = model_feature.predict(image)
    pre = loaded_model_SVM.predict(x_feature)[0]
    label.configure(foreground='#011638', text="Your traffic sign is " + label_names[pre]) 

def show_classify_button(file_path):
    classify_b=Button(top,text="Classify Image",command=lambda: classify(file_path),padx=10,pady=5)
    classify_b.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
    classify_b.place(relx=0.79,rely=0.46)
def upload_image():
    try:
        file_path=filedialog.askopenfilename()
        uploaded=Image.open(file_path)
        uploaded.thumbnail(((top.winfo_width()/2.25),(top.winfo_height()/2.25)))
        im=ImageTk.PhotoImage(uploaded)
        sign_image.configure(image=im)
        sign_image.image=im
        label.configure(text='')
        show_classify_button(file_path)
    except:
        pass

def visualize(label_names, prediction_1,k):
    global label_packed
    index_labels = np.argpartition(prediction_1, -k)[-k:]
    max_label_names = []
    proba_classes = prediction_1[index_labels]
    for i in range(len(index_labels)):
        max_label_names.append(label_names[index_labels[i]])
    # k max prediction
    x = np.arange(len(index_labels))  # the label locations
    width = 0.35  # the width of the bars
    fig, ax = plt.subplots()
    # rects1 = ax.bar(x - width/2, men_means, width, label='Men')
    print(proba_classes)
    
    rects2 = ax.bar(x + width/2, proba_classes, width, label='Your classes')
    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('probability')
    ax.set_title(str(k) + ' classes')
    ax.set_xticks(x)
    ax.set_xticklabels(max_label_names)
    ax.legend()
    autolabel(rects2,ax)
    fig.tight_layout()
    max_index = int(np.argmax(prediction_1))
    label.configure(foreground='#011638', text="Your traffic sign is " + label_names[max_index]) 
    plt.show()


def autolabel(rects,ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')

upload=Button(top,text="Upload an image",command=upload_image,padx=10,pady=5)
upload.configure(background='#364156', foreground='white',font=('arial',10,'bold'))
upload.pack(side=BOTTOM,pady=50)
sign_image.pack(side=BOTTOM,expand=True)
label.pack(side=BOTTOM,expand=True)
heading = Label(top, text="Know Your traffic sign",pady=20, font=('arial',20,'bold'))
heading.configure(background='#CDCDCD',foreground='#364156')
heading.pack()
plt.show()
top.mainloop()