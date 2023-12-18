import cv2 
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, models
import os
from PIL import Image
import time

(training_img, training_lab), (testing_img, testing_lab) = datasets.cifar10.load_data()
training_img, testing_img = training_img / 255, testing_img / 255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

for i in range(16):
    plt.subplot(4,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_img[i], cmap = plt.cm.binary)
    plt.xlabel(class_names[training_lab[i][0]])

#plt.show()


model = models.load_model('image_classifier.model')
handle = "testing_data_2"
while True:
    files = os.listdir(handle)
    for name in files:
        try:
            if ".jpg" in name or ".mp4" in name or ".gif" in name or ".png" in name or ".jpeg" in name or ".svg" in name:
                print(name)
                img_start = Image.open(handle+"/"+name)
                img_32 = img_start.resize((32,32))
                img_32.save("resized/32_"+name)
                img = cv2.imread('resized/32_'+name)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt.imshow(img, cmap=plt.cm.binary)
                prediction = model.predict(np.array([img]) / 255)
                pred_name = class_names[np.argmax(prediction)]
                holder = []
                folders = os.listdir("testing_data_2")
                for value in prediction:
                    for val in value:
                        if round(val, 5) != 0.0:
                            holder.append(round(val, 5))
                if len(holder) > 3 and max(holder) < 0.99:
                    if "pole_kindel" not in folders:
                        os.mkdir("testing_data_2/pole_kindel")
                    img_start.save("testing_data_2/pole_kindel/"+name)
                else:
                    if pred_name not in folders:
                        os.mkdir("testing_data_2/"+pred_name)
                    img_start.save("testing_data_2/"+pred_name+"/"+name)
                os.remove("resized/32_"+name)
                os.remove("testing_data_2/"+name)
                print(f"Prediction is {pred_name}")
            else:
                continue
        except OSError as e:
            print(" >>> Midagi läks valesti!")
    time.sleep(5)
