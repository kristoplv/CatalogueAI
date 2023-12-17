import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models

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


# RESOURCE SAVING START
training_img = training_img[:20000]
training_lab = training_lab[:20000]

testing_img = testing_img[:4000]
testing_lab = testing_lab[:4000]
# RESOURCE SAVING END


model = models.Sequential()
model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3))) ## 32 - neuronite arv kihis; (3,3) - kihi kontroll; relu - fn; input_shape - Sisendi kuju (32x32 pikslit ja 3 värvikaarti)
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu')) ## rectified linear unit, ehk kõik negatiivsed väärtused on 0-id ja positiivsed väärtused on kuni 1
model.add(layers.Flatten()) ## Väärtused üheks pikaks vektoriks(3x32x32 pikk)
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax')) ## väärtused viskame protsentideks kuni 100% (ehk väärtused 0 ja 1 vahel kuni 1ni); Samuti viimane tulemuste kiht

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(training_img, training_lab, epochs=30, validation_data=(testing_img, testing_lab)) ## Treenimiste väärtused sisse; epochs - Mitu korda vaatab närvivõrk samat andmehulka

#model.fit(training_img, training_lab, epochs=10, validation_data=(testing_img, testing_lab))

loss, accuracy = model.evaluate(testing_img, testing_lab)


print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier_720p.model')
#model = models.load_module()






