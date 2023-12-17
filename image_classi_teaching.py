import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from keras import datasets, layers, models

(training_img, training_lab), (testing_img, testing_lab) = datasets.cifar10.load_data()
training_img, testing_img = training_img / 255, testing_img / 255

class_names = ["Plane", "Car", "Bird", "Cat", "Deer", "Dog", "Frog", "Horse", "Ship", "Truck"]

# RESOURCE SAVING START
training_img = training_img[:20000]
training_lab = training_lab[:20000]

testing_img = testing_img[:4000]
testing_lab = testing_lab[:4000]
# RESOURCE SAVING END

model = models.load_model('image_classifier.model')
model.fit(training_img, training_lab, batch_size=20, epochs=30, validation_data=(testing_img, testing_lab))
loss, accuracy = model.evaluate(testing_img, testing_lab)
print(f"Loss: {loss}")
print(f"Accuracy: {accuracy}")

model.save('image_classifier.model')