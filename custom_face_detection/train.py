import os
from skimage.io import imread
from ViolaJones import ViolaJones
import numpy as np

images = []
labels = np.concatenate((np.ones(100), np.zeros(100)), axis=None)

for index, filename in enumerate(os.listdir('dataset/train/face')):
    if index < 100:
        image = imread('dataset/train/face/' + filename)
        images.append(image)

for index, filename in enumerate(os.listdir('dataset/train/non-face')):
    if index < 100:
        image = imread('dataset/train/non-face/' + filename)
        images.append(image)

VJ = ViolaJones()
VJ.train(images,labels,100,100)

VJ.save('result')