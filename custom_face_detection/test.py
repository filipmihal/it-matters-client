import os
from skimage.io import imread
from ViolaJones import ViolaJones
import numpy as np
import pickle

images = []
labels = np.concatenate((np.ones(400), np.zeros(1500)), axis=None)

def create_pkl():
    for index, filename in enumerate(os.listdir('dataset/test/face')):
        if index < 400:
            image = imread('dataset/test/face/' + filename)
            images.append(image)

    for index, filename in enumerate(os.listdir('dataset/test/non-face')):
        if index < 1500:
            image = imread('dataset/test/non-face/' + filename)
            images.append(image)
    
    with open('dataset/test_data.pkl','wb') as f:
        pickle.dump(images, f)

# create_pkl()

with open('dataset/test_data.pkl','rb') as f:
     images = pickle.load(f)

VJ = ViolaJones.load('result_6000.pkl')
VJ.test_model(images, labels)
