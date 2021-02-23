import os
from skimage.io import imread
from ViolaJones import ViolaJones
import numpy as np
import pickle

images = []
labels = np.concatenate((np.ones(2000), np.zeros(4000)), axis=None)

def create_pkl():
    for index, filename in enumerate(os.listdir('dataset/train/face')):
        if index < 2000:
            image = imread('dataset/train/face/' + filename)
            images.append(image)

    for index, filename in enumerate(os.listdir('dataset/train/non-face')):
        if index < 4000:
            image = imread('dataset/train/non-face/' + filename)
            images.append(image)

    with open('dataset/train_data.pkl','wb') as f:
        pickle.dump(images, f)

# create_pkl()

with open('dataset/train_data.pkl','rb') as f:
     images = pickle.load(f)

VJ = ViolaJones(30)
VJ.train(images,labels,2000,4000)

print(VJ.alphas)

image = imread('dataset/test/face/cmu_0021.pgm' )
print(VJ.classify(image))

image2 = imread('dataset/test/non-face/cmu_0080.pgm' )
print(VJ.classify(image2))

VJ.save('result_6000')
