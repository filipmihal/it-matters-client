from ViolaJones import ViolaJones
from skimage.io import imread


VJ = ViolaJones.load('result.pkl')

print(VJ.alphas)

image = imread('dataset/test/face/cmu_0021.pgm' )
print(VJ.classify(image))

image2 = imread('dataset/test/non-face/cmu_0080.pgm' )
print(VJ.classify(image2))