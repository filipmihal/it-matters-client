import numpy as np

class Integral:
    def __init__(self, image):
        self.image = self.get_integral_image(image) 
    
    def get_integral_image(self, image):
        integral_image = np.zeros(image.shape)
        for y in range(len(image)):
            for x in range(len(image[y])):
                integral_image[y][x] = image[y][x]
                if x != 0:
                    integral_image[y][x] += integral_image[y][x-1]
                if y != 0:
                    integral_image[y][x] += integral_image[y-1][x]
                if x != 0 and y != 0:
                    integral_image[y][x] -= integral_image[y-1][x-1]
        return integral_image