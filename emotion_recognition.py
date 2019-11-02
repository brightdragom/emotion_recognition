import cv2
import numpy as np
import pandas

data = np.genfromtxt('fer2013.csv', delimiter=',', dtype=None)
labels = data[1:,0].astype(np.int32)
image_strings = data[1:,1]
usage = data[1:,2]

stripe_images = np.array([np.fromstring(image_string, np.uint8, sep=' ') for image_string in image_strings])

for stripe_image in stripe_images:
    image = stripe_image.reshape((48,48))
    cv2.imshow('hello',image)
    cv2.waitKey(0)