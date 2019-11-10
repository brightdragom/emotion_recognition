import cv2
import tensorflow as tf
leaning_rate



img = cv2.imread('test_face.png', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
cv2.imshow(",",img)
cv2.waitKey(0)
