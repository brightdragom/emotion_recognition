import cv2
import dlib
import model_utils
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
# 모델 로드
#model=model_utils.createModel()
#model.load_weights('fer.h5')
#model.summary()
#model.compile(loss=categorical_crossentropy,
#              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
#              metrics=['accuracy'])

face_detector = dlib.get_frontal_face_detector()

while not cv2.VideoCapture(0).isOpened(): # 캠연결 안되있다고 가정함
    image = cv2.imread("test_image_2.jpg")
    faces=face_detector(image)
    cropping_faces=[]
    for f in faces:
        print(f)
        cv2.rectangle(image, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
        cropping_faces.append(image[f.top():f.bottom(),f.left():f.right()])
    for face in cropping_faces:
        cv2.imshow(" ",face)
        cv2.waitKey(0)

