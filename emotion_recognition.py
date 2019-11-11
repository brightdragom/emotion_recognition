import cv2
import dlib
import model_utils
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\NanumGothic.ttf").get_name()
rc('font', family=font_name)
# 모델 로드
model=model_utils.createModel()
model.load_weights('fer.h5')
model.summary()
model.compile(loss=categorical_crossentropy,
              optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
              metrics=['accuracy'])

face_detector = dlib.get_frontal_face_detector()

while not cv2.VideoCapture(0).isOpened(): # 캠연결 안되있다고 가정함
    image = cv2.imread("test_image/angry.jpg")
    faces=face_detector(image)
    cropping_faces=[]
    for f in faces:
        cv2.rectangle(image, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
        cropping_faces.append(image[f.top():f.bottom(),f.left():f.right()])
    for face in cropping_faces:
        face=cv2.resize(face,(48,48),interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        placehorder=[]
        placehorder.append(face.reshape((48, 48, 1))/255)
        placehorder=np.array(placehorder)

        #predict_classes로 예측하기
        # 예측값 리스트
        # 0 = 화남, 1 = 혐오, 2 = 공포, 3 = 행복, 4 = 슬픈, 5 = 서프라이즈, 6 = 중립
        emotion_list=["Angry","Disgust","Fear","Happy","Sad","Surprise", "Neutral"]
        result=model.predict(placehorder, batch_size=32)
        result=result[0,] # 표정 예측값
        result=np.round(result,3) # 소수점 버리기
        result=result/sum(result) # 백분률
        result=np.round(result,2)*100 # 소수점 버리기
        result=result.astype('uint8')
        print(result)
        #이미지 디스플레이
        cv2.imshow("show",face)
        cv2.moveWindow("show", 200, 200)
        #테이블 디스플레이
        fig=plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ypos = np.arange(7)
        rects = plt.barh(ypos, result, align='center', height=0.35)
        plt.yticks(ypos, emotion_list)
        plt.xlabel('emotion percentile')
        plt.show()

        cv2.waitKey(0)

