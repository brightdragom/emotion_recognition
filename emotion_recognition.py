import cv2
import operator
import dlib
import model_utils
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager, rc

def getEmotion(arr):
    # 0 = 화남, 1 = 혐오, 2 = 공포, 3 = 행복, 4 = 슬픈, 5 = 서프라이즈, 6 = 중립
    #        emotion_list=["Angry","Disgust","Fear","Happy","Sad","Surprise", "Neutral"]
    index, value = max(enumerate(result), key=operator.itemgetter(1))
    if index==0:
        return "angry"
    if index==1:
        return "Disgust"
    if index==2:
        return "Fear"
    if index==3:
        return "Happy"
    if index==4:
        return "Sad"
    if index==5:
        return "Surprise"
    if index==6:
        return "Natural"
    return "Natural"


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
cap=cv2.VideoCapture('test_video.mp4')
while cap.isOpened(): # 캠연결 안되있다고 가정함
    ret,full_size_image = cap.read()
    faces=face_detector(full_size_image,1)
    print(faces)
    cropping_faces=[]

    for f in faces: # 얼굴 찾아내기(여려명이 있을수 있음.)
        cv2.rectangle(full_size_image, (f.left(), f.top()), (f.right(), f.bottom()), (0,0,255), 2)
        cropping_faces.append(full_size_image[f.top():f.bottom(),f.left():f.right()])
    for face in cropping_faces: # 잘려진 얼굴영역 에 대한 표정 분석.
        face = cv2.resize(face,(48,48),interpolation=cv2.INTER_AREA)
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(face, (48, 48)), -1), 0)
        cv2.normalize(cropped_img, cropped_img, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        ##predict_classes로 예측하기
        emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
        result= model.predict(cropped_img)
        result=result[0,] # 표정 예측값
        result=np.round(result,3) # 소수점 버리기
        result=result/sum(result) # 백분률
        result=np.round(result,2)*100 # 소수점 버리기
        result=result.astype('uint8')
        #이미지 디스플레이
        retemotion=getEmotion(result)
        cv2.putText(full_size_image,retemotion,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),1,cv2.LINE_AA)
        cv2.imshow("show",full_size_image)
        cv2.moveWindow("show", 1000, 200)
        #테이블 디스플레이
        '''
        fig=plt.figure(figsize=(6,4))
        ax = fig.add_subplot(111)
        ypos = np.arange(7)
        rects = plt.barh(ypos, result, align='center', height=0.35)
        plt.yticks(ypos, emotion_list)
        plt.xlabel('emotion percentile')
        plt.show()
        '''

        if cv2.waitKey(10)==27:
            cv2.destroyAllWindows()


