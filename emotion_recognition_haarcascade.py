import cv2
import operator
import dlib
import numpy as np
from matplotlib import font_manager, rc
from keras.models import model_from_json

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

x=None
y=None
font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\NanumGothic.ttf").get_name()
rc('font', family=font_name)
# 모델 로드
print("모델 로딩중")
json_file = open('fer.json', 'r')
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
# 가중치 업로드
model.load_weights("fer.h5")
print("모델 로딩완료")

emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
face_detector = dlib.get_frontal_face_detector()
cap=cv2.VideoCapture('test_video/surprise.mp4')
while cap.isOpened(): # 캠연결 안되있다고 가정함
    ret,full_size_image = cap.read()
    # 얼굴영역찾기
    faces=face_detector(full_size_image,1)
    gray = cv2.cvtColor(full_size_image, cv2.COLOR_RGB2GRAY)
    classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = classifier.detectMultiScale(gray, 1.3, 10)

    for (x,y,w,h) in faces: # 얼굴 찾아내기(여려명이 있을수 있음.)
        roi_gray = gray[y:y + h, x:x + w]
        cropped_face = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        cv2.normalize(cropped_face, cropped_face, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
        cv2.rectangle(full_size_image, (x, y), (x + w, y + h), (0, 255, 0), 1)

        ##predict_classes로 예측하기
        result=model.predict(cropped_face)
        print(result)
        #이미지 디스플레이
        #retemotion=getEmotion(result)
        #print(retemotion)
        cv2.putText(full_size_image, emotion_list[int(np.argmax(result))], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        #cv2.putText(full_size_image,retemotion,(20,20),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0, 0, 255),1,cv2.LINE_AA)
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


