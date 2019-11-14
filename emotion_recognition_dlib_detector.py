import cv2
import operator
import dlib
import numpy as np
from matplotlib import font_manager, rc
from keras.models import model_from_json

# 모델 로드
def load_model():
    json_file = open('fer.json', 'r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)
    # 가중치 업로드
    model.load_weights("fer.h5")
    return model

def emotion_recognition(model,videosrc):
    emotion_dict = {"Angry": 0, "Disgust": 0, "Fear": 0, "Happy": 0, "Sad": 0, "Surprise": 0, "Neutural": 0}
    emotion_list = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutural"]
    face_detector = dlib.get_frontal_face_detector()
    cap = videosrc
    while cap.isOpened():
        ret, full_size_image = cap.read()
        if ret==False:
            return emotion_dict
        full_size_image = cv2.resize(full_size_image, dsize=(500, 500), interpolation=cv2.INTER_AREA)  # 정방향 화면입력시
        # full_size_image = cv2.resize(full_size_image, dsize=(180,320), interpolation=cv2.INTER_AREA) #16:9 화면잊ㅂ력시
        # 얼굴영역찾기
        faces = face_detector(full_size_image, 1)
        for f in faces:  # 얼굴 찾아내기(여려명이 있을수 있음.)
            x = f.right()
            y = f.top()
            cv2.rectangle(full_size_image, (f.left(), f.top()), (f.right(), f.bottom()), (0, 0, 255), 2)
            cropped_face = full_size_image[f.top():f.bottom(), f.left():f.right()]
            cropped_face = cv2.cvtColor(cropped_face, cv2.COLOR_RGB2GRAY)
            cropped_face = np.expand_dims(np.expand_dims(cv2.resize(cropped_face, (48, 48)), -1), 0)
            cv2.normalize(cropped_face, cropped_face, alpha=0, beta=1, norm_type=cv2.NORM_L2, dtype=cv2.CV_32F)
            result = model.predict(cropped_face)
            emotion_str = emotion_list[int(np.argmax(result))]
            cv2.putText(full_size_image, emotion_str, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1,
                        cv2.LINE_AA)
            emotion_dict[emotion_str] += 1
    return emotion_dict



font_name = font_manager.FontProperties(fname="C:\\Windows\\Fonts\\NanumGothic.ttf").get_name()
rc('font', family=font_name)

if __name__=="__main__":
    model=load_model()
    print(emotion_recognition(model,cv2.VideoCapture('test_video/sad.mp4')))

