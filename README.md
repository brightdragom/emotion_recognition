![python badge](https://img.shields.io/badge/Python-3.6.8-red)
![Tensorflow badge](https://img.shields.io/badge/Tensorflow-1.15.0-blue)
![Dlib badge](https://img.shields.io/badge/Dlib-19.18.0-green)
![OpenCV badge](https://img.shields.io/badge/OpenCV-4.1.1-blue)
![Dataset](https://img.shields.io/badge/Dataset-fer2013-yellow)

# face emotion recognition
CNN을 이용하여 얼굴 표정을 학습하고, Video 영상에서 얼굴 표정변화를 감지하고 이것을 딕셔너리화 하여 반환해주는 모듈입니다.
  
## 정확도
66% 

## Prerequisites
**버전에 유의하세요. 버전은 상기 뱃지에 표시되어있습니다.** 
```shell
pip install python
pip install tensorflow-gpu
pip install numpy
pip install opencv-python
pip install keras
pip install matplotlib
```
## Dlib 설치
anaconda 환경 설치 필요
```shell
conda install dlib
```

* anaconda가 설치 되지 않은 환경이라면, Dlib을 빌드하고 직접 인스톨 하셔야합니다. 

## 프로그램 작동
이 모듈은 2가지 버전이 있습니다
 1. dlib face detector Ver. - [emotion_recognition_dlib_detector.py](https://github.com/jaehyunup/emotion_recognition/blob/master/emotion_recognition_dlib_detector.py)
 2. openCV Haar detector Ver. - [emotion_recognition_haarcascade.py](https://github.com/jaehyunup/emotion_recognition/blob/master/emotion_recognition_haarcascade.py)  
 
자신의 환경에 맞게 두개중 하나를 실행시키시고  
imread Video 파라미터를 원하는 영상으로 바꿔준다면 얼굴이 있는 영상일때 emotion_dict로 반환됩니다.

 ![moduleimage](https://github.com/jaehyunup/emotion_recognition/blob/master/emotio_module.png)  
   


 
 