import cv2
import numpy as np
from sklearn.preprocessing import OneHotEncoder

# csv 불러와서 리스트화 시키기.
data = np.genfromtxt('fer2013.csv', delimiter=',', dtype=None)
emotion_labels = data[1:,0].astype(np.int32)
image_strings = data[1:,1]
usage = data[1:,2]
stripe_images = np.array([np.fromstring(image_string, np.uint8, sep=' ') for image_string in image_strings])


# training Data와 Test Data 분리.
data= list(zip(stripe_images,emotion_labels,usage))
train_x=[]
train_y=[]
for image,label,usage in data:
    train_x.append(image.reshape((48,48,1)))
    train_y.append(label)


# 분류된 데이터셋을 학습을 위해 정규화
# 이미지는 48*48로 28709개
# 라벨은 1개를 가지고있지만, One-Hot Encoding을 통해 7개중 한개를 가르치는값으로 변경

#train_x= np.reshape(train_x,(-1,(48,48)))
#test_x= np.reshape(test_x,(-1,(48,48)))

train_x=np.array(train_x)

# 라벨 원핫 인코딩
train_ohEncoder=OneHotEncoder()
train_y= np.array(train_y).reshape(-1,1)
train_ohEncoder.fit(train_y)
train_y=train_ohEncoder.transform(train_y).toarray()

#print(train_x.shape)
#print(train_y.shape)

train_x = np.array(train_x).astype(np.float32)
train_y = np.array(train_y).astype(np.float32)
print(train_x)
np.save("train_x.npy",train_x)
np.save("train_y.npy",train_y)
