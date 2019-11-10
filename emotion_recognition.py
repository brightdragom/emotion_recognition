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
train_data=[]
test_data=[]
for idx,value in enumerate(data):
    if value[2].decode('utf-8') !="Training":
        train_data=data[0:idx]
        test_data=data[idx:]
        break;
train_x=[]
train_y=[]
test_x=[]
test_y=[]

for image,label,usage in train_data:
    train_x.append(image)
    train_y.append(label)
for image,label,usage in test_data:
    test_x.append(image)
    test_y.append(label)

# 분류된 데이터셋을 학습을 위해 정규화
# 이미지는 64*64로 28709개의 2304픽셀을 가진다,즉 shape = (28709,2304)
# 라벨은 1개를 가지고있지만, One-Hot Encoding을 통해 7개중 한개를 가르치는값으로 변경
# 즉. shape= (28709,7)
train_x= np.reshape(train_x,(-1,2304))
# 라벨 원핫 인코딩
ohEncoder=OneHotEncoder()
train_y= np.array(train_y).reshape(-1,1)
ohEncoder.fit(train_y)
train_y=ohEncoder.transform(train_y).toarray()

print(train_x.shape)
print(train_y.shape)




img = cv2.imread('test_face.png', cv2.IMREAD_GRAYSCALE)
img=cv2.resize(img,(48,48),interpolation=cv2.INTER_AREA)
cv2.imshow(",",img)
cv2.waitKey(0)
