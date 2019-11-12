import sys, os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.regularizers import l2
from keras.losses import categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping, ModelCheckpoint
from keras.models import load_model
from keras.models import model_from_json
class model():
    def __init__(self):
        self.num_features = 64
        self.num_labels = 7
        self.batch_size = 64
        self.epochs = 100
        self.width= 48
        self.height = 48
        #desinging the CNN
        self.model = Sequential()
        self.model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', input_shape=(self.width, self.height, 1), data_format='channels_last', kernel_regularizer=l2(0.01)))
        self.model.add(Conv2D(self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Conv2D(2*2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(2*2*2*self.num_features, kernel_size=(3, 3), activation='relu', padding='same'))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Dropout(0.5))

        self.model.add(Flatten())

        self.model.add(Dense(2*2*2*self.num_features, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2*2*self.num_features, activation='relu'))
        self.model.add(Dropout(0.4))
        self.model.add(Dense(2*self.num_features, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(self.num_labels, activation='softmax'))

def model_trainning():
    x = np.load('train_x.npy')
    y = np.load('train_y.npy')
    # Z-score(표준 점수화)
    x -= np.mean(x,axis=0)
    x /= np.std(x,axis=0)

    # 트레이닝셋,테스트셋,검증셋 분리
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.1, random_state=41)

    # 테스트 데이터셋 저장
    np.save('test_x', X_test)
    np.save('test_y', y_test)

    num_features = 64
    num_labels = 7
    batch_size = 64
    epochs = 100
    width = 48
    height = 48
    # desinging the CNN
    model = Sequential()
    model.add(
        Conv2D(num_features, kernel_size=(3, 3), activation='relu', input_shape=(width, height, 1),
               data_format='channels_last', kernel_regularizer=l2(0.01)))
    model.add(Conv2D(num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(Conv2D(2 * 2 * 2 * num_features, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.5))

    model.add(Flatten())

    model.add(Dense(2 * 2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * 2 * num_features, activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(2 * num_features, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(num_labels, activation='softmax'))


    # Compliling the model with adam optimixer and categorical crossentropy loss
    model.compile(loss=categorical_crossentropy,
                  optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7),
                  metrics=['accuracy'])

    # training the model
    model.fit(np.array(X_train), np.array(y_train),
              batch_size=batch_size,
              epochs=epochs,
              verbose=1,
              validation_data=(np.array(X_valid), np.array(y_valid)),
              shuffle=True)

    # saving the  model to be used later
    fer_json = model.to_json()
    with open("fer.json", "w") as json_file:
        json_file.write(fer_json)
    model.save_weights("fer.h5")
    print("Saved model to disk")

model_trainning()