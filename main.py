# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 12:10:13 2019

@author: SWH
""" # 텐서플로우 경고 무시하는 코드
import keras
from keras.models import Sequential
import tensorflow as tf
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0


# 모듈 임포트
import os,random
import numpy as np
from keras.utils import np_utils
import keras.models as models
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from tensorflow.keras.utils import to_categorical
from keras.layers.convolutional import Conv1D
from keras.optimizers import Adam
from keras.regularizers import *
from keras.optimizers import adam
import matplotlib.pyplot as plt
import math
import pickle
import numpy as np
import pandas as pd
import cmath
import seaborn as sns

with open('RML2016.10a_dict.pkl', 'rb') as f:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    Xd = u.load()



data = input("<choice the vectors. a: IQ, b: phase_amp, c: FFT >") # 분류할 벡터 선택

snrs, mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])

X = []  
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))


x_all = np.vstack(X) # iq 벡터 데이터 저장 

x_all = x_all.swapaxes(2, 1) # cnn의 input으로 넣기 위해 축 변환

x_all = np.array(x_all)

x_r = x_all[:,:,0] # real 분리
x_i = x_all[:,:,1] # imagine 분리
     
x_r_flat = np.ravel(x_r, order='c') # phase vector를 계산하기 위해 1차원으로 폄
x_i_flat = np.ravel(x_i, order='c') # phase vector를 계산하기 위해 1차원으로 폄
x_phase = []

print("")
print("waiting...")
for i in range(len(x_i_flat)):
    x_phase.append(cmath.atan(x_r_flat[i]/x_i_flat[i])) # phase vector 계산

print("done.")    
x_phase = np.array(x_phase)

x_phase = x_phase.reshape(220000,128) 
x_amp = (x_r**2+x_i**2)**1/2 # amplitude vector 계산

x_fft_r = np.fft.fft(x_r) # fft vector 계산
x_fft_i = np.fft.fft(x_i) # fft vector 계산



x_phase_amp = np.dstack((x_phase, x_amp)) # phase와 amplitude vector 합침
x_fft = np.dstack((x_fft_r, x_fft_i)) # FFT vector 합침


np.random.seed(2016) # 랜덤으로 train을 뽑을 거기때문에 랜덤 시드 고정
n_examples = x_all.shape[0] # 총 데이터 갯수
n_train = n_examples * 0.67 # 67%만큼 train에 사용
train_idx = np.random.choice(220000, size=int(n_train), replace=False) # 전체에서 67%만큼 train index 뽑음
test_idx = list(set(range(0,n_examples))-set(train_idx)) # 전체에서 train 뽑은 나머지 (33%)


# a b c 선택시 분류에 사용할 vector가 달라짐
if data == 'a': 
    x_train = x_all[train_idx]
    x_test =  x_all[test_idx]
    print("IQ vector selected.")
if data == 'b':
    x_train = x_phase_amp[train_idx]
    x_test =  x_phase_amp[test_idx]
    print("phase and amp vector selected.")
if data == 'c':
    x_train = x_fft[train_idx]
    x_test =  x_fft[test_idx]
    print("FFT vector selected.")


# 원핫 인코딩으로 11가지 주파수를 변환
y_train = to_categorical(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
y_test = to_categorical(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))


print("train start.")  

# precision, recall, f1 score 함수

def recall_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

def precision_metric(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

def f1_metric(y_true, y_pred):
    precision = precision_metric(y_true, y_pred)
    recall = recall_metric(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))


from keras.models import Sequential

model = Sequential()

# 특정 snr일때 (18, 0, -8) 평가를 확인하기 위한 setting

test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
test_X_i = x_test[np.where(np.array(test_SNRs)==18)] #==18이 default.
test_Y_i = y_test[np.where(np.array(test_SNRs)==18)] #==18이 default.

# network 설정


model.add(Conv1D(filters=256, kernel_size=3, activation='relu', input_shape=(128,2)))
model.add(Dropout(0.6))
model.add(Conv1D(filters=80, kernel_size=3, activation='relu'))
model.add(Dropout(0.6))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.6))
model.add(Dense(11, activation='softmax'))
model.summary()
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy',  metrics=["accuracy", precision_metric, recall_metric, f1_metric])

history = model.fit(x_train, y_train, validation_data=(x_test, y_test), batch_size=1024, epochs=70, verbose=1)

loss, accuracy, precision, recall, f1 = model.evaluate(test_X_i, test_Y_i, verbose=1, batch_size=1024)



# accuracy, precision, recall, f1 score 값 출력
print(accuracy)
print(precision)
print(recall)
print(f1)


# confusision matrix 함수,아래 링크 참고함

#https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# snr -20부터 18까지의 정확도 출력,아래 링크 참고함
 #https://github.com/radioML/examples/blob/master/modulation_recognition/RML2016.10a_VTCNN2_example.ipynb


acc = {}
for snr in snrs:

    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = x_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = y_test[np.where(np.array(test_SNRs)==snr)]
    
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([11,11])
    confnorm = np.zeros([11,11])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,11):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    plt.figure()
    plot_confusion_matrix(confnorm, labels=mods, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Accuracy:", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

 # 해당 선택한 vector의 snr 종합 정확도 출력

plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Accuracy")
plt.title("Classification Accuracy in SNR")    


