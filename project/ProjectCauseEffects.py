from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import glob
import pandas as pd
from keras.layers.advanced_activations import PReLU
from keras.layers import LSTM
from keras.layers import merge
from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Input, Dropout, Convolution1D, GlobalMaxPooling1D, merge,LSTM

names=glob.glob('Data/*.txt')
data = []
maxlen=0
for f in names:
    df=pd.read_csv(f,delimiter= ' ', header=None).values #load all the files into an numpy ndarray with 2d (lines of file, pairs) separated by ' '. 
    df=df[:,0:2]
    
    if maxlen <  df.shape[0]:
        maxlen=df.shape[0]    #find the max length of all the files in order to do later zero padding to the same length
    data.append(df)#append all these numpy ndarray  into a list
data=np.array(data)#parshe the list into numpy ndarray type.

print('Choose training & testing set and make Zero Padding')
#########
p=0.85 ##the proportion of training set (0.8 means --> 80% training set and 20% testing set)
#########
X_train = sequence.pad_sequences(data[:int(len(data)*p)], maxlen=maxlen)
                                          
X_test = sequence.pad_sequences(data[(int(len(data)*p)+1):], maxlen=maxlen)

y_train=np.ones(len(X_train))
y_test=np.ones(len(X_test))
X_train_reverced = X_train[:,:,::-1] #we  reverse the pairs so as not to have causal direction
y_train_reverced = np.zeros(len(X_train)) #put targets zero that declare the lack of causal direction
merged_X_train= np.append(X_train,X_train_reverced,axis=0) #merge these data with causal directions and those without into one 
merged_y_train = np.append(y_train,y_train_reverced,axis=0) #merge the targets into one
X_train=merged_X_train
y_train=merged_y_train
X_test=X_test
y_test=y_test

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

print('Build model...')
batch_size = 32

inputs=Input(shape=(X_train.shape[1:]),)


x = inputs

x=Convolution1D(nb_filter=64, filter_length=3, border_mode='same')(x)
x= Activation("relu")(x)
x=GlobalMaxPooling1D()(x)
x=Dropout(0.25)(x)
x=Dense(64)(x)
x= Activation("relu")(x)
x=Dense(1)(x)
x= Activation("sigmoid")(x)
predictions = Activation("sigmoid")(x)

model = Model(input=inputs, output=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
print("Finish model...")

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))

print('Evaluate...')
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)