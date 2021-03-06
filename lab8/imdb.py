
from __future__ import print_function
from keras.layers.advanced_activations import PReLU
from keras.layers import LSTM
from keras.layers import merge

import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Model
from keras.layers import Dense, Activation, Embedding, Flatten, Input, Dropout, Convolution1D, GlobalMaxPooling1D
from keras.datasets import imdb

max_features = 1000
maxlen = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features)
#print(len(X_train), 'train sequences')
#print(len(X_test), 'test sequences')

print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print('y_train: \n',y_train)
print('length ytrain: ', len(y_train))
print('y_test: \n',y_test)
print('length ytest: ', len(y_test))
print('@@@@@@@@@@@@@@@@@@@@@@@@@')
print (X_train[0])

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#print('X_train shape:', X_train.shape)
#print('X_train', X_train)
#print('X_test shape:', X_test.shape)
#print('X_test',X_test)

print('Build model...')


inputs = Input(shape=(maxlen,))
x = inputs
x = Embedding(max_features, 128, dropout=0.2)(x)


#--------- LSTM layer --------------
y = inputs
y = Embedding(max_features, 128, dropout=0.2)(y)
y = LSTM(128, dropout_W=0.2, dropout_U=0.2)(y)

#----Convolution Layer------
x=Convolution1D(nb_filter=64, filter_length=3, border_mode='valid',activation='relu', subsample_length=1)(x)
x=GlobalMaxPooling1D()(x)
#---------------------------
#x = Flatten()(x)


x = merge([x,y], mode='concat')
#-------- layer of 64 relu & dropout layer------
x = Dense(64)(x)
x = PReLU()(x) # Non-linearity
#x=Dropout(0.5)(x)

#-----------------------------------------------
x = Dense(1)(x)
predictions = Activation("sigmoid")(x)




model = Model(input=inputs, output=predictions)
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=15,
          validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test,
                            batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)