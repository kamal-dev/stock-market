import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras import utils as np_utils

data = pd.read_csv('data.csv')

train_x = data.iloc[:2000,1:6].values
train_y = data.iloc[:2000,8:14].values

test_x = data.iloc[2000:,1:6].values
test_y = data.iloc[2000:,8:14].values


def model_2_layer():
    model = Sequential()
    model.add(Dense(30, input_dim = 5, activation='relu'))
    model.add(Dense(12, activation = 'tanh'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=len(train_x))
    scores = model.evaluate(test_x, test_y)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))
    return (scores[1]*100)

def model_3_layer():
    model = Sequential()
    model.add(Dense(15, input_dim = 5, activation='relu'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(21, activation = 'relu'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=len(train_x))
    scores = model.evaluate(test_x, test_y)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))
    return (scores[1]*100)

def model_4_layer():
    model = Sequential()
    model.add(Dense(15, input_dim = 5, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(19, activation = 'relu'))
    model.add(Dense(21, activation = 'relu'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=len(train_x))
    scores = model.evaluate(test_x, test_y)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))
    return (scores[1]*100)

def model_5_layer():
    model = Sequential()
    model.add(Dense(15, input_dim = 5, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(19, activation = 'relu'))
    model.add(Dense(29, activation = 'relu'))
    model.add(Dense(21, activation = 'relu'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=len(train_x))
    scores = model.evaluate(test_x, test_y)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))
    return (scores[1]*100)

def model_6_layer():
    model = Sequential()
    model.add(Dense(15, input_dim = 5, activation='relu'))
    model.add(Dense(10, activation = 'relu'))
    model.add(Dense(19, activation = 'relu'))
    model.add(Dense(29, activation = 'relu'))
    model.add(Dense(14, activation = 'relu'))
    model.add(Dense(21, activation = 'relu'))
    model.add(Dense(6, activation = 'sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(train_x, train_y, epochs=100, batch_size=len(train_x))
    scores = model.evaluate(test_x, test_y)
    #print("\nAccuracy: %.2f%%" % (scores[1]*100))
    return (scores[1]*100)
    
score2 = model_2_layer()
score3 = model_3_layer()
score4 = model_4_layer()
score5 = model_5_layer()
score6 = model_6_layer()
print("2 Layer -  %.2f" % score2)
print("3 Layer - %.2f" % score3)
print("4 Layer - %.2f" % score4)
print("5 Layer - %.2f" % score5)
print("6 Layer - %.2f" % score6)



#dataset
#Train dataset:test dataset::60:40
#Neural network 4,5,6,7: which is giving high accuracy
#Same dataset in rattle - categorise the difference in price in 5 levels(high, very high, low, etc)
#

