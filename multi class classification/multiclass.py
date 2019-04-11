import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.utils.np_utils import to_categorical
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense

#Importing dataset
df = pd.read_csv('Iris.csv')
df = df.drop(['Id'], axis=1)

#Preprocessing
X = df.drop(['Species'], axis=1)
Y = df['Species']

X = np.array(X)

encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)

#Splitting into train and test data
train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size = 0.1, random_state=0)

#Building the model
input_dimension = len(df.columns)-1

model = Sequential()
model.add(Dense(8, input_dim = input_dimension, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(10, activation = 'relu'))
model.add(Dense(3, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#Training the model
model.fit(train_x, train_y, epochs = 50, batch_size = 10)

scores = model.evaluate(test_x, test_y)

print(scores[1])