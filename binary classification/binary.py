import pandas as pd 
import numpy as np
from keras import backend as K

from sklearn.preprocessing import LabelEncoder
from sklearn import model_selection
from keras.models import Sequential
from keras.layers import Dense

print(K.tensorflow_backend._get_available_gpus())

df = pd.read_csv('data.csv')
df = df.drop(['id', 'Unnamed: 32'], axis=1)

X = df.drop(['diagnosis'], axis=1)
X = np.array(X)

Y = df['diagnosis']
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)

input_dimensions = len(df.columns)-1

train_x, test_x, train_y, test_y = model_selection.train_test_split(X, Y, test_size=0.1, random_state=0)

model = Sequential()
model.add(Dense(10, input_dim=input_dimensions, kernel_initializer='normal', activation='relu'))
model.add(Dense(10, kernel_initializer='normal', activation='relu'))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(train_x, train_y, epochs=100, batch_size=5)

scores = model.evaluate(test_x, test_y)

print(scores)